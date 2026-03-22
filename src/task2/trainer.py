import os
import json
import math
import random
import shutil
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.dataset import (
    build_plain_vocab,
    get_lm_dataloaders,
    PAD_IDX,
    MASK_IDX,
)
from src.utils.checkpoints import save_checkpoint, load_checkpoint
from src.utils.hf_wandb import (
    init_wandb, log_wandb, finish_wandb,
    save_and_push, load_from_hub, push_to_hub,
)
from src.task2.models import build_bilstm, build_ssm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 100))


# ── Training epochs ───────────────────────────────────────────────────────────
def train_epoch_mlm(model, loader, optimizer, criterion, device, clip):
    model.train()
    total_loss = 0
    for input_ids, labels in tqdm(loader, desc="  train", leave=False):
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_nwp(model, loader, optimizer, criterion, device, clip):
    model.train()
    total_loss = 0
    for input_ids, target_ids in tqdm(loader, desc="  train", leave=False):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_mlm(model, loader, criterion, device, vocab, num_samples=5):
    model.eval()
    total_loss = 0
    samples = []

    for input_ids, labels in tqdm(loader, desc="  eval ", leave=False):
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        total_loss += loss.item()

        # Collect sample predictions
        if len(samples) < num_samples:
            preds = logits.argmax(dim=-1)               # (batch, seq_len)
            for b in range(min(input_ids.shape[0], num_samples - len(samples))):
                inp_tokens  = input_ids[b].tolist()
                label_tokens = labels[b].tolist()
                pred_tokens  = preds[b].tolist()

                inp_str  = "".join(vocab.idx2token.get(i, "") for i in inp_tokens
                                   if i not in (0, 1, 2))
                # Show what was masked vs predicted
                masked_positions = [i for i, l in enumerate(label_tokens) if l != -100]
                target_chars = "".join(vocab.idx2token.get(label_tokens[i], "?")
                                       for i in masked_positions)
                pred_chars   = "".join(vocab.idx2token.get(pred_tokens[i], "?")
                                       for i in masked_positions)
                samples.append((inp_str[:80], target_chars[:40], pred_chars[:40]))

    avg_loss = total_loss / len(loader)
    return avg_loss, compute_perplexity(avg_loss), samples


@torch.no_grad()
def evaluate_nwp(model, loader, criterion, device, vocab, num_samples=5):
    model.eval()
    total_loss = 0
    samples = []

    for input_ids, target_ids in tqdm(loader, desc="  eval ", leave=False):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        total_loss += loss.item()

        # Collect sample predictions
        if len(samples) < num_samples:
            preds = logits.argmax(dim=-1)
            for b in range(min(input_ids.shape[0], num_samples - len(samples))):
                inp_str = "".join(vocab.idx2token.get(i, "") for i in input_ids[b].tolist()
                                  if i not in (0, 1, 2))
                tgt_str = "".join(vocab.idx2token.get(i, "") for i in target_ids[b].tolist()
                                  if i not in (0, 1, 2))
                prd_str = "".join(vocab.idx2token.get(i, "") for i in preds[b].tolist()
                                  if i not in (0, 1, 2))
                samples.append((inp_str[:80], tgt_str[:80], prd_str[:80]))

    avg_loss = total_loss / len(loader)
    return avg_loss, compute_perplexity(avg_loss), samples


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(cfg, metrics, model_type, samples):
    results_file = cfg["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    task = "MLM" if model_type == "bilstm" else "NWP"
    with open(results_file, "w") as f:
        f.write(f"Model      : {model_type.upper()}\n")
        f.write(f"Task       : {task}\n")
        f.write(f"val_loss   : {metrics['val_loss']:.4f}\n")
        f.write(f"perplexity : {metrics['perplexity']:.4f}\n\n")
        f.write("── Sample Predictions ──\n")
        if model_type == "bilstm":
            for inp, target, pred in samples:
                f.write(f"INPUT  : {inp}\n")
                f.write(f"MASKED : {target}\n")
                f.write(f"PREDICT: {pred}\n\n")
        else:
            for inp, target, pred in samples:
                f.write(f"INPUT  : {inp}\n")
                f.write(f"TARGET : {target}\n")
                f.write(f"PREDICT: {pred}\n\n")

    print(f"Results saved to {results_file}")

    if os.path.exists("/kaggle/working"):
        dest = os.path.join("/kaggle/working", os.path.basename(results_file))
        shutil.copy(results_file, dest)
        print(f"Results copied to {dest}")


# ── HuggingFace push ──────────────────────────────────────────────────────────
def push_to_hf(model, cfg, vocab=None):
    if not cfg["output"].get("huggingface_repo"):
        return

    task_name = cfg["logging"]["wandb_run_name"]
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    # Push model weights
    filename = f"{task_name}_best.pt"
    save_and_push(
        model,
        cfg["output"]["huggingface_repo"],
        filename=filename,
        local_dir=cfg["output"]["checkpoint_dir"],
    )
    print(f"Pushed {filename} to HuggingFace")

    # Push vocab
    if vocab is not None:
        vocab_path = os.path.join(
            cfg["output"]["checkpoint_dir"], f"{task_name}_vocab.json"
        )
        with open(vocab_path, "w") as f:
            json.dump({"plain": vocab.token2idx}, f, indent=2)
        push_to_hub(
            vocab_path,
            cfg["output"]["huggingface_repo"],
            f"{task_name}_vocab.json",
        )
        print(f"Pushed {task_name}_vocab.json to HuggingFace")


# ── Main train function ───────────────────────────────────────────────────────
def train(config_path: str, model_type: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Vocab
    vocab = build_plain_vocab(cfg["data"]["plain_file"])
    print(f"Vocab size: {len(vocab)}")

    # Data
    task = "mlm" if model_type == "bilstm" else "nwp"
    train_loader, val_loader, test_loader = get_lm_dataloaders(cfg, vocab, task)
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Model
    if model_type == "bilstm":
        model = build_bilstm(cfg, len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        model = build_ssm(cfg, len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # WandB
    init_wandb(
        project=cfg["logging"]["wandb_project"],
        config=cfg,
        name=cfg["logging"]["wandb_run_name"],
    )

    best_val_loss = float("inf")
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")

        if model_type == "bilstm":
            train_loss = train_epoch_mlm(
                model, train_loader, optimizer, criterion, device,
                cfg["training"]["clip_grad_norm"],
            )
            val_loss, val_ppl, _ = evaluate_mlm(model, val_loader, criterion, device, vocab)
        else:
            train_loss = train_epoch_nwp(
                model, train_loader, optimizer, criterion, device,
                cfg["training"]["clip_grad_norm"],
            )
            val_loss, val_ppl, _ = evaluate_nwp(model, val_loader, criterion, device, vocab)

        scheduler.step(val_loss)
        train_ppl = compute_perplexity(train_loss)

        print(f"  train loss: {train_loss:.4f} | train ppl: {train_ppl:.2f}")
        print(f"  val   loss: {val_loss:.4f} | val   ppl: {val_ppl:.2f}")

        log_wandb({
            "train/loss": train_loss,
            "train/perplexity": train_ppl,
            "val/loss": val_loss,
            "val/perplexity": val_ppl,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                cfg["output"]["checkpoint_file"],
            )
            print(f"  ✓ saved checkpoint (val_loss={best_val_loss:.4f})")

    finish_wandb()
    push_to_hf(model, cfg, vocab)
    return model, vocab, test_loader, cfg


# ── Evaluate and save ─────────────────────────────────────────────────────────
def evaluate_and_save(config_path: str, model_type: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_plain_vocab(cfg["data"]["plain_file"])
    task = "mlm" if model_type == "bilstm" else "nwp"
    _, _, test_loader = get_lm_dataloaders(cfg, vocab, task)

    if model_type == "bilstm":
        model = build_bilstm(cfg, len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        model = build_ssm(cfg, len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Load checkpoint
    if cfg["output"].get("huggingface_repo"):
        task_name = cfg["logging"]["wandb_run_name"]
        load_from_hub(
            model,
            cfg["output"]["huggingface_repo"],
            filename=f"{task_name}_best.pt",
            device=str(device),
        )
    else:
        load_checkpoint(cfg["output"]["checkpoint_file"], model, device=str(device))

    if model_type == "bilstm":
        val_loss, val_ppl, samples = evaluate_mlm(
            model, test_loader, criterion, device, vocab, num_samples=10
        )
    else:
        val_loss, val_ppl, samples = evaluate_nwp(
            model, test_loader, criterion, device, vocab, num_samples=10
        )

    print("\n── Test Results ──────────────────────────────")
    print(f"  val_loss   : {val_loss:.4f}")
    print(f"  perplexity : {val_ppl:.4f}")

    metrics = {"val_loss": val_loss, "perplexity": val_ppl}
    save_results(cfg, metrics, model_type, samples)
    return metrics
