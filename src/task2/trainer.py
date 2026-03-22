import os
import json
import math
import shutil
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.dataset import (
    build_plain_vocab,
    get_lm_dataloaders,
    PAD_IDX,
)
from src.utils.checkpoints import save_checkpoint, load_checkpoint
from src.utils.hf_wandb import init_wandb, log_wandb, finish_wandb, save_and_push, load_from_hub, push_to_hub
from src.task2.models import build_bilstm, build_ssm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(cross_entropy_loss)."""
    return math.exp(min(loss, 100))  # cap to avoid overflow


def train_epoch_mlm(model, loader, optimizer, criterion, device, clip):
    """Training epoch for BiLSTM MLM."""
    model.train()
    total_loss = 0

    for input_ids, labels in tqdm(loader, desc="  train", leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)                       # (batch, seq_len, vocab)

        # Only compute loss on masked positions (labels != -100)
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch_nwp(model, loader, optimizer, criterion, device, clip):
    """Training epoch for SSM NWP."""
    model.train()
    total_loss = 0

    for input_ids, target_ids in tqdm(loader, desc="  train", leave=False):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)                       # (batch, seq_len, vocab)

        logits_flat = logits.view(-1, logits.shape[-1])
        targets_flat = target_ids.view(-1)
        loss = criterion(logits_flat, targets_flat)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_mlm(model, loader, criterion, device):
    """Evaluate BiLSTM MLM."""
    model.eval()
    total_loss = 0

    for input_ids, labels in tqdm(loader, desc="  eval ", leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, compute_perplexity(avg_loss)


@torch.no_grad()
def evaluate_nwp(model, loader, criterion, device):
    """Evaluate SSM NWP."""
    model.eval()
    total_loss = 0

    for input_ids, target_ids in tqdm(loader, desc="  eval ", leave=False):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.shape[-1])
        targets_flat = target_ids.view(-1)
        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, compute_perplexity(avg_loss)


def save_results(cfg, metrics, model_type):
    """Save metrics to results file and copy to kaggle output."""
    results_file = cfg["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        f.write(f"Model      : {model_type.upper()}\n")
        f.write(f"Task       : {'MLM' if model_type == 'bilstm' else 'NWP'}\n")
        f.write(f"val_loss   : {metrics['val_loss']:.4f}\n")
        f.write(f"perplexity : {metrics['perplexity']:.4f}\n")

    print(f"Results saved to {results_file}")

    if os.path.exists("/kaggle/working"):
        dest = os.path.join("/kaggle/working", os.path.basename(results_file))
        shutil.copy(results_file, dest)
        print(f"Results copied to {dest}")


def push_to_hf(model, cfg, vocab=None):
    """Push model weights and vocab to HuggingFace."""
    if not cfg["output"].get("huggingface_repo"):
        return

    task_name = cfg["logging"]["wandb_run_name"]
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    filename = f"{task_name}_best.pt"
    save_and_push(
        model,
        cfg["output"]["huggingface_repo"],
        filename=filename,
        local_dir=cfg["output"]["checkpoint_dir"],
    )
    print(f"Pushed {filename} to HuggingFace")

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


def train(config_path: str, model_type: str):
    """Shared training function for both BiLSTM and SSM."""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Vocab ──────────────────────────────────────────────────────────────────
    vocab = build_plain_vocab(cfg["data"]["plain_file"])
    print(f"Vocab size: {len(vocab)}")

    # ── Data ───────────────────────────────────────────────────────────────────
    task = "mlm" if model_type == "bilstm" else "nwp"
    train_loader, val_loader, test_loader = get_lm_dataloaders(cfg, vocab, task)
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    if model_type == "bilstm":
        model = build_bilstm(cfg, len(vocab)).to(device)
    else:
        model = build_ssm(cfg, len(vocab)).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    # ── Training setup ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # MLM uses -100 to ignore non-masked positions
    if model_type == "bilstm":
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── WandB ──────────────────────────────────────────────────────────────────
    init_wandb(
        project=cfg["logging"]["wandb_project"],
        config=cfg,
        name=cfg["logging"]["wandb_run_name"],
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")

        if model_type == "bilstm":
            train_loss = train_epoch_mlm(
                model, train_loader, optimizer, criterion, device,
                cfg["training"]["clip_grad_norm"],
            )
            val_loss, val_ppl = evaluate_mlm(model, val_loader, criterion, device)
        else:
            train_loss = train_epoch_nwp(
                model, train_loader, optimizer, criterion, device,
                cfg["training"]["clip_grad_norm"],
            )
            val_loss, val_ppl = evaluate_nwp(model, val_loader, criterion, device)

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

    # Load from HuggingFace or local
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
        val_loss, val_ppl = evaluate_mlm(model, test_loader, criterion, device)
    else:
        val_loss, val_ppl = evaluate_nwp(model, test_loader, criterion, device)

    print("\n── Test Results ──────────────────────────────")
    print(f"  loss      : {val_loss:.4f}")
    print(f"  perplexity: {val_ppl:.4f}")

    save_results(cfg, {"val_loss": val_loss, "perplexity": val_ppl}, model_type)
    return {"val_loss": val_loss, "perplexity": val_ppl}
