import os
import json
import shutil
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.dataset import (
    build_plain_vocab,
    build_cipher_vocab,
    get_cipher_dataloaders,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX,
)
from src.utils.checkpoints import save_checkpoint, load_checkpoint
from src.utils.hf_wandb import init_wandb, log_wandb, finish_wandb, save_and_push, load_from_hub, push_to_hub
from src.task1.models import build_seq2seq
from src.task1.metrics import (
    char_accuracy,
    word_accuracy,
    avg_levenshtein,
    decode_predictions,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_vocab(cfg: dict, task_name: str):
    """Load vocab from HuggingFace or rebuild from data."""
    if cfg["output"].get("huggingface_repo"):
        try:
            from huggingface_hub import hf_hub_download
            vocab_path = hf_hub_download(
                repo_id=cfg["output"]["huggingface_repo"],
                filename=f"{task_name}_vocab.json",
                local_dir=cfg["output"]["checkpoint_dir"],
            )
            with open(vocab_path) as f:
                vocab_data = json.load(f)

            plain_vocab = build_plain_vocab(cfg["data"]["plain_file"])
            plain_vocab.token2idx = vocab_data["plain"]
            plain_vocab.idx2token = {int(v): k for k, v in vocab_data["plain"].items()}

            cipher_vocab = build_cipher_vocab()
            cipher_vocab.token2idx = vocab_data["cipher"]
            cipher_vocab.idx2token = {int(v): k for k, v in vocab_data["cipher"].items()}

            print("Loaded vocab from HuggingFace")
            return plain_vocab, cipher_vocab
        except Exception as e:
            print(f"Could not load vocab from HuggingFace: {e}, rebuilding from data")

    plain_vocab = build_plain_vocab(cfg["data"]["plain_file"])
    cipher_vocab = build_cipher_vocab()
    return plain_vocab, cipher_vocab


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, clip):
    model.train()
    total_loss = 0

    for cipher, plain in tqdm(loader, desc="  train", leave=False):
        cipher, plain = cipher.to(device), plain.to(device)

        optimizer.zero_grad()
        output = model(cipher, plain, teacher_forcing_ratio)
        output = output[:, 1:, :].contiguous().view(-1, output.shape[-1])
        target = plain[:, 1:].contiguous().view(-1)

        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, plain_vocab):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for cipher, plain in tqdm(loader, desc="  eval ", leave=False):
        cipher, plain = cipher.to(device), plain.to(device)

        output = model(cipher, plain, teacher_forcing_ratio=0.0)
        out_flat = output[:, 1:, :].contiguous().view(-1, output.shape[-1])
        tgt_flat = plain[:, 1:].contiguous().view(-1)
        loss = criterion(out_flat, tgt_flat)
        total_loss += loss.item()

        pred_ids = model.decode_greedy(cipher, SOS_IDX, EOS_IDX, plain.shape[1])
        preds = decode_predictions(pred_ids, plain_vocab, EOS_IDX, PAD_IDX)
        targets = decode_predictions(plain[:, 1:], plain_vocab, EOS_IDX, PAD_IDX)
        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = total_loss / len(loader)
    metrics = {
        "loss": avg_loss,
        "char_acc": char_accuracy(all_preds, all_targets),
        "word_acc": word_accuracy(all_preds, all_targets),
        "levenshtein": avg_levenshtein(all_preds, all_targets),
    }
    return metrics, all_preds, all_targets


def save_results(cfg, metrics, preds, targets):
    """Save metrics and sample predictions to results file and copy to kaggle output."""
    results_file = cfg["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        f.write(f"Model      : {cfg['model']['type'].upper()}\n")
        f.write(f"char_acc   : {metrics['char_acc']:.4f}\n")
        f.write(f"word_acc   : {metrics['word_acc']:.4f}\n")
        f.write(f"levenshtein: {metrics['levenshtein']:.2f}\n\n")
        f.write("── Sample Predictions ──\n")
        for pred, tgt in zip(preds[:20], targets[:20]):
            f.write(f"TARGET : {tgt}\n")
            f.write(f"PREDICT: {pred}\n\n")

    print(f"Results saved to {results_file}")

    # Auto-copy to /kaggle/working for easy download
    if os.path.exists("/kaggle/working"):
        dest = os.path.join("/kaggle/working", os.path.basename(results_file))
        shutil.copy(results_file, dest)
        print(f"Results copied to {dest}")


def push_to_hf(model, cfg, plain_vocab=None, cipher_vocab=None):
    """Push model weights and vocab to HuggingFace."""
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

    # Push vocab so results are reproducible
    if plain_vocab is not None and cipher_vocab is not None:
        vocab_path = os.path.join(
            cfg["output"]["checkpoint_dir"], f"{task_name}_vocab.json"
        )
        with open(vocab_path, "w") as f:
            json.dump({
                "plain": plain_vocab.token2idx,
                "cipher": cipher_vocab.token2idx,
            }, f, indent=2)
        push_to_hub(
            vocab_path,
            cfg["output"]["huggingface_repo"],
            f"{task_name}_vocab.json",
        )
        print(f"Pushed {task_name}_vocab.json to HuggingFace")


def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Vocab ──────────────────────────────────────────────────────────────────
    plain_vocab = build_plain_vocab(cfg["data"]["plain_file"])
    cipher_vocab = build_cipher_vocab()
    print(f"Plain vocab size : {len(plain_vocab)}")
    print(f"Cipher vocab size: {len(cipher_vocab)}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_cipher_dataloaders(
        cfg, cipher_vocab, plain_vocab
    )
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_seq2seq(cfg, len(cipher_vocab), len(plain_vocab)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    # ── Training setup ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── WandB ──────────────────────────────────────────────────────────────────
    run = init_wandb(
        project=cfg["logging"]["wandb_project"],
        config=cfg,
        name=cfg["logging"]["wandb_run_name"],
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")

        # Decay teacher forcing ratio each epoch, min 0.3
        tf_ratio = max(0.3, cfg["training"]["teacher_forcing_ratio"] - 0.02 * (epoch - 1))

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            tf_ratio,
            cfg["training"]["clip_grad_norm"],
        )

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, plain_vocab)
        scheduler.step(val_metrics["loss"])

        print(f"  train loss: {train_loss:.4f} | tf_ratio: {tf_ratio:.2f}")
        print(f"  val   loss: {val_metrics['loss']:.4f} | "
              f"char_acc: {val_metrics['char_acc']:.4f} | "
              f"word_acc: {val_metrics['word_acc']:.4f} | "
              f"lev: {val_metrics['levenshtein']:.2f}")

        log_wandb({
            "train/loss": train_loss,
            "val/loss": val_metrics["loss"],
            "val/char_acc": val_metrics["char_acc"],
            "val/word_acc": val_metrics["word_acc"],
            "val/levenshtein": val_metrics["levenshtein"],
            "lr": optimizer.param_groups[0]["lr"],
            "teacher_forcing_ratio": tf_ratio,
        }, step=epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics["loss"],
                cfg["output"]["checkpoint_file"],
            )
            print(f"  ✓ saved checkpoint (val_loss={best_val_loss:.4f})")

    finish_wandb()
    push_to_hf(model, cfg, plain_vocab, cipher_vocab)

    return model, plain_vocab, cipher_vocab, test_loader, cfg


def evaluate_and_save(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_name = cfg["logging"]["wandb_run_name"]

    # Load vocab from HF or rebuild
    plain_vocab, cipher_vocab = load_vocab(cfg, task_name)
    _, _, test_loader = get_cipher_dataloaders(cfg, cipher_vocab, plain_vocab)

    model = build_seq2seq(cfg, len(cipher_vocab), len(plain_vocab)).to(device)

    # Load weights from HuggingFace or local checkpoint
    if cfg["output"].get("huggingface_repo"):
        load_from_hub(
            model,
            cfg["output"]["huggingface_repo"],
            filename=f"{task_name}_best.pt",
            device=str(device),
        )
    else:
        load_checkpoint(cfg["output"]["checkpoint_file"], model, device=str(device))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    metrics, preds, targets = evaluate(model, test_loader, criterion, device, plain_vocab)

    print("\n── Test Results ──────────────────────────────")
    print(f"  char_acc   : {metrics['char_acc']:.4f}")
    print(f"  word_acc   : {metrics['word_acc']:.4f}")
    print(f"  levenshtein: {metrics['levenshtein']:.2f}")

    save_results(cfg, metrics, preds, targets)
    return metrics
