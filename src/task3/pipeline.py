import os
import json
import shutil
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.dataset import (
    build_plain_vocab,
    build_cipher_vocab,
    Vocab,
    PAD_IDX, SOS_IDX, EOS_IDX, MASK_IDX,
    tokenize_cipher, tokenize_plain,
)
from src.utils.checkpoints import load_checkpoint
from src.utils.hf_wandb import load_from_hub, init_wandb, log_wandb, finish_wandb
from src.task1.models import build_seq2seq
from src.task1.metrics import (
    char_accuracy, word_accuracy, avg_levenshtein, decode_predictions
)
from src.task2.models import build_bilstm, build_ssm


# ── BLEU and ROUGE ────────────────────────────────────────────────────────────
def compute_bleu(predictions, targets, max_n=4):
    """Corpus-level BLEU score (character-level n-grams)."""
    from collections import Counter
    import math

    def get_ngrams(seq, n):
        return Counter([seq[i:i+n] for i in range(len(seq)-n+1)])

    clipped_counts = {n: 0 for n in range(1, max_n+1)}
    total_counts   = {n: 0 for n in range(1, max_n+1)}
    pred_len = ref_len = 0

    for pred, ref in zip(predictions, targets):
        pred_len += len(pred)
        ref_len  += len(ref)
        for n in range(1, max_n+1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams  = get_ngrams(ref, n)
            for ngram, cnt in pred_ngrams.items():
                clipped_counts[n] += min(cnt, ref_ngrams.get(ngram, 0))
            total_counts[n] += max(len(pred) - n + 1, 0)

    if pred_len == 0:
        return 0.0

    bp = 1.0 if pred_len >= ref_len else math.exp(1 - ref_len / pred_len)
    log_bleu = 0.0
    for n in range(1, max_n+1):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_bleu += math.log(clipped_counts[n] / total_counts[n])

    return bp * math.exp(log_bleu / max_n)


def compute_rouge_l(predictions, targets):
    """Average ROUGE-L (character-level LCS) score."""
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n+1) for _ in range(2)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[i%2][j] = dp[(i-1)%2][j-1] + 1
                else:
                    dp[i%2][j] = max(dp[(i-1)%2][j], dp[i%2][j-1])
        return dp[m%2][n]

    scores = []
    for pred, ref in zip(predictions, targets):
        if len(ref) == 0:
            scores.append(0.0)
            continue
        lcs = lcs_length(pred, ref)
        precision = lcs / len(pred) if pred else 0
        recall    = lcs / len(ref)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


# ── Vocab loading ─────────────────────────────────────────────────────────────
def load_task1_vocab(cfg):
    """Load Task 1 vocab from HuggingFace or rebuild."""
    hf_repo = cfg["decryption"].get("huggingface_repo", "")
    plain_vocab  = build_plain_vocab(cfg["data"]["plain_file"])
    cipher_vocab = build_cipher_vocab()

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=hf_repo,
                filename="task1_lstm_vocab.json",
                local_dir="checkpoints/task1/lstm",
            )
            with open(path) as f:
                vocab_data = json.load(f)
            plain_vocab.token2idx  = vocab_data["plain"]
            plain_vocab.idx2token  = {int(v): k for k, v in vocab_data["plain"].items()}
            cipher_vocab.token2idx = vocab_data["cipher"]
            cipher_vocab.idx2token = {int(v): k for k, v in vocab_data["cipher"].items()}
            print("Loaded Task 1 vocab from HuggingFace")
        except Exception as e:
            print(f"Could not load Task 1 vocab from HF: {e}, rebuilding")

    return plain_vocab, cipher_vocab


def load_task2_vocab(cfg, model_type):
    """Load Task 2 vocab from HuggingFace or rebuild."""
    hf_repo = cfg["language_model"].get("huggingface_repo", "")
    vocab = build_plain_vocab(cfg["data"]["plain_file"])

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            filename = f"task2_{model_type}_vocab.json"
            path = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                local_dir=f"checkpoints/task2/{model_type}",
            )
            with open(path) as f:
                vocab_data = json.load(f)
            vocab.token2idx = vocab_data["plain"]
            vocab.idx2token = {int(v): k for k, v in vocab_data["plain"].items()}
            print(f"Loaded Task 2 {model_type} vocab from HuggingFace")
        except Exception as e:
            print(f"Could not load Task 2 vocab from HF: {e}, rebuilding")

    return vocab


# ── Model loading ─────────────────────────────────────────────────────────────
def load_decryption_model(cfg, plain_vocab, cipher_vocab, device):
    """Load Task 1 LSTM from HuggingFace or local checkpoint."""
    # Build config for task1 model
    task1_cfg = {
        "model": {
            "type": "lstm",
            "embedding_dim": 64,
            "hidden_dim": 512,
            "num_layers": 2,
            "dropout": 0.3,
        }
    }
    model = build_seq2seq(task1_cfg, len(cipher_vocab), len(plain_vocab)).to(device)

    hf_repo = cfg["decryption"].get("huggingface_repo", "")
    if hf_repo:
        load_from_hub(model, hf_repo, "task1_lstm_best.pt", device=str(device))
    else:
        load_checkpoint(cfg["decryption"]["checkpoint"], model, device=str(device))

    model.eval()
    print("Loaded Task 1 LSTM decryption model")
    return model


def load_lm_model(cfg, vocab, model_type, device):
    """Load Task 2 BiLSTM or SSM from HuggingFace or local checkpoint."""
    task2_cfg = {
        "model": {
            "type": model_type,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.3 if model_type == "bilstm" else 0.2,
            "state_dim": 64,
        }
    }

    if model_type == "bilstm":
        model = build_bilstm(task2_cfg, len(vocab)).to(device)
    else:
        model = build_ssm(task2_cfg, len(vocab)).to(device)

    hf_repo = cfg["language_model"].get("huggingface_repo", "")
    if hf_repo:
        load_from_hub(model, hf_repo, f"task2_{model_type}_best.pt", device=str(device))
    else:
        load_checkpoint(cfg["language_model"]["checkpoint"], model, device=str(device))

    model.eval()
    print(f"Loaded Task 2 {model_type} language model")
    return model


# ── Decryption ────────────────────────────────────────────────────────────────
@torch.no_grad()
def decrypt_lines(model, lines, cipher_vocab, plain_vocab, max_cipher_len, max_plain_len, device, batch_size=64):
    """Decrypt a list of cipher lines using the LSTM model."""
    all_preds = []
    all_confidences = []

    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        # Tokenize and encode
        encoded = []
        for line in batch_lines:
            tokens = tokenize_cipher(line.strip())[:max_cipher_len-2]
            ids = [SOS_IDX] + cipher_vocab.encode(tokens) + [EOS_IDX]
            encoded.append(torch.tensor(ids, dtype=torch.long))

        # Pad
        from torch.nn.utils.rnn import pad_sequence
        cipher_batch = pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX).to(device)

        # Decode
        pred_ids, conf_scores = model.decode_greedy(cipher_batch, SOS_IDX, EOS_IDX, max_plain_len)
        preds = decode_predictions(pred_ids, plain_vocab, EOS_IDX, PAD_IDX)
        all_preds.extend(preds)
        all_confidences.extend(conf_scores.cpu().tolist())

    return all_preds, all_confidences


# ── LM Correction ─────────────────────────────────────────────────────────────
@torch.no_grad()
def correct_with_bilstm(bilstm_model, predictions, vocab, device, confidence_threshold=0.5):
    """
    Use BiLSTM MLM to correct low-confidence characters.
    Strategy: mask each character one at a time, use BiLSTM to predict it,
    replace if BiLSTM is more confident than original.
    For efficiency, we only correct short sequences character by character.
    """
    corrected = []

    for pred in predictions:
        if len(pred) == 0:
            corrected.append(pred)
            continue

        tokens = list(pred)[:126]  # max seq len - 2
        ids = [SOS_IDX] + vocab.encode(tokens) + [EOS_IDX]
        input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        # Get BiLSTM predictions for all positions
        logits = bilstm_model(input_tensor)             # (1, seq_len, vocab_size)
        probs  = torch.softmax(logits, dim=-1)          # (1, seq_len, vocab_size)

        # For each non-special token position
        new_tokens = tokens.copy()
        for pos in range(len(tokens)):
            tensor_pos = pos + 1  # offset for SOS
            if tensor_pos >= probs.shape[1]:
                break

            # Current token confidence
            cur_idx = ids[tensor_pos]
            cur_conf = probs[0, tensor_pos, cur_idx].item()

            # BiLSTM top prediction
            top_idx  = probs[0, tensor_pos].argmax().item()
            top_conf = probs[0, tensor_pos, top_idx].item()

            # Replace if BiLSTM is significantly more confident and predicts different token
            if top_idx != cur_idx and top_conf > confidence_threshold and top_conf > cur_conf * 1.5:
                new_token = vocab.idx2token.get(top_idx, tokens[pos])
                if new_token not in ("<PAD>", "<SOS>", "<EOS>", "<MASK>"):
                    new_tokens[pos] = new_token

        corrected.append("".join(new_tokens))

    return corrected


@torch.no_grad()
def correct_with_ssm(ssm_model, predictions, vocab, device, top_k=3):
    """
    Use SSM NWP to correct predictions.
    Strategy: for each position, if the SSM strongly prefers a different
    character given the prefix, replace it.
    """
    corrected = []

    for pred in predictions:
        if len(pred) == 0:
            corrected.append(pred)
            continue

        tokens = list(pred)[:126]
        ids = [SOS_IDX] + vocab.encode(tokens) + [EOS_IDX]
        input_tensor = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)

        # Get SSM predictions
        logits = ssm_model(input_tensor)                # (1, seq_len, vocab_size)
        probs  = torch.softmax(logits, dim=-1)

        new_tokens = tokens.copy()
        for pos in range(len(tokens)):
            if pos >= probs.shape[1]:
                break

            cur_idx  = ids[pos+1] if pos+1 < len(ids) else PAD_IDX
            cur_prob = probs[0, pos, cur_idx].item()
            top_idx  = probs[0, pos].argmax().item()
            top_prob = probs[0, pos, top_idx].item()

            # Replace only if SSM is very confident and token is different
            if top_idx != cur_idx and top_prob > 0.6 and top_prob > cur_prob * 2.0:
                new_token = vocab.idx2token.get(top_idx, tokens[pos])
                if new_token not in ("<PAD>", "<SOS>", "<EOS>", "<MASK>"):
                    new_tokens[pos] = new_token

        corrected.append("".join(new_tokens))

    return corrected


# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_all_metrics(predictions, targets):
    return {
        "char_acc":    char_accuracy(predictions, targets),
        "word_acc":    word_accuracy(predictions, targets),
        "levenshtein": avg_levenshtein(predictions, targets),
        "bleu":        compute_bleu(predictions, targets),
        "rouge_l":     compute_rouge_l(predictions, targets),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(config_path: str, mode: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lm_type = cfg["language_model"]["model_type"]  # "bilstm" or "ssm"

    # ── Load vocabs ────────────────────────────────────────────────────────────
    plain_vocab, cipher_vocab = load_task1_vocab(cfg)
    lm_vocab = load_task2_vocab(cfg, lm_type)

    # ── Load models ────────────────────────────────────────────────────────────
    decrypt_model = load_decryption_model(cfg, plain_vocab, cipher_vocab, device)
    lm_model      = load_lm_model(cfg, lm_vocab, lm_type, device)

    # ── Load plain text targets ────────────────────────────────────────────────
    with open(cfg["data"]["plain_file"]) as f:
        plain_lines = [l.strip() for l in f.readlines() if l.strip()]

    max_cipher_len = cfg["data"]["max_cipher_len"]
    max_plain_len  = cfg["data"]["max_plain_len"]

    # Split — use test split only
    n = len(plain_lines)
    test_start = int(n * (cfg["data"]["train_split"] + cfg["data"]["val_split"]))
    plain_targets = plain_lines[test_start:]

    # ── WandB ──────────────────────────────────────────────────────────────────
    init_wandb(
        project=cfg["logging"]["wandb_project"],
        config=cfg,
        name=cfg["logging"]["wandb_run_name"],
    )

    # ── Run on each noise level ────────────────────────────────────────────────
    all_results = {}
    cipher_files = cfg["data"]["cipher_files"]

    for cipher_file in cipher_files:
        noise_level = os.path.basename(cipher_file).replace(".txt", "")  # e.g. cipher_01
        print(f"\n── Noise level: {noise_level} ──────────────────────────")

        # Load cipher lines (test split only)
        with open(cipher_file) as f:
            cipher_lines = [l.strip() for l in f.readlines() if l.strip()]
        cipher_test = cipher_lines[test_start:]

        # Align lengths
        min_len = min(len(cipher_test), len(plain_targets))
        cipher_test   = cipher_test[:min_len]
        targets_slice = plain_targets[:min_len]

        # 1. Decrypt only
        print("  Decrypting...")
        decrypted = decrypt_lines(
            decrypt_model, cipher_test, cipher_vocab, plain_vocab,
            max_cipher_len, max_plain_len, device,
        )

        metrics_decrypt = compute_all_metrics(decrypted, targets_slice)
        print(f"  Decrypt only — char_acc={metrics_decrypt['char_acc']:.4f} "
              f"bleu={metrics_decrypt['bleu']:.4f} rouge={metrics_decrypt['rouge_l']:.4f}")

        # 2. Decrypt + LM correction
        print(f"  Correcting with {lm_type}...")
        if lm_type == "bilstm":
            corrected = correct_with_bilstm(lm_model, decrypted, lm_vocab, device)
        else:
            corrected = correct_with_ssm(lm_model, decrypted, lm_vocab, device)

        metrics_corrected = compute_all_metrics(corrected, targets_slice)
        print(f"  Decrypt+{lm_type} — char_acc={metrics_corrected['char_acc']:.4f} "
              f"bleu={metrics_corrected['bleu']:.4f} rouge={metrics_corrected['rouge_l']:.4f}")

        all_results[noise_level] = {
            "decrypt_only": metrics_decrypt,
            f"decrypt_{lm_type}": metrics_corrected,
            "samples_decrypt":   list(zip(decrypted[:5], targets_slice[:5])),
            "samples_corrected": list(zip(corrected[:5], targets_slice[:5])),
        }

        # Log to WandB
        log_wandb({
            f"{noise_level}/decrypt/char_acc":    metrics_decrypt["char_acc"],
            f"{noise_level}/decrypt/word_acc":    metrics_decrypt["word_acc"],
            f"{noise_level}/decrypt/levenshtein": metrics_decrypt["levenshtein"],
            f"{noise_level}/decrypt/bleu":        metrics_decrypt["bleu"],
            f"{noise_level}/decrypt/rouge_l":     metrics_decrypt["rouge_l"],
            f"{noise_level}/{lm_type}/char_acc":    metrics_corrected["char_acc"],
            f"{noise_level}/{lm_type}/word_acc":    metrics_corrected["word_acc"],
            f"{noise_level}/{lm_type}/levenshtein": metrics_corrected["levenshtein"],
            f"{noise_level}/{lm_type}/bleu":        metrics_corrected["bleu"],
            f"{noise_level}/{lm_type}/rouge_l":     metrics_corrected["rouge_l"],
        })

    finish_wandb()

    # ── Save results ───────────────────────────────────────────────────────────
    results_file = cfg["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        f.write(f"Task 3 — Decryption + {lm_type.upper()} Correction\n")
        f.write("=" * 60 + "\n\n")

        for noise_level, results in all_results.items():
            f.write(f"Noise level: {noise_level}\n")
            f.write("-" * 40 + "\n")

            d = results["decrypt_only"]
            f.write(f"Decrypt only:\n")
            f.write(f"  char_acc={d['char_acc']:.4f} word_acc={d['word_acc']:.4f} "
                    f"lev={d['levenshtein']:.2f} bleu={d['bleu']:.4f} rouge={d['rouge_l']:.4f}\n")

            c = results[f"decrypt_{lm_type}"]
            f.write(f"Decrypt + {lm_type}:\n")
            f.write(f"  char_acc={c['char_acc']:.4f} word_acc={c['word_acc']:.4f} "
                    f"lev={c['levenshtein']:.2f} bleu={c['bleu']:.4f} rouge={c['rouge_l']:.4f}\n")

            f.write("\nSample predictions (decrypt only):\n")
            for pred, tgt in results["samples_decrypt"]:
                f.write(f"  TARGET : {tgt}\n")
                f.write(f"  PREDICT: {pred}\n\n")

            f.write(f"Sample predictions (decrypt + {lm_type}):\n")
            for pred, tgt in results["samples_corrected"]:
                f.write(f"  TARGET : {tgt}\n")
                f.write(f"  PREDICT: {pred}\n\n")

            f.write("\n")

    print(f"\nResults saved to {results_file}")

    if os.path.exists("/kaggle/working"):
        dest = os.path.join("/kaggle/working", os.path.basename(results_file))
        shutil.copy(results_file, dest)
        print(f"Results copied to {dest}")


def main(config_path: str, mode: str):
    run_pipeline(config_path, mode)
