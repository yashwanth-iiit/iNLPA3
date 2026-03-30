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
    PAD_IDX, SOS_IDX, EOS_IDX,
    tokenize_cipher,
)
from src.utils.checkpoints import load_checkpoint
from src.utils.hf_wandb import load_from_hub, init_wandb, log_wandb, finish_wandb
from src.task1.models import build_seq2seq
from src.task1.metrics import (
    char_accuracy, word_accuracy, avg_levenshtein, decode_predictions
)
from src.task2.models import build_bilstm, build_ssm


# ── BLEU ──────────────────────────────────────────────────────────────────────
def compute_bleu(predictions, targets, max_n=4):
    from collections import Counter
    import math

    def get_ngrams(seq, n):
        return Counter([seq[i:i+n] for i in range(len(seq)-n+1)])

    clipped = {n: 0 for n in range(1, max_n+1)}
    total   = {n: 0 for n in range(1, max_n+1)}
    pred_len = ref_len = 0

    for pred, ref in zip(predictions, targets):
        pred_len += len(pred)
        ref_len  += len(ref)
        for n in range(1, max_n+1):
            pg = get_ngrams(pred, n)
            rg = get_ngrams(ref, n)
            for ng, cnt in pg.items():
                clipped[n] += min(cnt, rg.get(ng, 0))
            total[n] += max(len(pred) - n + 1, 0)

    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len >= ref_len else math.exp(1 - ref_len / pred_len)
    log_bleu = 0.0
    for n in range(1, max_n+1):
        if total[n] == 0 or clipped[n] == 0:
            return 0.0
        log_bleu += math.log(clipped[n] / total[n])
    return bp * math.exp(log_bleu / max_n)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def compute_rouge_l(predictions, targets):
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(2)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i%2][j] = dp[(i-1)%2][j-1]+1 if a[i-1]==b[j-1] else max(dp[(i-1)%2][j], dp[i%2][j-1])
        return dp[m%2][n]

    scores = []
    for pred, ref in zip(predictions, targets):
        if not ref:
            scores.append(0.0)
            continue
        l = lcs(pred, ref)
        p = l/len(pred) if pred else 0
        r = l/len(ref)
        scores.append(2*p*r/(p+r) if p+r > 0 else 0.0)
    return sum(scores)/len(scores) if scores else 0.0


# ── Vocab loading ─────────────────────────────────────────────────────────────
def load_task1_vocab(cfg):
    plain_vocab  = build_plain_vocab(cfg["data"]["plain_file"])
    cipher_vocab = build_cipher_vocab()
    hf_repo = cfg["decryption"].get("huggingface_repo", "")
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=hf_repo, filename="task1_lstm_vocab.json",
                local_dir="checkpoints/task1/lstm",
            )
            with open(path) as f:
                d = json.load(f)
            plain_vocab.token2idx  = d["plain"]
            plain_vocab.idx2token  = {int(v): k for k, v in d["plain"].items()}
            cipher_vocab.token2idx = d["cipher"]
            cipher_vocab.idx2token = {int(v): k for k, v in d["cipher"].items()}
            print("Loaded Task 1 vocab from HuggingFace")
        except Exception as e:
            print(f"Could not load Task 1 vocab: {e}")
    return plain_vocab, cipher_vocab


def load_task2_vocab(cfg, model_type):
    vocab = build_plain_vocab(cfg["data"]["plain_file"])
    hf_repo = cfg["language_model"].get("huggingface_repo", "")
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=hf_repo, filename=f"task2_{model_type}_vocab.json",
                local_dir=f"checkpoints/task2/{model_type}",
            )
            with open(path) as f:
                d = json.load(f)
            vocab.token2idx = d["plain"]
            vocab.idx2token = {int(v): k for k, v in d["plain"].items()}
            print(f"Loaded Task 2 {model_type} vocab from HuggingFace")
        except Exception as e:
            print(f"Could not load Task 2 vocab: {e}")
    return vocab


# ── Model loading ─────────────────────────────────────────────────────────────
def load_decryption_model(cfg, plain_vocab, cipher_vocab, device):
    task1_cfg = {"model": {"type": "lstm", "embedding_dim": 64, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3}}
    model = build_seq2seq(task1_cfg, len(cipher_vocab), len(plain_vocab)).to(device)
    hf_repo = cfg["decryption"].get("huggingface_repo", "")
    if hf_repo:
        load_from_hub(model, hf_repo, "task1_lstm_best.pt", device=str(device))
    else:
        load_checkpoint(cfg["decryption"]["checkpoint"], model, device=str(device))
    model.eval()
    print("Loaded Task 1 LSTM")
    return model


def load_lm_model(cfg, vocab, model_type, device):
    task2_cfg = {"model": {"type": model_type, "embedding_dim": 128, "hidden_dim": 256,
                           "num_layers": 2, "dropout": 0.3 if model_type == "bilstm" else 0.2, "state_dim": 64}}
    model = build_bilstm(task2_cfg, len(vocab)).to(device) if model_type == "bilstm" \
            else build_ssm(task2_cfg, len(vocab)).to(device)
    hf_repo = cfg["language_model"].get("huggingface_repo", "")
    if hf_repo:
        load_from_hub(model, hf_repo, f"task2_{model_type}_best.pt", device=str(device))
    else:
        load_checkpoint(cfg["language_model"]["checkpoint"], model, device=str(device))
    model.eval()
    print(f"Loaded Task 2 {model_type}")
    return model


# ── Decryption ────────────────────────────────────────────────────────────────
@torch.no_grad()
def decrypt_lines(model, lines, cipher_vocab, plain_vocab, max_cipher_len, max_plain_len, device, batch_size=64):
    """Returns (predictions, confidences) where confidences[i][j] is decoder confidence at position j."""
    from torch.nn.utils.rnn import pad_sequence

    all_preds       = []
    all_confidences = []

    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        encoded = []
        for line in batch_lines:
            tokens = tokenize_cipher(line.strip())[:max_cipher_len-2]
            ids = [SOS_IDX] + cipher_vocab.encode(tokens) + [EOS_IDX]
            encoded.append(torch.tensor(ids, dtype=torch.long))

        cipher_batch = pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX).to(device)

        # decode_greedy now returns (pred_ids, conf_scores)
        pred_ids, conf_scores = model.decode_greedy(cipher_batch, SOS_IDX, EOS_IDX, max_plain_len)

        preds = decode_predictions(pred_ids, plain_vocab, EOS_IDX, PAD_IDX)
        all_preds.extend(preds)
        all_confidences.extend(conf_scores.cpu().tolist())

    return all_preds, all_confidences


# ── Correction ────────────────────────────────────────────────────────────────
@torch.no_grad()
def correct_with_bilstm(bilstm_model, predictions, confidences, vocab, device,
                        decoder_conf_threshold=0.7, lm_conf_threshold=0.8):
    """
    Only correct positions where decoder was uncertain (conf < decoder_conf_threshold).
    Replace with BiLSTM prediction only if BiLSTM is very confident (> lm_conf_threshold).
    """
    corrected = []

    for pred, conf in zip(predictions, confidences):
        if not pred:
            corrected.append(pred)
            continue

        tokens = list(pred)[:126]
        ids = [SOS_IDX] + vocab.encode(tokens) + [EOS_IDX]
        input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        logits = bilstm_model(input_tensor)             # (1, seq_len, vocab_size)
        probs  = torch.softmax(logits, dim=-1)

        new_tokens = tokens.copy()
        for pos in range(len(tokens)):
            tensor_pos = pos + 1                        # offset for SOS

            # Only correct if decoder was uncertain at this position
            decoder_conf = conf[pos] if pos < len(conf) else 1.0
            if decoder_conf >= decoder_conf_threshold:
                continue                                # decoder was confident, skip

            if tensor_pos >= probs.shape[1]:
                break

            top_idx  = probs[0, tensor_pos].argmax().item()
            top_conf = probs[0, tensor_pos, top_idx].item()

            # Only replace if BiLSTM is very confident
            if top_conf >= lm_conf_threshold:
                new_token = vocab.idx2token.get(top_idx, tokens[pos])
                if new_token not in ("<PAD>", "<SOS>", "<EOS>", "<MASK>"):
                    new_tokens[pos] = new_token

        corrected.append("".join(new_tokens))

    return corrected


@torch.no_grad()
def correct_with_ssm(ssm_model, predictions, confidences, vocab, device,
                     decoder_conf_threshold=0.7, lm_conf_threshold=0.8):
    """
    Only correct positions where decoder was uncertain (conf < decoder_conf_threshold).
    Replace with SSM prediction only if SSM is very confident (> lm_conf_threshold).
    """
    corrected = []

    for pred, conf in zip(predictions, confidences):
        if not pred:
            corrected.append(pred)
            continue

        tokens = list(pred)[:126]
        ids = [SOS_IDX] + vocab.encode(tokens) + [EOS_IDX]
        input_tensor = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)

        logits = ssm_model(input_tensor)
        probs  = torch.softmax(logits, dim=-1)

        new_tokens = tokens.copy()
        for pos in range(len(tokens)):
            # Only correct if decoder was uncertain
            decoder_conf = conf[pos] if pos < len(conf) else 1.0
            if decoder_conf >= decoder_conf_threshold:
                continue

            if pos >= probs.shape[1]:
                break

            top_idx  = probs[0, pos].argmax().item()
            top_conf = probs[0, pos, top_idx].item()

            if top_conf >= lm_conf_threshold:
                new_token = vocab.idx2token.get(top_idx, tokens[pos])
                if new_token not in ("<PAD>", "<SOS>", "<EOS>", "<MASK>"):
                    new_tokens[pos] = new_token

        corrected.append("".join(new_tokens))

    return corrected


# ── Metrics ───────────────────────────────────────────────────────────────────
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

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_type = cfg["language_model"]["model_type"]
    print(f"Device: {device} | LM: {lm_type}")

    # Vocabs
    plain_vocab, cipher_vocab = load_task1_vocab(cfg)
    lm_vocab = load_task2_vocab(cfg, lm_type)

    # Models
    decrypt_model = load_decryption_model(cfg, plain_vocab, cipher_vocab, device)
    lm_model      = load_lm_model(cfg, lm_vocab, lm_type, device)

    # Plain text targets (test split only)
    with open(cfg["data"]["plain_file"]) as f:
        plain_lines = [l.strip() for l in f if l.strip()]
    n = len(plain_lines)
    test_start    = int(n * (cfg["data"]["train_split"] + cfg["data"]["val_split"]))
    plain_targets = plain_lines[test_start:]

    max_cipher_len = cfg["data"]["max_cipher_len"]
    max_plain_len  = cfg["data"]["max_plain_len"]

    # WandB
    init_wandb(project=cfg["logging"]["wandb_project"], config=cfg, name=cfg["logging"]["wandb_run_name"])

    all_results  = {}
    cipher_files = cfg["data"]["cipher_files"]

    for cipher_file in cipher_files:
        noise_level = os.path.basename(cipher_file).replace(".txt", "")
        print(f"\n── {noise_level} ──────────────────────────────────────────")

        with open(cipher_file) as f:
            cipher_lines = [l.strip() for l in f if l.strip()]
        cipher_test   = cipher_lines[test_start:]
        min_len       = min(len(cipher_test), len(plain_targets))
        cipher_test   = cipher_test[:min_len]
        targets_slice = plain_targets[:min_len]

        # 1. Decrypt only
        print("  Decrypting...")
        decrypted, confidences = decrypt_lines(
            decrypt_model, cipher_test, cipher_vocab, plain_vocab,
            max_cipher_len, max_plain_len, device,
        )
        metrics_decrypt = compute_all_metrics(decrypted, targets_slice)
        print(f"  Decrypt only  — char_acc={metrics_decrypt['char_acc']:.4f} "
              f"word_acc={metrics_decrypt['word_acc']:.4f} "
              f"bleu={metrics_decrypt['bleu']:.4f} rouge={metrics_decrypt['rouge_l']:.4f}")

        # 2. Decrypt + LM correction (confidence-guided)
        print(f"  Correcting with {lm_type} (confidence-guided)...")
        if lm_type == "bilstm":
            corrected = correct_with_bilstm(lm_model, decrypted, confidences, lm_vocab, device)
        else:
            corrected = correct_with_ssm(lm_model, decrypted, confidences, lm_vocab, device)

        metrics_corrected = compute_all_metrics(corrected, targets_slice)
        print(f"  Decrypt+{lm_type} — char_acc={metrics_corrected['char_acc']:.4f} "
              f"word_acc={metrics_corrected['word_acc']:.4f} "
              f"bleu={metrics_corrected['bleu']:.4f} rouge={metrics_corrected['rouge_l']:.4f}")

        # Improvement
        improvement = metrics_corrected['char_acc'] - metrics_decrypt['char_acc']
        print(f"  Improvement   — char_acc: {improvement:+.4f}")

        all_results[noise_level] = {
            "decrypt_only":          metrics_decrypt,
            f"decrypt_{lm_type}":    metrics_corrected,
            "samples_decrypt":       list(zip(decrypted[:5], targets_slice[:5])),
            "samples_corrected":     list(zip(corrected[:5], targets_slice[:5])),
        }

        log_wandb({
            f"{noise_level}/decrypt/char_acc":      metrics_decrypt["char_acc"],
            f"{noise_level}/decrypt/word_acc":      metrics_decrypt["word_acc"],
            f"{noise_level}/decrypt/levenshtein":   metrics_decrypt["levenshtein"],
            f"{noise_level}/decrypt/bleu":          metrics_decrypt["bleu"],
            f"{noise_level}/decrypt/rouge_l":       metrics_decrypt["rouge_l"],
            f"{noise_level}/{lm_type}/char_acc":    metrics_corrected["char_acc"],
            f"{noise_level}/{lm_type}/word_acc":    metrics_corrected["word_acc"],
            f"{noise_level}/{lm_type}/levenshtein": metrics_corrected["levenshtein"],
            f"{noise_level}/{lm_type}/bleu":        metrics_corrected["bleu"],
            f"{noise_level}/{lm_type}/rouge_l":     metrics_corrected["rouge_l"],
        })

    finish_wandb()

    # Save results
    results_file = cfg["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        f.write(f"Task 3 — Decryption + {lm_type.upper()} Correction (Confidence-Guided)\n")
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

            improvement = c['char_acc'] - d['char_acc']
            f.write(f"Improvement: char_acc {improvement:+.4f}\n")

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
