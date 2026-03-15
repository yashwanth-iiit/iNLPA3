from typing import List
import torch


def levenshtein(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def char_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Fraction of characters predicted correctly (position-wise)."""
    correct = total = 0
    for pred, tgt in zip(predictions, targets):
        length = max(len(pred), len(tgt))
        if length == 0:
            continue
        for p, t in zip(pred.ljust(length), tgt.ljust(length)):
            correct += int(p == t)
            total += 1
    return correct / total if total > 0 else 0.0


def word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Fraction of whitespace-tokenized words predicted correctly."""
    correct = total = 0
    for pred, tgt in zip(predictions, targets):
        pred_words = pred.split()
        tgt_words = tgt.split()
        length = max(len(pred_words), len(tgt_words))
        if length == 0:
            continue
        for pw, tw in zip(
            pred_words + [""] * (length - len(pred_words)),
            tgt_words + [""] * (length - len(tgt_words)),
        ):
            correct += int(pw == tw)
            total += 1
    return correct / total if total > 0 else 0.0


def avg_levenshtein(predictions: List[str], targets: List[str]) -> float:
    """Average Levenshtein distance between predicted and target strings."""
    if not predictions:
        return 0.0
    return sum(levenshtein(p, t) for p, t in zip(predictions, targets)) / len(predictions)


def decode_predictions(
    token_ids: torch.Tensor,
    vocab,
    eos_idx: int,
    pad_idx: int,
) -> List[str]:
    """
    Convert batch of token id tensors to list of strings.
    Stops at EOS, strips PAD and special tokens.
    """
    results = []
    for seq in token_ids:
        chars = []
        for idx in seq.tolist():
            if idx == eos_idx:
                break
            if idx in (pad_idx, 1, 2, 3):  # PAD SOS EOS MASK
                continue
            token = vocab.idx2token.get(idx, "")
            chars.append(token)
        results.append("".join(chars))
    return results
