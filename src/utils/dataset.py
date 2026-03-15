import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
MASK_TOKEN = "<MASK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MASK_IDX = 3


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocab:
    """
    Bidirectional token <-> index mapping.
    Special tokens are always assigned the first indices:
        0: <PAD>, 1: <SOS>, 2: <EOS>, 3: <MASK>
    """

    def __init__(self):
        self.token2idx = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            MASK_TOKEN: MASK_IDX,
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def build(self, tokens: List[str]) -> None:
        for tok in sorted(set(tokens)):
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, PAD_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token.get(i, PAD_TOKEN) for i in indices]

    def __len__(self) -> int:
        return len(self.token2idx)


def build_plain_vocab(plain_file: str) -> Vocab:
    """Character-level vocab from plain text file."""
    vocab = Vocab()
    with open(plain_file, encoding="utf-8") as f:
        text = f.read()
    chars = list(set(text.replace("\n", "")))
    vocab.build(chars)
    return vocab


def build_cipher_vocab() -> Vocab:
    """Digit-level vocab for cipher text (0-9 only)."""
    vocab = Vocab()
    vocab.build([str(d) for d in range(10)])
    return vocab


# ── Tokenizers ────────────────────────────────────────────────────────────────
def tokenize_plain(line: str) -> List[str]:
    """Each character is a token."""
    return list(line)


def tokenize_cipher(line: str) -> List[str]:
    """Each digit character is a token."""
    return list(line.strip())


# ── Task 1: Cipher -> Plain (Seq2Seq) ─────────────────────────────────────────
class CipherDataset(Dataset):
    """
    Line-aligned (cipher, plain) pairs for seq2seq decryption.
    Encoder input : cipher digits  [SOS, d1, d2, ..., EOS]
    Decoder target: plain chars    [SOS, c1, c2, ..., EOS]
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        cipher_vocab: Vocab,
        plain_vocab: Vocab,
        max_cipher_len: int,
        max_plain_len: int,
    ):
        self.pairs = pairs
        self.cipher_vocab = cipher_vocab
        self.plain_vocab = plain_vocab
        self.max_cipher_len = max_cipher_len
        self.max_plain_len = max_plain_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        cipher_line, plain_line = self.pairs[idx]

        # Tokenize and truncate (leave room for SOS/EOS)
        cipher_tokens = tokenize_cipher(cipher_line)[: self.max_cipher_len - 2]
        plain_tokens = tokenize_plain(plain_line)[: self.max_plain_len - 2]

        # Encode with SOS and EOS
        cipher_ids = [SOS_IDX] + self.cipher_vocab.encode(cipher_tokens) + [EOS_IDX]
        plain_ids = [SOS_IDX] + self.plain_vocab.encode(plain_tokens) + [EOS_IDX]

        return (
            torch.tensor(cipher_ids, dtype=torch.long),
            torch.tensor(plain_ids, dtype=torch.long),
        )


def cipher_collate_fn(batch):
    """Pad cipher and plain sequences to max length in batch."""
    cipher_seqs, plain_seqs = zip(*batch)
    cipher_padded = pad_sequence(cipher_seqs, batch_first=True, padding_value=PAD_IDX)
    plain_padded = pad_sequence(plain_seqs, batch_first=True, padding_value=PAD_IDX)
    return cipher_padded, plain_padded


# ── Task 2: Plain text language modeling ──────────────────────────────────────
class MLMDataset(Dataset):
    """
    Masked Language Modeling dataset for Bi-LSTM.
    Randomly masks tokens and the model predicts original tokens.
    """

    def __init__(
        self,
        lines: List[str],
        vocab: Vocab,
        max_seq_len: int,
        mask_prob: float = 0.15,
    ):
        self.lines = lines
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        tokens = tokenize_plain(line)[: self.max_seq_len - 2]
        ids = [SOS_IDX] + self.vocab.encode(tokens) + [EOS_IDX]

        input_ids = ids.copy()
        labels = [-100] * len(ids)  # -100 = ignore in loss

        # Randomly mask tokens (not SOS/EOS/PAD)
        for i in range(1, len(ids) - 1):
            if random.random() < self.mask_prob:
                labels[i] = ids[i]
                input_ids[i] = MASK_IDX

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class NWPDataset(Dataset):
    """
    Next Word Prediction dataset for SSM.
    Input: tokens[0..n-1], Target: tokens[1..n]
    """

    def __init__(self, lines: List[str], vocab: Vocab, max_seq_len: int):
        self.lines = lines
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        tokens = tokenize_plain(line)[: self.max_seq_len - 1]
        ids = [SOS_IDX] + self.vocab.encode(tokens) + [EOS_IDX]

        input_ids = ids[:-1]   # all but last
        target_ids = ids[1:]   # all but first

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def lm_collate_fn(batch):
    """Pad language model sequences."""
    input_seqs, target_seqs = zip(*batch)
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=PAD_IDX)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=PAD_IDX)
    return input_padded, target_padded


# ── Data loading helpers ───────────────────────────────────────────────────────
def load_pairs(plain_file: str, cipher_file: str) -> List[Tuple[str, str]]:
    """Load and align (cipher, plain) line pairs, skip empty lines."""
    with open(plain_file, encoding="utf-8") as f:
        plain_lines = f.read().splitlines()
    with open(cipher_file, encoding="utf-8") as f:
        cipher_lines = f.read().splitlines()
    pairs = [
        (c.strip(), p.strip())
        for c, p in zip(cipher_lines, plain_lines)
        if c.strip() and p.strip()
    ]
    return pairs


def load_plain_lines(plain_file: str) -> List[str]:
    """Load plain text lines, skip empty."""
    with open(plain_file, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [l.strip() for l in lines if l.strip()]


def split_data(data: list, train: float, val: float, test: float):
    """Split data into train/val/test."""
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1"
    n = len(data)
    t1 = int(n * train)
    t2 = int(n * (train + val))
    return data[:t1], data[t1:t2], data[t2:]


def get_cipher_dataloaders(cfg: dict, cipher_vocab: Vocab, plain_vocab: Vocab):
    """Build train/val/test DataLoaders for Task 1."""
    pairs = load_pairs(cfg["data"]["plain_file"], cfg["data"]["cipher_file"])
    train_pairs, val_pairs, test_pairs = split_data(
        pairs,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        cfg["data"]["test_split"],
    )

    def make_loader(pairs, shuffle):
        ds = CipherDataset(
            pairs,
            cipher_vocab,
            plain_vocab,
            cfg["data"]["max_cipher_len"],
            cfg["data"]["max_plain_len"],
        )
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            collate_fn=cipher_collate_fn,
        )

    return (
        make_loader(train_pairs, shuffle=True),
        make_loader(val_pairs, shuffle=False),
        make_loader(test_pairs, shuffle=False),
    )


def get_lm_dataloaders(cfg: dict, vocab: Vocab, task: str = "mlm"):
    """Build train/val/test DataLoaders for Task 2 (mlm or nwp)."""
    lines = load_plain_lines(cfg["data"]["plain_file"])
    train_lines, val_lines, test_lines = split_data(
        lines,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        cfg["data"]["test_split"],
    )

    def make_loader(lines, shuffle):
        if task == "mlm":
            ds = MLMDataset(
                lines,
                vocab,
                cfg["data"]["max_seq_len"],
                cfg["data"].get("mask_prob", 0.15),
            )
        else:
            ds = NWPDataset(lines, vocab, cfg["data"]["max_seq_len"])
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            collate_fn=lm_collate_fn,
        )

    return (
        make_loader(train_lines, shuffle=True),
        make_loader(val_lines, shuffle=False),
        make_loader(test_lines, shuffle=False),
    )
