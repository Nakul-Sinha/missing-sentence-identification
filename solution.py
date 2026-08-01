"""Offline, from-scratch solver for Meridian Ashes.

The model is a compact Transformer encoder initialized randomly and trained only
on train.csv.  Test text is tokenized with the frozen train vocabulary and is
never used for fitting, model selection, or early stopping.

Examples
--------
Honest grouped validation::

    python solution.py validate --epochs 6

Train on every labelled row and create ./working/submission.csv::

    python solution.py train-predict --epochs 6 --ensemble 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from tokenizers import Tokenizer
from tokenizers import models as tokenizer_models
from tokenizers import pre_tokenizers as tokenizer_pre_tokenizers
from tokenizers import trainers as tokenizer_trainers
from transformers import BertConfig, BertForMaskedLM, BertModel, GPT2Config, GPT2LMHeadModel
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


LETTERS = "ABCDEF"
CANDIDATE_RE = re.compile(r"^\[([A-F])\]\s(.*)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+", re.UNICODE)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<cls>", "<left>", "<cand>", "<right>")
PAD, UNK, CLS, LEFT, CAND, RIGHT = range(len(SPECIAL_TOKENS))


@dataclass
class Config:
    seed: int = 2026
    vocab_size: int = 40_000
    min_frequency: int = 2
    left_tokens: int = 64
    candidate_tokens: int = 72
    right_tokens: int = 64
    candidate_bytes: int = 384
    d_model: int = 192
    heads: int = 6
    layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.12
    batch_size: int = 12
    epochs: int = 6
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.02
    warmup_fraction: float = 0.08
    grad_clip: float = 1.0
    originality_loss_weight: float = 1.0
    pair_loss_weight: float = 1.0
    validation_fraction: float = 0.15
    num_workers: int = 0
    subword_vocab_size: int = 8_000
    mlm_length: int = 192
    mlm_epochs: int = 5
    mlm_batch_size: int = 64
    rank_batch_size: int = 12
    causal_epochs: int = 8
    causal_batch_size: int = 64

    @property
    def max_length(self) -> int:
        return self.left_tokens + self.candidate_tokens + self.right_tokens + 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def tokenize(text: str) -> list[str]:
    return [token.lower().replace("\u2019", "'") for token in TOKEN_RE.findall(str(text))]


def normalized_words(text: str) -> str:
    return " ".join(token.lower().replace("\u2019", "'") for token in WORD_RE.findall(str(text)))


def parse_candidates(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in str(value).splitlines():
        match = CANDIDATE_RE.match(line)
        if match:
            result[match.group(1)] = match.group(2)
    if set(result) != set(LETTERS):
        raise ValueError("Every candidates field must contain exactly [A] through [F].")
    return result


def split_gap(value: str) -> tuple[str, str]:
    text = str(value)
    marker = " || RIGHT: "
    if not text.startswith("LEFT: ") or marker not in text:
        raise ValueError("Unexpected gap format; expected 'LEFT: ... || RIGHT: ...'.")
    left, right = text[len("LEFT: ") :].split(marker, 1)
    return left, right


def candidate_signature(text: str) -> tuple[str, ...]:
    # The benchmark guarantees that each original/counterfeit pair has exactly
    # the same whitespace-separated tokens.  This uses only row-local inputs.
    return tuple(sorted(str(text).split()))


def candidate_pairs(candidates: dict[str, str]) -> list[tuple[int, int]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, letter in enumerate(LETTERS):
        groups[candidate_signature(candidates[letter])].append(index)
    pairs = sorted((tuple(indices) for indices in groups.values()), key=lambda pair: pair[0])
    if len(pairs) != 3 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Expected exactly three same-token candidate pairs.")
    return [(int(pair[0]), int(pair[1])) for pair in pairs]


def locate_data_dir(requested: Path) -> Path:
    if (requested / "train.csv").is_file() and (requested / "test.csv").is_file():
        return requested
    roots = [Path("dataset"), Path("input"), Path("/kaggle/input"), Path("/kaggle/working/dataset")]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "train.csv").is_file() and (root / "test.csv").is_file():
            matches.append(root)
        if root == Path("/kaggle/input"):
            for train_path in root.rglob("train.csv"):
                if (train_path.parent / "test.csv").is_file():
                    matches.append(train_path.parent)
    matches = list(dict.fromkeys(path.resolve() for path in matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate train.csv and test.csv; pass --data-dir. Found: {matches}"
        )
    return matches[0]


def build_vocabulary(frame: pd.DataFrame, config: Config) -> tuple[dict[str, int], list[str]]:
    """Fit the vocabulary on labelled training text only."""
    counts: Counter[str] = Counter()
    columns = ("gap_1", "gap_2", "gap_3", "candidates")
    for column in columns:
        for value in frame[column].astype(str):
            counts.update(tokenize(value))
    capacity = max(0, config.vocab_size - len(SPECIAL_TOKENS))
    tokens = [
        token
        for token, frequency in counts.most_common()
        if frequency >= config.min_frequency
    ][:capacity]
    id_to_token = list(SPECIAL_TOKENS) + tokens
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


@dataclass
class EncodedRow:
    row_id: int
    left: list[list[int]]
    right: list[list[int]]
    candidates: list[list[int]]
    candidate_texts: list[str]
    candidate_bytes: list[list[int]]
    lexical_features: list[list[list[float]]]
    bridge_prior: list[list[float]]
    pairs: list[tuple[int, int]]
    targets: list[int] | None
    originals: list[float] | None


def encode_text(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, UNK) for token in tokenize(text)]


def encode_bytes(text: str) -> list[int]:
    # Zero is padding; raw UTF-8 bytes are shifted into 1..256.  This stateless
    # representation handles unseen names and archaic spelling without fitting
    # anything on test text.
    return [value + 1 for value in str(text).encode("utf-8", errors="replace")]


def lexical_pair_features(left: str, right: str, candidate: str) -> list[float]:
    """Stateless, non-TF-IDF cohesion features for one gap/candidate pair."""
    def words(value: str) -> list[str]:
        return [token.lower().replace("\u2019", "'") for token in WORD_RE.findall(value)]

    candidate_words = words(candidate)
    left_words = words(left)
    right_words = words(right)
    candidate_set = set(candidate_words)
    left_set = set(left_words)
    right_set = set(right_words)
    context_set = left_set | right_set
    content_set = {token for token in candidate_set if len(token) >= 4}
    context_content = {token for token in context_set if len(token) >= 4}
    candidate_caps = set(re.findall(r"\b[A-Z][A-Za-z'\u2019]{2,}\b", candidate))
    context_caps = set(re.findall(r"\b[A-Z][A-Za-z'\u2019]{2,}\b", f"{left} {right}"))
    denominator = max(1, len(candidate_set))
    content_denominator = max(1, len(content_set))
    return [
        math.log1p(len(candidate_words)) / 5.0,
        len(candidate_set & context_set) / denominator,
        len(candidate_set & left_set) / denominator,
        len(candidate_set & right_set) / denominator,
        len(content_set & context_content) / content_denominator,
        math.log1p(len(candidate_caps & context_caps)) / 3.0,
        float(candidate.lstrip().startswith(('"', "'", "\u201c"))),
        float(candidate.rstrip().endswith(('"', "'", "\u201d"))),
        float(left.rstrip().endswith(('"', "'", "\u201d"))),
        float(right.lstrip().startswith(('"', "'", "\u201c"))),
        float(bool(candidate_words) and candidate_words[0] in right_set),
        float(bool(candidate_words) and candidate_words[-1] in left_set),
    ]


def character_bridge_prior(left: str, right: str, candidate: str) -> float:
    context = re.sub(r"\s+", " ", f"{left} {right}".lower())
    sentence = re.sub(r"\s+", " ", candidate.lower())
    context_ngrams = {context[index : index + 5] for index in range(max(0, len(context) - 4))}
    sentence_ngrams = {sentence[index : index + 5] for index in range(max(0, len(sentence) - 4))}
    return len(context_ngrams & sentence_ngrams) / max(1, len(sentence_ngrams))


def encode_frame(frame: pd.DataFrame, vocabulary: dict[str, int], labelled: bool) -> list[EncodedRow]:
    encoded: list[EncodedRow] = []
    for row in frame.itertuples(index=False):
        parsed = parse_candidates(row.candidates)
        left_parts: list[list[int]] = []
        right_parts: list[list[int]] = []
        gap_texts: list[tuple[str, str]] = []
        for gap_index in range(1, 4):
            left, right = split_gap(getattr(row, f"gap_{gap_index}"))
            gap_texts.append((left, right))
            left_parts.append(encode_text(left, vocabulary))
            right_parts.append(encode_text(right, vocabulary))
        candidate_ids = [encode_text(parsed[letter], vocabulary) for letter in LETTERS]
        candidate_texts = [parsed[letter] for letter in LETTERS]
        candidate_byte_ids = [encode_bytes(parsed[letter]) for letter in LETTERS]
        lexical = [
            [lexical_pair_features(left, right, parsed[letter]) for letter in LETTERS]
            for left, right in gap_texts
        ]
        bridge_prior = [
            [character_bridge_prior(left, right, parsed[letter]) for letter in LETTERS]
            for left, right in gap_texts
        ]
        targets: list[int] | None = None
        originals: list[float] | None = None
        if labelled:
            binding = str(row.binding).split(">")
            if len(binding) != 3 or any(letter not in LETTERS for letter in binding):
                raise ValueError(f"Malformed training binding: {row.binding}")
            targets = [LETTERS.index(letter) for letter in binding]
            selected = set(binding)
            originals = [float(letter in selected) for letter in LETTERS]
        encoded.append(
            EncodedRow(
                row_id=int(row.id),
                left=left_parts,
                right=right_parts,
                candidates=candidate_ids,
                candidate_texts=candidate_texts,
                candidate_bytes=candidate_byte_ids,
                lexical_features=lexical,
                bridge_prior=bridge_prior,
                pairs=candidate_pairs(parsed),
                targets=targets,
                originals=originals,
            )
        )
    return encoded


class GapDataset(Dataset[tuple[int, int]]):
    def __init__(self, row_indices: Sequence[int]):
        self.items = [(int(row_index), gap) for row_index in row_indices for gap in range(3)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.items[index]


class RowDataset(Dataset[int]):
    def __init__(self, row_indices: Sequence[int]):
        self.items = [int(row_index) for row_index in row_indices]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> int:
        return self.items[index]


class RowBatchBuilder:
    """Encode each context and candidate once per row, not once per pairing."""

    def __init__(self, rows: list[EncodedRow], config: Config, labelled: bool):
        self.rows = rows
        self.config = config
        self.labelled = labelled

    @staticmethod
    def _pad(sequences: list[list[int]], limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = min(limit, max(len(sequence) for sequence in sequences))
        ids = torch.full((len(sequences), max_length), PAD, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
        for index, sequence in enumerate(sequences):
            length = min(len(sequence), max_length)
            ids[index, :length] = torch.tensor(sequence[:length], dtype=torch.long)
            mask[index, :length] = True
        return ids, mask

    def __call__(self, row_indices: Sequence[int]) -> dict[str, torch.Tensor]:
        batch_rows = [self.rows[int(row_index)] for row_index in row_indices]
        candidate_sequences = [
            candidate[: self.config.candidate_tokens]
            for row in batch_rows
            for candidate in row.candidates
        ]
        left_sequences = [
            context[-self.config.left_tokens :]
            for row in batch_rows
            for context in row.left
        ]
        right_sequences = [
            context[: self.config.right_tokens]
            for row in batch_rows
            for context in row.right
        ]
        candidate_ids, candidate_mask = self._pad(candidate_sequences, self.config.candidate_tokens)
        left_ids, left_mask = self._pad(left_sequences, self.config.left_tokens)
        right_ids, right_mask = self._pad(right_sequences, self.config.right_tokens)
        byte_sequences = [
            candidate[: self.config.candidate_bytes]
            for row in batch_rows
            for candidate in row.candidate_bytes
        ]
        byte_ids, byte_mask = self._pad(byte_sequences, self.config.candidate_bytes)
        batch_size = len(batch_rows)
        result = {
            "candidate_ids": candidate_ids.view(batch_size, 6, -1),
            "candidate_mask": candidate_mask.view(batch_size, 6, -1),
            "left_ids": left_ids.view(batch_size, 3, -1),
            "left_mask": left_mask.view(batch_size, 3, -1),
            "right_ids": right_ids.view(batch_size, 3, -1),
            "right_mask": right_mask.view(batch_size, 3, -1),
            "byte_ids": byte_ids.view(batch_size, 6, -1),
            "byte_mask": byte_mask.view(batch_size, 6, -1),
            "lexical_features": torch.tensor(
                [row.lexical_features for row in batch_rows], dtype=torch.float32
            ),
            "pairs": torch.tensor([row.pairs for row in batch_rows], dtype=torch.long),
            "row_indices": torch.tensor(row_indices, dtype=torch.long),
        }
        if self.labelled:
            result["targets"] = torch.tensor([row.targets for row in batch_rows], dtype=torch.long)
            result["originals"] = torch.tensor([row.originals for row in batch_rows], dtype=torch.float32)
            result["pair_targets"] = torch.tensor(
                [
                    [next(index for index, pair in enumerate(row.pairs) if target in pair) for target in row.targets]
                    for row in batch_rows
                ],
                dtype=torch.long,
            )
        return result


class BatchBuilder:
    def __init__(self, rows: list[EncodedRow], config: Config, labelled: bool):
        self.rows = rows
        self.config = config
        self.labelled = labelled

    def __call__(self, items: Sequence[tuple[int, int]]) -> dict[str, torch.Tensor]:
        sequences: list[list[int]] = []
        segments: list[list[int]] = []
        candidate_masks: list[list[bool]] = []
        targets: list[int] = []
        original_labels: list[list[float]] = []
        row_indices: list[int] = []
        gap_indices: list[int] = []

        for row_index, gap_index in items:
            row = self.rows[row_index]
            left = row.left[gap_index][-self.config.left_tokens :]
            right = row.right[gap_index][: self.config.right_tokens]
            for candidate in row.candidates:
                candidate = candidate[: self.config.candidate_tokens]
                sequence = [CLS, LEFT] + left + [CAND] + candidate + [RIGHT] + right
                segment = [0, 0] + [0] * len(left) + [1] + [1] * len(candidate) + [2] + [2] * len(right)
                candidate_mask = [False] * (2 + len(left)) + [False] + [True] * len(candidate) + [False] * (1 + len(right))
                sequences.append(sequence)
                segments.append(segment)
                candidate_masks.append(candidate_mask)
            if self.labelled:
                assert row.targets is not None and row.originals is not None
                targets.append(row.targets[gap_index])
                original_labels.append(row.originals)
            row_indices.append(row_index)
            gap_indices.append(gap_index)

        max_length = min(self.config.max_length, max(map(len, sequences)))
        input_ids = torch.full((len(sequences), max_length), PAD, dtype=torch.long)
        segment_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
        candidate_mask_tensor = torch.zeros((len(sequences), max_length), dtype=torch.bool)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
        for index, (sequence, segment, candidate_mask) in enumerate(zip(sequences, segments, candidate_masks)):
            length = min(len(sequence), max_length)
            input_ids[index, :length] = torch.tensor(sequence[:length])
            segment_ids[index, :length] = torch.tensor(segment[:length])
            candidate_mask_tensor[index, :length] = torch.tensor(candidate_mask[:length])
            attention_mask[index, :length] = True

        result = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "candidate_mask": candidate_mask_tensor,
            "attention_mask": attention_mask,
            "row_indices": torch.tensor(row_indices, dtype=torch.long),
            "gap_indices": torch.tensor(gap_indices, dtype=torch.long),
        }
        if self.labelled:
            result["targets"] = torch.tensor(targets, dtype=torch.long)
            result["originals"] = torch.tensor(original_labels, dtype=torch.float32)
        return result


class CoherenceTransformer(nn.Module):
    def __init__(self, vocabulary_size: int, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocabulary_size, config.d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.segment_embedding = nn.Embedding(3, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers, norm=nn.LayerNorm(config.d_model))
        feature_size = config.d_model * 8
        self.compatibility_head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, 1),
        )
        self.originality_head = nn.Sequential(
            nn.LayerNorm(config.d_model * 3),
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(segment_ids)
        )
        hidden = self.input_dropout(self.input_norm(hidden))
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)

        def masked_mean(mask: torch.Tensor) -> torch.Tensor:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        left_mask = attention_mask & segment_ids.eq(0)
        cand_mask = attention_mask & candidate_mask
        right_mask = attention_mask & segment_ids.eq(2)
        left_pool = masked_mean(left_mask)
        cand_pool = masked_mean(cand_mask)
        right_pool = masked_mean(right_mask)
        cls_pool = hidden[:, 0]

        compatibility_features = torch.cat(
            [
                cls_pool,
                left_pool,
                cand_pool,
                right_pool,
                left_pool * cand_pool,
                cand_pool * right_pool,
                torch.abs(left_pool - cand_pool),
                torch.abs(cand_pool - right_pool),
            ],
            dim=-1,
        )
        originality_features = torch.cat([cls_pool, cand_pool, left_pool * 0.0 + cand_pool - right_pool * 0.0], dim=-1)
        compatibility = self.compatibility_head(compatibility_features).squeeze(-1)
        originality = self.originality_head(originality_features).squeeze(-1)
        return compatibility, originality


class HybridCoherenceModel(nn.Module):
    """Hierarchical word-sequence matcher plus byte-level fluency detector."""

    def __init__(self, vocabulary_size: int, config: Config):
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(vocabulary_size, config.d_model, padding_idx=PAD)
        self.word_encoder = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.word_projection = nn.Sequential(
            nn.LayerNorm(config.d_model * 4),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.byte_embedding = nn.Embedding(257, 32, padding_idx=0)
        self.byte_convolutions = nn.ModuleList(
            [nn.Conv1d(32, 64, kernel_size=kernel, bias=False) for kernel in (2, 3, 4, 5, 7)]
        )
        self.byte_projection = nn.Sequential(
            nn.LayerNorm(64 * len(self.byte_convolutions)),
            nn.Linear(64 * len(self.byte_convolutions), config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.candidate_norm = nn.LayerNorm(config.d_model)
        lexical_size = 12
        compatibility_size = config.d_model * 7 + lexical_size
        self.compatibility_head = nn.Sequential(
            nn.LayerNorm(compatibility_size),
            nn.Linear(compatibility_size, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, 1),
        )
        self.originality_head = nn.Sequential(
            nn.LayerNorm(config.d_model * 3),
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.apply(CoherenceTransformer._initialize)

    def encode_words(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original_shape = ids.shape[:-1]
        flat_ids = ids.reshape(-1, ids.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        embedded = self.word_embedding(flat_ids)
        output, _ = self.word_encoder(embedded)
        weights = flat_mask.unsqueeze(-1).to(output.dtype)
        mean_pool = (output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        max_pool = output.masked_fill(~flat_mask.unsqueeze(-1), -1e4).max(dim=1).values
        first_pool = output[:, 0]
        lengths = flat_mask.sum(dim=1).clamp_min(1)
        last_pool = output[torch.arange(output.shape[0], device=output.device), lengths - 1]
        vector = self.word_projection(torch.cat([mean_pool, max_pool, first_pool, last_pool], dim=-1))
        return vector.view(*original_shape, -1)

    def encode_bytes(self, ids: torch.Tensor) -> torch.Tensor:
        original_shape = ids.shape[:-1]
        embedded = self.byte_embedding(ids.reshape(-1, ids.shape[-1])).transpose(1, 2)
        pooled = [F.relu(convolution(embedded)).amax(dim=-1) for convolution in self.byte_convolutions]
        vector = self.byte_projection(torch.cat(pooled, dim=-1))
        return vector.view(*original_shape, -1)

    def forward(
        self,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        left_ids: torch.Tensor,
        left_mask: torch.Tensor,
        right_ids: torch.Tensor,
        right_mask: torch.Tensor,
        byte_ids: torch.Tensor,
        byte_mask: torch.Tensor,
        lexical_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del byte_mask  # Padding has a fixed zero embedding and bias-free convolutions.
        candidate_words = self.encode_words(candidate_ids, candidate_mask)
        candidate_bytes = self.encode_bytes(byte_ids)
        candidates = self.candidate_norm(candidate_words + candidate_bytes)
        left = self.encode_words(left_ids, left_mask)
        right = self.encode_words(right_ids, right_mask)

        candidate_grid = candidates[:, None, :, :].expand(-1, 3, -1, -1)
        left_grid = left[:, :, None, :].expand(-1, -1, 6, -1)
        right_grid = right[:, :, None, :].expand(-1, -1, 6, -1)
        compatibility_features = torch.cat(
            [
                candidate_grid,
                left_grid,
                right_grid,
                candidate_grid * left_grid,
                candidate_grid * right_grid,
                torch.abs(candidate_grid - left_grid),
                torch.abs(candidate_grid - right_grid),
                lexical_features,
            ],
            dim=-1,
        )
        compatibility = self.compatibility_head(compatibility_features).squeeze(-1)
        originality_features = torch.cat(
            [candidate_words, candidate_bytes, torch.abs(candidate_words - candidate_bytes)], dim=-1
        )
        originality = self.originality_head(originality_features).squeeze(-1)
        return compatibility, originality


def make_loader(
    rows: list[EncodedRow],
    indices: Sequence[int],
    config: Config,
    labelled: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        GapDataset(indices),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=BatchBuilder(rows, config, labelled),
        drop_last=False,
    )


def make_row_loader(
    rows: list[EncodedRow],
    indices: Sequence[int],
    config: Config,
    labelled: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        RowDataset(indices),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=RowBatchBuilder(rows, config, labelled),
        drop_last=False,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    model_keys = ("input_ids", "segment_ids", "candidate_mask", "attention_mask")
    return {key: batch[key].to(device, non_blocking=True) for key in model_keys}


def move_row_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    model_keys = (
        "candidate_ids",
        "candidate_mask",
        "left_ids",
        "left_mask",
        "right_ids",
        "right_mask",
        "byte_ids",
        "byte_mask",
        "lexical_features",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in model_keys}


def learning_rate_factor(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return max(1e-3, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_model(
    rows: list[EncodedRow],
    train_indices: Sequence[int],
    config: Config,
    vocabulary_size: int,
    device: torch.device,
    validation_indices: Sequence[int] | None = None,
    validation_ngram: np.ndarray | None = None,
) -> tuple[HybridCoherenceModel, dict[str, float]]:
    seed_everything(config.seed)
    model = HybridCoherenceModel(vocabulary_size, config).to(device)
    loader = make_row_loader(rows, train_indices, config, labelled=True, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    total_steps = config.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, total_steps, config.warmup_fraction),
    )
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=amp_enabled)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_details: dict[str, float] = {}

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            model_inputs = move_row_batch(batch, device)
            target = batch["targets"].to(device, non_blocking=True)
            originals = batch["originals"].to(device, non_blocking=True)
            pairs = batch["pairs"].to(device, non_blocking=True)
            pair_targets = batch["pair_targets"].to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                compatibility, originality = model(**model_inputs)
                rank_loss = F.cross_entropy(
                    (compatibility + originality[:, None, :]).reshape(-1, 6),
                    target.reshape(-1),
                )
                pair_indices = pairs[:, None, :, :].expand(-1, 3, -1, -1)
                expanded_compatibility = compatibility[:, :, None, :].expand(-1, -1, 3, -1)
                pair_logits = torch.gather(expanded_compatibility, 3, pair_indices).mean(dim=-1)
                pair_loss = F.cross_entropy(pair_logits.reshape(-1, 3), pair_targets.reshape(-1))
                originality_loss = F.binary_cross_entropy_with_logits(originality, originals)
                loss = (
                    rank_loss
                    + config.pair_loss_weight * pair_loss
                    + config.originality_loss_weight * originality_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            group_count = target.shape[0]
            running_loss += float(loss.detach()) * group_count
            seen += group_count

        message: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": running_loss / max(1, seen),
            "seconds": time.time() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if validation_indices is not None:
            compatibility, originality = score_rows(model, rows, validation_indices, config, device)
            details = tune_and_evaluate(
                rows,
                validation_indices,
                compatibility,
                originality,
                ngram_originality=validation_ngram,
            )
            message.update(details)
            if details["meridian"] > best_score:
                best_score = details["meridian"]
                best_details = details
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(message, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_details


@torch.no_grad()
def score_rows(
    model: HybridCoherenceModel,
    rows: list[EncodedRow],
    indices: Sequence[int],
    config: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_row_loader(rows, indices, config, labelled=False, shuffle=False)
    local_position = {int(row_index): position for position, row_index in enumerate(indices)}
    compatibility_scores = np.zeros((len(indices), 3, 6), dtype=np.float32)
    originality_scores = np.zeros((len(indices), 3, 6), dtype=np.float32)
    amp_enabled = device.type == "cuda"
    for batch in loader:
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            compatibility, originality = model(**move_row_batch(batch, device))
        compatibility = compatibility.float().cpu().numpy()
        originality = originality.float().cpu().numpy()
        row_indices = batch["row_indices"].numpy()
        for batch_index, row_index in enumerate(row_indices):
            position = local_position[int(row_index)]
            compatibility_scores[position] = compatibility[batch_index]
            originality_scores[position] = np.repeat(originality[batch_index][None, :], 3, axis=0)
    return compatibility_scores, originality_scores


def fit_raw_count_originality(
    rows: list[EncodedRow], indices: Sequence[int], seed: int
) -> tuple[CountVectorizer, SGDClassifier]:
    """Fit a raw word-order model; deliberately uses no TF-IDF weighting."""
    texts: list[str] = []
    labels: list[float] = []
    for row_index in indices:
        row = rows[int(row_index)]
        assert row.originals is not None
        texts.extend(row.candidate_texts)
        labels.extend(row.originals)
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 7),
        min_df=2,
        max_features=300_000,
        binary=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=2e-6,
        max_iter=50,
        tol=1e-4,
        random_state=seed,
        average=True,
        class_weight="balanced",
    )
    classifier.fit(matrix, labels)
    return vectorizer, classifier


def score_raw_count_originality(
    rows: list[EncodedRow],
    indices: Sequence[int],
    model: tuple[CountVectorizer, SGDClassifier],
) -> np.ndarray:
    vectorizer, classifier = model
    texts = [text for row_index in indices for text in rows[int(row_index)].candidate_texts]
    scores = classifier.decision_function(vectorizer.transform(texts)).astype(np.float32)
    return scores.reshape(len(indices), 6)


def train_subword_tokenizer(frame: pd.DataFrame, config: Config) -> Tokenizer:
    """Train byte-BPE only on the supplied labelled-training subset."""
    tokenizer = Tokenizer(tokenizer_models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizer_pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = tokenizer_trainers.BpeTrainer(
        vocab_size=config.subword_vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=False,
    )

    def iterator() -> Iterable[str]:
        for row in frame.itertuples(index=False):
            for gap_index in range(1, 4):
                yield getattr(row, f"gap_{gap_index}")
            for text in parse_candidates(row.candidates).values():
                yield text

    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    return tokenizer


def original_training_excerpts(frame: pd.DataFrame) -> Iterable[str]:
    """Yield five-sentence original excerpts; counterfeits are excluded."""
    for row in frame.itertuples(index=False):
        candidates = parse_candidates(row.candidates)
        binding = str(row.binding).split(">")
        for gap_index, letter in enumerate(binding, start=1):
            left, right = split_gap(getattr(row, f"gap_{gap_index}"))
            yield f"{left} {candidates[letter]} {right}"


class MLMDataset(Dataset[list[int]]):
    def __init__(self, sequences: list[list[int]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


class MLMCollator:
    def __init__(self, tokenizer: Tokenizer, config: Config):
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.vocabulary_size = tokenizer.get_vocab_size()
        self.special_ids = torch.tensor(
            [tokenizer.token_to_id(token) for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")],
            dtype=torch.long,
        )
        self.config = config

    def __call__(self, sequences: Sequence[list[int]]) -> dict[str, torch.Tensor]:
        max_length = min(self.config.mlm_length, max(len(sequence) for sequence in sequences))
        input_ids = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, sequence in enumerate(sequences):
            length = min(len(sequence), max_length)
            input_ids[index, :length] = torch.tensor(sequence[:length], dtype=torch.long)
            attention_mask[index, :length] = True
        labels = input_ids.clone()
        special_mask = (input_ids[..., None] == self.special_ids).any(dim=-1)
        selected = (torch.rand(input_ids.shape) < 0.15) & attention_mask & ~special_mask
        labels[~selected] = -100
        replacement_draw = torch.rand(input_ids.shape)
        mask_replacement = selected & (replacement_draw < 0.80)
        random_replacement = selected & (replacement_draw >= 0.80) & (replacement_draw < 0.90)
        input_ids[mask_replacement] = self.mask_id
        random_tokens = torch.randint(5, self.vocabulary_size, input_ids.shape)
        input_ids[random_replacement] = random_tokens[random_replacement]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_masked_language_encoder(
    frame: pd.DataFrame,
    tokenizer: Tokenizer,
    config: Config,
    device: torch.device,
) -> BertModel:
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    sequences = []
    for text in original_training_excerpts(frame):
        content = tokenizer.encode(text, add_special_tokens=False).ids[: config.mlm_length - 2]
        sequences.append([cls_id] + content + [sep_id])
    loader = DataLoader(
        MLMDataset(sequences),
        batch_size=config.mlm_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=MLMCollator(tokenizer, config),
    )
    bert_config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.10,
        attention_probs_dropout_prob=0.10,
        max_position_embeddings=256,
        type_vocab_size=2,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        bos_token_id=cls_id,
        eos_token_id=sep_id,
    )
    model = BertForMaskedLM(bert_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = config.mlm_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, total_steps, config.warmup_fraction),
    )
    scaler = GradScaler(device.type, enabled=True)
    for epoch in range(config.mlm_epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        started = time.time()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = batch["input_ids"].shape[0]
            total_loss += float(loss.detach()) * count
            total_examples += count
        print(
            json.dumps(
                {
                    "mlm_epoch": epoch + 1,
                    "mlm_loss": total_loss / total_examples,
                    "seconds": time.time() - started,
                }
            ),
            flush=True,
        )
    encoder = model.bert
    del model.cls
    return encoder


@dataclass
class SubwordRow:
    row_id: int
    left: list[list[int]]
    right: list[list[int]]
    candidates: list[list[int]]
    pairs: list[tuple[int, int]]
    targets: list[int] | None
    originals: list[float] | None


def encode_subword_frame(frame: pd.DataFrame, tokenizer: Tokenizer, labelled: bool) -> list[SubwordRow]:
    result: list[SubwordRow] = []
    for row in frame.itertuples(index=False):
        parsed = parse_candidates(row.candidates)
        left_parts: list[list[int]] = []
        right_parts: list[list[int]] = []
        for gap_index in range(1, 4):
            left, right = split_gap(getattr(row, f"gap_{gap_index}"))
            left_parts.append(tokenizer.encode(left, add_special_tokens=False).ids)
            right_parts.append(tokenizer.encode(right, add_special_tokens=False).ids)
        candidate_parts = [tokenizer.encode(parsed[letter], add_special_tokens=False).ids for letter in LETTERS]
        targets = None
        originals = None
        if labelled:
            binding = str(row.binding).split(">")
            targets = [LETTERS.index(letter) for letter in binding]
            originals = [float(letter in set(binding)) for letter in LETTERS]
        result.append(
            SubwordRow(
                row_id=int(row.id),
                left=left_parts,
                right=right_parts,
                candidates=candidate_parts,
                pairs=candidate_pairs(parsed),
                targets=targets,
                originals=originals,
            )
        )
    return result


class SubwordBatchBuilder:
    def __init__(self, rows: list[SubwordRow], tokenizer: Tokenizer, config: Config, labelled: bool):
        self.rows = rows
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.config = config
        self.labelled = labelled

    def __call__(self, items: Sequence[tuple[int, int]]) -> dict[str, torch.Tensor]:
        sequences: list[list[int]] = []
        types: list[list[int]] = []
        candidate_masks: list[list[bool]] = []
        targets: list[int] = []
        originals: list[list[float]] = []
        pairs: list[list[tuple[int, int]]] = []
        pair_targets: list[int] = []
        row_indices: list[int] = []
        gap_indices: list[int] = []
        for row_index, gap_index in items:
            row = self.rows[row_index]
            left = row.left[gap_index][-self.config.left_tokens :]
            right = row.right[gap_index][: self.config.right_tokens]
            for candidate in row.candidates:
                candidate = candidate[: self.config.candidate_tokens]
                sequence = [self.cls_id] + left + [self.sep_id] + candidate + [self.sep_id] + right + [self.sep_id]
                token_types = [0] * (len(left) + 2) + [1] * (len(candidate) + 1) + [0] * (len(right) + 1)
                candidate_mask = [False] * (len(left) + 2) + [True] * len(candidate) + [False] * (len(right) + 2)
                sequences.append(sequence)
                types.append(token_types)
                candidate_masks.append(candidate_mask)
            if self.labelled:
                assert row.targets is not None and row.originals is not None
                target = row.targets[gap_index]
                targets.append(target)
                originals.append(row.originals)
                pairs.append(row.pairs)
                pair_targets.append(next(index for index, pair in enumerate(row.pairs) if target in pair))
            row_indices.append(row_index)
            gap_indices.append(gap_index)
        max_length = max(map(len, sequences))
        input_ids = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
        token_type_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
        candidate_mask_tensor = torch.zeros((len(sequences), max_length), dtype=torch.bool)
        for index, (sequence, token_types, candidate_mask) in enumerate(zip(sequences, types, candidate_masks)):
            length = len(sequence)
            input_ids[index, :length] = torch.tensor(sequence)
            attention_mask[index, :length] = True
            token_type_ids[index, :length] = torch.tensor(token_types)
            candidate_mask_tensor[index, :length] = torch.tensor(candidate_mask)
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "candidate_mask": candidate_mask_tensor,
            "row_indices": torch.tensor(row_indices, dtype=torch.long),
            "gap_indices": torch.tensor(gap_indices, dtype=torch.long),
        }
        if self.labelled:
            output["targets"] = torch.tensor(targets, dtype=torch.long)
            output["originals"] = torch.tensor(originals, dtype=torch.float32)
            output["pairs"] = torch.tensor(pairs, dtype=torch.long)
            output["pair_targets"] = torch.tensor(pair_targets, dtype=torch.long)
        return output


class BertBridgeRanker(nn.Module):
    def __init__(self, encoder: BertModel):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.compatibility_head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1)
        )
        self.originality_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1)
        )
        self.compatibility_head.apply(CoherenceTransformer._initialize)
        self.originality_head.apply(CoherenceTransformer._initialize)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        weights = candidate_mask.unsqueeze(-1).to(hidden.dtype)
        candidate_pool = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        cls_pool = hidden[:, 0]
        compatibility = self.compatibility_head(torch.cat([cls_pool, candidate_pool], dim=-1)).squeeze(-1)
        originality = self.originality_head(candidate_pool).squeeze(-1)
        return compatibility, originality


def make_subword_loader(
    rows: list[SubwordRow],
    indices: Sequence[int],
    tokenizer: Tokenizer,
    config: Config,
    labelled: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        GapDataset(indices),
        batch_size=config.rank_batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=SubwordBatchBuilder(rows, tokenizer, config, labelled),
    )


def move_subword_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("input_ids", "attention_mask", "token_type_ids", "candidate_mask")
    }


@torch.no_grad()
def score_subword_rows(
    model: BertBridgeRanker,
    rows: list[SubwordRow],
    indices: Sequence[int],
    tokenizer: Tokenizer,
    config: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_subword_loader(rows, indices, tokenizer, config, labelled=False, shuffle=False)
    local_position = {int(row_index): position for position, row_index in enumerate(indices)}
    compatibility_scores = np.zeros((len(indices), 3, 6), dtype=np.float32)
    originality_scores = np.zeros((len(indices), 3, 6), dtype=np.float32)
    for batch in loader:
        with autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            compatibility, originality = model(**move_subword_batch(batch, device))
        compatibility = compatibility.view(-1, 6).float().cpu().numpy()
        originality = originality.view(-1, 6).float().cpu().numpy()
        for batch_index, (row_index, gap_index) in enumerate(
            zip(batch["row_indices"].numpy(), batch["gap_indices"].numpy())
        ):
            position = local_position[int(row_index)]
            compatibility_scores[position, int(gap_index)] = compatibility[batch_index]
            originality_scores[position, int(gap_index)] = originality[batch_index]
    return compatibility_scores, originality_scores


def train_subword_ranker(
    rows: list[SubwordRow],
    train_indices: Sequence[int],
    tokenizer: Tokenizer,
    encoder: BertModel,
    config: Config,
    device: torch.device,
    evaluation_rows: list[EncodedRow] | None = None,
    validation_indices: Sequence[int] | None = None,
    validation_ngram: np.ndarray | None = None,
) -> tuple[BertBridgeRanker, dict[str, float]]:
    model = BertBridgeRanker(encoder).to(device)
    loader = make_subword_loader(rows, train_indices, tokenizer, config, labelled=True, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02, betas=(0.9, 0.98))
    total_steps = config.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total_steps, config.warmup_fraction)
    )
    scaler = GradScaler(device.type, enabled=True)
    best_score = -1.0
    best_details: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_groups = 0
        started = time.time()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = batch["targets"].to(device, non_blocking=True)
            originals = batch["originals"].to(device, non_blocking=True)
            pairs = batch["pairs"].to(device, non_blocking=True)
            pair_targets = batch["pair_targets"].to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                compatibility, originality = model(**move_subword_batch(batch, device))
                compatibility = compatibility.view(-1, 6)
                originality = originality.view(-1, 6)
                rank_loss = F.cross_entropy(compatibility + originality, targets)
                pair_logits = torch.gather(compatibility[:, None, :].expand(-1, 3, -1), 2, pairs).mean(dim=-1)
                pair_loss = F.cross_entropy(pair_logits, pair_targets)
                originality_loss = F.binary_cross_entropy_with_logits(originality, originals)
                loss = rank_loss + pair_loss + originality_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = targets.numel()
            total_loss += float(loss.detach()) * count
            total_groups += count
        message: dict[str, float | int] = {
            "rank_epoch": epoch + 1,
            "rank_loss": total_loss / total_groups,
            "seconds": time.time() - started,
        }
        if validation_indices is not None and evaluation_rows is not None:
            compatibility_scores, originality_scores = score_subword_rows(
                model, rows, validation_indices, tokenizer, config, device
            )
            details = tune_and_evaluate(
                evaluation_rows,
                validation_indices,
                compatibility_scores,
                originality_scores,
                ngram_originality=validation_ngram,
            )
            message.update(details)
            if details["meridian"] > best_score:
                best_score = details["meridian"]
                best_details = details
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(message, sort_keys=True), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_details


class CausalSequenceDataset(Dataset[list[int]]):
    def __init__(self, sequences: list[list[int]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


class CausalTrainCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, sequences: Sequence[list[int]]) -> dict[str, torch.Tensor]:
        max_length = max(map(len, sequences))
        input_ids = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, : len(sequence)] = True
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_bidirectional_causal_lm(
    frame: pd.DataFrame,
    tokenizer: Tokenizer,
    config: Config,
    device: torch.device,
) -> GPT2LMHeadModel:
    forward_id = tokenizer.token_to_id("[CLS]")
    reverse_id = tokenizer.token_to_id("[MASK]")
    sep_id = tokenizer.token_to_id("[SEP]")
    sequences: list[list[int]] = []
    for text in original_training_excerpts(frame):
        content = tokenizer.encode(text, add_special_tokens=False).ids[: config.mlm_length - 2]
        sequences.append([forward_id] + content + [sep_id])
        sequences.append([reverse_id] + list(reversed(content)) + [sep_id])
    loader = DataLoader(
        CausalSequenceDataset(sequences),
        batch_size=config.causal_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=CausalTrainCollator(tokenizer.token_to_id("[PAD]")),
    )
    model_config = GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=config.mlm_length,
        n_ctx=config.mlm_length,
        n_embd=256,
        n_layer=4,
        n_head=8,
        n_inner=768,
        resid_pdrop=0.10,
        embd_pdrop=0.10,
        attn_pdrop=0.10,
        bos_token_id=forward_id,
        eos_token_id=sep_id,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        use_cache=False,
    )
    model = GPT2LMHeadModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = config.causal_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total_steps, config.warmup_fraction)
    )
    scaler = GradScaler(device.type, enabled=True)
    for epoch in range(config.causal_epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        started = time.time()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = batch["input_ids"].shape[0]
            total_loss += float(loss.detach()) * count
            total_examples += count
        print(
            json.dumps(
                {
                    "causal_epoch": epoch + 1,
                    "causal_loss": total_loss / total_examples,
                    "seconds": time.time() - started,
                }
            ),
            flush=True,
        )
    return model


class CausalScoringDataset(Dataset[tuple[list[int], list[bool], int, int, int, int]]):
    def __init__(
        self,
        rows: list[SubwordRow],
        indices: Sequence[int],
        tokenizer: Tokenizer,
        config: Config,
    ):
        self.rows = rows
        self.indices = [int(index) for index in indices]
        self.forward_id = tokenizer.token_to_id("[CLS]")
        self.reverse_id = tokenizer.token_to_id("[MASK]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.config = config

    def __len__(self) -> int:
        return len(self.indices) * 3 * 6 * 2

    def __getitem__(self, index: int) -> tuple[list[int], list[bool], int, int, int, int]:
        direction = index % 2
        base = index // 2
        candidate_index = base % 6
        gap_index = (base // 6) % 3
        local_row = base // 18
        row_index = self.indices[local_row]
        row = self.rows[row_index]
        candidate = row.candidates[candidate_index][: self.config.candidate_tokens]
        if direction == 0:
            left = row.left[gap_index][-self.config.left_tokens :]
            right = row.right[gap_index][:32]
            sequence = [self.forward_id] + left + candidate + right + [self.sep_id]
            score_mask = (
                [False] * (1 + len(left))
                + [True] * len(candidate)
                + [True] * min(16, len(right))
                + [False] * (len(right) - min(16, len(right)) + 1)
            )
        else:
            right = list(reversed(row.right[gap_index][: self.config.right_tokens]))
            reversed_candidate = list(reversed(candidate))
            left = list(reversed(row.left[gap_index][-32:]))
            sequence = [self.reverse_id] + right + reversed_candidate + left + [self.sep_id]
            score_mask = (
                [False] * (1 + len(right))
                + [True] * len(reversed_candidate)
                + [True] * min(16, len(left))
                + [False] * (len(left) - min(16, len(left)) + 1)
            )
        return sequence, score_mask, local_row, gap_index, candidate_index, direction


class CausalScoreCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, items: Sequence[tuple[list[int], list[bool], int, int, int, int]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item[0]) for item in items)
        input_ids = torch.full((len(items), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        score_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        metadata = torch.zeros((len(items), 4), dtype=torch.long)
        for index, (sequence, mask, row, gap, candidate, direction) in enumerate(items):
            input_ids[index, : len(sequence)] = torch.tensor(sequence)
            attention_mask[index, : len(sequence)] = True
            score_mask[index, : len(mask)] = torch.tensor(mask)
            metadata[index] = torch.tensor([row, gap, candidate, direction])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "score_mask": score_mask,
            "metadata": metadata,
        }


@torch.no_grad()
def score_causal_insertions(
    model: GPT2LMHeadModel,
    rows: list[SubwordRow],
    indices: Sequence[int],
    tokenizer: Tokenizer,
    config: Config,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        CausalScoringDataset(rows, indices, tokenizer, config),
        batch_size=64,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=CausalScoreCollator(tokenizer.token_to_id("[PAD]")),
    )
    directional = np.zeros((len(indices), 3, 6, 2), dtype=np.float32)
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            token_log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            target_ids = input_ids[:, 1:]
            selected = torch.gather(token_log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            mask = batch["score_mask"][:, 1:].to(device, non_blocking=True)
            scores = (selected * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        for score, metadata in zip(scores.float().cpu().numpy(), batch["metadata"].numpy()):
            row, gap, candidate, direction = map(int, metadata)
            directional[row, gap, candidate, direction] = score
    return directional


def decode_row(
    row: EncodedRow,
    compatibility: np.ndarray,
    originality: np.ndarray,
    originality_weight: float,
    ngram_originality: np.ndarray | None = None,
    ngram_weight: float = 0.0,
    bridge_weight: float = 0.0,
) -> list[int]:
    # Average the auxiliary signal across gaps so that it reflects sentence
    # fluency rather than accidental compatibility with one context.
    stable_originality = originality.mean(axis=0)
    stable_originality = originality_weight * stable_originality
    if ngram_originality is not None:
        stable_originality = stable_originality + ngram_weight * ngram_originality
    scores = (
        compatibility
        + stable_originality[None, :]
        + bridge_weight * np.asarray(row.bridge_prior, dtype=np.float32)
    )
    best_score = -float("inf")
    best_assignment: list[int] | None = None
    # Exactly one member of every same-token pair, with the three selected
    # candidates assigned bijectively to the three gaps: only 48 hypotheses.
    for pair_order in permutations(range(3)):
        ordered_pairs = [row.pairs[pair_index] for pair_index in pair_order]
        for member_choices in product((0, 1), repeat=3):
            assignment = [ordered_pairs[gap][member_choices[gap]] for gap in range(3)]
            score = sum(float(scores[gap, candidate]) for gap, candidate in enumerate(assignment))
            if score > best_score:
                best_score = score
                best_assignment = assignment
    assert best_assignment is not None
    return best_assignment


def meridian_components(prediction: Sequence[int], truth: Sequence[int]) -> dict[str, float]:
    position = sum(left == right for left, right in zip(prediction, truth)) / 3.0
    membership = len(set(prediction).intersection(truth)) / 3.0
    precedence_hits = 0
    for first, second in ((truth[0], truth[1]), (truth[0], truth[2]), (truth[1], truth[2])):
        if first in prediction and second in prediction and prediction.index(first) < prediction.index(second):
            precedence_hits += 1
    precedence = precedence_hits / 3.0
    exact = float(list(prediction) == list(truth))
    meridian = 0.30 * position + 0.15 * membership + 0.05 * precedence + 0.50 * exact
    return {
        "position": position,
        "membership": membership,
        "precedence": precedence,
        "exact": exact,
        "meridian": meridian,
    }


def evaluate_predictions(rows: list[EncodedRow], indices: Sequence[int], predictions: Sequence[Sequence[int]]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for row_index, prediction in zip(indices, predictions):
        truth = rows[int(row_index)].targets
        assert truth is not None
        totals.update(meridian_components(prediction, truth))
    return {key: value / len(indices) for key, value in totals.items()}


def tune_and_evaluate(
    rows: list[EncodedRow],
    indices: Sequence[int],
    compatibility: np.ndarray,
    originality: np.ndarray,
    ngram_originality: np.ndarray | None = None,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    originality_weights = (0.0, 1.0, 2.0, 4.0, 8.0)
    ngram_weights = (0.0,) if ngram_originality is None else (0.0, 0.1, 0.25, 0.5, 1.0)
    bridge_weights = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
    for weight in originality_weights:
        for ngram_weight in ngram_weights:
            for bridge_weight in bridge_weights:
                predictions = [
                    decode_row(
                        rows[int(row_index)],
                        compatibility[position],
                        originality[position],
                        float(weight),
                        None if ngram_originality is None else ngram_originality[position],
                        float(ngram_weight),
                        float(bridge_weight),
                    )
                    for position, row_index in enumerate(indices)
                ]
                details = evaluate_predictions(rows, indices, predictions)
                details["originality_weight"] = float(weight)
                details["ngram_weight"] = float(ngram_weight)
                details["bridge_weight"] = float(bridge_weight)
                if best is None or details["meridian"] > best["meridian"]:
                    best = details
    assert best is not None
    return best


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def story_overlap_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group rows sharing a long exact context sentence.

    This prevents nearly identical excerpts from straddling train and
    validation.  It uses labelled training contexts only.
    """
    disjoint = DisjointSet(len(frame))
    owner: dict[str, int] = {}
    for row_index, row in enumerate(frame.itertuples(index=False)):
        for gap_index in range(1, 4):
            left, right = split_gap(getattr(row, f"gap_{gap_index}"))
            for sentence in re.split(r"(?<=[.!?])\s+", f"{left} {right}"):
                normalized = normalized_words(sentence)
                if len(normalized) < 60:
                    continue
                if normalized in owner:
                    disjoint.union(row_index, owner[normalized])
                else:
                    owner[normalized] = row_index
    return np.asarray([disjoint.find(index) for index in range(len(frame))], dtype=np.int64)


def grouped_split(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = story_overlap_groups(frame)
    members: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[int(group)].append(index)
    group_ids = list(members)
    random.Random(seed).shuffle(group_ids)
    target_rows = int(round(len(frame) * fraction))
    validation: list[int] = []
    for group in group_ids:
        if len(validation) >= target_rows:
            break
        validation.extend(members[group])
    validation_set = set(validation)
    train = np.asarray([index for index in range(len(frame)) if index not in validation_set], dtype=np.int64)
    validation_array = np.asarray(sorted(validation), dtype=np.int64)
    if set(groups[train]).intersection(groups[validation_array]):
        raise AssertionError("Grouped split leaked a connected story component.")
    return train, validation_array


def write_submission(
    output_path: Path,
    rows: list[EncodedRow],
    compatibility: np.ndarray,
    originality: np.ndarray,
    originality_weight: float,
    ngram_originality: np.ndarray | None = None,
    ngram_weight: float = 0.0,
    bridge_weight: float = 0.0,
) -> None:
    bindings: list[str] = []
    for position, row in enumerate(rows):
        assignment = decode_row(
            row,
            compatibility[position],
            originality[position],
            originality_weight,
            None if ngram_originality is None else ngram_originality[position],
            ngram_weight,
            bridge_weight,
        )
        bindings.append(">".join(LETTERS[index] for index in assignment))
    submission = pd.DataFrame({"id": [row.row_id for row in rows], "binding": bindings})
    validate_submission(submission, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def validate_submission(submission: pd.DataFrame, rows: list[EncodedRow]) -> None:
    if list(submission.columns) != ["id", "binding"]:
        raise ValueError("Submission columns must be exactly id,binding.")
    if len(submission) != len(rows) or submission["id"].duplicated().any():
        raise ValueError("Submission must contain every test id exactly once.")
    if set(submission["id"].astype(int)) != {row.row_id for row in rows}:
        raise ValueError("Submission ids do not exactly match test ids.")
    binding_pattern = re.compile(r"^[A-F]>[A-F]>[A-F]$")
    for binding in submission["binding"].astype(str):
        letters = binding.split(">")
        if not binding_pattern.fullmatch(binding) or len(set(letters)) != 3:
            raise ValueError(f"Invalid binding: {binding}")


def run_scratch_bert(
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
) -> None:
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rank_batch_size=args.batch_size,
        mlm_epochs=args.mlm_epochs,
    )
    if args.mode == "validate":
        train_indices, validation_indices = grouped_split(
            train_frame, config.validation_fraction, config.seed
        )
        fitting_frame = train_frame.iloc[train_indices].reset_index(drop=True)
        print(
            json.dumps(
                {
                    "architecture": "scratch-bert",
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "config": asdict(config),
                }
            ),
            flush=True,
        )
        tokenizer = train_subword_tokenizer(fitting_frame, config)
        evaluation_vocabulary, _ = build_vocabulary(fitting_frame, config)
        evaluation_rows = encode_frame(train_frame, evaluation_vocabulary, labelled=True)
        raw_count_model = fit_raw_count_originality(evaluation_rows, train_indices, config.seed)
        validation_ngram = score_raw_count_originality(
            evaluation_rows, validation_indices, raw_count_model
        )
        encoder = train_masked_language_encoder(fitting_frame, tokenizer, config, device)
        subword_rows = encode_subword_frame(train_frame, tokenizer, labelled=True)
        _, details = train_subword_ranker(
            subword_rows,
            train_indices,
            tokenizer,
            encoder,
            config,
            device,
            evaluation_rows=evaluation_rows,
            validation_indices=validation_indices,
            validation_ngram=validation_ngram,
        )
        print(json.dumps({"best_validation": details}, sort_keys=True), flush=True)
        return

    test_frame = pd.read_csv(data_dir / "test.csv")
    expected_test_columns = ["id", "gap_1", "gap_2", "gap_3", "candidates"]
    if list(test_frame.columns) != expected_test_columns:
        raise ValueError(f"Unexpected test columns: {list(test_frame.columns)}")
    tokenizer = train_subword_tokenizer(train_frame, config)
    evaluation_vocabulary, _ = build_vocabulary(train_frame, config)
    evaluation_train_rows = encode_frame(train_frame, evaluation_vocabulary, labelled=True)
    evaluation_test_rows = encode_frame(test_frame, evaluation_vocabulary, labelled=False)
    train_indices = np.arange(len(train_frame), dtype=np.int64)
    test_indices = np.arange(len(test_frame), dtype=np.int64)
    raw_count_model = fit_raw_count_originality(evaluation_train_rows, train_indices, config.seed)
    test_ngram = score_raw_count_originality(evaluation_test_rows, test_indices, raw_count_model)
    encoder = train_masked_language_encoder(train_frame, tokenizer, config, device)
    subword_train_rows = encode_subword_frame(train_frame, tokenizer, labelled=True)
    subword_test_rows = encode_subword_frame(test_frame, tokenizer, labelled=False)
    model, _ = train_subword_ranker(
        subword_train_rows,
        train_indices,
        tokenizer,
        encoder,
        config,
        device,
    )
    compatibility, originality = score_subword_rows(
        model, subword_test_rows, test_indices, tokenizer, config, device
    )
    write_submission(
        args.output,
        evaluation_test_rows,
        compatibility,
        originality,
        args.originality_weight,
        test_ngram,
        args.ngram_weight,
        args.bridge_weight,
    )
    print(json.dumps({"submission": str(args.output), "rows": len(test_frame)}), flush=True)


def run_causal_lm(
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
) -> None:
    config = Config(
        seed=args.seed,
        batch_size=args.batch_size,
        causal_batch_size=args.batch_size,
        causal_epochs=args.causal_epochs,
    )
    if args.mode == "validate":
        train_indices, validation_indices = grouped_split(
            train_frame, config.validation_fraction, config.seed
        )
        fitting_frame = train_frame.iloc[train_indices].reset_index(drop=True)
        print(
            json.dumps(
                {
                    "architecture": "causal-lm",
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "config": asdict(config),
                }
            ),
            flush=True,
        )
        tokenizer = train_subword_tokenizer(fitting_frame, config)
        evaluation_vocabulary, _ = build_vocabulary(fitting_frame, config)
        evaluation_rows = encode_frame(train_frame, evaluation_vocabulary, labelled=True)
        raw_count_model = fit_raw_count_originality(evaluation_rows, train_indices, config.seed)
        validation_ngram = score_raw_count_originality(
            evaluation_rows, validation_indices, raw_count_model
        )
        subword_rows = encode_subword_frame(train_frame, tokenizer, labelled=True)
        directional_sum = np.zeros((len(validation_indices), 3, 6, 2), dtype=np.float32)
        for ensemble_index in range(args.ensemble):
            seed_everything(config.seed + 1009 * ensemble_index)
            language_model = train_bidirectional_causal_lm(
                fitting_frame, tokenizer, config, device
            )
            directional_sum += score_causal_insertions(
                language_model,
                subword_rows,
                validation_indices,
                tokenizer,
                config,
                device,
            )
            del language_model
            torch.cuda.empty_cache()
        directional_sum /= args.ensemble
        details: dict[str, float] | None = None
        for forward_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            compatibility = (
                forward_weight * directional_sum[..., 0]
                + (1.0 - forward_weight) * directional_sum[..., 1]
            )
            candidate = tune_and_evaluate(
                evaluation_rows,
                validation_indices,
                compatibility,
                np.zeros_like(compatibility),
                ngram_originality=validation_ngram,
            )
            candidate["forward_weight"] = float(forward_weight)
            if details is None or candidate["meridian"] > details["meridian"]:
                details = candidate
        assert details is not None
        print(json.dumps({"best_validation": details}, sort_keys=True), flush=True)
        return

    test_frame = pd.read_csv(data_dir / "test.csv")
    tokenizer = train_subword_tokenizer(train_frame, config)
    evaluation_vocabulary, _ = build_vocabulary(train_frame, config)
    evaluation_train_rows = encode_frame(train_frame, evaluation_vocabulary, labelled=True)
    evaluation_test_rows = encode_frame(test_frame, evaluation_vocabulary, labelled=False)
    train_indices = np.arange(len(train_frame), dtype=np.int64)
    test_indices = np.arange(len(test_frame), dtype=np.int64)
    raw_count_model = fit_raw_count_originality(evaluation_train_rows, train_indices, config.seed)
    test_ngram = score_raw_count_originality(evaluation_test_rows, test_indices, raw_count_model)
    subword_test_rows = encode_subword_frame(test_frame, tokenizer, labelled=False)
    directional_sum = np.zeros((len(test_indices), 3, 6, 2), dtype=np.float32)
    for ensemble_index in range(args.ensemble):
        seed_everything(config.seed + 1009 * ensemble_index)
        language_model = train_bidirectional_causal_lm(train_frame, tokenizer, config, device)
        directional_sum += score_causal_insertions(
            language_model, subword_test_rows, test_indices, tokenizer, config, device
        )
        del language_model
        torch.cuda.empty_cache()
    directional_sum /= args.ensemble
    compatibility = (
        args.forward_weight * directional_sum[..., 0]
        + (1.0 - args.forward_weight) * directional_sum[..., 1]
    )
    write_submission(
        args.output,
        evaluation_test_rows,
        compatibility,
        np.zeros_like(compatibility),
        0.0,
        test_ngram,
        args.ngram_weight,
        args.bridge_weight,
    )
    print(json.dumps({"submission": str(args.output), "rows": len(test_frame)}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "train-predict"), nargs="?", default="train-predict")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--architecture", choices=("causal-lm", "scratch-bert", "hybrid"), default="hybrid"
    )
    parser.add_argument("--output", type=Path, default=Path("working/submission.csv"))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--ensemble", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--originality-weight", type=float, default=1.0)
    parser.add_argument("--ngram-weight", type=float, default=0.5)
    parser.add_argument("--bridge-weight", type=float, default=4.0)
    parser.add_argument("--mlm-epochs", type=int, default=Config.mlm_epochs)
    parser.add_argument("--causal-epochs", type=int, default=Config.causal_epochs)
    parser.add_argument("--forward-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
    seed_everything(config.seed)
    data_dir = locate_data_dir(args.data_dir)
    train_frame = pd.read_csv(data_dir / "train.csv")
    expected_train_columns = ["id", "gap_1", "gap_2", "gap_3", "candidates", "binding"]
    if list(train_frame.columns) != expected_train_columns:
        raise ValueError(f"Unexpected train columns: {list(train_frame.columns)}")
    if not torch.cuda.is_available():
        raise RuntimeError("The challenge requires a GPU for the primary sequence model, but CUDA is unavailable.")
    device = torch.device("cuda")
    if args.architecture == "causal-lm":
        run_causal_lm(args, train_frame, data_dir, device)
        return
    if args.architecture == "scratch-bert":
        run_scratch_bert(args, train_frame, data_dir, device)
        return
    vocabulary, id_to_token = build_vocabulary(train_frame, config)
    print(json.dumps({"config": asdict(config), "device": "cuda", "vocabulary": len(vocabulary)}))
    train_rows = encode_frame(train_frame, vocabulary, labelled=True)

    if args.mode == "validate":
        train_indices, validation_indices = grouped_split(train_frame, config.validation_fraction, config.seed)
        print(json.dumps({"train_rows": len(train_indices), "validation_rows": len(validation_indices)}))
        raw_count_model = fit_raw_count_originality(train_rows, train_indices, config.seed)
        validation_ngram = score_raw_count_originality(train_rows, validation_indices, raw_count_model)
        _, details = train_model(
            train_rows,
            train_indices,
            config,
            len(vocabulary),
            device,
            validation_indices=validation_indices,
            validation_ngram=validation_ngram,
        )
        print(json.dumps({"best_validation": details}, sort_keys=True))
        return

    test_frame = pd.read_csv(data_dir / "test.csv")
    expected_test_columns = ["id", "gap_1", "gap_2", "gap_3", "candidates"]
    if list(test_frame.columns) != expected_test_columns:
        raise ValueError(f"Unexpected test columns: {list(test_frame.columns)}")
    # This transformation uses the already-frozen training vocabulary.
    test_rows = encode_frame(test_frame, vocabulary, labelled=False)
    train_indices = np.arange(len(train_rows), dtype=np.int64)
    test_indices = np.arange(len(test_rows), dtype=np.int64)
    compatibility_sum = np.zeros((len(test_rows), 3, 6), dtype=np.float32)
    originality_sum = np.zeros((len(test_rows), 3, 6), dtype=np.float32)
    raw_count_model = fit_raw_count_originality(train_rows, train_indices, config.seed)
    test_ngram = score_raw_count_originality(test_rows, test_indices, raw_count_model)
    for ensemble_index in range(args.ensemble):
        member_config = Config(**{**asdict(config), "seed": config.seed + 1009 * ensemble_index})
        model, _ = train_model(train_rows, train_indices, member_config, len(vocabulary), device)
        compatibility, originality = score_rows(model, test_rows, test_indices, member_config, device)
        compatibility_sum += compatibility
        originality_sum += originality
        del model
        torch.cuda.empty_cache()
    compatibility_sum /= args.ensemble
    originality_sum /= args.ensemble
    write_submission(
        args.output,
        test_rows,
        compatibility_sum,
        originality_sum,
        args.originality_weight,
        test_ngram,
        args.ngram_weight,
        args.bridge_weight,
    )
    print(json.dumps({"submission": str(args.output), "rows": len(test_rows)}))


if __name__ == "__main__":
    main()
