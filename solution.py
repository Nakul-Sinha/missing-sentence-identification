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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


LETTERS = "ABCDEF"
CANDIDATE_RE = re.compile(r"^\[([A-F])\]\s(.*)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+", re.UNICODE)
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
    originality_loss_weight: float = 0.35
    validation_fraction: float = 0.15
    num_workers: int = 0

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
    return [token.lower().replace("’", "'") for token in TOKEN_RE.findall(str(text))]


def normalized_words(text: str) -> str:
    return " ".join(token.lower().replace("’", "'") for token in WORD_RE.findall(str(text)))


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
    candidates = [Path("dataset"), Path("input"), Path("/kaggle/working/dataset")]
    candidates.extend(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else None
    matches = [path for path in candidates if (path / "train.csv").is_file() and (path / "test.csv").is_file()]
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
    pairs: list[tuple[int, int]]
    targets: list[int] | None
    originals: list[float] | None


def encode_text(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, UNK) for token in tokenize(text)]


def encode_frame(frame: pd.DataFrame, vocabulary: dict[str, int], labelled: bool) -> list[EncodedRow]:
    encoded: list[EncodedRow] = []
    for row in frame.itertuples(index=False):
        parsed = parse_candidates(row.candidates)
        left_parts: list[list[int]] = []
        right_parts: list[list[int]] = []
        for gap_index in range(1, 4):
            left, right = split_gap(getattr(row, f"gap_{gap_index}"))
            left_parts.append(encode_text(left, vocabulary))
            right_parts.append(encode_text(right, vocabulary))
        candidate_ids = [encode_text(parsed[letter], vocabulary) for letter in LETTERS]
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


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    model_keys = ("input_ids", "segment_ids", "candidate_mask", "attention_mask")
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
) -> tuple[CoherenceTransformer, dict[str, float]]:
    seed_everything(config.seed)
    model = CoherenceTransformer(vocabulary_size, config).to(device)
    loader = make_loader(rows, train_indices, config, labelled=True, shuffle=True)
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
            model_inputs = move_batch(batch, device)
            target = batch["targets"].to(device, non_blocking=True)
            originals = batch["originals"].to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                compatibility, originality = model(**model_inputs)
                compatibility = compatibility.view(-1, 6)
                originality = originality.view(-1, 6)
                rank_loss = F.cross_entropy(compatibility + originality, target)
                originality_loss = F.binary_cross_entropy_with_logits(originality, originals)
                loss = rank_loss + config.originality_loss_weight * originality_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            group_count = target.numel()
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
            details = tune_and_evaluate(rows, validation_indices, compatibility, originality)
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
    model: CoherenceTransformer,
    rows: list[EncodedRow],
    indices: Sequence[int],
    config: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(rows, indices, config, labelled=False, shuffle=False)
    local_position = {int(row_index): position for position, row_index in enumerate(indices)}
    compatibility_scores = np.zeros((len(indices), 3, 6), dtype=np.float32)
    originality_scores = np.zeros((len(indices), 3, 6), dtype=np.float32)
    amp_enabled = device.type == "cuda"
    for batch in loader:
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            compatibility, originality = model(**move_batch(batch, device))
        compatibility = compatibility.view(-1, 6).float().cpu().numpy()
        originality = originality.view(-1, 6).float().cpu().numpy()
        row_indices = batch["row_indices"].numpy()
        gap_indices = batch["gap_indices"].numpy()
        for batch_index, (row_index, gap_index) in enumerate(zip(row_indices, gap_indices)):
            position = local_position[int(row_index)]
            compatibility_scores[position, int(gap_index)] = compatibility[batch_index]
            originality_scores[position, int(gap_index)] = originality[batch_index]
    return compatibility_scores, originality_scores


def decode_row(
    row: EncodedRow,
    compatibility: np.ndarray,
    originality: np.ndarray,
    originality_weight: float,
) -> list[int]:
    # Average the auxiliary signal across gaps so that it reflects sentence
    # fluency rather than accidental compatibility with one context.
    stable_originality = originality.mean(axis=0)
    scores = compatibility + originality_weight * stable_originality[None, :]
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
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for weight in np.arange(0.0, 3.01, 0.25):
        predictions = [
            decode_row(rows[int(row_index)], compatibility[position], originality[position], float(weight))
            for position, row_index in enumerate(indices)
        ]
        details = evaluate_predictions(rows, indices, predictions)
        details["originality_weight"] = float(weight)
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
) -> None:
    bindings: list[str] = []
    for position, row in enumerate(rows):
        assignment = decode_row(row, compatibility[position], originality[position], originality_weight)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "train-predict"), nargs="?", default="train-predict")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("working/submission.csv"))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--ensemble", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--originality-weight", type=float, default=1.0)
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
    vocabulary, id_to_token = build_vocabulary(train_frame, config)
    print(json.dumps({"config": asdict(config), "device": "cuda" if torch.cuda.is_available() else "cpu", "vocabulary": len(vocabulary)}))
    if not torch.cuda.is_available():
        raise RuntimeError("The challenge requires a GPU for the primary sequence model, but CUDA is unavailable.")
    device = torch.device("cuda")
    train_rows = encode_frame(train_frame, vocabulary, labelled=True)

    if args.mode == "validate":
        train_indices, validation_indices = grouped_split(train_frame, config.validation_fraction, config.seed)
        print(json.dumps({"train_rows": len(train_indices), "validation_rows": len(validation_indices)}))
        _, details = train_model(
            train_rows,
            train_indices,
            config,
            len(vocabulary),
            device,
            validation_indices=validation_indices,
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
    )
    print(json.dumps({"submission": str(args.output), "rows": len(test_rows)}))


if __name__ == "__main__":
    main()
