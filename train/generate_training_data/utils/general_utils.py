# general_utils.py
from __future__ import annotations

import glob
import os
import re
from typing import Iterable, Optional, Sequence, Tuple, TypeVar

import pyarrow.parquet as pq

T = TypeVar("T")

DEFAULT_CAPTION_COLUMN_CANDIDATES = (
    "caption",
    "description",
    "image_description",
    "text",
    "long_caption",
)

_PHRASE_PREFIX_RE = re.compile(r"^\s*PHRASE\s*[:=]\s*", flags=re.IGNORECASE)


# ============================== HF / distributed ==============================

def resolve_hf_snapshot_dir(repo_cache_dir: str) -> str:
    """
    Resolve a local Hugging Face snapshot dir.

    Works for:
    - an already-resolved model directory containing config.json
    - a HF cache repo dir with refs/main + snapshots/<commit>
    - a repo dir with only snapshots/*
    """
    if os.path.isfile(os.path.join(repo_cache_dir, "config.json")):
        return repo_cache_dir

    refs_main = os.path.join(repo_cache_dir, "refs", "main")
    snaps = os.path.join(repo_cache_dir, "snapshots")

    if os.path.exists(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            commit = f.read().strip()
        candidate = os.path.join(snaps, commit)
        if os.path.isdir(candidate):
            return candidate

    candidates = sorted(glob.glob(os.path.join(snaps, "*")))
    if candidates:
        return candidates[-1]

    return repo_cache_dir


def get_dist_info() -> Tuple[int, int, Optional[int]]:
    """
    Return (rank, world_size, local_rank) from standard env vars / SLURM.
    """
    rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))
    world = int(os.getenv("WORLD_SIZE", os.getenv("SLURM_NTASKS", "1")))
    local_rank_env = os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID")
    local_rank = int(local_rank_env) if local_rank_env is not None else None
    return rank, world, local_rank


# ============================== parquet helpers ==============================

def detect_first_available_column(
    pf: pq.ParquetFile,
    preferred: str,
    candidates: Sequence[str],
    *,
    column_kind: str = "column",
) -> str:
    """
    Return the first available column name from:
      [preferred] + candidates
    """
    names = set(pf.schema_arrow.names)

    ordered = []
    if preferred:
        ordered.append(preferred)
    ordered.extend(candidates)

    seen = set()
    deduped = []
    for col in ordered:
        if col and col not in seen:
            deduped.append(col)
            seen.add(col)

    for col in deduped:
        if col in names:
            return col

    raise ValueError(
        f"{column_kind.capitalize()} not found. "
        f"Tried {deduped}. Available: {sorted(names)}"
    )


def detect_caption_col(pf: pq.ParquetFile, preferred: str = "caption") -> str:
    return detect_first_available_column(
        pf,
        preferred=preferred,
        candidates=DEFAULT_CAPTION_COLUMN_CANDIDATES,
        column_kind="caption column",
    )


# ============================== text helpers =================================

def truncate_first_tokens(tokenizer, text: str, max_tokens: int = 512) -> str:
    """
    Keep only the first `max_tokens` tokenizer tokens.
    """
    ids = tokenizer(text or "", add_special_tokens=False).input_ids
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def strip_phrase_prefix(text: str) -> str:
    """
    Remove a leading 'PHRASE=' or 'PHRASE:' if present.
    """
    text = (text or "").strip()
    return _PHRASE_PREFIX_RE.sub("", text, count=1).strip()


def estimate_clause_count(phrase: str, max_clauses: int = 5) -> int:
    """
    Very lightweight clause counting heuristic:
      - if there are commas: clauses = commas + 1
      - else if there is an 'and': clauses = 2
      - else: clauses = 1
    Then clamp to [1, max_clauses].
    """
    core = strip_phrase_prefix(phrase)
    if not core:
        return 1

    comma_count = core.count(",")
    if comma_count > 0:
        n = comma_count + 1
    else:
        n = 2 if re.search(r"\band\b", core, flags=re.IGNORECASE) else 1

    return max(1, min(n, max_clauses))


def parse_tagged_output(
    text: str,
    *,
    allowed_tags: Sequence[str],
    max_chars: int = 512,
    split_on_semicolon: bool = False,
    strip_trailing_pattern: str = r"[.;:\s]+$",
) -> str:
    """
    Parse a generated single-line output like:
      PHRASE=...
      SENTENCE=...
      ACC_PHRASE=...
    by scanning for the first matching tagged line.

    Fallback:
      - if no tagged line is found, use the first non-empty line.

    Normalization:
      - strip outer quotes
      - optional split on first ';'
      - strip trailing punctuation via regex
      - collapse internal whitespace
      - clip to max_chars
    """
    raw = (text or "").strip()
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and line.strip() not in {"```", "~~~"}
    ]

    if not lines:
        return ""

    tag_pattern = "|".join(re.escape(tag) for tag in allowed_tags)
    pattern = re.compile(
        rf"^\s*(?:{tag_pattern})\s*[:=]\s*(.+)$",
        flags=re.IGNORECASE,
    )

    value = ""
    for line in lines:
        match = pattern.match(line)
        if match:
            value = match.group(1).strip()
            break

    if not value:
        value = lines[0].strip()

    value = (
        value.strip()
        .strip('"')
        .strip("'")
        .strip("“”‘’")
        .strip()
    )

    if split_on_semicolon:
        value = value.split(";", 1)[0].strip()

    if strip_trailing_pattern:
        value = re.sub(strip_trailing_pattern, "", value)

    value = re.sub(r"\s+", " ", value).strip()

    if max_chars is not None:
        value = value[:max_chars]

    return value


# ============================== batching =====================================

def chunked(xs: Sequence[T], n: int) -> Iterable[Sequence[T]]:
    """
    Yield xs in chunks of size n.
    """
    if n <= 0:
        raise ValueError(f"chunk size must be positive, got {n}")
    for i in range(0, len(xs), n):
        yield xs[i : i + n]