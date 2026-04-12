import os, re, glob
from typing import List, Optional, Tuple, Iterable
import random
import pyarrow.parquet as pq

from general_utils import (
    chunked as _chunked,
    estimate_clause_count,
    parse_tagged_output,
    strip_phrase_prefix as _strip_phrase_prefix,
    truncate_first_tokens as _truncate_caption_first_tokens,
)

from prompts.relation_prompts import (
    SYSTEM_MSG_REPHRASE,
    FEW_SHOT_REPHRASE,
    SYSTEM_MSG_NEGREL,
    FEW_SHOT_NEGREL,
    SYSTEM_MSG_CANON_REL,
    FEW_SHOT_CANON_REL,
)

# ======================== Helpers ===========================================

# -------- Helpers for clause counting & selection (outside the model) --------
def _choose_clause_indices(
    phrases: List[str],
    mode: str = "random",       # 'random' | 'alternate' | '<int>' (fixed 1-based)
    seed: Optional[int] = None,
    max_clauses: int = 5,
) -> List[int]:
    rng = random.Random(seed) if seed is not None else random
    indices: List[int] = []
    alt_toggle = False
    fixed = int(mode) if mode.isdigit() else None

    for p in phrases:
        k = estimate_clause_count(p, max_clauses=max_clauses)
        if fixed is not None:
            idx = min(max(1, fixed), k)
        elif mode == "alternate":
            idx = 1 if not alt_toggle else min(2, k)
            alt_toggle = not alt_toggle
        else:  # 'random'
            idx = rng.randint(1, k)
        indices.append(idx)
    return indices


def resolve_hf_snapshot_dir(repo_cache_dir: str) -> str:
    """Resolve a local HF snapshot dir (works for both dirs and bare repos)."""
    import os
    if os.path.isfile(os.path.join(repo_cache_dir, "config.json")):
        return repo_cache_dir
    refs_main = os.path.join(repo_cache_dir, "refs", "main")
    snaps = os.path.join(repo_cache_dir, "snapshots")
    if os.path.exists(refs_main):
        with open(refs_main) as f:
            commit = f.read().strip()
        cand = os.path.join(snaps, commit)
        if os.path.isdir(cand):
            return cand
    cand_list = sorted(glob.glob(os.path.join(snaps, "*")))
    if cand_list:
        return cand_list[-1]
    return repo_cache_dir


# ======================== Prompt building & parsing =========================

def _messages_rephrase(caption_text: str) -> List[dict]:
    """Chat messages for the 'rephrase' mode with one-shot."""
    msgs = [{"role": "system", "content": SYSTEM_MSG_REPHRASE}]
    cap, ans = FEW_SHOT_REPHRASE[0]
    demo_user = (
        "Caption:\n" + cap + "\n\n"
        "Task: Pick ONE object with at least one relation to another object and compose a SINGLE phrase "
        "listing 1–5 relations (spatial or action) that involve it. Return EXACTLY one line:\n"
        "PHRASE=<...>"
    )
    msgs.append({"role": "user", "content": demo_user})
    msgs.append({"role": "assistant", "content": ans})

    task_user = (
        "Caption:\n" + caption_text + "\n\n"
        "Task: Pick ONE object that has at least one relation with another object. "
        "Compose a SINGLE fluent phrase listing 1–5 relations (spatial or action) that involve the object, "
        "using only the caption.\n"
        "Return EXACTLY one line:\n"
        "PHRASE=<...>"
    )
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_rephrase_prompt(tokenizer, caption: str) -> str:
    """Final chat prompt using the FIRST 512 caption tokens."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption, 512)
    return tokenizer.apply_chat_template(
        _messages_rephrase(cap_trunc),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def parse_phrase_output(text: str) -> str:
    return parse_tagged_output(
        text,
        allowed_tags=("PHRASE",),
        max_chars=512,
    )

# ---------------- Prompt building for negative relations ----------------

def _messages_negrelation(rel_phrase: str, clause_index: int) -> List[dict]:
    """
    Chat messages for negative-relation editing with an explicit CLAUSE_INDEX.
    """
    msgs = [{"role": "system", "content": SYSTEM_MSG_NEGREL}]
    for demo_user, demo_assistant in FEW_SHOT_NEGREL:
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": demo_assistant})

    core = rel_phrase if rel_phrase.strip().lower().startswith("phrase=") else f"PHRASE={rel_phrase}"
    task_user = f"CLAUSE_INDEX={clause_index}\n{core}"
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_negrelation_prompt(tokenizer, rel_phrase: str, clause_index: int) -> str:
    """Final chat prompt for negative relation replacement with explicit index."""
    return tokenizer.apply_chat_template(
        _messages_negrelation(rel_phrase or "", clause_index),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
# ============================== Generation utils ============================

def batched_generate_phrases(
    llm,
    tokenizer,
    captions: List[str],
    sampling_params,
    gen_batch_size: int,
) -> List[str]:
    phrases: List[str] = []
    for batch_caps in _chunked(captions, gen_batch_size):
        prompts = [build_rephrase_prompt(tokenizer, c) for c in batch_caps]
        outs = llm.generate(prompts, sampling_params)
        for out in outs:
            txt = out.outputs[0].text if out.outputs else ""
            phrases.append(parse_phrase_output(txt))
    return phrases

# ---------------------- negative relation generation -------------------

def _head_prefix_and_count_is(phrase: str) -> Tuple[str, int]:
    """Return (prefix before first ' is ', count_of_' is '). Case-insensitive match for ' is '."""
    s = phrase or ""
    # Normalize spacing for counting but keep original for prefix slicing
    low = s.lower()
    first = low.find(" is ")
    if first == -1:
        return (s.strip(), 0)
    count_is = low.count(" is ")
    return (s[:first].strip(), count_is)

def batched_generate_negative_relation_phrases(
    llm,
    tokenizer,
    rel_phrases: List[str],
    sampling_params,
    gen_batch_size: int,
    clause_index_mode: str = "random",   # 'random' | 'alternate' | '<int>'
    seed: Optional[int] = None,
    max_clauses: int = 5,
    validate: bool = False, 
) -> List[str]:
    """
    For each relation phrase (e.g., 'The X is r1, is r2 and is r3'):
      - Choose a 1-based clause index via simple comma/and counting.
      - Instruct the model to edit exactly that clause.
      - Return ONLY the rewritten PHRASE string.
    """
    outputs: List[str] = []
    if not rel_phrases:
        return outputs

    # Pre-choose clause indices for the whole list (respects k ≤ max_clauses per phrase)
    chosen_idxs = _choose_clause_indices(
        rel_phrases, mode=clause_index_mode, seed=seed, max_clauses=max_clauses
    )

    for start in range(0, len(rel_phrases), gen_batch_size):
        end = min(start + gen_batch_size, len(rel_phrases))
        batch_rels = rel_phrases[start:end]
        batch_idxs = chosen_idxs[start:end]

        prompts = [
            build_negrelation_prompt(tokenizer, rp if rp is not None else "", idx)
            for rp, idx in zip(batch_rels, batch_idxs)
        ]
        gens = llm.generate(prompts, sampling_params)

        for out in gens:
            txt = out.outputs[0].text if out.outputs else ""
            outputs.append(parse_phrase_output(txt))

    return outputs

def _messages_canon_rel(raw_phrase: str) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_MSG_CANON_REL}]
    for u, a in FEW_SHOT_CANON_REL:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": (raw_phrase or "").strip()})
    return msgs

def build_canon_rel_prompt(tokenizer, raw_phrase: str) -> str:
    return tokenizer.apply_chat_template(
        _messages_canon_rel(raw_phrase or ""),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# Keep output handling minimal: first non-empty line, whitespace normalized.
def _parse_canon_rel_output(text: str) -> str:
    raw = (text or "").strip()
    line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
    # normalize internal whitespace only; do NOT change punctuation/words
    import re as _re
    return _re.sub(r"\s+", " ", line)

def batched_canonicalize_relation_phrases(
    llm,
    tokenizer,
    rel_phrases: List[str],
    sampling_params,
    gen_batch_size: int = 64,
    which: str = "pos",   # just for upstream bookkeeping if you need it
) -> List[str]:
    """
    LLM-only canonicalization to a noun phrase:
      - (maybe) lowercase first letter unless proper noun
      - insert exactly one ' that ' before the first verb
    No fallback, no validation, no PHRASE= wrapper.
    """
    outputs: List[str] = []
    if not rel_phrases:
        return outputs

    for start in range(0, len(rel_phrases), gen_batch_size):
        chunk = rel_phrases[start:start + gen_batch_size]
        prompts = [build_canon_rel_prompt(tokenizer, x if x is not None else "") for x in chunk]
        gens = llm.generate(prompts, sampling_params)
        for out in gens:
            gen_text = out.outputs[0].text if out.outputs else ""
            outputs.append(_parse_canon_rel_output(gen_text))
    return outputs
