# wh_utils.py
import os, re, glob
from typing import List, Optional, Tuple, Iterable
import random
import pyarrow.parquet as pq

from general_utils import (
    chunked as _chunked,
    parse_tagged_output,
    truncate_first_tokens as _truncate_caption_first_tokens,
)

from prompts.wh_prompts import (
    SYSTEM_MSG_REPHRASE,
    FEW_SHOT_REPHRASE,
    SYSTEM_MSG_WHQA_A,
    SYSTEM_MSG_WHQA_B,
    FEW_SHOT_WHQA_A,
    FEW_SHOT_WHQA_B,
    SYSTEM_MSG_NEG_WHQA_ATTR,
    FEW_SHOT_NEG_WHQA_ATTR,
)

# ======================== Prompt building & parsing =========================

def _truncate_caption_first_tokens(tokenizer, caption: str, max_tokens: int = 512) -> str:
    """Keep ONLY the first `max_tokens` tokens of the caption."""
    ids = tokenizer(caption or "", add_special_tokens=False).input_ids
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def _messages_rephrase(caption_text: str) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_MSG_REPHRASE}]

    # Few-shots (ALL exemplars)
    for cap, ans in FEW_SHOT_REPHRASE:
        demo_user = (
            "Example:\n"
            "caption:\n" + cap + "\n\n"
            "Task: Select one sentence with two objects and an explicit relation between them. "
            "Rewrite it as one coherent sentence including attributes for each of the two objects if stated. "
            "Return EXACTLY one line:\nSENTENCE=<...>"
        )
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": ans})

    # Actual task
    task_user = (
        "Caption:\n" + caption_text + "\n\n"
        "Task: Select one sentence with two objects and an explicit relation between them. "
        "Rewrite it as one coherent sentence including attributes for each of the two objects if stated. "
        "Important:\n"
        "• The examples above are FORMAT ONLY — do NOT copy phrases, numbers, or details from them. Your content should be taken from the current caption.\n"
        "Return EXACTLY one line:\nSENTENCE=<...>"
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
    )

def parse_phrase_output(text: str) -> str:
    return parse_tagged_output(
        text,
        allowed_tags=("SENTENCE", "PHRASE"),
        max_chars=1024,
        split_on_semicolon=True,
        strip_trailing_pattern=r"[.]\s*$",
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


# ------------------------- Prompt builders & parser --------------------------

_QA_PAIR_RE = re.compile(
    r'^\s*Q\s*[:=]\s*(.+?)\s*\|\|\s*A\s*[:=]\s*(.+?)\s*$',
    flags=re.IGNORECASE
)

def parse_whqa_pair_output(text: str) -> str:
    """
    Parse 'Q=... || A=...' into the same single-line format.
    On failure, return a best-effort 'Q=<first> || A=<second_or_empty>'.
    """
    raw = (text or "").strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Q= || A="
    m = _QA_PAIR_RE.match(lines[0])
    if m:
        q = m.group(1).strip()
        a = m.group(2).strip()
        return f"Q={q} || A={a}"
    if "||" in lines[0]:
        left, right = lines[0].split("||", 1)
        q = re.sub(r'^\s*Q\s*[:=]\s*', '', left).strip()
        a = re.sub(r'^\s*A\s*[:=]\s*', '', right).strip()
        return f"Q={q} || A={a}"
    q = re.sub(r'^\s*Q\s*[:=]\s*', '', lines[0]).strip()
    a = re.sub(r'^\s*A\s*[:=]\s*', '', (lines[1] if len(lines) > 1 else "")).strip()
    return f"Q={q} || A={a}"

def _messages_whqa_sentence(sentence_text: str, choose: str) -> List[dict]:
    """Build chat messages for one sentence, choosing A (answer=obj_a) or B (answer=obj_b)."""
    if choose.lower() == "b":
        msgs = [{"role": "system", "content": SYSTEM_MSG_WHQA_B}]
        few = FEW_SHOT_WHQA_B
    else:
        msgs = [{"role": "system", "content": SYSTEM_MSG_WHQA_A}]
        few = FEW_SHOT_WHQA_A
    for s, ans in few:
        msgs.append({"role": "user", "content": s})
        msgs.append({"role": "assistant", "content": ans})
    msgs.append({"role": "user", "content": sentence_text or ""})
    return msgs

def build_whqa_sentence_prompt(tokenizer, sentence_text: str, choose: str) -> str:
    """Apply the chat template; 'choose' is 'a' or 'b'."""
    msgs = _messages_whqa_sentence(sentence_text, choose)
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

# ----------------------------- Batch generation -----------------------------

def _resolve_choices(n: int, flip: str = "random", seed: int | None = None) -> List[str]:
    """
    Build a list of 'a'/'b' choices for n samples.
    flip: 'random' (default), 'alternate', 'a', or 'b'.
    """
    flip = (flip or "random").lower()
    if flip in ("a", "b"):
        return [flip] * n
    if flip == "alternate":
        return [("a" if i % 2 == 0 else "b") for i in range(n)]
    # random
    if seed is not None:
        rnd = random.Random(seed)
        return [rnd.choice(["a", "b"]) for _ in range(n)]
    return [random.choice(["a", "b"]) for _ in range(n)]

def batched_generate_whqa_from_wh_column(
    llm,
    tokenizer,
    wh_cells: List[str],
    sampling_params,
    gen_batch_size: int,
    flip: str = "random",
    random_seed: int | None = None,
) -> List[str]:
    """
    For each 'wh' cell (now a SINGLE sentence), generate ONE QA pair.
    Returns: ['Q=... || A=...', ...] aligned to input rows.

    flip: 'random' (default), 'alternate', 'a', or 'b' to control which side is chosen as the answer.
    random_seed: optional seed for deterministic random flipping.
    """
    n = len(wh_cells)
    if n == 0:
        return []
    # Choose A/B per item
    choices = _resolve_choices(n, flip=flip, seed=random_seed)

    outputs: List[str] = []
    # Chunked generation
    for start in range(0, n, gen_batch_size):
        end = min(start + gen_batch_size, n)
        batch_sentences = wh_cells[start:end]
        batch_choices = choices[start:end]
        prompts = [
            build_whqa_sentence_prompt(tokenizer, s or "", choose=c)
            for s, c in zip(batch_sentences, batch_choices)
        ]
        gens = llm.generate(prompts, sampling_params)
        for g in gens:
            txt = g.outputs[0].text if g.outputs else ""
            outputs.append(parse_whqa_pair_output(txt))
    return outputs

# ------------- shared utils (split POS pairs and extract the question) -------------
_QA_SPLIT_RE = re.compile(r'\s*;\s*')

def _split_whqa_cell(cell: str, max_pairs: int = 3) -> List[str]:
    """Split a semicolon-joined 'Q=... || A=...' cell into lines (max 3)."""
    raw = (cell or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _QA_SPLIT_RE.split(raw) if p.strip()]
    return parts[:max_pairs]

def _extract_question(pair_line: str) -> str:
    """Return just the Q=... portion as plain text."""
    norm = parse_whqa_pair_output(pair_line or "")
    q, _ = norm.split("||", 1)
    q = re.sub(r'^\s*Q\s*[:=]\s*', '', q).strip()
    return q

def _parse_multi_neg_pairs(text: str, expected_n: int) -> List[str]:
    """
    Parse model output lines; each must contain 'Q=' and '||'.
    Returns up to expected_n normalized 'Q=... || A=...' lines.
    """
    raw = (text or "").strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    pairs: List[str] = []
    for ln in lines:
        if "Q=" in ln and "||" in ln:
            pairs.append(parse_whqa_pair_output(ln))
            if len(pairs) >= expected_n:
                break
    return pairs



def _messages_neg_whqa_attr(question: str) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_MSG_NEG_WHQA_ATTR}]
    for q, out in FEW_SHOT_NEG_WHQA_ATTR:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": out})
    msgs.append({"role": "user", "content": question or ""})
    return msgs

def build_neg_whqa_attr_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        _messages_neg_whqa_attr(question),
        tokenize=False,
        add_generation_prompt=True,
    )

# --------------- Batched generator (ATTRIBUTE-only; grouped per row) ---------------

def batched_generate_neg_whqa_from_whqa_column(
    llm,
    tokenizer,
    whqa_cells: List[str],
    sampling_params,
    gen_batch_size: int,
) -> List[str]:
    """
    For each row:
      • Input: up to three POSITIVE QA pairs (semicolon-joined 'Q=... || A=...').
      • For each pair, build an ATTRIBUTE prompt (no caption).
      • Generate one NEG 'Q=... || A=...' line per pair.
      • Return per-row strings joined with '; ' in the original order.
    """
    flat_prompts: List[str] = []
    flat_positions: List[Tuple[int, int]] = []  # (row_idx, pair_idx)

    groups_questions: List[List[str]] = []
    for row_idx, cell in enumerate(whqa_cells):
        pos_pairs = _split_whqa_cell(cell or "", max_pairs=3)
        questions = [_extract_question(p) for p in pos_pairs]
        groups_questions.append(questions)

        for pair_idx, q in enumerate(questions):
            flat_prompts.append(build_neg_whqa_attr_prompt(tokenizer, q))
            flat_positions.append((row_idx, pair_idx))

    # If no pairs at all, return empty list aligned to rows
    if not flat_prompts:
        return ["" for _ in whqa_cells]

    # Generate in micro-batches
    neg_lines_flat: List[str] = ["" for _ in flat_prompts]
    for start in range(0, len(flat_prompts), gen_batch_size):
        end = min(start + gen_batch_size, len(flat_prompts))
        gens = llm.generate(flat_prompts[start:end], sampling_params)
        for i, g in enumerate(gens, start=start):
            txt = g.outputs[0].text if g.outputs else ""
            parsed = _parse_multi_neg_pairs(txt, expected_n=1)  # expect exactly one line
            neg_lines_flat[i] = parsed[0] if parsed else ""

    # Regroup per row in original order
    row_outputs: List[List[str]] = [[] for _ in whqa_cells]
    for (row_idx, _pair_idx), neg_line in zip(flat_positions, neg_lines_flat):
        row_outputs[row_idx].append(neg_line)

    return ["; ".join([s for s in seq if s]) for seq in row_outputs]

# ---- Optional: compatibility shim if your caller still imports the old name ----
def batched_generate_neg_whqa_from_whqa_column_with_captions(
    llm,
    tokenizer,
    whqa_cells: List[str],
    captions: List[str],          # IGNORED (kept for API compatibility)
    sampling_params,
    gen_batch_size: int,
    *_, **__,
) -> List[str]:
    """Backward-compatible wrapper that ignores captions and calls the attr-only path."""
    return batched_generate_neg_whqa_from_whqa_column(
        llm=llm,
        tokenizer=tokenizer,
        whqa_cells=whqa_cells,
        sampling_params=sampling_params,
        gen_batch_size=gen_batch_size,
    )