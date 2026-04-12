# attribute_utils.py
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

from prompts.attribute_prompts import (
    SYSTEM_MSG_PHRASE,
    FEW_SHOT_ATTRIBUTES,
    SYSTEM_MSG_NEGATTR,
    FEW_SHOT_NEGATTR,
    SYSTEM_MSG_NEGATTR_IDX,
    FEW_SHOT_NEGATTR_IDX,
    DIFF_SYSTEM_MSG,
    FEW_SHOT_DIFF
)

def _messages_attributes(caption_text: str) -> List[dict]:
    """Chat messages for the 'attributes' mode with few-shot."""
    msgs = [{"role": "system", "content": SYSTEM_MSG_PHRASE}]

    # Few-shots
    for cap, ans in FEW_SHOT_ATTRIBUTES:
        demo_user = (
            "Caption:\n" + cap + "\n\n"
            "Task: Pick ONE main object with at least one described attribute and write a SINGLE noun phrase with 1–5 attributes. "
            "Attributes must be adjectives or 'with ...' phrases. Keep it a noun phrase only—no full sentences. "
            "Return EXACTLY one line:\nPHRASE=<...>"
        )
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": ans})

    # Actual task
    task_user = (
        "Caption:\n" + caption_text + "\n\n"
        "Task: Pick ONE main object with at least one described attribute and write a SINGLE noun phrase with 1–5 attributes. "
        "Attributes must be adjectives or 'with ...' phrases. Keep it a noun phrase only—no full sentences. "
        "Return EXACTLY one line:\nPHRASE=<...>"
    )
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_attributes_prompt(tokenizer, caption: str) -> str:
    """Final chat prompt using the FIRST 512 caption tokens."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption, 512)
    return tokenizer.apply_chat_template(
        _messages_attributes(cap_trunc),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def parse_attribute_phrase_output(text: str) -> str:
    return parse_tagged_output(
        text,
        allowed_tags=("PHRASE",),
        max_chars=512,
    )

# ---------------- Prompt building for negative attributes ---------------

def _messages_negattr(attr_phrase: str) -> List[dict]:
    """
    Chat messages for negative-attribute editing (no caption input).
    Input is one noun phrase with attributes; output must be PHRASE=...
    """
    msgs = [{"role": "system", "content": SYSTEM_MSG_NEGATTR}]
    for demo_user, demo_assistant in FEW_SHOT_NEGATTR:
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": demo_assistant})

    # Actual task: supply just the existing attribute phrase
    task_user = f"PHRASE={attr_phrase}"
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_negattr_prompt(tokenizer, attr_phrase: str) -> str:
    """Final chat prompt for negative attribute replacement (no caption)."""
    # attr phrases are typically short; no truncation needed
    return tokenizer.apply_chat_template(
        _messages_negattr(attr_phrase),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# ============================== Generation utils ============================

def batched_generate_attribute_phrases(
    llm,
    tokenizer,
    captions: List[str],
    sampling_params,
    gen_batch_size: int,
) -> List[str]:
    phrases: List[str] = []
    for batch_caps in _chunked(captions, gen_batch_size):
        prompts = [build_attributes_prompt(tokenizer, c) for c in batch_caps]
        outs = llm.generate(prompts, sampling_params)
        for out in outs:
            txt = out.outputs[0].text if out.outputs else ""
            phrases.append(parse_attribute_phrase_output(txt))
    return phrases

# ----------------------- NEW: negative attribute generation -----------------

def batched_generate_negative_attribute_phrases(
    llm,
    tokenizer,
    attr_phrases: List[str],
    sampling_params,
    gen_batch_size: int,
    validate: bool = True,
) -> List[str]:
    """
    For each attribute phrase (a single noun phrase with 1–5 attributes):
      - Let the LLM randomly choose exactly ONE attribute and replace it with a negative attribute.
      - Return ONLY the rewritten PHRASE string.
    No caption is needed.
    """
    outputs: List[str] = []
    for batch_attrs in _chunked(attr_phrases, gen_batch_size):
        # Build prompts (skip empty gracefully)
        prompts = [build_negattr_prompt(tokenizer, a if a is not None else "") for a in batch_attrs]
        outs = llm.generate(prompts, sampling_params)
        for a, out in zip(batch_attrs, outs):
            old = a if a is not None else ""
            txt = out.outputs[0].text if out.outputs else ""
            new = parse_attribute_phrase_output(txt)
            if validate:
                # Minimal guard: require a non-empty change; otherwise keep original
                if not new or new.strip() == old.strip():
                    new = old
            outputs.append(new)
    return outputs

# ---------------- Index-based NEGATIVE attribute editing ---------------

def _has_premod_group(head_segment: str) -> bool:
    """
    Heuristic: detect if the text BEFORE the first 'with' contains a pre-nominal
    adjective group (e.g., 'a lightweight wooden chair').
    - Remove leading determiners/quantifiers.
    - If there's a comma in head -> True (e.g., 'a square, white ceramic plate').
    - Otherwise, if there are >= 2 tokens (adjective + noun or adj+adj+noun) -> True.
    """
    head = (head_segment or "").strip()
    if not head:
        return False
    head = _strip_phrase_prefix(head)
    head = re.sub(r'^\s*(?:a|an|the|this|that|these|those|\d+)\b\s*', '', head, flags=re.IGNORECASE)
    if not head:
        return False
    if "," in head:
        return True
    toks = [t for t in re.split(r'\s+', head) if t]
    return len(toks) >= 2

def estimate_attr_unit_count(attr_phrase: str, max_units: int = 5) -> int:
    """
    Count attribute units for a noun phrase of the form:
      [premod adj group]* HEAD [with ...], [with ...], ... and [with ...]
    Logic:
      • units = (# of 'with' clauses counted via estimate_clause_count on the tail)
      • +1 if a pre-nominal adjective group exists before the first 'with'
      • if neither premod nor 'with' present, default to 1 (conservative)
    """
    core = _strip_phrase_prefix(attr_phrase or "")
    if not core:
        return 1
    m = re.search(r'\bwith\b', core, flags=re.IGNORECASE)
    if m:
        head_seg = core[:m.start()]
        tail_seg = core[m.start():]
        with_units = estimate_clause_count(tail_seg, max_clauses=max_units)
        total = with_units + (1 if _has_premod_group(head_seg) else 0)
        if total > max_units:
            total = max_units # Anyhow we do not allow changing anything outside the 5-th attributes
    else:
        # no 'with' clauses; we still count 1 unit if there's any premod group, else 1 fallback
        total = 1
    return max(1, min(total, max_units))

def _choose_attribute_indices(
    phrases: List[str],
    mode: str = "random",       # 'random' | 'alternate' | '<int>' (fixed 1-based)
    seed: Optional[int] = None,
    max_units: int = 5,
) -> List[int]:
    """
    Mirror of _choose_clause_indices but for attribute units:
    uses estimate_attr_unit_count to cap the index properly.
    """
    rng = random.Random(seed) if seed is not None else random
    indices: List[int] = []
    alt_toggle = False
    fixed = int(mode) if str(mode).isdigit() else None

    for p in phrases:
        k = estimate_attr_unit_count(p, max_units=max_units)
        if fixed is not None:
            idx = min(max(1, fixed), k)
        elif mode == "alternate":
            idx = 1 if not alt_toggle else min(2, k)
            alt_toggle = not alt_toggle
        else:  # 'random'
            idx = rng.randint(1, k)
        indices.append(idx)
    return indices

def _messages_negattr_index(attr_phrase: str, attr_index: int) -> List[dict]:
    """
    Chat messages for index-based negative-attribute editing.
    """
    msgs = [{"role": "system", "content": SYSTEM_MSG_NEGATTR_IDX}]
    for demo_user, demo_assistant in FEW_SHOT_NEGATTR_IDX:
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": demo_assistant})

    core = attr_phrase if str(attr_phrase).strip().lower().startswith("phrase=") else f"PHRASE={attr_phrase}"
    task_user = f"ATTRIBUTE_INDEX={attr_index}\n{core}"
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_negattr_index_prompt(tokenizer, attr_phrase: str, attr_index: int) -> str:
    """Final chat prompt for index-based negative attribute replacement."""
    return tokenizer.apply_chat_template(
        _messages_negattr_index(attr_phrase or "", attr_index),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def batched_generate_negative_attribute_phrases_by_index(
    llm,
    tokenizer,
    attr_phrases: List[str],
    sampling_params,
    gen_batch_size: int,
    attr_index_mode: str = "random",   # 'random' | 'alternate' | '<int>'
    seed: Optional[int] = None,
    max_units: int = 5,
    validate: bool = True,
) -> List[str]:
    """
    Index-based variant:
      - For each attribute phrase, pick a 1-based ATTRIBUTE_INDEX via _choose_attribute_indices.
      - Instruct the model to edit exactly that unit.
      - Return ONLY the rewritten PHRASE string (fallback to original if validation fails and validate=True).
    """
    outputs: List[str] = []
    if not attr_phrases:
        return outputs

    chosen_idxs = _choose_attribute_indices(
        attr_phrases, mode=attr_index_mode, seed=seed, max_units=max_units
    )

    for start in range(0, len(attr_phrases), gen_batch_size):
        end = min(start + gen_batch_size, len(attr_phrases))
        batch_attrs = attr_phrases[start:end]
        batch_idxs = chosen_idxs[start:end]

        prompts = [
            build_negattr_index_prompt(tokenizer, a if a is not None else "", idx)
            for a, idx in zip(batch_attrs, batch_idxs)
        ]
        gens = llm.generate(prompts, sampling_params)

        for old_phrase, out in zip(batch_attrs, gens):
            old = old_phrase if old_phrase is not None else ""
            txt = out.outputs[0].text if out.outputs else ""
            new = parse_attribute_phrase_output(txt)
            if validate:
                if not new or new.strip() == old.strip():
                    new = old
            outputs.append(new)

    return outputs


# ======================== POS/NEG pair → 1-attr phrases ========================
# The following code should not be used, because it involves using LLM to construct the QA pairs, while in the end we use template to construct OBJ/ATTR/REL cases

def _messages_attr_diff(acc_answer: str, rej_answer: str) -> List[dict]:
    msgs = [{"role": "system", "content": DIFF_SYSTEM_MSG}]
    for user_demo, asst_demo in FEW_SHOT_DIFF:
        msgs.append({"role": "user", "content": user_demo})
        msgs.append({"role": "assistant", "content": asst_demo})

    prompt = f"Task:\nACC:\n{acc_answer.strip()}\n\nREJ:\n{rej_answer.strip()}"
    msgs.append({"role": "user", "content": prompt})
    return msgs


_DIFF_LINE_RE = re.compile(r'^\s*(ACC_PHRASE|REJ_PHRASE)\s*[:=]\s*(.+?)\s*$', re.IGNORECASE)

def parse_attr_diff_output(text: str) -> Tuple[str, str]:
    acc, rej = "", ""
    for line in (text or "").splitlines():
        m = _DIFF_LINE_RE.match(line.strip())
        if not m:
            continue
        tag, val = m.group(1).upper(), m.group(2).strip().strip('"').strip("'")
        if tag == "ACC_PHRASE":
            acc = val
        elif tag == "REJ_PHRASE":
            rej = val
    return acc[:512], rej[:512]

# Parse A_acc / A_rej from a cell like "Q=... || A_acc=... || A_rej=..."
_A_ACC_RE = re.compile(r'\bA_acc\s*=\s*(.+?)(?=\s*\|\|\s*A_rej\s*=|$)', re.DOTALL | re.IGNORECASE)
_A_REJ_RE = re.compile(r'\bA_rej\s*=\s*(.+)$', re.DOTALL | re.IGNORECASE)

def extract_acc_rej_from_cell(cell: str) -> Tuple[str, str]:
    cell = cell or ""
    m_acc = _A_ACC_RE.search(cell)
    m_rej = _A_REJ_RE.search(cell)
    acc = (m_acc.group(1).strip() if m_acc else "").replace("Assistant:", "").strip()
    rej = (m_rej.group(1).strip() if m_rej else "").replace("Assistant:", "").strip()
    return acc, rej

# Build prompt using your single DIFF_SYSTEM_MSG/_messages_attr_diff
def build_attr_diff_prompt(tokenizer, acc_answer: str, rej_answer: str) -> str:
    return tokenizer.apply_chat_template(
        _messages_attr_diff(acc_answer, rej_answer),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def batched_generate_attr_diff_phrases(
    llm,
    tokenizer,
    acc_rej_pairs: List[Tuple[str, str]],
    sampling_params,
    gen_batch_size: int = 64,
) -> List[Tuple[str, str]]:
    """
    For each (A_acc, A_rej) pair, produce (ACC_PHRASE, REJ_PHRASE) full sentences
    with only the head object + differing attribute, preserving surface format.
    """
    results: List[Tuple[str, str]] = []
    for i in range(0, len(acc_rej_pairs), gen_batch_size):
        batch = acc_rej_pairs[i:i+gen_batch_size]
        prompts = [build_attr_diff_prompt(tokenizer, a, r) for (a, r) in batch]
        outs = llm.generate(prompts, sampling_params)
        for out in outs:
            txt = out.outputs[0].text if out.outputs else ""
            results.append(parse_attr_diff_output(txt))
    return results
