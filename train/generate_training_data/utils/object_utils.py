# object_utils.py
import os, re, glob
from typing import List, Optional, Tuple, Iterable, Dict
import random
import pyarrow.parquet as pq

from general_utils import (
    chunked as _chunked,
    parse_tagged_output,
    truncate_first_tokens as _truncate_caption_first_tokens,
)

from prompts.object_prompts import (
    SYSTEM_MSG_OBJECTS,
    FEW_SHOT_OBJECTS,
    SYSTEM_MSG_NEGOBJ,
    FEW_SHOT_NEGOBJ,
    CATEGORIES,
    SYSTEM_MSG_CATEGORY,
    FEW_SHOT_CATEGORY,
)

# ============================== Phrase helpers (validation only) ============

def _split_objects_phrase(phrase: str) -> List[str]:
    """
    Split 'a dog, a cat and two birds' -> ['a dog','a cat','two birds'].
    Heuristic: split on the last ' and ' and commas on the left.
    Used only for lightweight validation; storage remains a single string.
    """
    s = (phrase or "").strip()
    if not s:
        return []
    if " and " in s:
        left, last = s.rsplit(" and ", 1)
        left_parts = [p.strip() for p in left.split(",") if p.strip()]
        items = left_parts + [last.strip()]
    else:
        items = [p.strip() for p in s.split(",") if p.strip()]
    return items

def _validate_single_replacement(original_phrase: str, new_phrase: str) -> bool:
    """
    Light sanity: same number of items and exactly one item differs (string-compare after strip).
    """
    old = _split_objects_phrase(original_phrase)
    new = _split_objects_phrase(new_phrase)
    if not old or not new or len(old) != len(new):
        return False
    diffs = sum(1 for a, b in zip(old, new) if a.strip() != b.strip())
    return diffs == 1

# ======================== Prompt building & parsing =========================

def _messages_objects(caption_text: str) -> List[dict]:
    """Chat messages for the 'objects' mode with few-shot."""
    msgs = [{"role": "system", "content": SYSTEM_MSG_OBJECTS}]

    for cap, ans in FEW_SHOT_OBJECTS:
        demo_user = (
            "Caption:\n" + cap + "\n\n"
            "Task: From this caption, select up to five main objects that have at least one descriptive attribute in the caption. "
            "Prefer plain object names; keep explicit numbers/quantifiers if present; pluralize repeated names without numbers. "
            "Return EXACTLY one line:\nPHRASE=<...>"
        )
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": ans})

    task_user = (
        "Caption:\n" + caption_text + "\n\n"
        "Task: Select up to five main objects that have at least one descriptive attribute in the caption. "
        "Prefer plain object names; keep explicit numbers/quantifiers if present; pluralize repeated names without numbers. "
        "Return EXACTLY one line:\nPHRASE=<...>"
    )
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_objects_prompt(tokenizer, caption: str) -> str:
    """Final chat prompt using the FIRST 512 caption tokens."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption, 512)
    return tokenizer.apply_chat_template(
        _messages_objects(cap_trunc),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def parse_objects_output(text: str) -> str:
    return parse_tagged_output(
        text,
        allowed_tags=("PHRASE",),
        max_chars=512,
    )

# ----------------- Prompt building for the negative objects -----------------

def _messages_negobj(caption_text: str, obj_phrase: str, replace_index_1based: int) -> List[dict]:
    """
    Chat messages for negative replacement with an explicit slot to replace.
    The model must replace exactly REPLACE_INDEX (1-based).
    """
    msgs = [{"role": "system", "content": SYSTEM_MSG_NEGOBJ}]
    for demo_user, demo_assistant in FEW_SHOT_NEGOBJ:
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": demo_assistant})
    task_user = f"Caption:\n{caption_text}\nPHRASE={obj_phrase}\nREPLACE_INDEX={replace_index_1based}"
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_negobj_prompt(tokenizer, caption: str, obj_phrase: str, replace_index_1based: int) -> str:
    """Final chat prompt for negative replacement, with an explicit index (1-based) to replace."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption, 512)
    return tokenizer.apply_chat_template(
        _messages_negobj(cap_trunc, obj_phrase, replace_index_1based),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# ============================== Generation utils ============================

def batched_generate_objects_phrases(
    llm,
    tokenizer,
    captions: List[str],
    sampling_params,
    gen_batch_size: int,
) -> List[str]:
    phrases: List[str] = []
    for batch_caps in _chunked(captions, gen_batch_size):
        prompts = [build_objects_prompt(tokenizer, c) for c in batch_caps]
        outs = llm.generate(prompts, sampling_params)
        for out in outs:
            txt = out.outputs[0].text if out.outputs else ""
            phrases.append(parse_objects_output(txt))
    return phrases

def batched_generate_negative_phrases(
    llm,
    tokenizer,
    captions: List[str],
    obj_phrases: List[str],
    sampling_params,
    gen_batch_size: int,
    validate: bool = False,
) -> List[str]:
    """
    For each (caption, obj_phrase), we uniformly choose one slot index in code,
    instruct the model to replace EXACTLY that slot, and return ONLY the new PHRASE string.
    """
    assert len(captions) == len(obj_phrases), "captions and obj_phrases must align"
    outputs: List[str] = []
    i = 0
    while i < len(captions):
        batch_caps = captions[i:i+gen_batch_size]
        batch_objs = obj_phrases[i:i+gen_batch_size]

        prompts = []
        meta_positions: List[int] = []  # 1-based per example
        passthrough_idx: List[int] = []

        for j, (cap, phr) in enumerate(zip(batch_caps, batch_objs)):
            if not phr:
                outputs.append("")  # no objects to replace
                passthrough_idx.append(j)
                continue
            items = _split_objects_phrase(phr)
            if not items:
                outputs.append("")
                passthrough_idx.append(j)
                continue
            # Uniformly choose a slot to replace (1..len(items))
            k = random.randint(1, len(items))
            meta_positions.append(k)
            prompts.append(build_negobj_prompt(tokenizer, cap, phr, k))

        if prompts:
            gens = llm.generate(prompts, sampling_params)
            gi = 0
            for j, (cap, phr) in enumerate(zip(batch_caps, batch_objs)):
                if j in passthrough_idx:
                    continue
                out = gens[gi]; gi += 1
                txt = out.outputs[0].text if out.outputs else ""
                new_phrase = parse_objects_output(txt)  # expect 'PHRASE=...'

                if validate:
                    old_items = _split_objects_phrase(phr)
                    new_items = _split_objects_phrase(new_phrase)
                    ok_len = len(new_items) == max(1, len(old_items))
                    ok_diff = _validate_single_replacement(phr, new_phrase) if len(old_items) >= 1 else bool(new_phrase)
                    if not (ok_len and ok_diff):
                        new_phrase = phr  # fallback

                outputs.append(new_phrase if new_phrase else phr)

        i += gen_batch_size

    return outputs


# ================= Existence QA with POS context (template-based) =============
# helpers that format answers exactly as requested:
#   POS: "Yes, I can see <pos objects> in this image"
#   NEG: "No, but I can see <pos objects> in this image"
#
# These DO NOT modify earlier helpers (which return plain 'yes'/'no').

def _item_to_existence_qa_with_pos_context(item: str, pos_phrase_for_answer: str, truth: bool) -> str:
    """
    Make one existence QA pair for a single object item, using the POSITIVE phrase as the
    answer context (even for negative questions).
      - truth=True  -> "Yes, I can see <pos_phrase> in this image"
      - truth=False -> "No, but I can see <pos_phrase> in this image"
    """
    it = _normalize_item_for_question(item)
    pos_ans = (pos_phrase_for_answer or "").strip().rstrip(" ,.;:")
    if not it or not pos_ans:
        return ""
    q = f"Q=Can you see {it} in this image?"
    if truth:
        a = f"Yes, I can see {pos_ans} in this image"
    else:
        a = f"No, but I can see {pos_ans} in this image"
    return f"{q} || A={a}"

def build_existence_qa_pairs_pos_with_context(pos_phrase: str) -> str:
    """
    From POSITIVE PHRASE -> per-item existence Qs, answers reference the FULL positive phrase:
      Q=Can you see <item> in this image? || A=Yes, I can see <pos_phrase> in this image
    Returns a single semicolon-joined string ('' if no valid items).
    """
    items = _split_objects_phrase(pos_phrase)
    pairs = [
        _item_to_existence_qa_with_pos_context(it, pos_phrase, truth=True)
        for it in items
    ]
    pairs = [p for p in pairs if p]
    return "; ".join(pairs)

def build_existence_qa_pairs_neg_with_context(neg_phrase: str, pos_phrase_for_answer: str) -> str:
    """
    From NEGATIVE PHRASE -> per-item existence Qs, answers reference the FULL positive phrase:
      Q=Can you see <neg_item> in this image? || A=No, but I can see <pos_phrase> in this image
    Returns a single semicolon-joined string ('' if no valid items).
    """
    items = _split_objects_phrase(neg_phrase)
    pairs = [
        _item_to_existence_qa_with_pos_context(it, pos_phrase_for_answer, truth=False)
        for it in items
    ]
    pairs = [p for p in pairs if p]
    return "; ".join(pairs)

def batched_build_existence_qa_pairs_pos_with_context(
    pos_phrases: List[str],
) -> List[str]:
    """
    Batched POS existence QA using the full positive phrase in the answer.
    """
    out: List[str] = []
    for phr in pos_phrases:
        out.append(build_existence_qa_pairs_pos_with_context(phr or ""))
    return out

def batched_build_existence_qa_pairs_neg_with_context(
    neg_phrases: List[str],
    pos_phrases_for_answer: List[str],
) -> List[str]:
    """
    Batched NEG existence QA using the aligned positive phrase for each row in the answer.
    neg_phrases and pos_phrases_for_answer must be aligned row-wise.
    """
    assert len(neg_phrases) == len(pos_phrases_for_answer), "neg and pos phrases must align row-wise"
    out: List[str] = []
    for neg, pos in zip(neg_phrases, pos_phrases_for_answer):
        out.append(build_existence_qa_pairs_neg_with_context(neg or "", pos or ""))
    return out

def batched_build_pos_and_neg_existence_qa_with_context(
    pos_phrases: List[str],
    neg_phrases: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Convenience: given aligned (obj, obj_neg), build:
      - POS existence QA (answers: Yes, I can see <obj> in this image)
      - NEG existence QA (answers: No, but I can see <obj> in this image)
    Returns (pos_list, neg_list).
    """
    assert len(pos_phrases) == len(neg_phrases), "pos and neg phrases must align row-wise"
    pos_out = batched_build_existence_qa_pairs_pos_with_context(pos_phrases)
    neg_out = batched_build_existence_qa_pairs_neg_with_context(neg_phrases, pos_phrases)
    return pos_out, neg_out

# ========================= QA templates & builders =========================
# Number-neutral templates avoid singular/plural agreement issues.
QA_TEMPLATES: List[Dict[str, str]] = [
    {
        "q":     "Can you see <PHRASE> in this image?",
        "yes":   "Yes, I can see <POS> in this image.",
        "nobut": "No, but I can see <PHRASE> in this image.",
    },
    {
        "q":     "Does this image contain <PHRASE>?",
        "yes":   "Yes, this image contains <POS>.",
        "nobut": "No, but this image contains <PHRASE>.",
    },
    {
        "q":     "Does this image show <PHRASE>?",
        "yes":   "Yes, this image shows <POS>.",
        "nobut": "No, but this image shows <PHRASE>.",
    },
    {
        "q":     "Can <PHRASE> be seen in this image?",
        "yes":   "Yes, <POS> can be seen in this image.",
        "nobut": "No, but <PHRASE> can be seen in this image.",
    },
    {
        "q":     "Does this image include <PHRASE>?",
        "yes":   "Yes, this image includes <POS>.",
        "nobut": "No, but this image includes <PHRASE>.",
    }
]

def _pick_qa_template(rng: Optional[random.Random] = None) -> Dict[str, str]:
    r = rng if rng is not None else random
    return r.choice(QA_TEMPLATES)

def _format_qa_line(q_text: str, a_text: str) -> str:
    """Return a single line `Q=... || A=...` with trailing punctuation normalized."""
    q = (q_text or "").strip()
    a = (a_text or "").strip()
    # ensure terminal punctuation (q: '?', a: '.')
    if q and not q.endswith("?"):
        q = q.rstrip(".") + "?"
    if a and not a.endswith("."):
        a = a.rstrip(".") + "."
    return f"Q={q} || A={a}"

def _fill_template_for_question(tmpl_q: str, phrase_for_q: str) -> str:
    # Use your existing normalizers to keep behavior consistent
    q_phrase = _lowercase_first_word(_normalize_phrase_for_slot(phrase_for_q))
    return tmpl_q.replace("<PHRASE>", q_phrase)

def _fill_answer_yes(tmpl_yes: str, pos_phrase: str) -> str:
    pos = _normalize_phrase_for_slot(pos_phrase)
    return tmpl_yes.replace("<POS>", pos)

def _fill_answer_nobut(tmpl_nobut: str, phrase_for_nobut: str) -> str:
    # The 'No, but ...' template uses <PHRASE>, so we pass either NEG (pos_question mode)
    # or POS (neg_question mode), depending on what you want to contrast with.
    ph = _normalize_phrase_for_slot(phrase_for_nobut)
    return tmpl_nobut.replace("<PHRASE>", ph)

def build_qa_triplet_for_pair(
    pos_phrase: str,
    neg_phrase: str,
    mode: str = "pos_question",
    rng: Optional[random.Random] = None,
    # Optional grammar-polish hook: callable(List[str]) -> List[str]
    grammar_polish_fn: Optional[callable] = None,
) -> Tuple[str, str]:
    """
    Build two complete QA lines (same question, two answers):
      - `accept_line`: Q=... || A=...
      - `reject_line`: Q=... || A=...

    Modes:
      - "pos_question": Q uses POS;   accept = YES(+POS);   reject = NOBUT(+NEG)
      - "neg_question": Q uses NEG;   accept = NOBUT(+POS); reject = YES(+NEG)

    Returns: (accept_line, reject_line). Empty strings if missing phrases.
    """
    pos = (pos_phrase or "").strip()
    neg = (neg_phrase or "").strip()
    if not pos or not neg:
        return "", ""

    tmpl = _pick_qa_template(rng)
    if mode == "pos_question":
        q_text = _fill_template_for_question(tmpl["q"], pos)
        a_accept = _fill_answer_yes(tmpl["yes"], pos)           # Yes, ... <POS> ...
        a_reject = _fill_answer_nobut(tmpl["nobut"], neg)       # No, but ... <NEG> ...
    elif mode == "neg_question":
        q_text = _fill_template_for_question(tmpl["q"], neg)
        a_accept = _fill_answer_nobut(tmpl["nobut"], pos)       # No, but ... <POS> ...
        a_reject = _fill_answer_yes(tmpl["yes"], neg)           # Yes, ... <NEG> ...
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'pos_question' or 'neg_question'.")

    accept_line = _format_qa_line(q_text, a_accept)
    reject_line = _format_qa_line(q_text, a_reject)

    if grammar_polish_fn is not None:
        accept_line, reject_line = grammar_polish_fn([accept_line, reject_line])

    return accept_line, reject_line

def batched_build_qa_triplets_for_pairs(
    pos_list: List[str],
    neg_list: List[str],
    mode: str = "pos_question",
    base_rng: Optional[random.Random] = None,
    grammar_polish_fn: Optional[callable] = None,
) -> Tuple[List[str], List[str]]:
    """
    Batched wrapper over `build_qa_triplet_for_pair`. Deterministic variety if `base_rng` is given.
    Returns (accept_lines, reject_lines).
    """
    assert len(pos_list) == len(neg_list), "pos_list and neg_list must align"
    acc, rej = [], []
    for p, n in zip(pos_list, neg_list):
        rng = None
        if base_rng is not None:
            rng = random.Random(base_rng.randrange(1 << 30))
        a, r = build_qa_triplet_for_pair(
            pos_phrase=p or "",
            neg_phrase=n or "",
            mode=mode,
            rng=rng,
            grammar_polish_fn=grammar_polish_fn,
        )
        acc.append(a)
        rej.append(r)
    return acc, rej



# ============================== Caption categorization (for ablation study) ==============================

def _messages_category(caption_text: str) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_MSG_CATEGORY}]
    for cap, lab in FEW_SHOT_CATEGORY:
        msgs.append({"role": "user", "content": f"Caption:\n{cap}\n\nReturn ONLY the label."})
        msgs.append({"role": "assistant", "content": lab})
    msgs.append({"role": "user", "content": f"Caption:\n{caption_text}\n\nReturn ONLY the label."})
    return msgs

def build_category_prompt(tokenizer, caption: str) -> str:
    """Build chat prompt (truncate caption to first 512 tokens for safety)."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption or "", 512)
    return tokenizer.apply_chat_template(
        _messages_category(cap_trunc),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def parse_category_output(text: str) -> str:
    raw = (text or "").strip().strip('"').strip("'")
    norm = re.sub(r"\s+", "_", raw.lower())

    if norm in CATEGORIES:
        return norm

    if any(k in norm for k in ["diagram", "flowchart", "schematic", "uml", "illustration", "block_diagram"]):
        return "chart_graph"
    if any(k in norm for k in ["chart", "graph", "plot", "histogram", "bar_chart", "line_chart", "scatter"]):
        return "chart_graph"
    if any(k in norm for k in ["screenshot", "screen_capture", "ui", "app", "ide", "terminal", "webpage", "dashboard"]):
        return "screenshot_ui"
    if any(k in norm for k in ["scan", "scanned", "document", "page", "paper", "invoice", "receipt", "whiteboard", "handwriting", "handwritten", "resume"]):
        return "document_text"

    # final fallback
    return "natural_image"

def batched_categorize_captions(
    llm,
    tokenizer,
    captions: List[str],
    sampling_params,
    gen_batch_size: int,
) -> List[str]:
    """
    Categorize each caption into one of:
      ['natural_image','chart_graph','screenshot_ui','document_text']
    If the model returns anything else, fallback to 'natural_image'.
    """
    out: List[str] = []
    for batch_caps in _chunked(captions, gen_batch_size):
        prompts = [build_category_prompt(tokenizer, c or "") for c in batch_caps]
        gens = llm.generate(prompts, sampling_params)
        for g in gens:
            txt = g.outputs[0].text.strip() if g.outputs else ""
            out.append(parse_category_output(txt))
    return out


