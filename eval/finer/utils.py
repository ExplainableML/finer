#!/usr/bin/env python3
from pathlib import Path
import argparse, re, torch, pandas as pd
from tqdm import tqdm
from collections import defaultdict
from io import StringIO


LETTER    = ["A", "B", "C", "D", "E"]
LETTER_RE = re.compile(r"[A-Ea-e]")


def mcq_text(question: str, choices: list[str]) -> str:
    lines = ["Select the best answer for the multiple-choice question.", "",
             question, ""]
    lines += [f"({LETTER[i]}) {c}" for i, c in enumerate(choices)]
    lines += ["", "Please answer with a single capital letter (A, B, C, D, or E):"]
    return "\n".join(lines)

def build_msgs(img_path: str, prompt: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text",  "text": prompt},
        ],
    }]

def match_choice(gen_text: str):
    # We could make this more robust, however, in practice, the model will always output A, B, C, D, or E. So I decided to make it simple.
    m = LETTER_RE.search(gen_text)
    return LETTER.index(m.group(0).upper()) if m else -1


def pct(n, d):
    return f"{(n/d):.3%}  ({n}/{d})" if d else "n/a"


def strip_pos_neg(q):
    return re.sub(r"_(pos|neg)$", "", str(q))


def suffix(q):
    m = re.search(r"_(pos|neg)$", str(q))
    return m.group(1) if m else None


def compute_pair_accuracy(
    df: pd.DataFrame,
    name: str = "current run",
    print_report: bool = True,
):

    required = {"image", "gt_index", "pred_idx", "qtype"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for paired accuracy: {sorted(missing)}")

    df = df.reset_index(drop=True)

    n = len(df)
    if n % 2 != 0:
        raise ValueError(f"Row count must be even for strict 2-row pairing. Got {n} rows.")

    correct = (df["pred_idx"] == df["gt_index"]).astype(int).to_list()
    images = df["image"].astype(str).to_list()
    qtypes = df["qtype"].astype(str).to_list()

    total_pairs = n // 2
    both_correct = 0

    for i in range(0, n, 2):
        # sanity check 1: this pair MUST from the same image
        if images[i] != images[i + 1]:
            raise ValueError(
                f"Row {i} and {i + 1} have different images: "
                f"{images[i]} vs {images[i + 1]}"
            )

        qt1, qt2 = qtypes[i], qtypes[i + 1]
        base1, base2 = strip_pos_neg(qt1), strip_pos_neg(qt2)

        # sanity check 2: this pair MUST have the same question type
        if base1 != base2:
            raise ValueError(
                f"Pair {i}//{i + 1} mismatched base qtype: "
                f"{qt1!r} -> {base1!r} vs {qt2!r} -> {base2!r}"
            )

        # snity check 3: this pair MUST be one POS and one NEG
        s1, s2 = suffix(qt1), suffix(qt2)
        if {s1, s2} != {"pos", "neg"}:
            raise ValueError(
                f"Pair {i}//{i + 1} must be (pos, neg) in any order; "
                f"got suffixes {s1!r}, {s2!r} "
                f"(qtypes: {qt1!r}, {qt2!r})"
            )

        # paired accuracy: both rows in the pair must be correct
        if correct[i] == 1 and correct[i + 1] == 1:
            both_correct += 1

    pair_acc = both_correct / total_pairs if total_pairs else 0.0

    if print_report:
        print(f"Paired accuracy     : {pct(both_correct, total_pairs)}")

    return pair_acc