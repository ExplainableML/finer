#!/usr/bin/env python3
# save as: parquet_to_dpo_jsonl_select.py
# Usage examples:
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 4 --mode select_all
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode either_pos_or_neg --seed 42
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode 25_pos_75_neg --seed 42
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode only_neg
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode only_obj
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode only_attr
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode only_rel
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode only_wh
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode pon_except_wh --seed 42
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode either_rel_or_wh --seed 42
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode 30_rel_70_wh --seed 42
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode 15_attr_15_rel_70_wh --seed 42
#   python3 parquet_to_dpo_jsonl_select.py --max_shards 10 --mode 10_obj_10_attr_10_rel_70_wh --seed 42
#
# Output: one JSONL file compatible with LLaMA-Factory DPO (ShareGPT+images+ranking):
#   {
#     "conversations":[{"from":"human","value":"<image>\n{question}"}],
#     "chosen":{"from":"gpt","value":"..."},
#     "rejected":{"from":"gpt","value":"..."},
#     "images":["/abs/path/to/image.jpg"],
#     "meta":{...}
#   }

import json
import re
import random
import argparse
from pathlib import Path

import pyarrow.parquet as pq

# ====== DEFAULT PATHS (edit if needed) ======
IMAGES_DIR = Path("/dss/dssmcmlfs01/pn39yu/pn39yu-dss-0000/datasets/pixmo/images")
PARQUET_DIR  = Path("/dss/dssmcmlfs01/pn39yu/pn39yu-dss-0000/datasets/pixmo/annotations/parquet")
OUT_JSONL   = Path("/p/scratch/taco-vlm/datasets/pixmo/jsonl/dpo_selected.jsonl")

# Column names per QA type
QA_MAP = {
    "obj":  ("obj_qa_pos",  "obj_qa_neg"),
    "attr": ("attr_qa_pos", "attr_qa_neg"),
    "rel":  ("rel_qa_pos",  "rel_qa_neg"),
    "wh":   ("wh_pos_qa",   "wh_neg_qa"),
}

# Parse strings like:  Q=... || A_acc=... || A_rej=...
PAT = re.compile(r"\s*Q=(.*?)\s*\|\|\s*A_acc=(.*?)\s*\|\|\s*A_rej=(.*)\s*$")

def parse_qa(s: str):
    """Return (question, chosen, rejected) or None if not parseable/usable."""
    if not isinstance(s, str) or not s.strip():
        return None
    m = PAT.match(s)
    if not m:
        return None
    q, acc, rej = [x.strip() for x in m.groups()]
    if not q or not acc or not rej:
        return None
    if acc == rej:
        return None
    return q, acc, rej

def build_record(image_path: str, q: str, acc: str, rej: str,
                 shard: str, row_idx, qa_type: str, polarity: str):
    """Build one LLaMA-Factory DPO (ShareGPT+images+ranking) record."""
    # Ensure strings are clean
    q = str(q).strip()
    acc = str(acc).strip()
    rej = str(rej).strip()
    return {
        "conversations": [
            {"from": "human", "value": f"<image>\n{q}"}
        ],
        "chosen":   {"from": "gpt", "value": acc},
        "rejected": {"from": "gpt", "value": rej},
        "images": [image_path],
        # meta is optional but handy for analysis
        "meta": {"shard": shard, "row_idx": int(row_idx), "qa_type": qa_type, "polarity": polarity},
    }

def all_qa_cols():
    """Flatten QA_MAP into a list of column names."""
    return [c for pair in QA_MAP.values() for c in pair]

def iter_rows_streaming(pq_path: Path, needed_cols):
    """Yield (row_idx, image_file, qa_values_dict) streaming through batches."""
    pf = pq.ParquetFile(pq_path)
    has_row_idx = "row_idx" in set(pf.schema.names)
    row_base = 0
    for batch in pf.iter_batches(columns=list(needed_cols), batch_size=1024):
        cols = {name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                for name in batch.schema.names}
        n = len(next(iter(cols.values()))) if cols else 0
        for i in range(n):
            ridx = cols["row_idx"][i] if has_row_idx else (row_base + i)
            image_file = cols.get("image_file", [None]*n)[i]
            qa_vals = {k: cols.get(k, [None]*n)[i] for k in all_qa_cols()}
            yield ridx, image_file, qa_vals
        row_base += n


def load_sidecar_allowed_rows(sidecar_dir: Path, shard_stem: str, keep_category: str) -> set[int] | None:
    """
    Load the sidecar parquet for this shard and return a set of row ids to KEEP.
    Expects columns: 'row' (int), 'category' (str). Returns None if file missing.
    """
    sidecar_path = sidecar_dir / f"{shard_stem}.parquet"
    if not sidecar_path.exists():
        print(f"[WARN] Sidecar not found for shard {shard_stem}: {sidecar_path} — skipping filter for this shard.")
        return None
    pf = pq.ParquetFile(sidecar_path)
    keep = set()
    for batch in pf.iter_batches(columns=["row", "category"], batch_size=8192):
        rows = batch.column(batch.schema.get_field_index("row")).to_pylist()
        cats = batch.column(batch.schema.get_field_index("category")).to_pylist()
        for r, c in zip(rows, cats):
            if c == keep_category:
                # 'row' matches the 0-based row index within the shard
                keep.add(int(r))
    return keep

# ===================== NEW FUNCTIONS (added only; originals untouched) =====================

def pick_rel_or_wh_for_image(rng: random.Random) -> str:
    """
    Randomly pick which QA type to use for this image in 'either_rel_or_wh' mode.
    Returns 'rel' or 'wh'.
    """
    return rng.choice(["rel", "wh"])

def emit_for_rel_or_wh_for_image(rng: random.Random, qa_vals, img_path_str, shard, row_idx,
                                 emit_for_single_type_fn):
    """
    For one image/row: randomly choose 'rel' OR 'wh', then emit BOTH pos and neg
    (i.e., call the existing 'emit_for_single_type' for the chosen type).
    """
    chosen_type = pick_rel_or_wh_for_image(rng)
    emit_for_single_type_fn(chosen_type, qa_vals, img_path_str, shard, row_idx)
    return chosen_type

def pick_weighted_qa_type_for_image(rng: random.Random, weighted_choices):
    """
    Pick a QA type from weighted_choices = [(qa_type, weight), ...], where weights sum to 1.0 (or ~1.0).
    Uses rng.random() so it is reproducible w.r.t. --seed.
    """
    r = rng.random()
    cum = 0.0
    last_key = weighted_choices[-1][0]
    for key, w in weighted_choices:
        cum += float(w)
        if r < cum:
            return key
    return last_key  # fallback for numerical edge cases

def emit_for_weighted_choice_for_image(rng: random.Random, qa_vals, img_path_str, shard, row_idx,
                                       weighted_choices, emit_for_single_type_fn):
    """
    For one image/row: pick a QA type using the provided weights, then emit BOTH pos and neg
    by delegating to the existing emit_for_single_type.
    """
    chosen_type = pick_weighted_qa_type_for_image(rng, weighted_choices)
    emit_for_single_type_fn(chosen_type, qa_vals, img_path_str, shard, row_idx)
    return chosen_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", type=str, default=str(PARQUET_DIR))
    ap.add_argument("--images_dir",  type=str, default=str(IMAGES_DIR))
    ap.add_argument("--out_jsonl",   type=str, default=str(OUT_JSONL))
    ap.add_argument("--max_shards",  type=int, default=4, help="Take first N shards after sorting by name.")
    ap.add_argument(
        "--mode",
        type=str,
        choices=(
            "select_all",
            "either_pos_or_neg",
            "25_pos_75_neg",             # NEW: weighted PoN (25% pos, 75% neg) with same fallback logic
            "only_neg",
            "only_obj",
            "only_attr",
            "only_rel",
            "only_wh",
            "pon_except_wh",             # NEW: PoN for obj/attr/rel, only NEG for wh
            "either_rel_or_wh",          # NEW: for each image, choose rel OR wh, then emit both pos+neg of chosen type
            "30_rel_70_wh",              # NEW: weighted rel/wh
            "15_attr_15_rel_70_wh",      # NEW: weighted attr/rel/wh
            "10_obj_10_attr_10_rel_70_wh",  # NEW: weighted obj/attr/rel/wh
        ),
        default="select_all",
        help=(
            "select_all: emit all parsed pos/neg for all QA types; "
            "either_pos_or_neg: randomly choose pos or neg per QA type; "
            "25_pos_75_neg: choose pos with 25% and neg with 75% per QA type (fallback if only one exists); "
            "only_neg: emit only neg per QA type; "
            "only_obj/only_attr/only_rel/only_wh: emit pos and/or neg only for that QA type; "
            "pon_except_wh: for obj/attr/rel choose pos or neg randomly (like either_pos_or_neg); for wh emit only neg; "
            "either_rel_or_wh: per image, randomly choose rel or wh, then emit BOTH pos and neg for that chosen type; "
            "30_rel_70_wh: per image, choose rel with 30% and wh with 70%, then emit BOTH pos and neg; "
            "15_attr_15_rel_70_wh: per image, choose attr 15%, rel 15%, wh 70%, then emit BOTH pos and neg; "
            "10_obj_10_attr_10_rel_70_wh: per image, choose obj 10%, attr 10%, rel 10%, wh 70%, then emit BOTH pos and neg."
        ),
    )
    ap.add_argument("--seed",        type=int, default=1234, help="Only used in either_pos_or_neg, 25_pos_75_neg and pon_except_wh modes.")
    ap.add_argument("--require_image_exists", action="store_true",
                    help="Skip record if image path not found.")
    ap.add_argument("--relative_paths", action="store_true",
                    help="Store image paths relative to --images_dir (pair with --media_dir at train time).")
    ap.add_argument("--sidecar_dir", type=str, default=None,
                help="Directory containing sidecar parquet shards with columns [row, category]. If set with --only_natural_images, will load the matching shard (same stem) to filter rows.")
    ap.add_argument("--only_natural_images", action="store_true",
                    help="If set, keep only rows whose sidecar category == 'natural_image'. Requires --sidecar_dir.")
    ap.add_argument("--sidecar_keep_category", type=str, default="natural_image",
                    help="Category to keep from sidecars when --only_natural_images is set.")
    ap.add_argument("--start_shard", type=int, default=0,
                help="0-based index into the sorted shard list (after name sort). "
                     "Use together with --max_shards. Example: --start_shard 4 --max_shards 2 "
                     "will pick the 5th and 6th shards.")
    args = ap.parse_args()
    
    if args.only_natural_images and not args.sidecar_dir:
        raise SystemExit("[ERR] --only_natural_images requires --sidecar_dir")
    sidecar_dir = Path(args.sidecar_dir).resolve() if args.sidecar_dir else None
    parquet_dir = Path(args.parquet_dir).resolve()
    images_dir  = Path(args.images_dir).resolve()
    out_path    = Path(args.out_jsonl).resolve()

    rng = random.Random(args.seed)

    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"[ERR] No shards found in {parquet_dir}")

    start = max(0, int(args.start_shard))
    if start >= len(shards):
        raise SystemExit(f"[ERR] --start_shard {start} out of range (found {len(shards)} shards).")

    if args.max_shards is not None and args.max_shards >= 0:
        shards = shards[start:start + args.max_shards]
    else:
        shards = shards[start:]

    needed_cols = {"image_file", "row_idx", *all_qa_cols()}

    num_written = 0
    num_skipped = 0

    def emit_for_single_type(qa_type_key: str, qa_vals, img_path_str, shard, row_idx):
        nonlocal num_written
        pos_col, neg_col = QA_MAP[qa_type_key]
        for polarity, col in (("pos", pos_col), ("neg", neg_col)):
            parsed = parse_qa(qa_vals.get(col))
            if not parsed:
                continue
            q, acc, rej = parsed
            rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type_key, polarity)
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1

    with out_path.open("w", encoding="utf-8") as w:
        for pq_path in shards:
            shard = pq_path.stem
            print(f"[DO] {pq_path.name}")

            allowed_rows = None
            if args.only_natural_images and sidecar_dir is not None:
                allowed_rows = load_sidecar_allowed_rows(sidecar_dir, shard, args.sidecar_keep_category)

            for row_idx, image_file, qa_vals in iter_rows_streaming(pq_path, needed_cols):
                if not image_file:
                    num_skipped += 1
                    continue
                
                if allowed_rows is not None and row_idx not in allowed_rows:
                    continue

                abs_img_path = images_dir / shard / image_file
                if args.require_image_exists and not abs_img_path.exists():
                    num_skipped += 1
                    continue

                if args.relative_paths:
                    try:
                        img_path_str = str(abs_img_path.relative_to(images_dir))
                    except ValueError:
                        img_path_str = str(abs_img_path)
                else:
                    img_path_str = str(abs_img_path)

                if args.mode == "select_all":
                    for qa_type, (pos_col, neg_col) in QA_MAP.items():
                        for polarity, col in (("pos", pos_col), ("neg", neg_col)):
                            parsed = parse_qa(qa_vals.get(col))
                            if not parsed:
                                continue
                            q, acc, rej = parsed
                            rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type, polarity)
                            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            num_written += 1

                elif args.mode == "either_pos_or_neg":
                    for qa_type, (pos_col, neg_col) in QA_MAP.items():
                        pos_parsed = parse_qa(qa_vals.get(pos_col))
                        neg_parsed = parse_qa(qa_vals.get(neg_col))

                        choice = None
                        if pos_parsed and neg_parsed:
                            choice = rng.choice(["pos","neg"])
                        elif pos_parsed:
                            choice = "pos"
                        elif neg_parsed:
                            choice = "neg"

                        if not choice:
                            continue

                        q, acc, rej = pos_parsed if choice == "pos" else neg_parsed
                        rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type, choice)
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        num_written += 1

                elif args.mode == "25_pos_75_neg":
                    for qa_type, (pos_col, neg_col) in QA_MAP.items():
                        pos_parsed = parse_qa(qa_vals.get(pos_col))
                        neg_parsed = parse_qa(qa_vals.get(neg_col))

                        choice = None
                        if pos_parsed and neg_parsed:
                            # weighted choice: 25% pos, 75% neg
                            choice = "pos" if rng.random() < 0.25 else "neg"
                        elif pos_parsed:
                            choice = "pos"
                        elif neg_parsed:
                            choice = "neg"

                        if not choice:
                            continue

                        q, acc, rej = pos_parsed if choice == "pos" else neg_parsed
                        rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type, choice)
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        num_written += 1

                elif args.mode == "only_neg":
                    # for each qa_type, emit only the negative pair if present
                    for qa_type, (_pos_col, neg_col) in QA_MAP.items():
                        neg_parsed = parse_qa(qa_vals.get(neg_col))
                        if not neg_parsed:
                            continue
                        q, acc, rej = neg_parsed
                        rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type, "neg")
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        num_written += 1

                elif args.mode in ("only_obj", "only_attr", "only_rel", "only_wh"):
                    # emit pos and/or neg only for the selected QA type (mirror select_all behavior but restricted)
                    mode_to_key = {
                        "only_obj": "obj",
                        "only_attr": "attr",
                        "only_rel": "rel",
                        "only_wh": "wh",
                    }
                    qa_key = mode_to_key[args.mode]
                    emit_for_single_type(qa_key, qa_vals, img_path_str, shard, row_idx)

                elif args.mode == "pon_except_wh":
                    for qa_type, (pos_col, neg_col) in QA_MAP.items():
                        if qa_type != "wh":
                            pos_parsed = parse_qa(qa_vals.get(pos_col))
                            neg_parsed = parse_qa(qa_vals.get(neg_col))

                            choice = None
                            if pos_parsed and neg_parsed:
                                choice = rng.choice(["pos", "neg"])
                            elif pos_parsed:
                                choice = "pos"
                            elif neg_parsed:
                                choice = "neg"

                            if not choice:
                                continue

                            q, acc, rej = pos_parsed if choice == "pos" else neg_parsed
                            rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type, choice)
                            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            num_written += 1
                        else:
                            # wh: only negative
                            neg_parsed = parse_qa(qa_vals.get(neg_col))
                            if not neg_parsed:
                                continue
                            q, acc, rej = neg_parsed
                            rec = build_record(img_path_str, q, acc, rej, shard, row_idx, qa_type, "neg")
                            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            num_written += 1

                elif args.mode == "either_rel_or_wh":
                    # For each image (row), randomly choose rel OR wh, then emit BOTH pos and neg of that chosen type.
                    emit_for_rel_or_wh_for_image(
                        rng=rng,
                        qa_vals=qa_vals,
                        img_path_str=img_path_str,
                        shard=shard,
                        row_idx=row_idx,
                        emit_for_single_type_fn=emit_for_single_type,
                    )

                elif args.mode == "30_rel_70_wh":
                    emit_for_weighted_choice_for_image(
                        rng=rng,
                        qa_vals=qa_vals,
                        img_path_str=img_path_str,
                        shard=shard,
                        row_idx=row_idx,
                        weighted_choices=[("rel", 0.30), ("wh", 0.70)],
                        emit_for_single_type_fn=emit_for_single_type,
                    )

                elif args.mode == "15_attr_15_rel_70_wh":
                    emit_for_weighted_choice_for_image(
                        rng=rng,
                        qa_vals=qa_vals,
                        img_path_str=img_path_str,
                        shard=shard,
                        row_idx=row_idx,
                        weighted_choices=[("attr", 0.15), ("rel", 0.15), ("wh", 0.70)],
                        emit_for_single_type_fn=emit_for_single_type,
                    )

                elif args.mode == "10_obj_10_attr_10_rel_70_wh":
                    emit_for_weighted_choice_for_image(
                        rng=rng,
                        qa_vals=qa_vals,
                        img_path_str=img_path_str,
                        shard=shard,
                        row_idx=row_idx,
                        weighted_choices=[("obj", 0.10), ("attr", 0.10), ("rel", 0.10), ("wh", 0.70)],
                        emit_for_single_type_fn=emit_for_single_type,
                    )

                else:
                    # Shouldn't reach here due to argparse choices
                    pass

    print(f"[DONE] Wrote {num_written} records to {out_path}")
    if num_skipped:
        print(f"[NOTE] Skipped {num_skipped} rows due to missing/invalid image_file (and/or missing file if required).")

if __name__ == "__main__":
    main()
