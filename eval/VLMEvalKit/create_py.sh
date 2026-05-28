cat > find_qual_cases.py <<'PY'
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() in [".csv"]:
        return pd.read_csv(path)
    if path.suffix.lower() in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path}")


def normalize_colnames(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df, candidates):
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    return None


def infer_key_col(df1, df2):
    candidates = [
        "index", "question_id", "id", "sample_id", "record_id",
        "line_idx", "line_id", "uid"
    ]
    for c in candidates:
        c1 = find_col(df1, [c])
        c2 = find_col(df2, [c])
        if c1 is not None and c2 is not None:
            return c1, c2

    # fallback: use question if available
    c1 = find_col(df1, ["question"])
    c2 = find_col(df2, ["question"])
    if c1 is not None and c2 is not None:
        return c1, c2

    raise ValueError(
        "Could not infer key column. Please inspect columns and pass --key-col manually."
    )


def infer_correct_col(df):
    candidates = [
        "hit", "score", "correct", "is_correct", "accuracy",
        "acc", "match", "auxmatch", "judge", "rating"
    ]
    for c in candidates:
        col = find_col(df, [c])
        if col is not None:
            return col

    # fuzzy fallback
    for col in df.columns:
        name = col.lower()
        if any(x in name for x in ["correct", "hit", "score", "match"]):
            return col

    raise ValueError(
        f"Could not infer correctness column. Available columns:\n{list(df.columns)}\n"
        "Please pass --ours-correct-col and --base-correct-col."
    )


def to_bool_correct(x):
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        return float(x) > 0

    s = str(x).strip().lower()

    correct_values = {
        "1", "true", "yes", "y", "correct", "right",
        "success", "matched", "match", "pass", "passed"
    }
    incorrect_values = {
        "0", "false", "no", "n", "incorrect", "wrong",
        "fail", "failed", "unmatched", "mismatch"
    }

    if s in correct_values:
        return True
    if s in incorrect_values:
        return False

    # Sometimes score is stored as string like "1.0"
    try:
        return float(s) > 0
    except Exception:
        return False


def safe_name(x, max_len=80):
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:max_len].strip("_") or "unknown"


def parse_image_cell(cell):
    """Return a list of possible image references from a dataframe cell."""
    if pd.isna(cell):
        return []

    if isinstance(cell, list):
        return [str(x) for x in cell]

    s = str(cell).strip()
    if not s:
        return []

    # Try JSON/list-like strings
    for parser in [json.loads]:
        try:
            obj = parser(s)
            if isinstance(obj, list):
                return [str(x) for x in obj]
            if isinstance(obj, str):
                return [obj]
        except Exception:
            pass

    # Common separators
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s and not os.path.exists(s):
        parts = [x.strip() for x in s.split(",") if x.strip()]
        if len(parts) > 1:
            return parts

    return [s]


def locate_image(ref, lmu_data, image_root):
    ref = str(ref).strip()
    if not ref:
        return None

    candidates = []

    p = Path(ref)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / ref,
            Path(lmu_data) / ref,
            Path(image_root) / ref,
            Path(image_root) / Path(ref).name,
            Path(lmu_data) / "images" / ref,
            Path(lmu_data) / "images" / Path(ref).name,
        ])

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Last fallback: search by filename under image_root
    fname = Path(ref).name
    if fname:
        matches = list(Path(image_root).rglob(fname)) if Path(image_root).exists() else []
        if matches:
            return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="Custom model auxmatch/result xlsx")
    parser.add_argument("--base", required=True, help="Original model auxmatch/result xlsx")
    parser.add_argument("--dataset-tsv", default=None, help="Optional VLMEvalKit dataset TSV")
    parser.add_argument("--lmu-data", default=os.environ.get("LMUData", "/data/rxiao/LMUData"))
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--key-col", default=None)
    parser.add_argument("--ours-correct-col", default=None)
    parser.add_argument("--base-correct-col", default=None)
    parser.add_argument("--max-cases", type=int, default=50)

    args = parser.parse_args()

    ours = normalize_colnames(read_table(args.ours))
    base = normalize_colnames(read_table(args.base))

    print("OURS columns:", list(ours.columns))
    print("BASE columns:", list(base.columns))

    if args.key_col is not None:
        ours_key = base_key = args.key_col
    else:
        ours_key, base_key = infer_key_col(ours, base)

    ours_correct_col = args.ours_correct_col or infer_correct_col(ours)
    base_correct_col = args.base_correct_col or infer_correct_col(base)

    print(f"Using key columns: ours={ours_key}, base={base_key}")
    print(f"Using correctness columns: ours={ours_correct_col}, base={base_correct_col}")

    ours["_ours_correct_bool"] = ours[ours_correct_col].apply(to_bool_correct)
    base["_base_correct_bool"] = base[base_correct_col].apply(to_bool_correct)

    # Keep all useful columns, adding prefixes to avoid conflicts
    ours_pref = ours.add_prefix("ours__")
    base_pref = base.add_prefix("base__")

    merged = ours_pref.merge(
        base_pref,
        left_on=f"ours__{ours_key}",
        right_on=f"base__{base_key}",
        how="inner",
    )

    selected = merged[
        (merged["ours___ours_correct_bool"] == True)
        & (merged["base___base_correct_bool"] == False)
    ].copy()

    print(f"Matched samples: {len(merged)}")
    print(f"Ours correct & base wrong: {len(selected)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all selected rows
    selected_csv = out_dir / "ours_correct_base_wrong.csv"
    selected.to_csv(selected_csv, index=False)
    print(f"Saved table: {selected_csv}")

    # Load dataset TSV for image lookup if provided
    dataset = None
    if args.dataset_tsv is not None and Path(args.dataset_tsv).exists():
        dataset = normalize_colnames(read_table(args.dataset_tsv))
        print("DATASET columns:", list(dataset.columns))

    image_root = args.image_root
    if image_root is None:
        image_root = str(Path(args.lmu_data) / "images" / "HallusionBench")

    # possible columns containing image paths
    image_candidates = [
        "image", "images", "img", "img_path", "image_path", "filename", "file_name"
    ]

    saved_cases = []

    for rank, (_, row) in enumerate(selected.head(args.max_cases).iterrows(), start=1):
        sample_key = row[f"ours__{ours_key}"]
        case_dir = out_dir / f"case_{rank:04d}_id_{safe_name(sample_key)}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Try to find image refs from ours/base rows first
        image_refs = []

        for prefix in ["ours__", "base__"]:
            for cand in image_candidates:
                col = prefix + cand
                if col in selected.columns:
                    image_refs.extend(parse_image_cell(row[col]))

        # If not found, try dataset TSV by key/index
        if dataset is not None:
            ds_key = find_col(dataset, [ours_key, base_key, "index", "id", "question_id"])
            if ds_key is not None:
                ds_match = dataset[dataset[ds_key].astype(str) == str(sample_key)]
                if len(ds_match) > 0:
                    ds_row = ds_match.iloc[0]
                    for cand in image_candidates:
                        ds_img_col = find_col(dataset, [cand])
                        if ds_img_col is not None:
                            image_refs.extend(parse_image_cell(ds_row[ds_img_col]))

        # Deduplicate refs
        image_refs = list(dict.fromkeys([x for x in image_refs if x]))

        copied_images = []
        for img_i, ref in enumerate(image_refs):
            img_path = locate_image(ref, args.lmu_data, image_root)
            if img_path is None:
                continue
            suffix = img_path.suffix or ".png"
            dst = case_dir / f"image_{img_i:02d}{suffix}"
            shutil.copy2(img_path, dst)
            copied_images.append(str(dst))

        # Save metadata for this case
        meta = row.to_dict()
        meta["_copied_images"] = copied_images
        meta["_raw_image_refs"] = image_refs

        with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        # Create a readable markdown summary
        def get_first_existing(row, names):
            for n in names:
                if n in row and not pd.isna(row[n]):
                    return row[n]
            return ""

        question = get_first_existing(row, [
            "ours__question", "ours__Question", "ours__query", "ours__prompt",
            "base__question", "base__Question", "base__query", "base__prompt",
        ])
        gt = get_first_existing(row, [
            "ours__answer", "ours__Answer", "ours__gt", "ours__label",
            "base__answer", "base__Answer", "base__gt", "base__label",
        ])
        ours_pred = get_first_existing(row, [
            "ours__prediction", "ours__pred", "ours__response", "ours__model_answer",
            "ours__answer_pred",
        ])
        base_pred = get_first_existing(row, [
            "base__prediction", "base__pred", "base__response", "base__model_answer",
            "base__answer_pred",
        ])

        with open(case_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write(f"# Case {rank}: {sample_key}\n\n")
            f.write(f"## Question\n{question}\n\n")
            f.write(f"## Ground truth\n{gt}\n\n")
            f.write(f"## Custom prediction\n{ours_pred}\n\n")
            f.write(f"## Base prediction\n{base_pred}\n\n")
            f.write(f"## Images\n")
            for img in copied_images:
                f.write(f"- {Path(img).name}\n")

        saved_cases.append(str(case_dir))

    with open(out_dir / "case_dirs.txt", "w", encoding="utf-8") as f:
        for d in saved_cases:
            f.write(d + "\n")

    print(f"Saved {len(saved_cases)} qualitative case folders under: {out_dir}")


if __name__ == "__main__":
    main()
PY