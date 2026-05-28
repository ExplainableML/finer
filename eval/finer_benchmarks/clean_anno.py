from pathlib import Path
import pandas as pd

# Input / output directories
in_dir = Path("/data/rxiao/finer/eval/finer/improved_annotations/finer_comprecap")
out_dir = in_dir / "cleaned"

out_dir.mkdir(parents=True, exist_ok=True)

# Columns to keep, in this exact order
keep_cols = [
    "image",
    "qtype",
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "choice5",
    "gt_index",
]

for csv_path in sorted(in_dir.glob("*.csv")):
    print(f"Processing: {csv_path.name}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in keep_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_path.name} is missing columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    df_clean = df[keep_cols]

    out_path = out_dir / csv_path.name
    df_clean.to_csv(out_path, index=False)

    print(f"  saved to: {out_path}")

print("Done.")