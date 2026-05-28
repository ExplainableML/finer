from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Image

# =========================
# Paths
# =========================

CSV_DIR = Path("/data/rxiao/finer/eval/finer/improved_annotations/finer_comprecap")
IMAGE_DIR = Path("/data/rxiao/comprecap_images")

# Change this to your Hugging Face dataset repo name.
REPO_ID = "xiaorui638/finer-comprecap"

PRIVATE = True

CSV_FILES = {
    "multi_attr": "multi_attr.csv",
    "multi_obj": "multi_obj.csv",
    "multi_rel": "multi_rel.csv",
    "wh": "wh.csv",
}

REQUIRED_COLS = [
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


def build_split(split_name: str, csv_name: str) -> Dataset:
    csv_path = CSV_DIR / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, keep_default_na=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df = df[REQUIRED_COLS].copy()

    # Keep the original image filename for debugging / traceability.
    df["image_filename"] = df["image"].astype(str)

    # Replace image column with absolute local paths.
    # After cast_column("image", Image()), this becomes an HF image feature.
    df["image"] = df["image_filename"].apply(lambda x: str(IMAGE_DIR / x))

    # Normalize gt_index.
    df["gt_index"] = pd.to_numeric(df["gt_index"], errors="raise").astype(int)

    # Optional metadata column.
    df["subset"] = split_name

    # Verify all images exist before uploading.
    missing_images = []
    for p in df["image"]:
        if not Path(p).exists():
            missing_images.append(p)

    if missing_images:
        print(f"\nFound {len(missing_images)} missing images. First 20:")
        for p in missing_images[:20]:
            print(p)
        raise FileNotFoundError("Some images are missing. Fix paths before uploading.")

    ds = Dataset.from_pandas(df, preserve_index=False)

    # This makes the column an actual image feature.
    # When pushed to HF, image bytes are embedded in Parquet by default.
    ds = ds.cast_column("image", Image())

    return ds


def main():
    dataset_dict = DatasetDict()

    for split_name, csv_name in CSV_FILES.items():
        print(f"Building split: {split_name} from {csv_name}")
        dataset_dict[split_name] = build_split(split_name, csv_name)
        print(dataset_dict[split_name])

    print("\nDatasetDict:")
    print(dataset_dict)

    print(f"\nPushing to Hugging Face: {REPO_ID}")
    dataset_dict.push_to_hub(
        REPO_ID,
        private=PRIVATE,
        max_shard_size="500MB",
    )

    print("\nDone.")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()