#!/usr/bin/env python3
import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


REPO_ID = "xiaorui638/FINER-Tuning-data"


def rewrite_image_path(old_path: str, data_root: Path) -> str:
    p = Path(old_path)
    shard = p.parent.name
    filename = p.name
    return str(data_root / "images_extracted" / shard / filename)


def extract_tars(images_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(images_dir.glob("*.tar"))
    print(f"Found {len(tar_files)} tar files.")

    for tar_path in tqdm(tar_files, desc="Extracting image tars"):
        shard_name = tar_path.stem
        shard_out = out_dir / shard_name

        if shard_out.exists() and any(shard_out.iterdir()):
            continue

        shard_out.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(shard_out)


def rewrite_jsonl_file(path: Path, data_root: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    n = 0
    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue

            sample = json.loads(line)

            if "images" in sample:
                sample["images"] = [
                    rewrite_image_path(img_path, data_root)
                    for img_path in sample["images"]
                ]

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    tmp_path.replace(path)
    print(f"Rewrote {n} samples in place: {path}")


def rewrite_all_jsonl(data_root: Path):
    for split_dir in ["sampled_annotations"]:
        src_root = data_root / split_dir
        if not src_root.exists():
            continue

        for src_path in sorted(src_root.rglob("*.jsonl")):
            rewrite_jsonl_file(src_path, data_root)


def update_llamafactory_dataset_info(
    llamafactory_dir: Path,
    dataset_name: str,
    jsonl_path: Path,
):
    dataset_info_path = llamafactory_dir / "data" / "dataset_info.json"

    if not dataset_info_path.exists():
        raise FileNotFoundError(f"dataset_info.json not found: {dataset_info_path}")

    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if dataset_name not in dataset_info:
        raise KeyError(f"Dataset key '{dataset_name}' not found in {dataset_info_path}")

    dataset_info[dataset_name]["file_name"] = str(jsonl_path.resolve())

    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated LlamaFactory dataset_info.json:")
    print(f"  dataset key : {dataset_name}")
    print(f"  file_name   : {jsonl_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to download and prepare FINER-Tuning-data.",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip tar extraction if images are already extracted.",
    )
    parser.add_argument(
        "--llamafactory_dir",
        type=str,
        default=None,
        help="Path to local LlamaFactory root. If provided, dataset_info.json will be updated.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pixmo_dpo_220k_pon",
        help="Dataset key in LlamaFactory/data/dataset_info.json to update.",
    )
    parser.add_argument(
        "--jsonl_relpath",
        type=str,
        default="sampled_annotations/main_experiments/first6_pon.jsonl",
        help="Path relative to output_dir for the JSONL file to register in LlamaFactory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to: {output_dir}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=[
            "images/*.tar",
            "sampled_annotations/**/*.jsonl",
            "full_annotations/**/*.jsonl",
            "README.md",
        ],
    )

    if not args.skip_extract:
        extract_tars(
            images_dir=output_dir / "images",
            out_dir=output_dir / "images_extracted",
        )

    rewrite_all_jsonl(output_dir)

    if args.llamafactory_dir is not None:
        llamafactory_dir = Path(args.llamafactory_dir).expanduser().resolve()
        jsonl_path = output_dir / args.jsonl_relpath
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        update_llamafactory_dataset_info(
            llamafactory_dir=llamafactory_dir,
            dataset_name=args.dataset_name,
            jsonl_path=jsonl_path,
        )

    print("\nDone.")
    print(f"Updated annotation files in place under: {output_dir / 'sampled_annotations'}")
    print(f"Updated annotation files in place under: {output_dir / 'full_annotations'}")
    print(f"Images are under: {output_dir / 'images_extracted'}")


if __name__ == "__main__":
    main()