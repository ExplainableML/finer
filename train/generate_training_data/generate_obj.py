import os, time, glob
from params import parse_generate_obj_args
from typing import Optional, List

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.object_utils import (
    batched_generate_objects_phrases,
    batched_generate_negative_phrases,
    batched_categorize_captions, 
)

from  utils.general_utils import get_dist_info, resolve_hf_snapshot_dir, detect_caption_col


def detect_objects_col(pf: pq.ParquetFile, preferred: str) -> str:
    """
    Detect the column that holds the objects phrase.
    Tries: <preferred>, 'obj', 'objects_phrase', 'objects', 'object_phrase'.
    """
    names = set(pf.schema_arrow.names)
    candidates = [preferred, "obj", "objects_phrase", "objects", "object_phrase"]
    for c in candidates:
        if c and c in names:
            return c
    raise ValueError(
        f"Objects phrase column not found. Tried {candidates}. "
        f"Available: {sorted(names)}"
    )

# ======================== per-file pipeline (objects) =======================

def process_file_sidecar(
    path: str,
    caption_col_name: str,
    tokenizer,
    llm: LLM,
    sampling: SamplingParams,
    out_dir: Optional[str],
    read_batch_rows: int = 2048,
    gen_batch_size: int = 64,
    neg_objects: bool = False,
    objects_col_name: Optional[str] = None,
    categorize: bool = False,                
    category_col_name: str = "category", 
):
    """
    Modes:
      - neg_objects=False and categorize=False (default):
          For each caption, generate 'objects_phrase'. Writes sidecar: (row, objects_phrase).
      - neg_objects=True:
          For each (caption, obj_phrase), replace exactly one object with a negative. Writes sidecar: (row, obj_neg).
      - categorize=True:
          For each caption, predict one category in {'natural_image','chart_graph','screenshot_ui','document_text'}.
          Writes sidecar: (row, <category-column>).
    """
    t0 = time.time()
    base = os.path.basename(path)
    pf = pq.ParquetFile(path)
    cap_col = detect_caption_col(pf, caption_col_name)

    obj_col = None
    if neg_objects:
        obj_col = detect_objects_col(pf, objects_col_name or "obj")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sidecar_name = os.path.splitext(base)[0] + ".parquet"
    out_path = os.path.join(out_dir or os.path.dirname(path), sidecar_name)

    writer = None
    emitted = 0
    row_offset = 0
    acc_rows: List[int] = []
    acc_caps: List[str] = []
    acc_objs: List[str] = []

    def flush():
        nonlocal writer, emitted, acc_rows, acc_caps, acc_objs
        if not acc_caps:
            return

        if neg_objects:
            phrases = batched_generate_negative_phrases(
                llm=llm,
                tokenizer=tokenizer,
                captions=acc_caps,
                obj_phrases=acc_objs,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
                validate=True,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "obj_neg": pa.array(phrases, type=pa.string()),
            })
        elif categorize:
            cats = batched_categorize_captions(
                llm=llm,
                tokenizer=tokenizer,
                captions=acc_caps,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                category_col_name: pa.array(cats, type=pa.string()),
            })
        else:
            phrases = batched_generate_objects_phrases(
                llm=llm,
                tokenizer=tokenizer,
                captions=acc_caps,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "objects_phrase": pa.array(phrases, type=pa.string()),
            })

        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        emitted += len(acc_rows)
        acc_rows.clear()
        acc_caps.clear()
        acc_objs.clear()

    mode = ("NEG-OBJECTS" if neg_objects else ("CATEGORIZE" if categorize else "OBJECTS"))
    print(f"[START:{mode}] {base} -> {sidecar_name}")

    if neg_objects:
        for batch in pf.iter_batches(columns=[cap_col, obj_col], batch_size=read_batch_rows):
            caps = batch.column(0).to_pylist()
            objs = batch.column(1).to_pylist()
            n = len(caps)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_caps.extend([c if c is not None else "" for c in caps])
            acc_objs.extend([o if o is not None else "" for o in objs])
            row_offset += n
            if len(acc_caps) >= 10_000:
                flush()
    else:
        for batch in pf.iter_batches(columns=[cap_col], batch_size=read_batch_rows):
            caps = batch.column(0).to_pylist()
            n = len(caps)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_caps.extend([c if c is not None else "" for c in caps])
            row_offset += n
            if len(acc_caps) >= 10_000:
                flush()

    flush()
    if writer is not None:
        writer.close()
    print(f"[DONE:{mode}]  {base} -> {sidecar_name} | {emitted} rows | {time.time()-t0:.1f}s")


def main():
    args = parse_generate_obj_args()
    rank, world, local_rank = get_dist_info()
    if "CUDA_VISIBLE_DEVICES" not in os.environ and local_rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        print(f"[Info] Set CUDA_VISIBLE_DEVICES={local_rank}")

    files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError("No parquet files matched.")

    my_files = [f for i, f in enumerate(files) if (i % world) == rank]
    print(f"[Rank {rank}/{world}] files assigned: {len(my_files)}")

    model_path = args.model
    if os.path.isdir(model_path):
        model_path = resolve_hf_snapshot_dir(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=args.inference_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=1,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    if args.neg_objects and args.temperature <= 0.0:
        print("[Warn] --neg-objects with temperature=0.0 may replace the same position repeatedly. "
              "Consider setting --temperature 0.4–0.8 for more randomness.")

    for f in my_files:
        base = os.path.basename(f)
        sidecar_name = os.path.splitext(base)[0] + ".parquet"
        out_path = os.path.join(args.out_dir or os.path.dirname(f), sidecar_name)
        if args.skip_if_exists and os.path.exists(out_path):
            print(f"[SKIP] {base} -> {sidecar_name} already exists.")
            continue

        process_file_sidecar(
            path=f,
            caption_col_name=args.caption_column,
            tokenizer=tokenizer,
            llm=llm,
            sampling=sampling,
            out_dir=args.out_dir,
            read_batch_rows=args.read_batch_rows,
            gen_batch_size=args.gen_batch_size,
            neg_objects=args.neg_objects,            
            objects_col_name=args.objects_column,
            categorize=args.categorize,                 
            category_col_name=args.category_column,  
        )

if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
