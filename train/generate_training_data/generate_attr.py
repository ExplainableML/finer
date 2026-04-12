#!/usr/bin/env python3
# summarize_attributes.py
import os, time, glob
from params import parse_generate_attr_args
from typing import Optional, List

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.attribute_utils import (
    batched_generate_attribute_phrases,
    batched_generate_negative_attribute_phrases,
    batched_generate_negative_attribute_phrases_by_index,
    extract_acc_rej_from_cell,
    batched_generate_attr_diff_phrases,
)

from  utils.general_utils import get_dist_info, resolve_hf_snapshot_dir, detect_caption_col

# ------------------------------ local helpers -------------------------------

def detect_qa_col(pf: pq.ParquetFile, preferred: str) -> str:
    """
    Find the column that contains the 'Q=... || A_acc=... || A_rej=...' cell.
    Tries: <preferred>, 'attr_qa_pos', 'qa_pos', 'pos_qa', 'attr_pos'.
    """
    names = set(pf.schema_arrow.names)
    for c in [preferred, "attr_qa_pos", "qa_pos", "pos_qa", "attr_pos"]:
        if c in names:
            return c
    raise ValueError(f"QA column not found. Available: {sorted(names)}")


def detect_attr_phrase_col(pf: pq.ParquetFile, preferred: str) -> str:
    """
    Detect the column that holds the attribute phrase (the positive 'attr' string).
    Tries: <preferred>, 'attr', 'attribute_phrase', 'attributes', 'attr_phrase'.
    """
    names = set(pf.schema_arrow.names)
    candidates = [preferred, "attr", "attribute_phrase", "attributes", "attr_phrase"]
    for c in candidates:
        if c and c in names:
            return c
    raise ValueError(
        f"Attribute phrase column not found. Tried {candidates}. "
        f"Available: {sorted(names)}"
    )

# ======================== per-file pipeline (attributes) ====================

def process_file_sidecar(
    path: str,
    caption_col_name: str,
    tokenizer,
    llm: LLM,
    sampling: SamplingParams,
    out_dir: Optional[str],
    read_batch_rows: int = 2048,
    gen_batch_size: int = 64,
    neg_attributes: bool = False,
    attr_col_name: Optional[str] = None,
    neg_attr_edit_mode: str = "random",      
    attr_index_mode: str = "random",         
    attr_index_seed: Optional[int] = None,   
    max_attr_units: int = 5,                 
):
    """
    If neg_attributes=False (default/original):
        For each caption, generate a single 'attribute phrase' describing ONE main object with 1–5 attributes.
        Writes sidecar with columns: row, attribute_phrase.

    If neg_attributes=True:
        For each existing attribute phrase (no caption needed), replace exactly one attribute with a negative.
        Writes sidecar with columns: row, attr_neg.
    """
    t0 = time.time()
    base = os.path.basename(path)
    pf = pq.ParquetFile(path)

    # Columns needed per mode
    if neg_attributes:
        attr_col = detect_attr_phrase_col(pf, attr_col_name or "attr")
    else:
        cap_col = detect_caption_col(pf, caption_col_name)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sidecar_name = os.path.splitext(base)[0] + ".parquet"
    out_path = os.path.join(out_dir or os.path.dirname(path), sidecar_name)

    writer = None
    emitted = 0
    row_offset = 0
    acc_rows: List[int] = []
    acc_caps: List[str] = []   # used in positive mode
    acc_attrs: List[str] = []  # used in negative mode

    def flush():
        nonlocal writer, emitted, acc_rows, acc_caps, acc_attrs
        if not acc_rows:
            return

        if neg_attributes:
            if neg_attr_edit_mode == "index":
                # Index-based negative attribute editing
                phrases = batched_generate_negative_attribute_phrases_by_index(
                    llm=llm,
                    tokenizer=tokenizer,
                    attr_phrases=acc_attrs,
                    sampling_params=sampling,
                    gen_batch_size=gen_batch_size,
                    attr_index_mode=attr_index_mode,
                    seed=attr_index_seed,
                    max_units=max_attr_units,
                    validate=True,
                )
            else:
                # Random-position negative attribute editing (existing behavior)
                phrases = batched_generate_negative_attribute_phrases(
                    llm=llm,
                    tokenizer=tokenizer,
                    attr_phrases=acc_attrs,
                    sampling_params=sampling,
                    gen_batch_size=gen_batch_size,
                    validate=True,
                )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "attr_neg": pa.array(phrases, type=pa.string()),
            })
        
        else:
            # Original: generate attribute phrases from captions
            phrases = batched_generate_attribute_phrases(
                llm=llm,
                tokenizer=tokenizer,
                captions=acc_caps,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "attribute_phrase": pa.array(phrases, type=pa.string()),
            })

        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        emitted += len(acc_rows)
        acc_rows.clear()
        acc_caps.clear()
        acc_attrs.clear()

    mode = "NEG-ATTR" if neg_attributes else "ATTR"
    print(f"[START:{mode}] {base} -> {sidecar_name}")

    if neg_attributes:
        # Read attribute phrase only
        for batch in pf.iter_batches(columns=[attr_col], batch_size=read_batch_rows):
            attrs = batch.column(0).to_pylist()
            n = len(attrs)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_attrs.extend([a if a is not None else "" for a in attrs])
            row_offset += n
            if len(acc_rows) >= 10_000:
                flush()
    else:
        # Original behavior: read captions only
        for batch in pf.iter_batches(columns=[cap_col], batch_size=read_batch_rows):
            caps = batch.column(0).to_pylist()
            n = len(caps)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_caps.extend([c if c is not None else "" for c in caps])
            row_offset += n
            if len(acc_rows) >= 10_000:
                flush()

    flush()
    if writer is not None:
        writer.close()
    print(f"[DONE:{mode}]  {base} -> {sidecar_name} | {emitted} rows | {time.time()-t0:.1f}s")


def process_file_attr_diff(
    path: str,
    qa_col_name: str,
    tokenizer,
    llm: LLM,
    sampling: SamplingParams,
    out_dir: Optional[str],
    read_batch_rows: int = 2048,
    gen_batch_size: int = 64,
):
    """
    Reads a QA column with cells like: 'Q=... || A_acc=... || A_rej=...'
    Uses your diff prompts to produce two MINIMAL full-sentence outputs per row:
      - acc_phrase_1attr  (keeps surface format of ACC answer)
      - rej_phrase_1attr  (keeps surface format of REJ answer)
    Writes a sidecar parquet with columns: row, acc_phrase_1attr, rej_phrase_1attr
    """
    t0 = time.time()
    base = os.path.basename(path)
    pf = pq.ParquetFile(path)

    qa_col = detect_qa_col(pf, qa_col_name)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sidecar_name = os.path.splitext(base)[0] + ".parquet"
    out_path = os.path.join(out_dir or os.path.dirname(path), sidecar_name)

    writer = None
    emitted = 0
    row_offset = 0
    acc_rows: List[int] = []
    accrej_pairs: List[tuple] = []

    def flush():
        nonlocal writer, emitted, acc_rows, accrej_pairs
        if not acc_rows:
            return
        pairs = batched_generate_attr_diff_phrases(
            llm=llm,
            tokenizer=tokenizer,
            acc_rej_pairs=accrej_pairs,
            sampling_params=sampling,
            gen_batch_size=gen_batch_size,
        )
        acc_list = [p[0] for p in pairs]
        rej_list = [p[1] for p in pairs]
        tbl = pa.table({
            "row": pa.array(acc_rows, type=pa.int64()),
            "acc_phrase_1attr": pa.array(acc_list, type=pa.string()),
            "rej_phrase_1attr": pa.array(rej_list, type=pa.string()),
        })
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        emitted += len(acc_rows)
        acc_rows.clear()
        accrej_pairs.clear()

    print(f"[START:ATTR-DIFF] {base} -> {sidecar_name}")
    for batch in pf.iter_batches(columns=[qa_col], batch_size=read_batch_rows):
        cells = batch.column(0).to_pylist()
        n = len(cells)
        acc_rows.extend(range(row_offset, row_offset + n))
        for cell in cells:
            a_acc, a_rej = extract_acc_rej_from_cell(cell or "")
            accrej_pairs.append((a_acc, a_rej))
        row_offset += n
        if len(acc_rows) >= 10_000:
            flush()

    flush()
    if writer is not None:
        writer.close()
    print(f"[DONE:ATTR-DIFF]  {base} -> {sidecar_name} | {emitted} rows | {time.time()-t0:.1f}s")


def main():
    args = parse_generate_attr_args()
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
    if args.neg_attributes and args.temperature <= 0.0:
        print("[Warn] --neg-attributes with temperature=0.0 may change the same position repeatedly. "
              "Consider setting --temperature 0.4–0.8 for more randomness.")

    for f in my_files:
        base = os.path.basename(f)
        sidecar_name = os.path.splitext(base)[0] + ".parquet"
        out_path = os.path.join(args.out_dir or os.path.dirname(f), sidecar_name)
        if args.skip_if_exists and os.path.exists(out_path):
            print(f"[SKIP] {base} -> {sidecar_name} already exists.")
            continue
        
        if args.attr_diff:
            process_file_attr_diff(
                path=f,
                qa_col_name=args.qa_column,
                tokenizer=tokenizer,
                llm=llm,
                sampling=sampling,
                out_dir=args.out_dir,
                read_batch_rows=args.read_batch_rows,
                gen_batch_size=args.gen_batch_size,
            )
        else:
            process_file_sidecar(
                path=f,
                caption_col_name=args.caption_column,
                tokenizer=tokenizer,
                llm=llm,
                sampling=sampling,
                out_dir=args.out_dir,
                read_batch_rows=args.read_batch_rows,
                gen_batch_size=args.gen_batch_size,
                neg_attributes=args.neg_attributes,       
                attr_col_name=args.attr_column,           
                neg_attr_edit_mode=args.neg_attr_edit_mode,   
                attr_index_mode=args.attr_index_mode,         
                attr_index_seed=args.attr_index_seed,        
                max_attr_units=args.max_attr_units,      
            )

if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
