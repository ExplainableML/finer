#!/usr/bin/env python3
import os, time, glob
from params import parse_generate_wh_args
from typing import Optional, List

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.wh_utils import (
    batched_generate_phrases,                
    batched_generate_whqa_from_wh_column,  
    batched_generate_neg_whqa_from_whqa_column,
)

from  utils.general_utils import get_dist_info, resolve_hf_snapshot_dir, detect_caption_col


# ------------------------------ local helpers -------------------------------

def detect_wh_col(pf: pq.ParquetFile, preferred: str) -> str:
    """
    Detect the column that holds the ';'-separated wh sentences.
    Tries: <preferred>, 'wh', 'wh_sentences', 'wh_phrase', 'relations_wh'.
    """
    names = set(pf.schema_arrow.names)
    candidates = [preferred, "wh", "wh_sentences", "wh_phrase", "relations_wh"]
    for c in candidates:
        if c and c in names:
            return c
    raise ValueError(
        f"WH sentences column not found. Tried {candidates}. "
        f"Available: {sorted(names)}"
    )

def detect_whqa_col(pf: pq.ParquetFile, preferred: str) -> str:
    """
    Detect the column that holds POSITIVE QA pairs (semicolon-joined 'Q=... || A=...').
    Tries: <preferred>, 'wh_qa', 'qa', 'whqa'.
    """
    names = set(pf.schema_arrow.names)
    candidates = [preferred, "wh_qa", "qa", "whqa"]
    for c in candidates:
        if c and c in names:
            return c
    raise ValueError(
        f"Positive QA column not found. Tried {candidates}. "
        f"Available: {sorted(names)}"
    )

# ======================== per-file pipeline (rephrase/QA) ===================

def process_file_sidecar(
    path: str,
    caption_col_name: str,
    tokenizer,
    llm: LLM,
    sampling: SamplingParams,
    out_dir: Optional[str],
    read_batch_rows: int = 2048,
    gen_batch_size: int = 64,
    # --- existing ---
    wh_qa: bool = False,
    wh_col_name: Optional[str] = None,
    neg_qa: bool = False,
    whqa_col_name: Optional[str] = None,
):
    """
    Modes:
      Default (wh_qa=False, neg_qa=False): from captions -> generate WH sentences.
        Writes: columns [row, wh_sentences]

      QA mode (wh_qa=True): from existing 'wh' column -> generate one POS QA per ';'-separated sentence.
        Writes: columns [row, wh_qa] where value is 'Q=... || A=...; Q=... || A=...; ...'

      NEG-QA mode (neg_qa=True): from POS QA column -> generate NEGATIVE QA pairs (one per positive pair).
        Writes: columns [row, wh_qa_neg] where value is 'Q=... || A=...; ...'
    """
    t0 = time.time()
    base = os.path.basename(path)
    pf = pq.ParquetFile(path)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sidecar_name = os.path.splitext(base)[0] + ".parquet"
    out_path = os.path.join(out_dir or os.path.dirname(path), sidecar_name)

    writer = None
    emitted = 0
    row_offset = 0
    acc_rows: List[int] = []
    acc_caps: List[str] = []
    acc_wh:   List[str] = []
    acc_whqa: List[str] = []  # used for POS/NEG QA

    if neg_qa:
        whqa_col = detect_whqa_col(pf, whqa_col_name or "wh_qa")
        mode = "WH-NEG-QA"
        print(f"[START:{mode}] {base} -> {sidecar_name}  (input col: pos_qa='{whqa_col}')")
    elif wh_qa:
        wh_col = detect_wh_col(pf, wh_col_name or "wh")
        mode = "WH-QA"
        print(f"[START:{mode}] {base} -> {sidecar_name}  (input col: {wh_col})")
    else:
        cap_col = detect_caption_col(pf, caption_col_name)
        mode = "WH-SENTENCES"
        print(f"[START:{mode}] {base} -> {sidecar_name}  (input col: {cap_col})")

    def flush():
        nonlocal writer, emitted, acc_rows, acc_caps, acc_wh, acc_whqa
        if not acc_rows:
            return

        if neg_qa:
            neg_cells = batched_generate_neg_whqa_from_whqa_column(
                llm=llm,
                tokenizer=tokenizer,
                whqa_cells=acc_whqa,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "wh_qa_neg": pa.array(neg_cells, type=pa.string()),
            })
        elif wh_qa:
            qa_cells = batched_generate_whqa_from_wh_column(
                llm=llm,
                tokenizer=tokenizer,
                wh_cells=acc_wh,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "wh_qa": pa.array(qa_cells, type=pa.string()),
            })
        else:
            phrases = batched_generate_phrases(
                llm=llm,
                tokenizer=tokenizer,
                captions=acc_caps,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "wh_sentences": pa.array(phrases, type=pa.string()),
            })

        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        emitted += len(acc_rows)
        acc_rows.clear()
        acc_caps.clear()
        acc_wh.clear()
        acc_whqa.clear()

    # Read and queue
    if neg_qa:
        # Generate negative WH QA pairs based on negative WH QA pairs
        for batch in pf.iter_batches(columns=[whqa_col], batch_size=read_batch_rows):
            whqas = batch.column(0).to_pylist()
            n = len(whqas)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_whqa.extend([q if q is not None else "" for q in whqas])
            row_offset += n
            if len(acc_rows) >= 10_000:
                flush()
    elif wh_qa:
        # Generate positive WH QA pairs based on positive sentences
        for batch in pf.iter_batches(columns=[wh_col], batch_size=read_batch_rows):
            wh_cells = batch.column(0).to_pylist()
            n = len(wh_cells)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_wh.extend([w if w is not None else "" for w in wh_cells])
            row_offset += n
            if len(acc_rows) >= 10_000:
                flush()
    else:
        # Summarite sentences
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

# ================================= CLI =====================================

def main():
    args = parse_generate_wh_args()
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
    if args.wh_qa and args.temperature <= 0.0:
        print("[Warn] --wh-qa with temperature=0.0 may reduce randomness in which component gets selected. "
              "Consider --temperature 0.3–0.7.")
    if args.neg_qa and args.temperature <= 0.0:
        print("[Warn] --neg-qa with temperature=0.0 may reduce variability in replacements. "
              "Consider --temperature 0.3–0.7.")

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
            wh_qa=args.wh_qa,
            wh_col_name=args.wh_column,
            neg_qa=args.neg_qa,
            whqa_col_name=args.whqa_column,
        )

if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
