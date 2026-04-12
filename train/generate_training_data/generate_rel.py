#!/usr/bin/env python3
import os, time, glob
from params import parse_generate_rel_args
from typing import Optional, List

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.relation_utils import (
    batched_generate_phrases,
    batched_generate_negative_relation_phrases, 
    batched_canonicalize_relation_phrases,      
    _strip_phrase_prefix,                 
)

from  utils.general_utils import get_dist_info, resolve_hf_snapshot_dir, detect_caption_col


# ------------------------------ local helpers -------------------------------

def detect_relation_phrase_col(pf: pq.ParquetFile, preferred: str) -> str:
    """
    Detect the column that holds the relation phrase.
    Tries: <preferred>, 'rel', 'rel_neg', 'relation_phrase', 'relations', 'rel_phrase'.
    """
    names = set(pf.schema_arrow.names)
    candidates = [preferred, "rel", "rel_neg", "relation_phrase", "relations", "rel_phrase"]
    for c in candidates:
        if c and c in names:
            return c
    raise ValueError(
        f"Relation phrase column not found. Tried {candidates}. "
        f"Available: {sorted(names)}"
    )

def _looks_neg_col(name: str) -> bool:
    return "neg" in (name or "").lower()

# ======================== per-file pipeline (rephrase) ======================

def process_file_sidecar(
    path: str,
    caption_col_name: str,
    tokenizer,
    llm: LLM,
    sampling: SamplingParams,
    out_dir: Optional[str],
    read_batch_rows: int = 2048,
    gen_batch_size: int = 64,
    # --- modes ---
    neg_relations: bool = False,
    relation_col_name: Optional[str] = None,
    canon_relations: bool = False,         
    canon_col_name: Optional[str] = None,   
):
    """
    Modes (mutually exclusive):
      1) Default (neg_relations=False, canon_relations=False):
           For each caption -> generate 1 relation phrase.
           Writes: row, relation_phrase
      2) Negative relations (neg_relations=True):
           For each existing relation phrase -> replace exactly one clause.
           Reads column: relation_col_name (e.g., 'rel'), Writes: row, rel_neg
      3) Canonicalize relations for Q (canon_relations=True):
           For each relation phrase (POS or NEG) -> minimally edit:
             - lowercase initial letter if not a proper noun
             - insert 'that' before first verb
           Reads column: canon_col_name (e.g., 'rel' or 'rel_neg')
           Writes:
             - if POS col -> row, rel_for_q
             - if NEG col -> row, rel_neg_for_q
    """
    t0 = time.time()
    base = os.path.basename(path)
    pf = pq.ParquetFile(path)

    # Determine required input column for the chosen mode
    if canon_relations:
        canon_col = detect_relation_phrase_col(pf, canon_col_name or "rel")
        is_neg_canon = _looks_neg_col(canon_col)
    elif neg_relations:
        rel_col = detect_relation_phrase_col(pf, relation_col_name or "rel")
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
    acc_caps: List[str] = []   # used in positive generation mode
    acc_rels: List[str] = []   # used in neg-relations and canon-relations

    def flush():
        nonlocal writer, emitted, acc_rows, acc_caps, acc_rels
        if not acc_rows:
            return

        if canon_relations:
            # LLM canonicalization (POS or NEG col)
            which = "neg" if is_neg_canon else "pos"
            canon = batched_canonicalize_relation_phrases(
                llm=llm,
                tokenizer=tokenizer,
                rel_phrases=acc_rels,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
                which=which,
            )
            canon_core = [_strip_phrase_prefix(s or "") for s in canon]
            out_col = "rel_neg_for_q" if is_neg_canon else "rel_for_q"
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                out_col: pa.array(canon_core, type=pa.string()),
            })

        elif neg_relations:
            # Generate negative relation phrases from existing relation phrases
            phrases = batched_generate_negative_relation_phrases(
                llm=llm,
                tokenizer=tokenizer,
                rel_phrases=acc_rels,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
                validate=False,  # keep behavior same as your original
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "rel_neg": pa.array(phrases, type=pa.string()),
            })

        else:
            # summarize positive relation phrases from captions
            phrases = batched_generate_phrases(
                llm=llm,
                tokenizer=tokenizer,
                captions=acc_caps,
                sampling_params=sampling,
                gen_batch_size=gen_batch_size,
            )
            tbl = pa.table({
                "row": pa.array(acc_rows, type=pa.int64()),
                "relation_phrase": pa.array(phrases, type=pa.string()),
            })

        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        emitted += len(acc_rows)
        acc_rows.clear()
        acc_caps.clear()
        acc_rels.clear()

    mode = "CANON-REL" if canon_relations else ("NEG-REL" if neg_relations else "REL")
    print(f"[START:{mode}] {base} -> {sidecar_name}")

    if canon_relations:
        for batch in pf.iter_batches(columns=[canon_col], batch_size=read_batch_rows):
            rels = batch.column(0).to_pylist()
            n = len(rels)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_rels.extend([r if r is not None else "" for r in rels])
            row_offset += n
            if len(acc_rows) >= 10_000:
                flush()

    elif neg_relations:
        for batch in pf.iter_batches(columns=[rel_col], batch_size=read_batch_rows):
            rels = batch.column(0).to_pylist()
            n = len(rels)
            acc_rows.extend(range(row_offset, row_offset + n))
            acc_rels.extend([r if r is not None else "" for r in rels])
            row_offset += n
            if len(acc_rows) >= 10_000:
                flush()

    else:
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


def main():
    args = parse_generate_rel_args()
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
    if args.neg_relations and args.temperature <= 0.0:
        print("[Warn] --neg-relations with temperature=0.0 may change the same position repeatedly. "
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
            neg_relations=args.neg_relations,
            relation_col_name=args.relation_column,
            canon_relations=args.canon_relations,    
            canon_col_name=args.canon_column,      
        )

if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
