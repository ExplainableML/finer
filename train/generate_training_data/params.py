# params.py
from __future__ import annotations

import argparse


def _add_common_generation_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_max_model_len: int,
    default_max_new_tokens: int,
) -> None:
    """Add arguments shared by all sidecar-generation scripts."""
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--glob", default="train-*.parquet")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--read-batch-rows", type=int, default=2048)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=default_max_model_len)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument(
        "--inference-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help="Generation cap for the phrase.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--skip-if-exists", action="store_true")


def build_generate_attr_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate attribute sidecars from parquet files."
    )
    _add_common_generation_args(
        parser,
        default_model="microsoft/phi-4",
        default_max_model_len=2048,
        default_max_new_tokens=128,
    )

    parser.add_argument(
        "--attr-diff",
        action="store_true",
        help="Enable ACC/REJ -> minimal 1-attr full-sentence generation.",
    )
    parser.add_argument(
        "--qa-column",
        default="attr_qa_pos",
        help="Column containing 'Q=... || A_acc=... || A_rej=...' "
        "(default: attr_qa_pos).",
    )

    parser.add_argument(
        "--neg-attributes",
        action="store_true",
        help="Enable negative-attribute generation mode.",
    )
    parser.add_argument(
        "--attr-column",
        default="attr",
        help="Attribute phrase column to read when --neg-attributes is set. "
        "Fallbacks: attribute_phrase, attributes, attr_phrase.",
    )

    parser.add_argument(
        "--neg-attr-edit-mode",
        choices=["random", "index"],
        default="random",
        help="When --neg-attributes: 'random' = model picks a random attribute; "
        "'index' = edit a specific attribute index per phrase.",
    )
    parser.add_argument(
        "--attr-index-mode",
        default="random",
        help="When --neg-attr-edit-mode index: how to pick the 1-based index for "
        "each phrase: 'random' | 'alternate' | '<int>' (e.g., '1').",
    )
    parser.add_argument(
        "--attr-index-seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic index selection when using "
        "'random' attr-index-mode.",
    )
    parser.add_argument(
        "--max-attr-units",
        type=int,
        default=5,
        help="Maximum attribute units per phrase (caps index).",
    )

    return parser


def parse_generate_attr_args() -> argparse.Namespace:
    return build_generate_attr_parser().parse_args()


def build_generate_obj_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate object sidecars from parquet files."
    )
    _add_common_generation_args(
        parser,
        default_model="microsoft/phi-4",
        default_max_model_len=2048,
        default_max_new_tokens=64,
    )

    parser.add_argument(
        "--neg-objects",
        action="store_true",
        help="Enable negative-object generation mode.",
    )
    parser.add_argument(
        "--objects-column",
        default="obj",
        help="Objects phrase column to read when --neg-objects is set. "
        "Fallbacks: objects_phrase, objects, object_phrase.",
    )

    parser.add_argument(
        "--categorize",
        action="store_true",
        help="Enable caption-based category sidecar.",
    )
    parser.add_argument(
        "--category-column",
        default="category",
        help="Output column name for categories.",
    )

    return parser


def parse_generate_obj_args() -> argparse.Namespace:
    return build_generate_obj_parser().parse_args()


def build_generate_rel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate relation sidecars from parquet files."
    )
    _add_common_generation_args(
        parser,
        default_model="Qwen/Qwen3-8B",
        default_max_model_len=2048,
        default_max_new_tokens=128,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--neg-relations",
        action="store_true",
        help="Enable negative-relation generation mode from an existing relation phrase column.",
    )
    mode.add_argument(
        "--canon-relations",
        action="store_true",
        help="Enable canonicalization: lowercase initial (unless proper noun) + "
        "insert 'that' before first verb.",
    )

    parser.add_argument(
        "--relation-column",
        default="rel",
        help="Relation phrase column to read when --neg-relations is set. "
        "Fallbacks: rel, rel_neg, relation_phrase, relations, rel_phrase.",
    )
    parser.add_argument(
        "--canon-column",
        default="rel",
        help="Relation phrase column to canonicalize when --canon-relations is set "
        "(e.g., 'rel' or 'rel_neg').",
    )

    return parser


def parse_generate_rel_args() -> argparse.Namespace:
    return build_generate_rel_parser().parse_args()


def build_generate_wh_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate WH sidecars from parquet files."
    )
    _add_common_generation_args(
        parser,
        default_model="microsoft/phi-4",
        default_max_model_len=4096,
        default_max_new_tokens=144,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--wh-qa",
        action="store_true",
        help="Enable QA generation mode from existing ';'-separated WH sentences.",
    )
    mode.add_argument(
        "--neg-qa",
        action="store_true",
        help="Enable NEGATIVE QA generation from POSITIVE QA ('wh_qa') column.",
    )

    parser.add_argument(
        "--wh-column",
        default="wh",
        help="Column name to read when --wh-qa is set. "
        "Fallbacks: wh, wh_sentences, wh_phrase, relations_wh.",
    )
    parser.add_argument(
        "--whqa-column",
        default="wh_pos_qa",
        help="Column name to read POSITIVE QA from when --neg-qa is set. "
        "Fallbacks: wh_qa, qa, whqa.",
    )

    return parser


def parse_generate_wh_args() -> argparse.Namespace:
    return build_generate_wh_parser().parse_args()