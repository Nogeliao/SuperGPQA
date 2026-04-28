#!/usr/bin/env python3
"""
Re-run inference for JSONL result lines whose response is an error dict, e.g.
`{"error": "Request timed out."}`.

Run from repository root with PYTHONPATH set, same as `infer.py`:

  export PYTHONPATH=$(pwd)
  python infer/retry_timeout_responses.py \\
    --config config/config_default.yaml \\
    --input results/Model_SuperGPQA-all_zero-shot.jsonl \\
    --model_name YourModel --mode zero-shot --split SuperGPQA-all

If `prompt` is present in the JSONL (the default config has save_prompt: true),
`--split` is not required. Otherwise pass `--split` and `--mode` to rebuild prompts
from the dataset (one full pass over the split is done to resolve UUIDs).

Modes that use a post-processor (e.g. zero-shot-bon) are retried; BoN may need
multiple rounds — if anything looks wrong, re-run the main `infer.py` for those rows.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import RetryError
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config_wrapper import initialize_config, get_config_wrapper
from data_loader import load_data
from models import load_model, infer
from post_process.custom_post_processor import PostProcessorRegistry
from utils.common import write_jsonl_lines

# Aligned with infer.infer_batch (uses config_wrapper from get_config_wrapper after init)


def _infer_batch(model_components, model_name, batch):
    config_wrapper = get_config_wrapper()
    results = []
    prompts = [sample[config_wrapper.prompt_key] for sample in batch]
    historys = [sample.get(config_wrapper.history_key, {}) for sample in batch]
    try:
        responses, meta_responses = infer(model_name)(prompts, historys, **model_components)
        for sample, response, meta_response in zip(batch, responses, meta_responses):
            sample[config_wrapper.response_key] = response
            if (
                config_wrapper.save_meta_response
                and config_wrapper.meta_response_key
                and meta_response
            ):
                sample[config_wrapper.meta_response_key] = meta_response
            results.append(sample)
    except RetryError as e:
        last_attempt = e.last_attempt
        exception = last_attempt.exception() if last_attempt else None
        if exception:
            print(f"Error: {str(exception)}")
        for sample in batch:
            sample[config_wrapper.response_key] = {"error": str(exception) if exception else "RetryError"}
            results.append(sample)
    except Exception as e:
        print(f"Error: {str(e)}")
        for sample in batch:
            sample[config_wrapper.response_key] = {"error": str(e)}
            results.append(sample)
    return results


def _error_matches_filter(sample, substring: str) -> bool:
    cw = get_config_wrapper()
    r = sample.get(cw.response_key)
    if not isinstance(r, dict):
        return False
    err = r.get(cw.error_key) if cw.error_key in r else r.get("error")
    if err is None:
        return False
    return substring in str(err)


def _load_uuid_to_prompt(split: str, mode: str, uuids: set[str]) -> dict[str, str]:
    config_wrapper = get_config_wrapper()
    out: dict[str, str] = {}
    for prompt, item in tqdm(
        load_data(split=split, mode=mode),
        desc="Scanning dataset for UUIDs / prompts",
    ):
        uid = str(config_wrapper.get_id(item)) if config_wrapper.get_id(item) is not None else None
        if uid and uid in uuids and uid not in out and prompt:
            out[uid] = prompt
    return out


def main():
    parser = argparse.ArgumentParser(description="Re-infer failed / timeout lines in a JSONL.")
    parser.add_argument("--config", type=str, default="config/config_default.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to an existing result .jsonl")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Where to write merged results. Default: overwrite --input (after .bak copy).",
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="zero-shot",
        help="Mode used to build prompts if missing (e.g. zero-shot, five-shot).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Dataset split for prompt lookup if lines lack `prompt` (e.g. SuperGPQA-all).",
    )
    parser.add_argument(
        "--error_substring",
        type=str,
        default="Request timed out",
        help="Only re-run lines whose error message contains this substring.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_accel", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="If set, do not write a .bak file when overwriting the input file.",
    )
    args = parser.parse_args()

    initialize_config(args.config)
    config_wrapper = get_config_wrapper()
    config_wrapper.mode = args.mode

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    to_retry: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        r = row.get(config_wrapper.response_key)
        if not isinstance(r, dict):
            continue
        if config_wrapper.error_key not in r and "error" not in r:
            continue
        if not _error_matches_filter(row, args.error_substring):
            continue
        uid = config_wrapper.get_id(row)
        to_retry.append((i, str(uid) if uid is not None else f"__line_{i}__"))

    if not to_retry:
        print("No lines matched the error filter. Nothing to do.")
        return

    print(f"Found {len(to_retry)} line(s) to re-infer (filter: {args.error_substring!r}).")

    uuids = {u for _, u in to_retry if not u.startswith("__line_")}
    prompt_map: dict[str, str] = {}
    need_rebuild = any(
        not rows[i].get(config_wrapper.prompt_key) for i, _ in to_retry
    )
    if need_rebuild:
        if not args.split:
            print(
                "Some rows are missing the `prompt` field; pass e.g. "
                "--split SuperGPQA-all --mode zero-shot to rebuild prompts from data.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Building prompts from dataset (this can take a while)…")
        prompt_map = _load_uuid_to_prompt(args.split, args.mode, uuids)
        for i, uid in to_retry:
            if rows[i].get(config_wrapper.prompt_key):
                continue
            if uid.startswith("__line_") or uid not in prompt_map:
                print(
                    f"Warning: no prompt for uuid={uid!r} (line {i+1}); skip.",
                    file=sys.stderr,
                )
            else:
                rows[i][config_wrapper.prompt_key] = prompt_map[uid]

    to_retry = [
        (i, u)
        for (i, u) in to_retry
        if rows[i].get(config_wrapper.prompt_key)
    ]
    if not to_retry:
        print("No rows left with a prompt. Exiting.", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or input_path
    if out_path == input_path and not args.no_backup:
        bak = input_path + ".bak"
        shutil.copy2(input_path, bak)
        print(f"Backed up input to {bak}")

    processor = PostProcessorRegistry.get_processor(args.mode)
    if processor:
        print(
            "Note: a post-processor is registered for this mode; this script only re-runs raw inference. "
            "For BoN / multi-round modes, re-run the main `infer.py` if results look incomplete.",
            file=sys.stderr,
        )

    model_components = load_model(args.model_name, args.use_accel)

    batches: list[list[dict]] = []
    current: list[dict] = []
    for i, _ in to_retry:
        current.append(rows[i])
        if len(current) >= args.batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(_infer_batch, model_components, args.model_name, b) for b in batches
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Re-inference batches"):
            fut.result()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as outf:
        for row in rows:
            # write_jsonl_lines may pop prompt; use a copy so we do not mutate `rows`
            write_jsonl_lines(outf, copy.deepcopy(row))

    print(f"Wrote {len(rows)} line(s) to {out_path}")


if __name__ == "__main__":
    main()
