# vLLM vs ATOM Rollout Performance Reproduction Runbook

This runbook reproduces the current long-decode comparison from a clean machine. It does not depend on files or models from the current working directory.

It will:

- pull `rocm/atom-dev:vllm-v0.22.0-nightly_20260712`
- clone ATOM from GitHub and checkout the PR head
- create the benchmark script locally from the code embedded below
- use a caller-provided model path, or automatically download `Qwen/Qwen3-8B-Base`
- run vLLM and ATOM with no-eager; run ATOM with `level=3`
- force 4096 generated tokens per request by using long prompts and `ignore_eos`

## Tested Configuration

- Image: `rocm/atom-dev:vllm-v0.22.0-nightly_20260712`
- ATOM repo: `https://github.com/ROCm/ATOM.git`
- ATOM PR: `1411`
- Model default: `Qwen/Qwen3-8B-Base`
- Benchmark shape: `8` prompts, `4096` max output tokens, `8192` max model length
- vLLM: `enforce_eager=false`
- ATOM: `level=3`, `enforce_eager=false`, `per_block_fp8`

Reference result on the original MI325X test machine:

```text
atom/bf16:          32768 tokens, 29.068s, 1127.28 tok/s
atom/per_block_fp8: 32768 tokens, 36.643s,  894.26 tok/s
vllm/bf16:          32768 tokens, 27.572s, 1188.47 tok/s
vllm/per_block_fp8: 32768 tokens, 56.626s,  578.67 tok/s
```

Exact numbers may vary with GPU SKU, ROCm driver, clocks, container cache state, and filesystem.

## Prerequisites

The new machine should have:

- AMD GPU with ROCm runtime available to Docker
- Docker
- `git`
- enough disk for the image, model, and benchmark outputs
- optional `HF_TOKEN` if the model download requires authentication in your environment

## Quick Start

```bash
mkdir -p "$HOME/atom_vllm_rollout_repro"
cd "$HOME/atom_vllm_rollout_repro"
```

Create `direct_compare.py`:

```bash
cat > direct_compare.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

BACKENDS = ("vllm", "atom")
PRECISIONS = ("bf16", "per_block_fp8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def long_output_questions() -> list[str]:
    return [
        (
            "Generate a very long deterministic plain-text sequence for a decode throughput benchmark. "
            "Write one numbered line per item, starting at 000001. Each line must contain the item number, "
            "the phrase 'rollout benchmark keeps decoding', and a short checksum equal to the item number "
            "modulo 97. Do not summarize, do not stop early, and do not write a conclusion."
        ),
        (
            "Continue the following synthetic log stream for as long as possible. Use monotonically increasing "
            "step numbers, one line per step, with fields step, rank, batch, token, latency_us, and status=ok. "
            "Do not explain the format and do not finish with a final answer."
        ),
        (
            "Write a long CSV-like dataset with columns row_id, shard_id, request_id, token_count, and note. "
            "Start row_id at 1 and keep producing rows. The note field should be a short repeated benchmark "
            "sentence. Do not add markdown fences or a summary."
        ),
        (
            "Produce a long list of JSONL records. Each record must be a single line object with keys id, "
            "worker, prompt_tokens, generated_tokens, and message. Keep incrementing id and keep the message "
            "short. Do not close with commentary."
        ),
        (
            "Generate a long Chinese benchmark text. 每一行以递增编号开头，后面写一句关于 rollout 解码吞吐、"
            "显存预算、KV cache、批处理调度的短句。不要总结，不要提前结束，持续输出编号行。"
        ),
        (
            "Write a long appendix for a performance report. Use sections named Observation 1, Observation 2, "
            "and so on. Each observation should be two concise sentences about GPU inference benchmarking. "
            "Keep going without a conclusion."
        ),
        (
            "Create a long deterministic table in plain text. Each row should be: index | queue_depth | "
            "prefill_tokens | decode_tokens | throughput_note. Increment index by one and vary the numeric "
            "fields using simple arithmetic. No final summary."
        ),
        (
            "Write an extended sequence of Python comments only. Each comment line should describe one small "
            "step in a hypothetical model-serving benchmark pipeline. Number every line and keep going."
        ),
    ]


def prepare_prompts(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for idx, question in enumerate(long_output_questions()[: args.num_prompts]):
        messages = [{"role": "user", "content": question}]
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = question
        prompts.append({"prompt_index": idx, "text": text})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "prompts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote {out} prompts={len(prompts)}")


def load_prompts(path: Path, repeat: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        base = [json.loads(line) for line in f if line.strip()]
    rows = []
    for rep in range(repeat):
        for row in base:
            item = dict(row)
            item["repeat_index"] = rep
            item["request_index"] = len(rows)
            rows.append(item)
    return rows


def sampling_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
    }


def run_vllm(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    os.environ["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
    from vllm import LLM, SamplingParams

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "kv_cache_dtype": "auto",
    }
    if args.precision == "per_block_fp8":
        kwargs["quantization"] = "fp8_per_block"

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    init_s = time.perf_counter() - t0
    sp = SamplingParams(**sampling_args(args), logprobs=(0 if args.calculate_logprobs else None))
    texts = [p["text"] for p in prompts]
    if args.warmup_tokens:
        warm = SamplingParams(max_tokens=args.warmup_tokens, temperature=0.0, seed=args.seed)
        llm.generate([texts[0]], warm)
    t1 = time.perf_counter()
    outs = llm.generate(texts, sp)
    gen_s = time.perf_counter() - t1

    records = []
    for req, out in zip(prompts, outs):
        comp = out.outputs[0]
        records.append({
            **req,
            "prompt_text": req["text"],
            "text": comp.text,
            "token_ids": [int(x) for x in comp.token_ids],
        })
    try:
        llm.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    return kwargs, init_s, add_elapsed(records, gen_s)


def run_atom(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    os.environ.pop("ATOM_DISABLE_VLLM_PLUGIN", None)
    from atom import SamplingParams
    from atom.model_engine.llm_engine import LLMEngine

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.atom_enforce_eager if args.atom_enforce_eager is not None else args.enforce_eager,
        "trust_remote_code": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "bf16",
        "cudagraph_capture_sizes": str([1, 2, 4, 8, 16]),
        "level": args.atom_level,
    }
    if args.precision == "per_block_fp8":
        kwargs["online_quant_config"] = {"global_quant_config": "per_block_fp8"}

    t0 = time.perf_counter()
    llm = LLMEngine(**kwargs)
    init_s = time.perf_counter() - t0
    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ignore_eos=args.ignore_eos,
        logprobs=(0 if args.calculate_logprobs else None),
    )
    texts = [p["text"] for p in prompts]
    if args.warmup_tokens:
        llm.generate([texts[0]], SamplingParams(max_tokens=args.warmup_tokens, temperature=0.0))
    t1 = time.perf_counter()
    outs = llm.generate(texts, sp)
    gen_s = time.perf_counter() - t1

    records = []
    for req, out in zip(prompts, outs):
        records.append({
            **req,
            "prompt_text": req["text"],
            "text": out.get("text", "") if isinstance(out, dict) else str(out),
            "token_ids": [int(x) for x in out.get("token_ids", [])] if isinstance(out, dict) else [],
        })
    llm.close()
    del llm
    gc.collect()
    return kwargs, init_s, add_elapsed(records, gen_s)


def add_elapsed(records: list[dict[str, Any]], elapsed: float) -> list[dict[str, Any]]:
    for row in records:
        row["_generate_s"] = elapsed
    return records


def summarize(backend: str, precision: str, init_s: float, records: list[dict[str, Any]]) -> dict[str, Any]:
    gen_s = records[0].pop("_generate_s") if records else 0.0
    out_tokens = sum(len(r["token_ids"]) for r in records)
    return {
        "backend": backend,
        "precision": precision,
        "requests": len(records),
        "init_s": init_s,
        "generate_s": gen_s,
        "output_tokens": out_tokens,
        "output_tokens_per_s": out_tokens / gen_s if gen_s else None,
        "requests_per_s": len(records) / gen_s if gen_s else None,
        "response_len_mean": mean([len(r["token_ids"]) for r in records]) if records else 0,
    }


def single(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.repeat)
    if args.backend == "vllm":
        engine_kwargs, init_s, records = run_vllm(args, prompts)
    else:
        engine_kwargs, init_s, records = run_atom(args, prompts)
    payload = {
        "status": "ok",
        "backend": args.backend,
        "precision": args.precision,
        "args": vars(args),
        "engine_kwargs": engine_kwargs,
        "summary": summarize(args.backend, args.precision, init_s, records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(f"[single] {args.backend}/{args.precision} -> {args.output}")
    print(json.dumps(payload["summary"], indent=2))


def compare_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    from difflib import SequenceMatcher

    rows = []
    for ra, rb in zip(a["records"], b["records"]):
        rows.append({
            "request_index": ra["request_index"],
            "exact": ra["text"] == rb["text"],
            "similarity": SequenceMatcher(None, ra["text"], rb["text"]).ratio(),
            "len_a": len(ra["token_ids"]),
            "len_b": len(rb["token_ids"]),
        })
    return {
        "a": f"{a['backend']}/{a['precision']}",
        "b": f"{b['backend']}/{b['precision']}",
        "requests": len(rows),
        "exact_rate": mean([float(r["exact"]) for r in rows]) if rows else 0,
        "similarity_mean": mean([r["similarity"] for r in rows]) if rows else 0,
    }


def summarize_dir(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw"
    payloads = []
    failures = []
    for path in sorted(raw.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "ok":
            payloads.append(data)
        else:
            failures.append(data)
    by_key = {f"{p['backend']}/{p['precision']}": p for p in payloads}
    comparisons = []
    for precision in PRECISIONS:
        va, aa = f"vllm/{precision}", f"atom/{precision}"
        if va in by_key and aa in by_key:
            comparisons.append(compare_pair(by_key[va], by_key[aa]))
    for backend in BACKENDS:
        ba, fa = f"{backend}/bf16", f"{backend}/per_block_fp8"
        if ba in by_key and fa in by_key:
            comparisons.append(compare_pair(by_key[ba], by_key[fa]))
    summary = {
        "generation": [p["summary"] for p in payloads],
        "comparisons": comparisons,
        "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Direct vLLM vs ATOM Inference", "", "## Generation", ""]
    for row in summary["generation"]:
        lines.append(
            f"- `{row['backend']}/{row['precision']}`: requests={row['requests']}, "
            f"tokens={row['output_tokens']}, generate_s={row['generate_s']:.3f}, "
            f"tok/s={row['output_tokens_per_s']:.2f}, init_s={row['init_s']:.3f}"
        )
    lines.extend(["", "## Similarity", ""])
    for row in comparisons:
        lines.append(
            f"- `{row['a']}` vs `{row['b']}`: exact={row['exact_rate']:.3f}, "
            f"similarity={row['similarity_mean']:.3f}"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for f in failures:
            lines.append(f"- `{f.get('backend')}/{f.get('precision')}`: {f.get('error')}")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_all(args: argparse.Namespace) -> None:
    prepare_prompts(args)
    raw = args.output_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "single",
        "--model",
        args.model,
        "--prompts",
        str(args.output_dir / "prompts.jsonl"),
        "--seed",
        str(args.seed),
        "--repeat",
        str(args.repeat),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--atom-level",
        str(args.atom_level),
        "--no-atom-enforce-eager",
        "--no-enforce-eager",
        "--ignore-eos",
    ]
    if not args.calculate_logprobs:
        base.append("--no-calculate-logprobs")
    for precision in PRECISIONS:
        for backend in BACKENDS:
            out = raw / f"{backend}_{precision}.json"
            cmd = base + ["--backend", backend, "--precision", precision, "--output", str(out)]
            env = os.environ.copy()
            compare_dir = str(Path(__file__).resolve().parent)
            if backend == "atom":
                atom_dir = env.get("ATOM_DIR", str(Path(compare_dir) / "ATOM"))
                pieces = [compare_dir, atom_dir]
                if env.get("PYTHONPATH"):
                    pieces.append(env["PYTHONPATH"])
                env["PYTHONPATH"] = os.pathsep.join(pieces)
                env.pop("ATOM_DISABLE_VLLM_PLUGIN", None)
            else:
                env["PYTHONPATH"] = compare_dir
                env["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
                env.pop("TORCHDYNAMO_DISABLE", None)
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                out.write_text(json.dumps({
                    "status": "failed",
                    "backend": backend,
                    "precision": precision,
                    "error": f"exit_code={exc.returncode}",
                }, indent=2), encoding="utf-8")
    summarize_dir(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--model", required=True)
        q.add_argument("--seed", type=int, default=10086)
        q.add_argument("--repeat", type=int, default=1)
        q.add_argument("--max-tokens", type=int, default=4096)
        q.add_argument("--max-model-len", type=int, default=8192)
        q.add_argument("--max-num-batched-tokens", type=int, default=8192)
        q.add_argument("--max-num-seqs", type=int, default=16)
        q.add_argument("--gpu-memory-utilization", type=float, default=0.3)
        q.add_argument("--temperature", type=float, default=0.0)
        q.add_argument("--top-p", type=float, default=1.0)
        q.add_argument("--top-k", type=int, default=-1)
        q.add_argument("--warmup-tokens", type=int, default=32)
        q.add_argument("--tensor-parallel-size", type=int, default=1)
        q.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
        q.add_argument("--atom-level", type=int, default=3)
        q.add_argument("--atom-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
        q.add_argument("--ignore-eos", action="store_true", default=True)
        q.add_argument("--calculate-logprobs", action=argparse.BooleanOptionalAction, default=True)

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--num-prompts", type=int, default=8)
    r.add_argument("--output-dir", type=Path, default=Path("results_long_4096"))

    s = sub.add_parser("single")
    common(s)
    s.add_argument("--prompts", type=Path, required=True)
    s.add_argument("--backend", choices=BACKENDS, required=True)
    s.add_argument("--precision", choices=PRECISIONS, required=True)
    s.add_argument("--output", type=Path, required=True)

    sm = sub.add_parser("summarize")
    sm.add_argument("--output-dir", type=Path, default=Path("results_long_4096"))
    return p


def main() -> None:
    args = parser().parse_args()
    try:
        if args.phase == "run":
            run_all(args)
        elif args.phase == "single":
            single(args)
        elif args.phase == "summarize":
            summarize_dir(args)
    except Exception as exc:
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "status": "failed",
                "backend": getattr(args, "backend", None),
                "precision": getattr(args, "precision", None),
                "error": repr(exc),
            }, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
PY
chmod +x direct_compare.py
```

Create `run_repro.sh`:

```bash
cat > run_repro.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$PWD}"
DATA_ROOT="${DATA_ROOT:-$WORKDIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/results_long_ignore_eos_8p4096}"
IMAGE="${IMAGE:-rocm/atom-dev:vllm-v0.22.0-nightly_20260712}"
ATOM_REPO="${ATOM_REPO:-https://github.com/ROCm/ATOM.git}"
ATOM_PR="${ATOM_PR:-1411}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-Base}"
MODEL="${MODEL_PATH:-$MODEL_ID}"

mkdir -p "$DATA_ROOT" "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd -P)"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd -P)"

if [ ! -d "$WORKDIR/ATOM/.git" ]; then
  git clone "$ATOM_REPO" "$WORKDIR/ATOM"
fi
git -C "$WORKDIR/ATOM" fetch origin "pull/$ATOM_PR/head:pr-$ATOM_PR"
git -C "$WORKDIR/ATOM" checkout "pr-$ATOM_PR"

sudo docker pull "$IMAGE"

MODEL_MOUNT_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
  MODEL_PATH="$(realpath "$MODEL_PATH")"
  test -e "$MODEL_PATH"
  MODEL="$MODEL_PATH"
  MODEL_MOUNT_ARGS=(-v "$MODEL_PATH:$MODEL_PATH:ro")
fi

sudo docker run --rm --name atom-vllm-rollout-repro \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$WORKDIR":"$WORKDIR" \
  -v "$DATA_ROOT":"$DATA_ROOT" \
  "${MODEL_MOUNT_ARGS[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME="$DATA_ROOT/hf_home" \
  -e HF_HUB_CACHE="$DATA_ROOT/cache/hub" \
  -e HF_DATASETS_CACHE="$DATA_ROOT/cache/datasets" \
  "$IMAGE" bash -lc "
set -euo pipefail
cd '$WORKDIR'
python3 -m pip install -e '$WORKDIR/ATOM' --no-deps

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_TARGET_DEVICE=rocm
export HF_HOME='$DATA_ROOT/hf_home' HF_HUB_CACHE='$DATA_ROOT/cache/hub' HF_DATASETS_CACHE='$DATA_ROOT/cache/datasets'
export ATOM_DIR='$WORKDIR/ATOM'
export PYTHONPATH='$WORKDIR'

python3 -u direct_compare.py run \
  --model '$MODEL' \
  --num-prompts 8 \
  --max-tokens 4096 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.30 \
  --warmup-tokens 32 \
  --seed 10086 \
  --no-enforce-eager \
  --atom-level 3 \
  --no-atom-enforce-eager \
  --ignore-eos \
  --output-dir '$RESULTS_DIR'
"

echo "REPORT=$RESULTS_DIR/REPORT.md"
echo "SUMMARY=$RESULTS_DIR/summary.json"
SH
chmod +x run_repro.sh
```

Run with automatic model download:

```bash
./run_repro.sh
```

Run with a local model path:

```bash
MODEL_PATH=/path/to/Qwen3-8B-Base ./run_repro.sh
```

After completion:

```bash
cat results_long_ignore_eos_8p4096/REPORT.md
```

## Expected Output Checks

The run is the intended 4096-token decode test only if all four rows show:

```text
requests=8
tokens=32768
response_len_mean=4096
```

If `tokens` is much smaller than `32768`, check that `--ignore-eos` is present and that the generated prompts are the long-output prompts.

## Common Issues

### vLLM no-eager fails with `aot_compile is not supported`

For vLLM no-eager, `TORCHDYNAMO_DISABLE` must not be set in the vLLM subprocess. The embedded `direct_compare.py` removes it for vLLM when `--no-enforce-eager` is used. Keep that behavior.

### Model download fails

Set `HF_TOKEN` if needed:

```bash
HF_TOKEN=... ./run_repro.sh
```

Or provide a local model:

```bash
MODEL_PATH=/models/Qwen3-8B-Base ./run_repro.sh
```

### ATOM per-block fp8 fails

Confirm the checked-out ATOM PR branch:

```bash
git -C ATOM branch --show-current
```

It should be:

```text
pr-1411
```

The image's preinstalled ATOM did not support this per-block fp8 path in the original test; the editable install from this PR is required.
