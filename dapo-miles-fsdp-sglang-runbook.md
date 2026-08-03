# Miles 原生 DAPO Runbook（FSDP2 + SGLang · BF16，跨机快速复现）

> 在一台**全新 8 卡 AMD GPU 机器**上，用 **Miles**（slime 的 fork，SGLang rollout + Megatron/FSDP 训练）
> 复现与 `dapo-lumenrl-native-vllm-fsdp-runbook.md` **同模型、同数据、同超参**的 DAPO 数学 RL 训练，
> 用于和 LumenRL 那条线做逐项对比。
>
> - **训练后端 FSDP2**（`--train-backend fsdp`）：纯数据并行，无 TP/PP/EP/CP，**直接读 HF checkpoint**，
>   不需要 Megatron 的 `torch_dist` 权重转换。
> - **rollout 引擎 SGLang**（Miles 唯一支持的引擎，没有 vLLM 选项），8 个 TP=1 引擎与 8 个训练 actor
>   **colocate**；训练时通过 TorchMemorySaver 把引擎显存释放掉，训练完再唤醒。
> - 模型 Qwen3-8B-Base，数据 DAPO-Math-17k / AIME-2024，与 LumenRL 完全一致。
>
> **一句话复现**：选对镜像 → clone miles → 起容器（**注意 `--ulimit nofile`**）→ 装 miles →
> 下模型/数据 → 转数据布局 → 写 1 个启动脚本 → smoke → 长跑。
>
> **已验证**：8×MI355X（gfx950）+ ROCm 7.2，smoke 3 步 exit 0，长跑跑到 step 9 + AIME eval 正常，
> 指标与 LumenRL BF16 基线重合（见 §10、§11.3）。

---

## 1. 架构与对应关系

| 维度 | LumenRL 原生（对照组） | 本 runbook（Miles） |
|---|---|---|
| 入口 | `python -m lumenrl.trainer.main` | `ray job submit -- python3 train.py`（Ray 作业） |
| 编排 | 单 Ray driver：8 FSDP2 actor + 8 colocated vLLM AsyncLLM | 同构：8 FSDP2 actor + 8 colocated SGLang engine（`--colocate`） |
| 训练后端 | Lumen FSDP2（fp32 master + bf16 compute） | PyTorch FSDP2（`fsdp_utils`，fp32 master，`--disable-fp32-master` 可关） |
| 推理后端 | vLLM `AsyncLLM`（TP=1） | **SGLang**（TP=1，`--rollout-num-gpus-per-engine 1`），前面挂 `sglang_router` |
| 权重同步 | ZMQ CUDA-IPC 分桶 | CUDA-IPC 分桶（`--update-weight-buffer-size`），colocate 下配 sleep/wake |
| 显存策略 | 训练与 vLLM **同时常驻**（`gpu_memory_utilization=0.30`，不 sleep） | **offload/sleep**：rollout 时训练权重下 CPU，训练时释放引擎显存 |
| 权重加载 | HF 直读 | HF 直读（**FSDP 不需要 `torch_dist` 转换**，这是它相对 Megatron 的主要卖点） |
| 动态采样 | `algorithm.dapo.filter_groups`（acc） | `--over-sampling-batch-size` + `--dynamic-sampling-filter-path ...check_reward_nonzero_std` |
| TIS 修正 | `rollout_correction.rollout_is=token` / `threshold=2.0` | `--use-tis --tis-clip 2.0` |
| 策略损失 | clip-higher + dual-clip + **token-mean** | clip-higher + dual-clip + **seq-mean**（token-mean 在 FSDP 上不可用，见 §13.2） |
| overlong 奖励 | `algorithm.dapo.overlong_buffer` | **没有这个功能**（见 §13.1） |
| 优势估计 | grpo（按 uid 组归一化） | grpo（`--advantage-estimator grpo`） |
| reward | `lumenrl.rewards.math_reward.dapo_math_reward` | `--rm-type dapo --reward-key score`（**同一份 verl `math_dapo` 移植**） |

> 两边的 reward 打分函数是同一份代码的两次移植：都从 verl `math_dapo.py` 来，都抽最后一个
> `Answer:` / `\boxed{}` 归一化后字符串比较，返回 `{"score": ±1.0, "acc": bool, "pred": str}`。
> 这一点是两条线可比的前提，已实测确认（`compute_score("blah Answer: 34", "34")` →
> `{'score': 1.0, 'acc': True, 'pred': '34'}`）。

---

## 2. 路径变量（所有后续命令都用这三个变量，换机只改这里）

```bash
export MILES_ROOT=/path/to/miles_rl      # 代码根（内含 miles 仓库 + 启动脚本）
export DATA_ROOT=/path/to/data           # 模型 / 数据 / 日志 根
export CONTAINER=miles-dev
mkdir -p "$MILES_ROOT" "$DATA_ROOT/logs"
```

---

## 3. 选镜像（**这一节踩坑最多，先读**）

### 3.1 不要用 `rlsys/miles:MI350-355-latest`

文档 `docs/platforms/amd.md` 让你 `docker pull rlsys/miles:MI350-355-latest`。
**这个 tag 是陈旧的**：实测它指向 2026-05 构建的镜像（sglang 0.5.11 / transformers 5.3 /
内置 miles 停在 5 月的 commit），而仓库 HEAD 的 `requirements.txt` 已经要求
`transformers==5.12.1`。把当前仓库装上去会一路缺依赖，且 sglang API 已经漂移。

### 3.2 用 `rocm/sgl-dev` 的 miles CI 镜像（每日构建）

CI 每天从仓库 HEAD 构建并推到 `rocm/sgl-dev`，tag 形如 `miles-<profile>-<YYYYMMDD>`：

| GPU | profile | tag 样例 |
|---|---|---|
| MI350 / MI355（gfx950）+ ROCm 7.2 | `rocm720-mi35x` | `miles-rocm720-mi35x-20260802` |
| MI350 / MI355（gfx950）+ ROCm 7.0 | `rocm700-mi35x` | `miles-rocm700-mi35x-20260802` |
| MI300 / MI325（gfx942）+ ROCm 7.0 | `rocm700-mi30x` | `miles-rocm700-mi30x-20260802` |

列出可用 tag（挑一个日期 **≥ 你的仓库 HEAD 日期**的）：

```bash
curl -s "https://hub.docker.com/v2/repositories/rocm/sgl-dev/tags?page_size=100&name=miles" \
  | python3 -c 'import sys,json;[print(t["name"], t["last_updated"]) for t in json.load(sys.stdin)["results"]]'
```

```bash
export IMAGE=rocm/sgl-dev:miles-rocm720-mi35x-20260802
docker pull "$IMAGE"      # 约 30GB 压缩 / 115GB 解压
```

> 注意 `rocm/sgl-dev:miles-rocm720-mi35x`（不带日期）**不存在**，只有带日期的 tag。

镜像内实测版本（`miles-rocm720-mi35x-20260802`）：

```
python 3.10.12 · torch 2.9.1+rocm7.2.0 · sglang 0.5.17.dev32
transformers 5.12.1 · ray 2.56.1 · flash_attn 2.8.3
内置 miles: /root/miles（= 仓库 HEAD 的上一个 commit）
```

镜像已经内置了这些 ROCm 环境变量，**不需要自己设**：
`HIP_FORCE_DEV_KERNARG=1`、`HSA_NO_SCRATCH_RECLAIM=1`、`SGLANG_USE_AITER=1`、
`SGLANG_MOE_PADDING=1`、`SGLANG_ROCM_FUSED_DECODE_MLA=1`、`NCCL_MIN_NCHANNELS=112`、
`TORCHINDUCTOR_MAX_AUTOTUNE=1`、`RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`。

---

## 4. 拉取代码

```bash
cd "$MILES_ROOT"
git clone https://github.com/radixark/miles.git
```

镜像里已经有一份 `/root/miles`，但它落后仓库 HEAD。**用你自己 clone 的那份**，装成 editable
覆盖掉内置的（§6）。如果你不改代码，直接用内置 `/root/miles` 也行，把后面所有
`$MILES_ROOT/miles` 换成 `/root/miles` 即可。

---

## 5. 启动 Docker（**`--ulimit nofile` 是必需的**）

```bash
docker rm -f "$CONTAINER" 2>/dev/null
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --device=/dev/dri --device=/dev/kfd --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --shm-size 128G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "$MILES_ROOT":"$MILES_ROOT" -v "$DATA_ROOT":"$DATA_ROOT" \
  -e MILES_ROOT="$MILES_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  -e HF_HOME="$DATA_ROOT/hf_home" \
  "$IMAGE" -lc 'sleep infinity'
```

> ⚠️ **`--ulimit nofile=1048576:1048576` 不能省**。容器默认 `ulimit -n` 软限制是 1024，
> 8 个 SGLang 引擎 + 8 个 FSDP actor 会在**第一次 rollout 结束时**把它耗尽，raylet 直接崩：
> ```
> (raylet) logging.cc:118: Unhandled exception: ... what(): open: Too many open files [system:24]
> (FSDPTrainRayActor) Check failed: raylet_ipc_client_->ActorCreationTaskDone() ... Broken pipe
> ```
> 这个报错看起来像 Ray 的 bug（它自己也建议你去提 issue），实际就是 fd 不够。
> 建完容器先 `docker exec "$CONTAINER" bash -lc 'ulimit -n'` 确认不是 1024。

---

## 6. 装 miles + 验证

```bash
docker exec "$CONTAINER" bash -lc '
set -e
git config --global --add safe.directory "$MILES_ROOT/miles"
cd "$MILES_ROOT/miles" && pip install -e . --no-deps
'
```

`--no-deps` 是刻意的：镜像里的 torch / sglang / transformers 是配套编译好的，
让 pip 按 `requirements.txt` 解依赖会把它们换掉。用对了日期的镜像就不缺任何东西。

**验证导入 + reward 打分**：

```bash
docker exec "$CONTAINER" bash -lc '
cd "$MILES_ROOT/miles"
python3 - <<PY
import torch, sglang, transformers, ray, miles
print("torch", torch.__version__, "sglang", sglang.__version__,
      "transformers", transformers.__version__, "gpus", torch.cuda.device_count())
print("miles at", miles.__file__)
from miles.backends.experimental.fsdp_utils.actor import FSDPTrainRayActor
from miles.rollout.rm_hub.math_dapo_utils import compute_score
print("dapo rm:", compute_score("blah Answer: 34", "34"))
print("import OK")
PY
'
```

> 期望：`gpus 8`、`miles at $MILES_ROOT/miles/miles/__init__.py`（**不是** `/root/miles`）、
> `dapo rm: {'score': 1.0, 'acc': True, 'pred': '34'}`、`import OK`。
> 中间会刷一堆无害告警（`deep_ep is not installed`、`aiter` RoPE 精度提示、
> `modelopt` 与 transformers 版本不匹配、Qwen3ASR docstring `[ERROR]`），都可以忽略。

**确认 Qwen3 在 FSDP 后端的已验证模型列表里**（`fsdp_utils/adaptations/class_patches.py`）：

```python
VERIFIED_MODEL_TYPES = frozenset({"glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"})
```

---

## 7. 下载模型 / 数据 + 转成 Miles 布局

### 7.1 模型 + 原始数据

非国内网络：

```bash
docker exec "$CONTAINER" bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D=os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY
'
```

中国内网走 ModelScope，ID 同名，用法照抄 LumenRL runbook §6.1 的 modelscope 片段。

> **必须用 Base 版**。`--rm-type dapo` 抽的是 `Answer:` 行，instruct/thinking 版会先写
> `<think>` 且在预算内不闭合，reward 恒为 −1，动态采样会一直丢组。

### 7.2 转成 Miles 的列布局（**必做**）

Miles 读每行是**扁平索引**的：`data[--input-key]` 和 `data[--label-key]`，不支持嵌套路径。
而 verl 的 parquet 把答案放在 `reward_model.ground_truth` 这个 struct 里，所以要把它提到顶层
`label` 列。`prompt` 列（chat 消息数组）原样保留，这样喂给两个框架的 prompt 是**逐字节相同**的。

Miles 原生支持 `.parquet` 和 `.jsonl`（`miles/utils/data.py:28`），parquet 更小更快，这里用 parquet。

```bash
cat > "$MILES_ROOT/convert_dapo_to_miles.py" <<'PYEOF'
"""Lift verl's nested reward_model.ground_truth to a top-level `label` column.

Miles indexes each row flat (data[input_key] / data[label_key]), so the answer has to
be a top-level column. The `prompt` column is passed through untouched, which keeps the
prompts byte-identical to the ones LumenRL trains on.
"""

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq


def first_parquet(*dir_globs: str) -> str:
    for g in dir_globs:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no parquet under {dir_globs}")


def convert(src: str, dst: str) -> int:
    table = pq.read_table(src)
    label = table.column("reward_model").combine_chunks().field("ground_truth")
    out = pa.table(
        {
            "prompt": table.column("prompt"),
            "label": label,
            "data_source": table.column("data_source"),
        }
    )
    pq.write_table(out, dst)
    return out.num_rows


def main() -> None:
    data = os.environ["DATA_ROOT"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{data}/data_miles/qwen3-8b")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [
        (first_parquet(f"{data}/data_cached/**/dapo-math-17k*.parquet",
                       f"{data}/raw/DAPO-Math-17k/**/*.parquet"),
         os.path.join(args.out_dir, "dapo-math-17k.parquet")),
        (first_parquet(f"{data}/data_cached/**/aime-2024*.parquet",
                       f"{data}/raw/AIME-2024/**/*.parquet"),
         os.path.join(args.out_dir, "aime-2024.parquet")),
    ]
    for src, dst in jobs:
        print(f"[{src}] -> {dst}: {convert(src, dst)} rows")


if __name__ == "__main__":
    main()
PYEOF
docker exec "$CONTAINER" bash -lc 'cd "$MILES_ROOT" && python3 convert_dapo_to_miles.py'
```

产出（即启动脚本默认的 `PROMPT_DATA` / `EVAL_DATA`）：

```
$DATA_ROOT/data_miles/qwen3-8b/dapo-math-17k.parquet    # train, 1,791,700 行
$DATA_ROOT/data_miles/qwen3-8b/aime-2024.parquet        # eval,  960 行
```

脚本会**优先复用 LumenRL 的 `data_cached/` 过滤产物**（如果这台机器上跑过 LumenRL runbook），
否则直接吃 `raw/`。两者等价：DAPO-Math-17k 里没有超过 1024 token 的 prompt，
LumenRL runbook §13.3 实测按 2048 重新过滤一行都不删（1,791,700 → 1,791,700）。
所以 **Miles 侧不需要单独做 prompt 长度过滤**。

### 7.3 去重 AIME eval 集（greedy 评测时必做，省 97% eval 时间）

verl 的 AIME-2024 parquet 是 **960 行但只有 30 个不同 prompt** —— 每道题复制了 32 份，
目的是让 `temperature>0` 的评测一遍拿到 pass@k 样本。本 runbook 的验证是 **greedy**
（`--eval-temperature 0`），32 份输出逐字节相同，**96.9% 的 eval 算力是白烧的**。

实测：全量 960 条一次 eval 要 **12 分 12 秒**，去重到 30 条后 eval 时间可忽略。
按 `--eval-interval 10` 算，1000 步能省下约 16 小时。

```bash
cat > "$MILES_ROOT/dedup_aime_eval.py" <<'PYEOF'
"""Deduplicate the AIME eval parquet down to its distinct prompts.

The verl AIME-2024 parquet carries each of the 30 problems 32 times so that
temperature>0 evaluation gets pass@k samples out of one pass. Our eval is greedy
(--eval-temperature 0), so all 32 copies produce byte-identical output and 31/32 of
the eval budget is wasted. Only run this when the eval is greedy.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    data = os.environ["DATA_ROOT"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{data}/data_miles/qwen3-8b/aime-2024.parquet")
    ap.add_argument("--dst", default=f"{data}/data_miles/qwen3-8b/aime-2024.dedup.parquet")
    args = ap.parse_args()

    table = pq.read_table(args.src)
    rows, seen = [], set()
    for row in table.to_pylist():
        key = row["prompt"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), args.dst)
    print(f"[{args.src}] -> {args.dst}: {table.num_rows} -> {len(rows)} rows")


if __name__ == "__main__":
    main()
PYEOF
docker exec "$CONTAINER" bash -lc 'cd "$MILES_ROOT" && python3 dedup_aime_eval.py'
```

> 期望 `960 -> 30 rows`（30 个唯一 prompt / 29 个唯一答案 —— 有两道题答案恰好相同，正常）。
>
> ⚠️ **不能用 `aime-2024.parquet@[0:30]` 切片代替**：实测前 30 行里只有 21 个不同 prompt，
> 复制份是交错排列的，必须按 prompt 内容去重。
>
> ⚠️ **只在 greedy 评测下去重**。如果你改成
> `--eval-temperature 0.6 --n-samples-per-eval-prompt 16` 这类采样评测，
> 32 份重复等价于多次采样、是有意义的，那就用回 `aime-2024.parquet`。

### 7.4 关于 chat template 与 prompt 长度过滤

- `--apply-chat-template` 会把消息数组渲染成字符串。Qwen3-8B-Base **自带 chat template**
  （ChatML：`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`），和 LumenRL 侧
  `tokenizer.apply_chat_template` 用的是同一个，无需 `--chat-template-path`。
- **不要传 `--rollout-max-prompt-len`**。它会触发 `filter_long_prompt()` 把全部 179 万条
  prompt tokenize 一遍，启动要等很久，而这份数据本来就没有超长 prompt。

---

## 8. 参数映射表（LumenRL config → Miles flag）

这是两条线可比的核心。左列是 LumenRL 的 yaml 字段，右列是等价的 miles 命令行。

| LumenRL | smoke 值 | longrun 值 | Miles flag |
|---|---|---|---|
| `num_training_steps` | 3 | 1000 | `--num-rollout`（每 rollout 恰好 1 个优化步，见下） |
| `train_global_batch_size` | 128 | 512 | `--global-batch-size` |
| `dapo.num_generations` | 16 | 16 | `--n-samples-per-prompt` |
| （= batch ÷ generations，prompt 数） | 8 | 32 | `--rollout-batch-size` |
| `gen_batch_size` | 24 | 96 | `--over-sampling-batch-size` |
| `filter_groups.enable / metric=acc` | true | true | `--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` |
| `max_response_length` | 512 | 20480 | `--rollout-max-response-len` |
| `max_total_sequence_length` | 2048 | 21504 | `--sglang-context-length` |
| `max_token_len_per_gpu` | 2048 | 21504 | `--use-dynamic-batch-size --max-tokens-per-gpu` |
| `train_micro_batch_size: 1` | — | — | 动态批下由 `--max-tokens-per-gpu` 决定，`--micro-batch-size` 被忽略 |
| `learning_rate` | 1e-6 | 1e-6 | `--lr` |
| `lr_warmup_steps` | 10 | 10 | `--lr-warmup-iters`（**见下方注意**） |
| `lr_decay_style: constant` | — | — | `--lr-decay-style constant` |
| `weight_decay` | 0.1 | 0.1 | `--weight-decay` |
| `max_grad_norm` | 1.0 | 1.0 | `--clip-grad` |
| `clip_ratio_low / high / c` | 0.2 / 0.28 / 10 | 同 | `--eps-clip / --eps-clip-high / --eps-clip-c` |
| `adv_estimator: grpo` | — | — | `--advantage-estimator grpo` |
| `kl_coeff: 0` / `use_kl_in_reward: false` | — | — | `--kl-coef 0.0`（且不传 `--use-kl-loss`） |
| `entropy` 系数 0 | — | — | `--entropy-coef 0.0` |
| `rollout_is: token` / `threshold: 2.0` | — | — | `--use-tis --tis-clip 2.0` |
| `temperature / top_p / top_k` | 1.0 / 1.0 / −1 | 同 | `--rollout-temperature / --rollout-top-p / --rollout-top-k` |
| `max_num_batched_tokens` | 4096 | 32768 | `--sglang-chunked-prefill-size` |
| `max_num_seqs` | 64 | 64 | `--sglang-max-running-requests` |
| `gpu_memory_utilization` | 0.30 | 0.30 | `--sglang-mem-fraction-static 0.70`（**语义不同，见 §13.3**） |
| `val_steps` | —（关） | 10 | `--eval-interval` |
| `seed` | 10086 | 10086 | `--seed` |
| `checkpointing.save_steps` | 1000 | 50 | `--save` + `--save-interval`（都不传 = 不落盘） |
| `overlong_buffer.len / penalty_factor` | 256 / 1.0 | 512 / 1.0 | **无对应实现，见 §13.1** |
| `loss_agg_mode: token-mean` | — | — | **FSDP 上不可用，见 §13.2** |

**批量换算的不变式**（`docs/user-guide/cli-reference.md:35`）：

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

上面两套配置都让 `num_steps_per_rollout = 1`，所以 `--num-rollout` 直接等于 LumenRL 的
`num_training_steps`，两边的 step 编号可以一一对应。

**`--lr-warmup-iters` 的两个坑**：

1. `FSDPLRScheduler` 有 `assert lr_warmup_steps < lr_decay_steps`，而 `lr_decay_iters` 不传时
   会等于 `train_iters`。smoke 只有 3 步 → `10 < 3` 断言失败。**必须显式传
   `--lr-decay-iters`**（脚本里传了一个跑不到的大数，`constant` 衰减下没有副作用）。
2. 两边 warmup 有 1 步的错位：LumenRL 第 1 步 lr = 2e-7，Miles 第 0 步 lr = 1e-7。
   都是 10 步线性 warmup，只是起点索引差 1，第 10 步后都到 1e-6。3 步 smoke 里影响可忽略。

---

## 9. 启动脚本

Miles 仓库里**没有**现成的 AMD + FSDP 启动脚本（`scripts/amd/*` 全是 Megatron，
`scripts/*_fsdp.py` 全是 NVIDIA 假设）。所以下面两个脚本由本 runbook 生成，
**本文档是它们的唯一来源**。

### 9.0 为什么不用 `scripts/run_*_fsdp.py`

仓库的 python 启动器走 `U.execute_train()`，它会：`pkill` 残留进程 → `ray start --head` →
拼 `runtime_env` → `ray job submit -- python3 <绝对路径>/train.py <flags>`。
我们的脚本做的是同一件事，只是把 AMD 需要的差异固化进去。三个 NVIDIA 假设必须改掉：

- `--sglang-attention-backend fa3` 是 Hopper FA3，ROCm 上要去掉（镜像已设 `SGLANG_USE_AITER=1`，
  让 SGLang 自己选 ROCm 默认后端）。
- `--attn-implementation flash_attention_3` → **`flash_attention_2`**（镜像装的是 ROCm flash-attn 2.8.3）。
- `check_has_nvlink()` 靠 `nvidia-smi` 判断，AMD 上必然返回 0，`NCCL_NVLS_ENABLE=0`，无害。

另外 `scripts/run_mcore_fsdp.py` 的 FSDP 分支无条件传了 `--save-retain-interval`，
那是 Megatron parser 的参数，FSDP parser 用 `parse_args()`（不是 `parse_known_args()`），
会直接 `unrecognized arguments` 退出。**别照抄那一行。**

### 9.1 公共环境（两个脚本共用的头部）

```bash
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export MASTER_ADDR=127.0.0.1
export no_proxy=127.0.0.1,localhost
export HAS_NVLINK=0
ulimit -n 1048576
```

三条硬约束，写在脚本里，别自己改：

1. **绝对路径的 `train.py`**。`ray job submit` 的作业工作目录是 `/root`，不是你 `cd` 到的目录。
   写 `-- python3 train.py` 会 `can't open file '/root/train.py'`。仓库自己的
   `command_utils.py:123` 也是先把它转成绝对路径的。
2. **不要在 `--runtime-env-json` 里设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。**
   colocate 下 SGLang 用 TorchMemorySaver 释放引擎显存，开着 expandable segments 会直接拒绝启动：
   `RuntimeError: TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet`。
   参考脚本是通过 `--train-env-vars` 只给训练 actor 设的；而 ROCm/HIP 本来就不实现
   expandable segments，这里干脆全不设。
3. **`ulimit -n`**（见 §5）。

### 9.2 Smoke 脚本

```bash
cat > "$MILES_ROOT/run_miles_bf16_smoke.sh" <<'EOF'
#!/bin/bash
# Miles FSDP2 + SGLang BF16 DAPO smoke, matched to LumenRL
# examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml.
#
# 8 prompts x 16 generations = 128 sequences = global_batch_size
#   => 1 optimizer step per rollout, so --num-rollout == LumenRL num_training_steps.
set -euo pipefail

MILES_DIR=${MILES_DIR:-$MILES_ROOT/miles}
DATA_ROOT=${DATA_ROOT:?}
MODEL_PATH=${MODEL_PATH:-$DATA_ROOT/models/Qwen3-8B-Base}
DC=${DC:-$DATA_ROOT/data_miles/qwen3-8b}
STEPS=${STEPS:-3}
EXTRA_ARGS=${EXTRA_ARGS:-}

export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export HF_HOME=$DATA_ROOT/hf_home
export HAS_NVLINK=0
ulimit -n 1048576 || true

pkill -9 -f sglang || true
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f raylet || true
pkill -9 -f gcs_server || true
sleep 5

ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MILES_DIR\",
  \"no_proxy\": \"127.0.0.1,localhost\", \"MASTER_ADDR\": \"$MASTER_ADDR\",
  \"NCCL_NVLS_ENABLE\": \"0\"}}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 "$MILES_DIR/train.py" \
  --train-backend fsdp \
  --hf-checkpoint "$MODEL_PATH" \
  --gradient-checkpointing \
  --attn-implementation flash_attention_2 \
  --update-weight-buffer-size 536870912 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate \
  --prompt-data "$DC/dapo-math-17k.parquet" \
  --input-key prompt --label-key label --apply-chat-template \
  --rollout-shuffle --balance-data \
  --rm-type dapo --reward-key score \
  --num-rollout "$STEPS" \
  --rollout-batch-size 8 --n-samples-per-prompt 16 --global-batch-size 128 \
  --over-sampling-batch-size 24 \
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
  --rollout-max-response-len 512 \
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 2048 \
  --advantage-estimator grpo \
  --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 \
  --kl-coef 0.0 --entropy-coef 0.0 \
  --use-tis --tis-clip 2.0 --observe-training-entropy \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --lr-warmup-iters 10 --lr-decay-iters 1000 \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0 \
  --seed 10086 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.70 \
  --sglang-context-length 2048 \
  --sglang-chunked-prefill-size 4096 \
  --sglang-max-running-requests 64 \
  --sglang-decode-log-interval 1000 \
  --skip-eval-before-train \
  $EXTRA_ARGS
EOF
chmod +x "$MILES_ROOT/run_miles_bf16_smoke.sh"
```

### 9.3 Longrun 脚本

相对 smoke 只改规模 + 加 eval + 加 wandb：

```bash
sed -e 's/^STEPS=${STEPS:-3}$/STEPS=${STEPS:-1000}/' \
    -e 's/--rollout-batch-size 8 --n-samples-per-prompt 16 --global-batch-size 128/--rollout-batch-size 32 --n-samples-per-prompt 16 --global-batch-size 512/' \
    -e 's/--over-sampling-batch-size 24/--over-sampling-batch-size 96/' \
    -e 's/--rollout-max-response-len 512/--rollout-max-response-len 20480/' \
    -e 's/--max-tokens-per-gpu 2048/--max-tokens-per-gpu 21504/' \
    -e 's/--sglang-context-length 2048/--sglang-context-length 21504/' \
    -e 's/--sglang-chunked-prefill-size 4096/--sglang-chunked-prefill-size 32768/' \
    -e 's|--lr-decay-iters 1000|--lr-decay-iters "$STEPS"|' \
    "$MILES_ROOT/run_miles_bf16_smoke.sh" > "$MILES_ROOT/run_miles_bf16_longrun.sh"

# 追加 eval + wandb（不传 --save / --save-interval 就是不落盘）
python3 - "$MILES_ROOT/run_miles_bf16_longrun.sh" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()

# W&B key 从文件读，不要用 --wandb-key：那会进 ray 的作业命令行和 dashboard。
s = s.replace(
    'ulimit -n 1048576 || true\n',
    'ulimit -n 1048576 || true\n\n'
    'export WANDB_DIR=$DATA_ROOT/wandb\n'
    'WANDB_KEY_FILE=${WANDB_KEY_FILE:-$MILES_ROOT/wandb.key}\n'
    'if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_KEY_FILE" ]; then\n'
    '  export WANDB_API_KEY="$(cut -d= -f2- "$WANDB_KEY_FILE" | tr -d \'[:space:]\')"\n'
    'fi\n'
    ': "${WANDB_API_KEY:?WANDB_API_KEY unset and $WANDB_KEY_FILE not readable}"\n',
)
s = s.replace(
    '\\"NCCL_NVLS_ENABLE\\": \\"0\\"}}"',
    '\\"NCCL_NVLS_ENABLE\\": \\"0\\", \\"WANDB_API_KEY\\": \\"$WANDB_API_KEY\\",\n'
    '  \\"WANDB_DIR\\": \\"$WANDB_DIR\\"}}"',
)
s = s.replace(
    '  --skip-eval-before-train \\\n',
    '  --skip-eval-before-train \\\n'
    '  --eval-interval 10 \\\n'
    '  --eval-prompt-data aime "$DC/aime-2024.dedup.parquet" \\\n'
    '  --n-samples-per-eval-prompt 1 \\\n'
    '  --eval-max-response-len 20480 \\\n'
    '  --eval-temperature 0 --eval-top-p 1 \\\n'
    '  --use-wandb \\\n'
    '  --wandb-project "${WANDB_PROJECT:-MilesRl}" \\\n'
    '  --wandb-group "${WANDB_GROUP:-qwen3-8b-base-dapo-bf16-fsdp-sglang}" \\\n',
)
open(p, "w").write(s)
print("wrote", p)
PYEOF
chmod +x "$MILES_ROOT/run_miles_bf16_longrun.sh"
```

W&B key 放 `$MILES_ROOT/wandb.key`，格式 `WANDB_API_KEY=xxxx`（脚本自动读）。

> ⚠️ key 是通过 `--runtime-env-json` 注入的，所以它会出现在 `ray job list` 输出和 Ray dashboard 里。
> 单用户机器无所谓；共享机器上改成 `docker exec -e WANDB_API_KEY=... ` 从容器环境继承。

---

## 10. Smoke（3 步，前台等结果）

```bash
docker exec -d "$CONTAINER" bash -lc \
  "bash $MILES_ROOT/run_miles_bf16_smoke.sh > $DATA_ROOT/logs/miles-bf16-smoke.log 2>&1"

# 看指标（三类日志行）
L=$DATA_ROOT/logs/miles-bf16-smoke.log
grep -aoE "log_utils.py:460 - step [0-9]+: \{[^}]*\}" $L      # 训练侧
grep -aoE "log_utils.py:66 - rollout [0-9]+: \{[^}]*\}" $L    # rollout 侧
grep -aoE "metrics.py:79 - perf [0-9]+: \{[^}]*\}" $L         # 引擎吞吐 / 动态采样
```

**耗时**：首次约 12 分钟（8 个引擎启动 + aiter kernel JIT + 179 万行数据加载），
JIT 产物落盘后重跑约 6 分钟。

### 10.1 Smoke 期望证据（8×MI355X 实测）

| step | `grad_norm` | `train_rollout_kl` | `tis_abs` | `tis` | `entropy_loss` | `raw_reward` | resp_len | `step_time` |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.110 | 0.001200 | 0.01961 | 1.00069 | 0.809 | −0.734 | 396 | 60.3 s |
| 1 | 1.803 | 0.001196 | 0.01887 | 1.00054 | 0.676 | −0.844 | 413 | 36.7 s |
| 2 | 1.569 | 0.001279 | 0.02015 | 1.00062 | 0.737 | −0.797 | 421 | 44.4 s |

同机 LumenRL BF16 smoke 基线（对照）：

| step | `grad_norm` | `rollout_corr/kl` | `mismatch/abs_diff` | `is_weight_mean` | `entropy` | `reward/accuracy` | resp_len | `step_s` |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.754 | 0.001175 | 0.01778 | 0.99997 | 0.614 | 0.0859 | 402 | 38.4 s |
| 2 | 0.856 | 0.000996 | 0.01808 | 1.00025 | 0.691 | 0.1172 | 406 | 16.8 s |
| 3 | 0.989 | 0.001402 | 0.01902 | 0.99971 | 0.737 | 0.0938 | 386 | 22.5 s |

判据：

- **`train_rollout_kl` 在 1.1–1.3e-3**（对应 LumenRL 的 `rollout_corr/kl`）。这是训练/推理一致性
  的核心指标，BF16 下就该是这个量级；跑到 1e-2 说明权重同步或 kernel 对不齐。
- **`tis_abs` 在 1.8–2.0e-2**，`tis` ≈ 1.000，`tis_clipfrac` ≈ 0。
- `ppo_kl` ≈ 0（单步优化，old == train），实测 ~1e-10。
- `raw_reward` 是 ±1 分的均值，换算准确率 `acc = (raw_reward + 1) / 2`，
  上表对应 0.133 / 0.078 / 0.102，和 LumenRL 的 0.086 / 0.117 / 0.094 同分布。
- `rollout/dynamic_filter/drop_zero_std_-1.0` 应该 > 0 且能凑够 batch（过采样 24 取 8，
  通常 2 轮）。如果一直凑不够，看 §12。
- `grad_norm` 1.5–2.1，**如果看到 4000+，是踩了 §13.2 的坑**。

> **数据顺序说明**：两个框架的 shuffle 实现不同，3 步里各自取到的 prompt 子集不一样，
> 所以 reward / entropy 的逐步数值只能按分布比，不能逐点对齐。要逐点可比得固定同一批 prompt。

---

## 11. 长跑

```bash
docker exec -d "$CONTAINER" bash -lc \
  "bash $MILES_ROOT/run_miles_bf16_longrun.sh > $DATA_ROOT/logs/miles-bf16-longrun.log 2>&1"
```

覆盖用法：

```bash
STEPS=30 WANDB_GROUP=my-run bash $MILES_ROOT/run_miles_bf16_longrun.sh   # 先跑 30 步验证
```

> 建议先 `STEPS=30` 确认显存/指标健康，再上 1000 步。

### 11.1 落盘（默认关）

脚本不传 `--save` / `--save-interval`，所以**不写 checkpoint，崩了从头再来**。要开：

```bash
EXTRA_ARGS="--save $DATA_ROOT/ckpts/miles-qwen3-8b --save-interval 50"
```

FSDP 存的是 DCP 格式（`iter_XXXXXXX/{model,optimizer,lr_scheduler}/*.distcp`），
不是 HF；要转回 HF 用 `tools/convert_fsdp_to_hf.py`。

### 11.2 监控

```bash
L=$DATA_ROOT/logs/miles-bf16-longrun.log
grep -aoE "log_utils.py:460 - step [0-9]+: \{[^}]*\}" $L | tail -5
grep -aoE "train_metric_utils.py:50 - perf [0-9]+: \{[^}]*\}" $L | tail -3
grep -aiE "Traceback|RuntimeError|OutOfMemory" $L | tail
```

W&B 面板在 `MilesRl` 项目。关键曲线：`train/grad_norm`、`train/train_rollout_kl`、
`train/entropy_loss`、`rollout/raw_reward`、`rollout/response_lengths`、`perf/step_time`。

### 11.3 长跑健康判据（前 10 步实测）

| step | `grad_norm` | `train_rollout_kl` | `tis_abs` | `entropy_loss` | `lr` | `step_time` |
|---|---|---|---|---|---|---|
| 0 | 0.994 | 0.001201 | 0.01917 | 0.734 | 1e-7 | 98.7 s |
| 2 | 1.355 | 0.001251 | 0.01968 | 0.757 | 3e-7 | 57.3 s |
| 3 | 1.048 | 0.001223 | 0.01937 | 0.740 | 4e-7 | 58.3 s |
| 4 | 1.468 | 0.001114 | 0.01757 | 0.673 | 5e-7 | 68.7 s |
| 9 | 0.660 | 0.001185 | 0.01873 | 0.738 | 1e-6 | — |
| 12 | 0.741 | 0.001167 | 0.01816 | 0.690 | 1e-6 | — |

（step 9 之后插入了一次 12 分钟的 AIME eval，见 §11.4。）

> 上表是 `--sglang-mem-fraction-static 0.70` 下的数字。改成 0.36 会让 `train_rollout_kl`
> 变成 0.015–0.027，见 §13.3 —— 那不是"更对齐"，是坏了。

- **`grad_norm` 0.66–1.47**，和 LumenRL 长跑的 ~0.85 基本重合。smoke 里偏高（1.5–2.1）
  是小 batch 下 seq-mean 与 token-mean 的差异被放大，512 序列下这个差异基本消失。
- `train_rollout_kl` 全程 1.1–1.3e-3，**不随步数单调爬升**。爬上去说明权重同步漏了。
  换过配置之后第一步就该落在这个区间；如果 step 0 就是 1e-2 量级，先回头看 §13.3。
- 第 0 步 `truncated_ratio = 0.0`、平均回复 641 token —— 20480 的预算前期完全用不满，
  所以早期 step 只要 ~1 分钟。**回复长度会随训练增长，step 时间会明显变慢**，
  LumenRL 同规模长跑的稳态是 4–5 分钟/步。
- 显存：rollout 期间每卡约 200GB（`--sglang-mem-fraction-static 0.70` × 288GB），
  训练期间引擎显存被 TorchMemorySaver 释放。空闲基线 298MB/卡。

### 11.4 AIME eval 的代价

长跑脚本默认用 §7.3 产出的 **`aime-2024.dedup.parquet`（30 条）**，不是原始的 960 条。
两者在 greedy 下评测结果等价，但耗时差两个数量级：

| eval 集 | 条数 | 唯一 prompt | 单次 eval 耗时 | 1000 步累计 |
|---|---|---|---|---|
| `aime-2024.parquet` | 960 | 30 | **12 分 12 秒** | 约 16 小时 |
| `aime-2024.dedup.parquet` | 30 | 30 | 可忽略 | 约 0.5 小时 |

全量那版最后 5% 就占了约 4 分钟 —— greedy 下有几条会一路生成到 20480 上限（base 模型的
重复退化），这条长尾在 30 条集里同样存在，只是不再乘 32。

如果你要改成采样评测（`--eval-temperature 0.6 --n-samples-per-eval-prompt 16`），
把 `--eval-prompt-data` 换回 `aime-2024.parquet`，理由见 §7.3。

### 11.5 停止

```bash
docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f sglang; pkill -9 -f raylet; pkill -9 -f gcs_server
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -3'   # 应回到 ~298MB/卡
```

---

## 12. 排障速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `python3: can't open file '/root/train.py'` | `ray job submit` 的作业 cwd 是 `/root`，不是你 `cd` 的目录 | 用绝对路径 `python3 "$MILES_DIR/train.py"` |
| `RuntimeError: TorchMemorySaver is disabled ... expandable_segments is not supported yet` | 全局设了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，与 colocate 的 TorchMemorySaver 冲突 | 从 `runtime-env-json` 里删掉；ROCm 本来也不支持 |
| `(raylet) Unhandled exception ... open: Too many open files` → actor `Broken pipe` | 容器 `ulimit -n` 默认 1024 | `docker run --ulimit nofile=1048576:1048576`，脚本里再 `ulimit -n 1048576` |
| `AssertionError: custom_tis_function_path must be set when get_mismatch_metrics is set` | `--get-mismatch-metrics` 只覆盖 MIS/RS 路径 | 去掉它；`--use-tis` 单独就会输出 `tis` / `tis_abs` / `tis_clipfrac` |
| `unrecognized arguments: --save-retain-interval` | 抄了 `run_mcore_fsdp.py` 的 Megatron 专属参数 | FSDP parser 是 `parse_args()`，不吃 Megatron flag，删掉 |
| `grad_norm` 4000+ | 开了 `--calculate-per-token-loss`（FSDP 上没归一化） | 去掉它，见 §13.2 |
| `train_rollout_kl` 从 step 0 就是 1e-2 量级（正常是 1.2e-3） | 把 `--sglang-mem-fraction-static` 调低了（例如为了对齐 LumenRL 的 0.30 而设成 0.36） | 回到 0.70，见 §13.3。不是 KV 压力，也不是权重同步问题 |
| `assert self.lr_warmup_steps < self.lr_decay_steps` | 步数少于 warmup 且没传 `--lr-decay-iters` | 显式传 `--lr-decay-iters`（constant 衰减下随便给个大数） |
| reward 恒为 −1 / 动态采样一直凑不够组 | 用了 instruct/thinking 版模型，或 `--rm-type deepscaler`（它要求响应里有 `</think>`，base 模型不会产出） | 换 Base 版；用 `--rm-type dapo --reward-key score` |
| `--rm-type dapo` 但 reward 是 dict 报错 | `compute_score` 返回 dict | 必须配 `--reward-key score` |
| 启动卡在 tokenize 很久 | 传了 `--rollout-max-prompt-len`，触发全量 179 万条 prompt 过滤 | 这份数据没有超长 prompt，别传这个参数 |
| `pip install -e .` 后一路 `ModuleNotFoundError` | 镜像太旧（`rlsys/miles:MI350-355-latest`） | 换 §3.2 的日期 tag |
| SGLang 报 fa3 不可用 | `--sglang-attention-backend fa3` 是 Hopper 专属 | ROCm 上删掉这个参数 |
| 训练前向挂在 flash-attn | `--attn-implementation flash_attention_3` | 改 `flash_attention_2` |

---

## 13. 与 LumenRL 对不齐的地方（**做对比前必读**）

### 13.1 Miles 没有 overlong buffer

LumenRL / verl 的 DAPO 有 soft overlong punishment（`overlong_buffer.len` / `penalty_factor`）：
回复长度进入末尾缓冲区就按比例扣分。**Miles 整棵树里没有这个实现**
（按 `overlong|over_long|soft_punish|length_penalty` 全文搜索，零命中），截断样本只是被标记
`truncated`。

在 smoke 配置下这个差异是实打实的：resp=512 时两边截断率都在 41%–53%，LumenRL 对这批样本
额外扣了分，Miles 没有。长跑配置下 resp=20480，前期截断率 0，差异暂时不显现。

要对齐得自己写 `--custom-rm-path`（`async def custom_rm(args, sample) -> float`，
优先级高于 `--rm-type`）。

### 13.2 `--calculate-per-token-loss` 在 FSDP 后端上是坏的

verl 的 `loss_agg_mode: token-mean` 对应 miles 的 `--calculate-per-token-loss`。
**但它在 FSDP 后端上不生效，而且会静默放大梯度。**

机制：`training_utils/loss.py:178` 的 `loss_function()` 在 `calculate_per_token_loss=True` 时
返回的 loss 是 `sum_of_token`（token 原始求和，**不做任何除法**），同时把 `num_tokens`
作为第二个返回值 `normalizer` 交出去 —— Megatron 靠 `finalize_model_grads` 用它做归一化。
而 FSDP 后端的 `fsdp_utils/actor.py:552-560`：

```python
loss, normalizer, log_dict = loss_function(..., apply_megatron_loss_scaling=False)
loss.backward()          # normalizer 被丢弃
```

实测后果：`grad_norm` 从 ~1 飙到 **4749–5808**，正好是每卡 token 数的量级
（128 序列 × ~410 token ÷ 8 卡 ≈ 6.5k）。`--clip-grad 1.0` 会把每一步都裁掉约 5000 倍，
训练方向还对，但幅度语义和 LumenRL 完全不同。

**所以这里用 miles 的默认聚合**（`sum_of_sample_mean`：每条序列内取均值、跨序列求和，
再除以 `global_batch_size`），也就是 verl 语义里的 **seq-mean**，不是 token-mean。
512 序列的长跑下两者数值上很接近（`grad_norm` 0.66–1.47 vs LumenRL ~0.85），
128 序列的 smoke 下差一截（1.5–2.1 vs 0.75–0.99）。

### 13.3 显存份额：**刻意不对齐**，改成 0.30 会让 mismatch 劣化 20 倍

先说清楚对标对象。LumenRL 有两条 BF16 长跑线，选错会得出相反的结论：

| | `MODE=bf16`（vLLM） | `MODE=atombf16`（ATOM） |
|---|---|---|
| CUDA graph | **关**（`enforce_eager: true`） | **开**（`run_dapo.sh` 覆盖 `enforce_eager=false` + `compilation_config.level=3`） |
| sleep / offload | **关**（`enable_sleep_mode: false`，引擎常驻） | **开**（覆盖 `enable_sleep_mode=true`、`sleep_level=2`） |
| 与 Miles 的可比性 | 差：两项都相反 | **好：两项都与 Miles 一致** |

Miles 的 colocate 是 CUDA graph 开 + sleep/offload 开，所以**跨框架对比要用 ATOM 那条线**
（COMPARE-RL 里的 `lumen-rl-atom-fsdp-bf16`）。拿 vLLM 线做参考会把 CUDA graph 和 sleep
误判成"未对齐项"。两条线的规模/超参完全相同，只有 rollout 引擎不同，所以 §8 的映射表两者通用。

**显存份额没有对齐，这是刻意的。** LumenRL ATOM 是 `atom_cfg.gpu_memory_utilization: 0.30`；
本 runbook 用 `--sglang-mem-fraction-static 0.70`。SGLang 在 `enable_memory_saver`（`--colocate`
会打开）下把请求值乘 0.85，所以 0.70 落到 0.595，而 0.36 会落到 0.306 ≈ LumenRL 的 0.30。

试过对齐，结论是**不能对齐**：

| | `0.70`（→0.595） | `0.36`（→0.306） |
|---|---|---|
| `train_rollout_kl` | 0.0012 | **0.015–0.027** |
| `tis_abs` | 0.019 | **0.052–0.067** |
| `tis_clipfrac` | ~0 | **0.0026–0.0046** |
| KV 池 | 1,132,707 token | 528,245 token |

13–20 倍的训练/推理 mismatch 劣化，**从 step 0 就是常数偏移，不是漂移**。因果是双向验证过的：
改下去劣化，改回来三项指标同时完全复原（0.00118 / 0.0190 / 0）。而且用一个同为 0.70、
只换了 eval 集的 run 做过对照，确认 mem_fraction 是唯一变量。

**不是 KV 压力**：零 preemption / retraction 事件，KV 池峰值只用到 13%。
CUDA graph 配置（`max_bs=512`、`backend='full'`）、`chunked_prefill_size`、`max_prefill_tokens`、
`max_running_requests`、prefix cache 命中率（0.48–0.78 vs 0.54–0.82）在两次之间**全部相同**。

**机制未解释。** 在解释清楚之前保持 0.70 —— 20 倍的 mismatch 会淹没这个对比想测的任何东西，
而且在当前回复长度下 KV 根本不是瓶颈，对齐这个参数换不来任何东西。

### 13.4 每步生成量差 1.6 倍，且 Miles 的批次偏向短回复

`n=16`、32 prompt、过采样 96 prompt —— 这些参数两边完全一致。但**每步实际跑完的生成量不一致**，
因为两个框架实现 DAPO 动态采样的方式根本不同：

| | LumenRL | Miles |
|---|---|---|
| 做法 | 把整个 96-prompt 的 gen_batch **跑到底**，再过滤，从幸存者里取 32 组 | 提交 96 组，**凑够 32 组通过过滤就 `abort()` 掉在飞请求** |
| 每步跑完 | **恒定 96 组 / 1536 条**（日志里 `N=1536` 出现 207 次，一次没变） | **41–70 组 / 656–1120 条**（约 62%） |
| 进入训练的 32 组 | 完整批次过滤后的子集 | **最先跑完的那 32 组** |

代码在 `miles/rollout/sglang_rollout.py`：`while len(data) < target_data_size` 一退出就
`aborted_samples = await abort(args, rollout_id)`。**现有参数改不了这个行为**，
唯一的扩展点是 `--rollout-function-path`（要自己写函数）。

**由此带来选择偏差。** 一个组要等它 16 个样本里最慢的那个结束才算完成，所以"先完成"≈"最长回复较短"。
过滤器大约刷掉一半，凑够 32 个幸存者需要约 64 组完成，也就是**最慢的约 1/3 从来没进过训练** ——
而那恰恰是最难的题、和模型往长度上限跑偏的那些。miles 又没有 overlong buffer（§13.1），
等于对长回复既不惩罚也不训练，直接不看。

**这个偏差目前没有量化。** `rollout/response_len/*` 只统计保留的 32 组
（`RolloutFnTrainOutput(samples=data)`），全量生成的分布没有落日志；要量化得挂
`--rollout-all-samples-process-path`。

**也不要急着把它当成 reward 差距的原因。** 前 20 步 LumenRL 的 reward 从 −0.67 爬到 −0.46，
miles 平在 −0.75 → −0.70，看着像，但至少有四个混淆项能独立解释：seq-mean vs token-mean（§13.2）、
miles 没有 overlong buffer（§13.1）、两边数据洗牌顺序不同、20 步的 reward 噪声本来就大。
反证也在：miles 的回复长度反而涨得更多（641→923 vs 716→789），真在往短里塌不该是这个走向。

**不写代码能缓解的办法是 `--partial-rollout`**：被 abort 的组连同已生成部分回收进 buffer，
下一轮接着生成，慢组最终还是会进训练，只是晚一步且带 off-policy 数据
（`--mask-offpolicy-in-partial-rollout` 可以把这些 token 从 loss 里 mask 掉）。默认关。

判据要看长跑后期：如果长度偏差真的起作用，miles 的 `response_len` 会先于 LumenRL 走平甚至回落，
AIME 准确率也会更早饱和。LumenRL 参考线在 step 196 是 `response_length_mean=5020`、
`val acc=0.228`。

### 13.5 其它小差异

| 项 | LumenRL | Miles | 影响 |
|---|---|---|---|
| rollout 引擎 | vLLM 0.23.0 / ATOM | SGLang 0.5.17 | kernel/采样实现不同，`train_rollout_kl` 已验证同量级 |
| prefix caching | ATOM 线 `enable_prefix_caching: false` | SGLang radix cache **开**，命中率 0.48–0.82 | 16 个 generation 共享同一 prompt 前缀，对 prefill 成本影响不小；要对齐用 `--sglang-disable-radix-cache` |
| 动态采样重试上限 | `max_num_gen_batches: 10`，超了抛异常 | 无上限，`while len(data) < target` 一直转 | 极端情况下 miles 会一直采样不报错 |
| 权重同步分桶 | ATOM 线 `update_weights_bucket_megabytes: 2048` | `--update-weight-buffer-size` 512MB | 只影响 `perf/update_weights_time`，不影响 rollout |
| 过滤判据 | acc 的组内 std | reward 的组内 std | reward = ±1 且由 acc 决定，等价（除非启用 overlong 惩罚） |
| lr warmup 起点 | 第 1 步 = 2e-7 | 第 0 步 = 1e-7 | 差 1 步索引，第 10 步后都到 1e-6 |
| 验证采样 | greedy，全量 960 | 同（本 runbook 设成 greedy n=1） | 见 §11.4 的去重建议 |
| Adam betas | `torch.optim.AdamW` 默认 (0.9, 0.999) | FSDP 默认 (0.9, **0.95**) | 本 runbook 显式传 `--adam-beta2 0.999` 对齐 |

---

## 14. 与 LumenRL 的实测对比（COMPARE-RL）

### 14.1 W&B 接线

跨框架对比走 `xysheng/COMPARE-RL` 项目，group `dapo-8b-precision-comparison`。各框架先记到自己的
项目，再由一个拷贝脚本搬进 COMPARE-RL，并把指标名映射到统一的 `compare/*` 命名空间。
Miles 侧的映射（记录在拷贝 run 的 config 里）：

| `compare/*` | Miles 源指标 | LumenRL 源指标 |
|---|---|---|
| `compare/kl` | `train/train_rollout_kl` | `core/kl` |
| `compare/entropy` | `train/entropy_loss` | `core/entropy` |
| `compare/reward_mean` | `rollout/raw_reward` | `core/reward_mean` |
| `compare/step_time_s` | `perf/step_time` | `core/step_time_s` |
| `compare/response_length_mean` | `rollout/response_len/mean` | `core/response_len_mean` |
| `compare/val_core_acc_mean_at_1` | `eval/aime`（`(v+1)/2`） | `train/val-core/acc/mean@1` |

> `rollout/raw_reward` 是 ±1 分的均值，准确率 = `(raw_reward + 1) / 2`。

**必须按 step index 对齐比较**，不能比绝对墙钟：回复长度随训练增长，rollout 成本跟着涨，
不同步数之间的耗时没有可比性。LumenRL 参考线 step 1 的平均回复是 716 token，step 196 是 5020。

### 14.2 实测（vs `lumen-rl-atom-fsdp-bf16`，同为 8×MI355X）

| step | resp_len（Lumen / Miles） | step_s（Lumen / Miles） | kl（Lumen / Miles） |
|---|---|---|---|
| 1 | 716 / 641 | 131.8 / 98.7 | 0.00090 / 0.00120 |
| 3 | 778 / 601 | 206.7 / 57.3 | 0.00092 / 0.00125 |
| 10 | 729 / 800 | 168.0 / 58.8 | 0.00104 / 0.00119 |
| 20 | 789 / 923 | 196.1 / 62.0 | 0.00069 / 0.00110 |

- **回复长度相当的前提下，Miles 每步快 2.5–3 倍。** 主因不是算子快慢，而是 §13.4 的生成量差异：
  LumenRL 每步跑完 1536 条，Miles 只跑完 656–1120 条。其次是 prefix caching（§13.5）。
- `kl` 两边同量级，Miles 略高约 20%。这是判断"对比是否成立"的关键指标 —— 它对得上，
  才说明两边的训练/推理一致性处在同一水平。
- `entropy` 20 步内 LumenRL 0.63→0.41、Miles 0.73→0.63；`reward` LumenRL −0.67→−0.46、
  Miles −0.75→−0.70。**这个差距目前无法归因**，混淆项见 §13.4 末尾。

---

## 15. 一句话流程

```bash
# 1. 选镜像（别用 rlsys/miles:*-latest）
export IMAGE=rocm/sgl-dev:miles-rocm720-mi35x-<YYYYMMDD>
export MILES_ROOT=/path/to/miles_rl DATA_ROOT=/path/to/data CONTAINER=miles-dev
docker pull "$IMAGE"

# 2. clone + 起容器（--ulimit nofile 必需）+ 装 miles           §4 §5 §6
# 3. 下模型/数据 + convert_dapo_to_miles.py                      §7
# 4. 生成两个启动脚本                                            §9
# 5. smoke：3 步，看 train_rollout_kl ~1.2e-3、grad_norm 1.5-2.1  §10
docker exec -d "$CONTAINER" bash -lc "bash $MILES_ROOT/run_miles_bf16_smoke.sh > $DATA_ROOT/logs/miles-bf16-smoke.log 2>&1"
# 6. 长跑：先 STEPS=30 验证，再 1000 步                          §11
docker exec -d "$CONTAINER" bash -lc "bash $MILES_ROOT/run_miles_bf16_longrun.sh > $DATA_ROOT/logs/miles-bf16-longrun.log 2>&1"
```
