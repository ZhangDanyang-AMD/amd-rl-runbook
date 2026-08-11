# LumenRL 原生 DAPO Runbook（BF16 + FP8，跨机快速复现）

> 在一台**全新 8 卡 AMD GPU 机器**上,**只用 LumenRL**(不依赖 verl)复现 verl `recipe/dapo` 的
> DAPO 数学 RL 训练,架构对齐 verl 正式长跑。**BF16 与 FP8 两条路线共用一套流程**,只在启动时用
> `MODE=bf16|fp8` / `TRAIN_FP8=0|1` 切换。
>
> - **单 Ray-driver 进程**:8 个 FSDP2 训练 actor + 8 个**同卡 colocated vLLM `AsyncLLM`** 在线 rollout
>   replica(TP=1);训练→rollout 权重经 **ZMQ CUDA-IPC** 同步。
> - 训练后端 **Lumen FSDP2**(fp32 master 权重 + bf16 compute);可选 **FP8 E2E**(blockwise2d 线性)。
> - rollout 可选 **vLLM 在线 `fp8_per_block`**(= verl 的 per_block_fp8)。
> - 规模/超参 1:1 对齐 verl BF16 formal long run(1000 步、batch 32×16=512、resp 20480 …)。
>
> 模型 Qwen3-8B-Base / 8×AMD(gfx950/MI350X 或 MI300X)。镜像 `vllm/vllm-openai-rocm:v0.23.0`
> (含 vllm v1 + torch-ROCm)。
>
> **一句话复现**:设路径变量 → clone 3 仓库 → 起容器装依赖 → 打 vLLM RMSNorm patch → 下模型/数据
> → 写 1 个启动脚本 → smoke → `docker exec -d` 起长跑。
>
> **MoE 见 §13**:同一套环境换成 Qwen3-30B-A3B-Base（BF16 FSDP2 + vLLM），第 2–5 节环境完全复用，
> 只需补装 `flydsl==0.1.8` 并下模型。第 13 节自带 smoke、长跑、健康判据和专属排障。
>
> **Megatron-Native + EP=8 见 §13.9–13.15**:同一个 MoE 配方、同一份数据、同一个 rollout，
> 只把训练后端换掉（`training_backend: megatron_native`，TP=PP=CP=1 · EP=8 → DP=8，和 FSDP2 的
> DP8 对齐）。额外要装 megatron-core / Apex / TransformerEngine，见 §13.10。

---

## 1. 架构与对应关系

| 维度 | verl 正式长跑 | 本 runbook（LumenRL 原生） |
|---|---|---|
| 入口 | `python -m recipe.dapo.main_dapo` | `python -m lumenrl.trainer.main`（Ray 控制器，非 torchrun） |
| 编排 | Ray + HybridEngine | **单 Ray-driver：8 FSDP2 actor + 8 colocated AsyncLLM replica** |
| 训练后端 | FSDP | **Lumen FSDP2**（fp32 master + bf16 compute；可选 FP8 blockwise2d 线性） |
| 推理后端 | vLLM `mode=async` | **vLLM `AsyncLLM` 在线**（`transport=ray_http`，逐请求并发）；**可选 ATOM `AsyncLLMEngine`**（`MODE=atomfp8`，同 ray_http + ZMQ IPC） |
| 权重同步 | ZMQ CUDA-IPC 分桶 | **同**（`update_weights_ipc_send` + colocate worker ext，`NCCL_CUMEM_ENABLE=0`） |
| 前向 | remove-padding + varlen | **packed 单序列 varlen**（micro_batch=1，纯 causal，训练/logprob 一致） |
| rollout 输入 | token-in（`prompt_token_ids`） | **token-in**（HF tokenize，`add_special_tokens=False`） |
| rollout seed | `replica_rank + data.seed` | **同**（每 replica 引擎 seed = base_seed + replica_rank） |
| 动态采样 | `algorithm.filter_groups` | `algorithm.dapo.filter_groups`（同语义） |
| TIS 修正 | `rollout_correction.rollout_is=token` | 同 + vLLM `calculate_log_probs=true` |
| overlong 奖励 | reward_manager `overlong_buffer_cfg` | `algorithm.dapo.overlong_buffer`（同公式） |
| 策略损失 | clip-higher + dual-clip + token-mean | 同（`asymmetric_clip_loss` + `loss_agg_mode=token-mean`） |
| 优势估计 | grpo（按 uid 组归一化） | grpo（同） |
| **FP8 rollout** | `rollout.quantization=fp8_per_block` | 同（vLLM `vllm_cfg.quantization=fp8_per_block`；或 ATOM `atom_cfg.online_quant_config.global_quant_config=per_block_fp8`，均 verl-free 移植） |
| **FP8 训练** | Lumen FP8 blockwise2d + `FP8_PARAM_MANAGER=1` | Lumen FP8 blockwise2d + **`FP8_PARAM_MANAGER=0`**（native FSDP2 fp32-master 要求） |

三条 rollout 路线（训练侧一致，只换 rollout 引擎/精度）：

| `MODE` | rollout 引擎 | rollout 精度 | 说明 |
|---|---|---|---|
| `bf16` | vLLM AsyncLLM | BF16 | 基线 |
| `fp8` | vLLM AsyncLLM | per-block FP8 | verl `fp8_per_block` 对齐 |
| `atomfp8` | **ATOM AsyncLLMEngine** | per-block FP8 | ATOM 引擎，默认 `no-eager + compilation_config.level=3 + sleep_level=2`，KL/收敛与 `fp8` 等价（见 §12 注意事项） |

> 代码全部在 `Lumen-RL` 仓库内（clone 即得）。本 runbook 自包含,启动脚本由第 8 节 heredoc 生成。

---

## 2. 路径变量（所有后续命令都用这三个变量，换机只改这里）

```bash
export RL_ROOT=/path/to/lumen_rl      # 代码根（内含 Lumen-RL / Lumen / aiter + 启动脚本）
export DATA_ROOT=/path/to/data        # 模型 / 数据 / 日志 / ckpt 根
export CONTAINER=rl-vllm-fsdp
mkdir -p "$RL_ROOT" "$DATA_ROOT/logs"
```

---

## 3. 拉取代码（仓库各自分支；`aiter` 需要 submodule）

默认非国内网络直接从 GitHub 拉取：

```bash
cd "$RL_ROOT"
git clone -b dev/vllm-fsdp-dapo   https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone -b amd-atom-rollout     https://github.com/ZhangDanyang-AMD/Lumen.git
git clone -b lumen/triton_kernels https://github.com/ZhangDanyang-AMD/aiter.git
git clone -b lumen-rl             https://github.com/xysheng-AMD/ATOM.git   # 仅 MODE=atomfp8（ATOM rollout）需要

# aiter 的 JIT 依赖 composable_kernel；必须补齐，否则 ATOM / FP8 触发
# module_rmsnorm 时会找不到 3rdparty/composable_kernel/.../generate.py。
cd "$RL_ROOT/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel
```

中国内网机器如果 GitHub 直连不稳定，可只对本次命令使用代理镜像（不要写死进仓库 remote）：

```bash
cd "$RL_ROOT"
GHP=https://gh-proxy.com/https://github.com
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b dev/vllm-fsdp-dapo   "$GHP/ZhangDanyang-AMD/Lumen-RL.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b amd-atom-rollout     "$GHP/ZhangDanyang-AMD/Lumen.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen/triton_kernels "$GHP/ZhangDanyang-AMD/aiter.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen-rl             "$GHP/xysheng-AMD/ATOM.git"

cd "$RL_ROOT/aiter"
git -c http.version=HTTP/1.1 \
  -c url."$GHP/".insteadOf=https://github.com/ \
  submodule update --init --depth 1 3rdparty/composable_kernel
```

> 容器内用 root 访问宿主挂载仓库时，`pip install -e` / `setuptools_scm` 可能遇到
> `fatal: detected dubious ownership`。这类一次性实验机器可以在容器内设置全局
> `safe.directory`（见第 5 节）；若是共享机器，也可改用临时 `GIT_CONFIG_GLOBAL`。

| 仓库 | 分支 | 用途 |
|---|---|---|
| `Lumen-RL` | `dev/vllm-fsdp-dapo`（跟踪分支最新 HEAD） | RL 主框架 |
| `Lumen` | `amd-atom-rollout` | FSDP2 训练后端（FP8） |
| `aiter` | `lumen/triton_kernels` | AMD kernel |
| `ATOM` | `lumen-rl` | **仅 `MODE=atomfp8`**：ATOM rollout 引擎；BF16/vLLM-FP8 路线不需要 |

已有 checkout 获取后续代码分支更新：

```bash
git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
git -C "$RL_ROOT/Lumen" pull --ff-only origin amd-atom-rollout
git -C "$RL_ROOT/aiter" pull --ff-only origin lumen/triton_kernels
git -C "$RL_ROOT/aiter" submodule update --init --depth 1 3rdparty/composable_kernel
```

---

## 4. 启动 Docker（把 `$RL_ROOT/$DATA_ROOT` 及派生变量注入容器）

非国内网络直接拉官方镜像：

```bash
sudo docker pull vllm/vllm-openai-rocm:v0.23.0
```

中国内网如果 Docker Hub 慢或卡住，可用国内 registry 镜像拉取后重新打本地 tag：

```bash
sudo docker pull docker.m.daocloud.io/vllm/vllm-openai-rocm:v0.23.0
sudo docker tag docker.m.daocloud.io/vllm/vllm-openai-rocm:v0.23.0 \
  vllm/vllm-openai-rocm:v0.23.0
```

> 启动前务必确认容器内版本是 `vllm 0.23.0`。本机若已有 `vllm/vllm-openai-rocm:latest`
> 不代表可用，可能是旧版 vLLM。

```bash
sudo docker rm -f "$CONTAINER" 2>/dev/null
sudo docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$RL_ROOT":"$RL_ROOT" -v "$DATA_ROOT":"$DATA_ROOT" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" \
  vllm/vllm-openai-rocm:v0.23.0 -lc 'sleep infinity'
```
> `-e` 注入的变量对后续所有 `docker exec` 会话可见,脚本无需再硬编码路径。
> 容器 `stop/start` 不丢依赖;只有 `docker rm` 才丢(丢了要重跑第 5、6 节)。

---

## 5. 装依赖 + vLLM RMSNorm patch + 验证（一次搞定）

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
# 容器 root 访问宿主挂载仓库时允许 git introspection（editable install / setuptools_scm）。
git config --global --add safe.directory "$RL_ROOT/Lumen-RL" || true
git config --global --add safe.directory "$LUMEN_DIR" || true
git config --global --add safe.directory "$AITER_DIR" || true
git config --global --add safe.directory "$RL_ROOT/ATOM" || true
# --- 依赖(不覆盖镜像内 vLLM/torch) ---
cd "$AITER_DIR" && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop || pip install -e .
pip install -e "$LUMEN_DIR" --no-deps || true
cd "$RL_ROOT/Lumen-RL" && pip install -e . --no-deps
pip install "ray[default]>=2.9" "accelerate>=0.28" datasets \
  "math_verify[antlr4_13_2]" "omegaconf>=2.3,<2.4" safetensors wandb
'
```
> `MODE=atomfp8` 的 ATOM 引擎**无需单独 pip 安装**——`run_dapo.sh` 自动把 `$RL_ROOT/ATOM` 及
> `examples/DAPO/atom_aiter_shim` 加到 `PYTHONPATH`(`import atom` 即可用)。仅需先按 §3 `git clone` ATOM。

**vLLM AITER RMSNorm model-sensitive patch**(FP8 rollout 必需;BF16 路线无害)。FP8 rollout 走
`VLLM_ROCM_USE_AITER=1`,vLLM 用 AITER RMSNorm,必须传 `use_model_sensitive_rmsnorm=1` 才能与训练侧
Lumen 的 model-sensitive RMSNorm 对齐,否则 `rollout_corr/kl` 偏大。

> ⚠️ 该 patch 改的是**容器内 vllm wheel**,`docker rm` 重建容器后会丢,**新机器/新容器必打一次**。

在宿主机写出 patch 脚本(幂等,只改 `kernels/aiter_ops.py` / `_aiter_ops.py` 两条 plain RMSNorm
路径,不碰任何 quant-fusion 路径),脚本随 `$RL_ROOT` 挂载进容器:

```bash
cat > "$RL_ROOT/patch_vllm_aiter_rmsnorm.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM AITER RMSNorm Patch (model-sensitive / T5-like) for the rollout side.

Patches the *plain* (non-quant) AITER RMSNorm paths inside the container's
installed vLLM wheel so they pass ``use_model_sensitive_rmsnorm=1``. Quant-fusion
paths (rmsnorm2d_fwd_with_dynamicquant / *_fp8_group_quant / ...) are NOT touched.
Idempotent: safe to run repeatedly.
"""

import importlib.util
import os
import sys

RMS_OLD = """    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)"""

RMS_NEW = """    if not getattr(_rms_norm_impl, "_lumen_logged", False):
        print("[vllm-aiter] rms_norm use_model_sensitive_rmsnorm=1", flush=True)
        _rms_norm_impl._lumen_logged = True

    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)"""

ADD_OLD = """    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return out, residual_out"""

ADD_NEW = """    if not getattr(_rocm_aiter_rmsnorm2d_fwd_with_add_impl, "_lumen_logged", False):
        print(
            "[vllm-aiter] rmsnorm2d_fwd_with_add use_model_sensitive_rmsnorm=1",
            flush=True,
        )
        _rocm_aiter_rmsnorm2d_fwd_with_add_impl._lumen_logged = True
    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
        use_model_sensitive_rmsnorm=1,
    )
    return out, residual_out"""

REPLACEMENTS = ((RMS_OLD, RMS_NEW), (ADD_OLD, ADD_NEW))


def _vllm_dir() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("ERROR: vllm is not importable in this interpreter")
    return spec.submodule_search_locations[0]


def _patch_file(path: str) -> int:
    if not os.path.isfile(path):
        print(f"[skip] {path} (not found)")
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    changed = 0
    for old, new in REPLACEMENTS:
        if new in src:
            continue  # already patched
        if old in src:
            src = src.replace(old, new, 1)
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print(f"[patched] {path} ({changed} site(s))")
    else:
        already = sum(new in src for _, new in REPLACEMENTS)
        print(f"[ok] {path} (already patched)" if already else f"[skip] {path} (no plain call sites)")
    return changed


def main() -> None:
    vdir = _vllm_dir()
    targets = [os.path.join(vdir, "kernels", "aiter_ops.py"), os.path.join(vdir, "_aiter_ops.py")]
    total = sum(_patch_file(p) for p in targets)
    print(f"[done] vLLM AITER RMSNorm patch: {total} replacement(s) applied")


if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 patch_vllm_aiter_rmsnorm.py'
```
> 期望:`[patched] .../kernels/aiter_ops.py (2 site(s))`(或已打过时 `[ok] ... (already patched)`)。
> BF16-only 复现可跳过本 patch(BF16 走 `VLLM_ROCM_USE_AITER=0`)。

**验证依赖 + patch + 导入一起过**:
```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import torch, vllm, ray, lumenrl, inspect
from lumenrl.engine.inference.vllm_ray_server import VLLMRayServer
from vllm.kernels import aiter_ops as k
print("torch", torch.__version__, "vllm", vllm.__version__, "ray", ray.__version__, "GPUs", torch.cuda.device_count())
ms = all("use_model_sensitive_rmsnorm=1" in inspect.getsource(getattr(k,a))
         for a in ["_rms_norm_impl","_rocm_aiter_rmsnorm2d_fwd_with_add_impl"])
print("RMSNorm model-sensitive patch:", ms, "(FP8 需为 True)")
print("import OK")
PY
'
```
> 期望:`GPUs 8`、vllm 0.23.0、`RMSNorm model-sensitive patch: True`、`import OK`。

---

## 6. 下载模型 / 数据（BF16 与 FP8 都需要）

**6.1 模型 + 原始数据**(模型 Qwen3-8B-Base;数据 = verl DAPO 同款 train/val)。

非国内网络可继续用 Hugging Face：
```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D=os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
# 原始 parquet -> $DATA_ROOT/raw/<name>/data/*.parquet
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY
'
```

中国内网机器建议使用 ModelScope，避免 HF 慢/断连：

```bash
sudo docker exec "$CONTAINER" bash -lc '
pip install modelscope
python3 - <<PY
from modelscope.hub.snapshot_download import snapshot_download
import os

D = os.environ["DATA_ROOT"]
snapshot_download(
    "Qwen/Qwen3-8B-Base",
    local_dir=f"{D}/models/Qwen3-8B-Base",
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*", "*.py", "*.tiktoken"],
    max_workers=8,
)
snapshot_download(
    repo_id="BytedTsinghua-SIA/DAPO-Math-17k",
    repo_type="dataset",
    local_dir=f"{D}/raw/DAPO-Math-17k",
    allow_patterns=["*.parquet", "*.json", "*.jsonl", "*.md", "*.txt"],
    max_workers=4,
)
snapshot_download(
    repo_id="BytedTsinghua-SIA/AIME-2024",
    repo_type="dataset",
    local_dir=f"{D}/raw/AIME-2024",
    allow_patterns=["*.parquet", "*.json", "*.jsonl", "*.md", "*.txt"],
    max_workers=4,
)
PY
'
```

> ModelScope 上已验证 ID：`Qwen/Qwen3-8B-Base`、
> `BytedTsinghua-SIA/DAPO-Math-17k`、`BytedTsinghua-SIA/AIME-2024`。
> 下载完成后仍走同一套本地路径，后续过滤与训练命令不需要改。

**6.2 过滤 prompt ≤1024**(复刻 verl `RLHFDataset.maybe_filter_out_long_prompts`;不预过滤则启动会进入
耗时的 overlong-prompt 扫描)。写出脚本并在容器内运行:
```bash
cat > "$RL_ROOT/filter_prompts.py" <<'PYEOF'
import os, glob
import datasets
from transformers import AutoTokenizer

DATA = os.environ["DATA_ROOT"]
MODEL_PATH = f"{DATA}/models/Qwen3-8B-Base"
MAX_PROMPT_LENGTH = 1024
PROMPT_KEY = "prompt"
OUT_DIR = f"{DATA}/data_cached/qwen3-8b-maxprompt1024"

def first_parquet(*dir_globs):
    for g in dir_globs:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no parquet under {dir_globs}")

JOBS = [
    (first_parquet(f"{DATA}/raw/DAPO-Math-17k/**/*.parquet"),
     os.path.join(OUT_DIR, "dapo-math-17k.filtered.parquet")),
    (first_parquet(f"{DATA}/raw/AIME-2024/**/*.parquet"),
     os.path.join(OUT_DIR, "aime-2024.filtered.parquet")),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def doc2len(doc) -> int:
    return len(tokenizer.apply_chat_template(doc[PROMPT_KEY], add_generation_prompt=True, tokenize=True))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
    for src, dst in JOBS:
        ds = datasets.Dataset.from_parquet(src)
        before = len(ds)
        ds = ds.filter(lambda d: doc2len(d) <= MAX_PROMPT_LENGTH, num_proc=nproc,
                       desc=f"Filtering prompts > {MAX_PROMPT_LENGTH} tokens")
        ds.to_parquet(dst)
        print(f"[{src}] -> {dst}: {before} -> {len(ds)} (removed {before-len(ds)})")

if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 filter_prompts.py'
```
产出(即 config / `run_dapo.sh` 默认 `TRAIN_FILE` / `VAL_FILE`):
```
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val
```

---

## 7. 训练配置（已提交在 Lumen-RL 仓库）

| 路线 | config 文件 |
|---|---|
| BF16 | `examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_longrun.yaml` |
| FP8  | `examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml`（只差 `vllm_cfg.quantization=fp8_per_block`） |
| ATOM FP8 | `examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_longrun.yaml`（`generation_backend=atom`、`atom_cfg.transport=ray_http`、`online_quant_config.global_quant_config=per_block_fp8`；启动脚本覆盖为 `no-eager + compilation_config.level=3 + sleep_level=2`；规模/超参与 vLLM FP8 1:1） |

> Qwen3-30B-A3B MoE 的 config 是 `dapo_qwen3moe_a3b_ray_vllm_verlref_{4k_smoke,longrun}.yaml`，
> 规模/超参不是照抄本节的 8B 表格，而是对齐 verl 的 FP8 参考实验，见 **§13**。

规模/超参(两个 config 相同):

| 项 | 值 |
|---|---|
| `num_training_steps` | 1000 |
| `train_global_batch_size` / `dapo.num_generations` | 512（=32 × 16） |
| `gen_batch_size` | 96 |
| `max_response_length` / `max_total_sequence_length` / `max_token_len_per_gpu` | 20480 / 21504 / 21504 |
| `learning_rate` / `lr_warmup_steps` / `weight_decay` / `max_grad_norm` | 1e-6 / 10 / 0.1 / 1.0 |
| `clip_ratio_low/high/c`,`loss_agg_mode` | 0.2 / 0.28 / 10.0，token-mean |
| `overlong_buffer.len` / `penalty_factor` | 512 / 1.0 |
| `filter_groups` | enable，metric=acc，max_num_gen_batches=10 |
| `rollout_correction.rollout_is` / `threshold` | token / 2.0 |
| `training.optimizer_dtype` | bf16（master 权重 engine 内部 fp32） |
| `fsdp_cfg.param_offload / optimizer_offload` | **false**（Ray 路径不支持 offload） |
| vllm_cfg | vLLM FP8:`gpu_memory_utilization=0.30`, `max_model_len=21504`, `enforce_eager=true`, `max_num_batched_tokens=32768`, `max_num_seqs=64`, `calculate_log_probs=true`, `enable_sleep_mode=false`; ATOM FP8 启动时覆盖 `enforce_eager=false`, `atom_cfg.engine_kwargs.compilation_config.level=3`, `enable_sleep_mode=true`, `sleep_level=2` |
| val / checkpoint / seed | `val_steps=10` / `save_steps=50`,`save_total_limit=5`,`resume=true` / 10086 |

BF16 vs FP8 的差异(仅两处):

| 项 | BF16 | FP8 |
|---|---|---|
| `vllm_cfg.quantization` | `""` | `fp8_per_block` |
| 训练线性层 | BF16 | 可选 FP8 blockwise2d（`TRAIN_FP8=1`，`FP8_PARAM_MANAGER=0`） |

> 其余(fp32 master、attention 前向、规模、超参、TIS)全同。`mem/actor` 两者都 ~11.6GB。

---

## 8. 统一启动脚本（BF16/FP8/smoke/长跑 一个脚本）

启动脚本 **已在 Lumen-RL 仓库内**:`examples/DAPO/run_dapo.sh`(`git clone` 即得)。用 `MODE`
(bf16|fp8|atomfp8|atombf16)、`TRAIN_FP8`(0|1)、`STEPS` 控制,**没有重复的 env 块**;所有路径走
`$RL_ROOT`/`$DATA_ROOT` 变量+仓库标准布局自动定位,**无任何机器专属路径**。

> ⚠️ **本 runbook 不再内嵌 `run_dapo.sh` 副本**。这里曾放过一段 heredoc 用于"重建脚本",结果与仓库
> HEAD 漂移了 117 行:丢掉 `atombf16` 模式与 `EXTRA_OVERRIDE`,多带一套仓库已移除的 `CKPT_DIR`
> 逻辑,并且会把已修好的 `TORCHDYNAMO_DISABLE` / `PYTORCH_CUDA_ALLOC_CONF` 行为**写回旧版**。
> **以仓库文件为唯一来源**;脚本被误改就还原它:
>
> ```bash
> git -C "$RL_ROOT/Lumen-RL" checkout -- examples/DAPO/run_dapo.sh
> ```

脚本的启动开关全部通过环境变量给,**不需要改脚本内容**:

| 变量 | 默认 | 作用 |
|---|---|---|
| `MODE` | `bf16` | `bf16` / `fp8` / `atomfp8` / `atombf16`:选 config + rollout 引擎与精度 |
| `TRAIN_FP8` | `0` | `1` = 训练侧 Lumen FP8 blockwise2d(自动带 `FP8_PARAM_MANAGER=0`) |
| `STEPS` | `1000` | 覆盖 `num_training_steps` |
| `CONFIG_OVERRIDE` | 按 `MODE` 推导 | 直接指定 config 路径。**smoke 必须用它**(见 §9) |
| `EXTRA_OVERRIDE` | 空 | 追加任意 Hydra 覆盖,空格分隔。例:`"checkpointing.save_steps=10 checkpointing.save_total_limit=2"` |
| `LOG` | `$DATA_ROOT/logs/$RUN_ID.log` | 日志路径;脚本同时把它写进 `/tmp/run_dapo_log.txt` |
| `MODEL_PATH` / `TRAIN_FILE` / `VAL_FILE` | `$DATA_ROOT/...` 标准布局 | 换模型/数据集 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | **本平台建议启动时置空**,见下 |
| `ATOM_TORCH_COMPILE_CACHE_ROOT` | `/tmp/atom_torch_compile_cache` | 每 replica 独立 compile cache 根目录 |

> checkpoint 目录**由 config 的 `checkpointing.checkpoint_dir` 决定**(脚本不再覆盖它);要换位置或
> 改落盘频率用 `EXTRA_OVERRIDE`。`save_steps` 小(如 10)时注意磁盘:8B FSDP2 单个 checkpoint 约 90GB。

**`PYTORCH_CUDA_ALLOC_CONF` 建议启动时置空**:ROCm/HIP allocator **不支持** `expandable_segments`,
留默认值只会让每个 actor 刷 `expandable_segments not supported on this platform` 警告(实测一次
smoke 25 条)。脚本用 `${VAR-default}` 判定,所以"显式传空值"算已设置并会被 `unset`:

```bash
PYTORCH_CUDA_ALLOC_CONF= STEPS=1000 MODE=atomfp8 TRAIN_FP8=1 \
  bash "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
```

**`TORCHDYNAMO_DISABLE` 不要手工设置**。脚本全局保持 `=1`(训练 actor 关 dynamo);`MODE=atomfp8`
的 no-eager level=3 rollout 所需的 dynamo 由 `ATOMReplicaManager` 在创建 ATOM Ray actor 时通过
`runtime_env` 注入 `TORCHDYNAMO_DISABLE=0`,**只作用于 rollout 进程**(判据与 per-replica compile
cache 隔离共用:`compilation_config.level>0` 或 `enforce_eager=false`)。
**不要在顶层 `export TORCHDYNAMO_DISABLE=0`** —— 那会让训练 actor 一并继承;dynamo 对训练侧纯属
副作用,这也是它一度被全局打开的原因。

### 8.1 已验证 helper scripts（可选，本机 run area）

如果使用本 run area（`lumenrl_native_vllm_fsdp_run/`）而不是直接在 `$RL_ROOT/Lumen-RL` 内手敲
`docker exec`，可用 `scripts/` 下的两个 wrapper；它们只是把本节 `run_dapo.sh` 的 ATOM FP8
no-eager level=3 方案固化成可重复命令。

> 这个 run area **不随仓库分发**，新机器上通常不存在（`ls $RL_ROOT/../lumenrl_native_vllm_fsdp_run`
> 确认）。没有它就直接用 §9 / §10 的 `docker exec` + `run_dapo.sh` 命令，功能完全等价。

```bash
# 4k smoke：默认 3 step、batch=64、gen_batch=24、num_generations=8、no eval/no save。
# 打开 no-eager level=3 时需同时打开 sleep2。
# 注意：不要再传 TORCHDYNAMO_DISABLE=0 —— dynamo 现在由 ATOM Ray actor 的 runtime_env
# 按进程注入（见 §8），顶层传 0 会让训练 actor 一并继承。
SMOKE_STEPS=1 \
ATOM_ENFORCE_EAGER=false \
ATOM_COMPILATION_LEVEL=3 \
ENABLE_SLEEP_MODE=true \
SLEEP_LEVEL=2 \
./scripts/run_atom_fp8_4k_smoke.sh

# 正式长跑：使用 runbook formal config 原始规模
# 1000 steps、batch=512、gen_batch=96、num_generations=16、resp=20480。
./scripts/run_atom_fp8_noeager_level3_longrun.sh
```

正式长跑 wrapper 默认写入：

```bash
LOG=$DATA_ROOT/logs/lumenrl_native_vllm_fsdp/<run_id>.log
CKPT=$DATA_ROOT/ckpts/lumenrl_native_vllm_fsdp/atomfp8-noeager-level3-8b
```

可覆盖：

```bash
RUN_ID=my-run LONGRUN_STEPS=1000 CKPT=/path/to/ckpt ./scripts/run_atom_fp8_noeager_level3_longrun.sh
```

### 8.2 ATOM / AITER 首次 JIT 预编译（训练前做）

`MODE=atomfp8` 使用本地 `ATOM` + 本地 `aiter` 源码，不同于 vLLM wheel 内置 kernel。
ATOM 启动 Qwen3 FP8 rollout 时会按需 JIT 编译 `aiter` kernel，尤其是：

- `module_rmsnorm`：ATOM `atom/model_ops/layernorm.py` 直接调用本地
  `aiter.rmsnorm2d_fwd(..., use_model_sensitive_rmsnorm=1)`，首次会编译；
- `module_gemm_a8w8_blockscale_bpreshuffle`、`module_activation`、`module_sample`、
  `module_rope_*` / `module_cache` 等：首次 ATOM FP8 rollout 会继续触发。

这些是**环境安装 / 预热成本**，不要把它们算进训练性能。建议在正式 smoke / longrun 前先预编译
RMSNorm，并让一次 ATOM 小 smoke 完整跑完，把后续常用 JIT 产物也落盘。

```bash
# 先确认 aiter submodule 已拉取（见第 3 节），否则 module_rmsnorm 会找不到 generate.py。
test -f "$RL_ROOT/aiter/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d/generate.py"

# 单进程预编译 module_rmsnorm。首次可能耗时十几到二十分钟；看到 PRECOMPILE_DONE 才算完成。
sudo docker exec "$CONTAINER" bash -lc '
export PYTHONPATH="$AITER_DIR:${PYTHONPATH:-}"
python3 - <<PY
import torch
from aiter import rmsnorm2d_fwd
print("PRECOMPILE_START", flush=True)
x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
w = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
y = rmsnorm2d_fwd(x, w, 1e-6, use_model_sensitive_rmsnorm=1)
torch.cuda.synchronize()
print("PRECOMPILE_DONE", y.shape, y.dtype, flush=True)
PY
'
```

若中途被 `Ctrl-C`、容器重启或 `pkill` 打断，可能留下 stale lock，表现为后续进程一直打印
`waiting for baton release at .../lock_module_rmsnorm`，但没有 `ninja` / `hipcc` / `clang-22`
编译进程。清理后重跑预编译：

```bash
sudo docker exec "$CONTAINER" bash -lc '
rm -rf "$AITER_DIR/aiter/jit/build/lock_module_rmsnorm" \
       "$AITER_DIR/aiter/jit/build/module_rmsnorm"
'
```

> vLLM FP8 rollout 不需要这一步，因为它走镜像内已安装的 vLLM/AITER 路径；ATOM FP8 rollout
> 走本地 `aiter` 源码 JIT，因此需要预编译。

---

## 9. Smoke（小配置验证整链路，前台等结果）

> ⚠️ Smoke 必须显式使用 `CONFIG_OVERRIDE=..._smoke.yaml` 小配置。只设置 `STEPS=5`
> 会继续使用 longrun config（resp=20480、batch=512、gen_batch=96），不是快速 smoke。
> 小配置只用于验证链路；第 10 节长跑仍使用 `run_dapo.sh` 默认 longrun config，不受影响。

| 路线 | smoke config | 规模 |
|---|---|---|
| BF16 vLLM | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | resp=512, batch=128, gen_batch=24, generations=16 |
| vLLM FP8 rollout | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | resp=512, batch=128, gen_batch=24, generations=16 |
| vLLM FP8 4k rollout | `dapo_qwen3_8b_ray_vllm_fp8_4k_smoke.yaml` | resp=4096, batch=64, gen_batch=24, generations=8 |
| ATOM FP8 4k rollout | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | resp=4096, batch=64, gen_batch=24, generations=8, no-eager level=3（由 `MODE=atomfp8` 覆盖） |

推荐先跑三条“链路 smoke”（训练侧保持 BF16，避免把 `TRAIN_FP8=1` 的训练 kernel JIT 混入快速验证）：

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
# 显式注入，避免容器内 RL_ROOT 为空；同时把 PYTORCH_CUDA_ALLOC_CONF 置空（§8：本平台不支持
# expandable_segments）。注意 `export VAR=;` 是"设为空值"，脚本会据此 unset 掉它。
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"

# BF16 vLLM：512-token 小配置。
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  STEPS=1 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-bf16-512.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# vLLM FP8 rollout：512-token 小配置；TRAIN_FP8=0 仅验证 rollout fp8_per_block。
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  STEPS=1 MODE=fp8 TRAIN_FP8=0 LOG=$DATA_ROOT/logs/smoke-vllm-fp8-512.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# ATOM FP8 rollout：4k 小配置；MODE=atomfp8 会覆盖为 no-eager + compilation_config.level=3 + sleep2。
# 运行前建议先完成第 8.2 节 AITER JIT 预编译；首次仍可能继续编译 GEMM/activation/sample/rope/cache 等小模块。
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  STEPS=1 MODE=atomfp8 TRAIN_FP8=0 LOG=$DATA_ROOT/logs/smoke-atom-fp8-4k.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# 确认 ATOM 正在跑 no-eager level=3（run_dapo.sh 的 MODE=atomfp8 覆盖项）。
# 期望有一行 "rollout-scoped TORCHDYNAMO_DISABLE=0 (driver keeps 1)"：driver 是 1 才是对的。
sudo docker exec "$CONTAINER" bash -lc '
L=$(cat /tmp/run_dapo_log.txt)
grep -aE "enforce_eager=false|compilation_config|level=3|rollout-scoped TORCHDYNAMO|torch compile cache_dir|Ray ATOM rollout ready" "$L" | tail -20
'

# 更硬的证据：直接读进程 env —— 训练 actor 必须是 1，rollout actor 必须是 0（跑起来时执行）。
sudo docker exec "$CONTAINER" bash -lc '
for pat in LumenActorWorker ATOMRayServer; do
  p=$(pgrep -f $pat | head -1)
  echo "--- $pat pid=$p"
  [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ | grep -E "^(TORCHDYNAMO_DISABLE|PYTORCH_CUDA_ALLOC_CONF|LUMEN_FP8)="
done
'
```

如需对比 vLLM FP8 与 ATOM FP8 的 4k rollout，可把第二条换成
`dapo_qwen3_8b_ray_vllm_fp8_4k_smoke.yaml`。如需验证训练侧 FP8 E2E，再单独设置
`TRAIN_FP8=1`；这会触发 Lumen 训练侧 FP8 kernel/JIT，耗时应视为环境预热或单独测试，不要混入
rollout smoke 性能对比。

本机 run area 也可直接用 §8.1 的 `scripts/run_atom_fp8_4k_smoke.sh`，适合验证
ATOM no-eager level=3 的短配置。

Smoke 期望证据:
- `RLTrainer.setup (ray-controller) complete: ... actor_workers=8`、`VLLMRayServer[i]: engine seed=<10086+i>`
  - ATOM:`ATOMRayServer[...] AsyncLLMEngine ready`、`Ray ATOM rollout ready: ... online_quant={'global_quant_config': 'per_block_fp8'}`
  - ATOM no-eager level=3:日志中应能看到 `enforce_eager=false` / `compilation_config.level=3`
    或等价的 ATOM engine kwargs;以及每 replica 一行 `torch compile cache_dir=...`。
  - dynamo 作用域:`ATOMReplicaManager: rollout-scoped TORCHDYNAMO_DISABLE=0 (driver keeps 1)`。
    **driver 显示 1 才正确** —— 只有 rollout actor 该拿到 0。要更硬的证据就读进程 env(上面第二条命令):
    `LumenActorWorker` = `TORCHDYNAMO_DISABLE=1`,`ATOMRayServer` = `0`。
- 小配置可能关闭 `filter_groups`；若开启，`kept` 应 >0。若 `kept 0/...` + `Rollout reward: accuracy=0.0000` + 大量 `finished with reason max`,是 rollout 退化,见 §12。
- `callbacks: step=1 ... entropy=... grad_norm~0.85 ppo_kl=0`,exit 0
  - BF16:`rollout_corr/kl≈0.001`;vLLM-FP8:`≈0.003–0.004`;ATOM-FP8:`≈0.004`
- FP8 训练额外:`[verl] Restored lm_head to BF16`、`online fp8 reload: ...weights=399`
  - `Lumen optimizations applied (fp8=True, ...)` 这行来自 `fsdp_backend` 的 worker logger,**默认不一定
    出现在 driver 日志里**(历史上能看到是因为挂了 `probe_apply_lumen.py`)。别把"没看到"当成 FP8 没生效。
  - 可靠判据是训练 actor 走了哪套 GEMM:`grep -a LumenActorWorker "$L" | grep -c a8w8_blockscale_tuned_gemm`
    应 >0(FP8 blockwise);`TRAIN_FP8=0` 时该值为 0、只出现 `bf16_tuned_gemm.csv`。
- **不应**出现 `materialized on CPU`、`has no attribute`、entropy≈0.04 / grad_norm 1e4+(见第 12 节排障)

---

## 10. 启动长跑（`docker exec -d` 分离，防中断）

```bash
# 三选一：
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
# 显式把 RL_ROOT/DATA_ROOT 注入容器命令（从宿主 §2 变量展开），detached exec 不再依赖 §4 的 -e 注入；
# 同时把 PYTORCH_CUDA_ALLOC_CONF 置空（§8）。
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=bf16                bash '$S'"   # BF16
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8                bash '$S'"   # vLLM FP8 rollout + BF16 训练
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8     TRAIN_FP8=1 bash '$S'"   # vLLM FP8 rollout + FP8 训练
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8 TRAIN_FP8=1 bash '$S'"   # ATOM FP8 rollout + FP8 训练（no-eager level=3 + sleep2）
# W&B(可选):把 key 放 $RL_ROOT/wandb.key,格式 WANDB_API_KEY=xxxx(脚本自动读)

# 更密的落盘 / 换 ckpt 位置：用 EXTRA_OVERRIDE 追加 Hydra 覆盖（空格分隔）。
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8 TRAIN_FP8=1 \
  EXTRA_OVERRIDE='checkpointing.save_steps=10 checkpointing.save_total_limit=2' bash '$S'"
```

> ⚠️ **`DATA_ROOT` 想换盘时必须无条件覆盖**。§4 的 `docker run -e DATA_ROOT=...` 把值烤进了容器环境,
> 所以自建 wrapper 里写 `export DATA_ROOT="${DATA_ROOT:-/new/disk}"` **不会生效**(变量已存在),
> ckpt 会静默写回旧盘。上面的 `ENVX` 是无条件赋值,安全;自己写 wrapper 请直接
> `export DATA_ROOT=/new/disk`。
>
> 代码留在 `$HOME`、数据/ckpt/日志放大容量盘时,`DATA_ROOT` 指到那块盘即可(§2),模型与
> `data_cached/` 需一并搬过去。`save_steps=10` + 8B FSDP2 约 90GB/ckpt,务必先 `df -h` 看余量。

本机 run area 可直接：

```bash
./scripts/run_atom_fp8_noeager_level3_longrun.sh
```

> 建议先 `STEPS=30` 起一版确认显存/指标健康,再上 1000 步。
> ⚠️ `run_dapo.sh` 开头有 `: "${RL_ROOT:?}"`,容器内 `RL_ROOT` 为空会直接报错退出;上面的 `ENVX` 就是为
> 防这个坑(曾遇到容器 `RL_ROOT` 未注入导致 detached 长跑起不来)。

确认已在跑:
```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 200
  grep -aE "setup .ray-controller. complete|filter_groups round|View run" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|CUDA error" "$L" | tail'
```

---

## 11. 监控 / 停止 / 续跑

**监控**(W&B `core/` 面板含 `core/entropy`、`core/kl`(=rollout_corr/kl)、`core/ppo_kl`、
`core/reward_mean`、`core/grad_norm`、`core/val_accuracy` 等):
```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); grep -aE "callbacks: step=" "$L" | tail -5'
```

健康判据:

| 指标 | BF16 | FP8 |
|---|---|---|
| `entropy` | ~0.8 缓降（趋势 ~25 步后显现） | 同趋势 |
| `grad_norm` | ~0.85，无持续尖峰 | ~0.85 |
| `ppo_kl` | ≈0（单步优化、old==train） | ≈0 |
| `rollout_corr/kl` | ≈0.001 | **≈0.003–0.004**（FP8 gap，正常；逼近 2.0 才警惕；ATOM-FP8≈0.004，比 vLLM 略高） |
| step 时间 | ~4–5 min/步（resp 20480） | 同量级；ATOM no-eager level=3 主要加速 rollout,但 sleep/wake + 权重同步会增加固定开销 |

**停止**:
```bash
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f lumenrl.trainer.main; pkill -9 -f VLLMRayServer; pkill -9 -f raylet; pkill -9 -f gcs_server
  for pid in $(ps -eo pid,stat,comm | awk "\$3==\"python3\" && \$2!~/Z/ {print \$1}"); do kill -9 $pid 2>/dev/null; done
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -1'   # 应回到 ~298MB/卡
```
**续跑**:`checkpointing.resume=true`,重跑第 10 节同一命令即从最近 checkpoint 恢复(`save_steps=50`)。

---

## 12. 排障 / 风险 / 显存回退

**FP8 训练发散**(entropy≈0.04 / grad_norm 1e4+ / `rollout_corr/kl` 1e4+):基本是
①`FP8_PARAM_MANAGER` 没设成 0(它与 native FSDP2 fp32-master 冲突),或
②第 5 节 vLLM RMSNorm patch 没打(新容器)。

**长跑后期发散**:resp=20480 长序列历史(~step 278)出现过 `grad_norm` 尖峰 → `rollout_corr/kl` 越 TIS
阈值 2.0 → entropy 爆升 → 策略塌缩。缓解:盯 `rollout_corr/kl` 与 `grad_norm`,必要时后期降 lr /
给 `rollout_is_threshold` 加下界。

**显存回退(OOM)**:
- `policy.max_response_length=8192` + `max_total_sequence_length=9216` + `max_token_len_per_gpu=9216`;
- 或降 `train_global_batch_size` / `gen_batch_size`;
- Ray 路径 **不要**开 `fsdp_cfg.param_offload/optimizer_offload`(会报 `parameters should be materialized on CPU`)。

**ATOM no-eager / level=3 注意事项**:`MODE=atomfp8` 正式方案默认 `enforce_eager=false` +
`atom_cfg.engine_kwargs.compilation_config.level=3`。必须同时满足:
- `ATOM_ISOLATE_TORCH_COMPILE_CACHE=1`（LumenRL Ray adapter 会给每个 colocated replica 设置独立
  `compilation_config.cache_dir`;否则 8 个单卡 replica 都是 `rank_0/backbone`,会并发写同一
  torch compile cache,触发 Inductor `write_atomic -> rename` 的 `FileNotFoundError`）。
- `policy.generation.vllm_cfg.enable_sleep_mode=true` 且 `sleep_level=2`（rollout 后释放 KV cache /
  weights / CUDA graph;否则 no-eager CUDA graph 与 rollout 权重常驻,训练 backward 容易
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`）。
- dynamo **不要手工开**:`ATOMReplicaManager` 会在建 ATOM Ray actor 时通过 `runtime_env` 注入
  `TORCHDYNAMO_DISABLE=0`,只作用于 rollout 进程;脚本全局保持 `=1`。顶层 `export TORCHDYNAMO_DISABLE=0`
  会让训练 actor 一并继承(纯副作用),见 §8。
- 若失败后显存仍高,清理 `spawn_main` / `torch/_inductor/compile_worker` 孤儿进程,或直接
  `docker restart "$CONTAINER"`；容器 stop/start 不丢依赖。

**跑完/中断后显存不释放**（`rocm-smi` 每卡仍 ~90GB，但 `ps` 里已无 trainer）:`run_dapo.sh` 只在
**启动前**清理进程,收尾不清,所以 ATOM EngineCore 的 `spawn_main` 子进程（及其 inductor
compile worker）会变成孤儿继续占显存。下次 `run_dapo.sh` 启动时会被清掉,但期间这台机器等于没卡。
手动清（注意用 `spawn[_]main` 这类写法，否则 `pkill -f` 会匹配到自己的命令行而自杀）:

```bash
sudo docker exec "$CONTAINER" bash -lc '
  pkill -9 -f "compile_[w]orker" || true
  pkill -9 -f "spawn[_]main"     || true
  pkill -9 -f "resource[_]tracker" || true
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -3'   # 应回到 ~298MB/卡
```

**ATOM rollout 退化**(`MODE=atomfp8` 时 `filter_groups: kept 0/96` + `Rollout reward: accuracy=0.0000`
+ 日志大量 `finished with reason max`、无 `eos`):rollout 生成崩坏(全部答错、打满 max length)。优先检查
ATOM `layernorm.py` 的 model-sensitive RMSNorm 是否生效(见 §14);未对齐会表现为
`rollout_corr/kl` 偏大(~0.007 而非 ~0.004),严重时可能导致生成质量退化。

**ATOM `rollout_corr/kl` 偏大**(~0.007 而非 ~0.004):ATOM `atom/model_ops/layernorm.py` 的 plain
RMSNorm 未传 `use_model_sensitive_rmsnorm=1`,与训练侧 T5-like RMSNorm 不一致(见 §14 修复)。

---

## 13. Qwen3-30B-A3B MoE（BF16 FSDP2 + vLLM）：smoke 与长跑

同一套环境换一个 MoE policy：Qwen3-30B-A3B-Base（30.5B 总参 / 3.3B 激活，48 层、128 专家、top-8）。
训练侧仍是 FSDP2，rollout 仍是同卡 colocated vLLM `AsyncLLM`(TP=1) + ZMQ CUDA-IPC 权重同步。
**第 2–5 节的环境完全复用**，新机器只需再做两件事：装 `flydsl==0.1.8`（§13.1）和下模型（§13.2）。
数据、启动脚本、监控、停止全部照旧。

规模/超参**不再照抄 §7 的 8B 表格**，而是对齐 verl 的 FP8 参考实验
（`recipe/low_precision/run_dapo_qwen3_moe_30b_megatron_fp8e2e.sh`）的 BF16 基线，见 §13.3。

> §13.1–13.8 这条线**不是** Megatron+EP。同一个模型换 Megatron-Native 后端见
> **§13.9–13.15**（两条线的 config 除了 `training_backend` 和 `megatron_cfg` 逐字段相同，
> 所以可以直接对照）；MoE+EP 的实现说明见 `dapo-lumenrl-megatron-moe-ep-handoff.md`，
> 环境从零构建见 `dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md`。
> 调试历史与已排除的假设见 `dapo-lumenrl-moe-fsdp-vllm-handoff.md`
> 和 `megatron-moe-length-collapse-handoff.md`。

### 13.1 相对 8B 路线的差异（只有这些）

| 项 | 8B dense | Qwen3-30B-A3B MoE |
|---|---|---|
| 模型 | `Qwen/Qwen3-8B-Base` | **`Qwen/Qwen3-30B-A3B-Base`**，必须 Base 版 |
| `flydsl` | 镜像自带 0.1.4.2 可用 | **必须升到 0.1.8** |
| `aiter setup.py develop` | FP8/ATOM 需要 | BF16 **不需要**(`VLLM_ROCM_USE_AITER=0`) |
| §5 vLLM RMSNorm patch | FP8 必需 | BF16 **不需要** |
| 数据 | `data_cached/qwen3-8b-maxprompt1024/` | **同一份直接复用**（两个模型 tokenizer 字节相同） |
| MoE 专属 | — | R3 路由复放 + 融合专家权重同步 + 同步覆盖率断言 |

**为什么必须是 Base 版**：instruct/thinking 版的 Qwen3-30B-A3B 在 `max_response_length` 内**永远不闭合
`</think>`**（实测给到 3072 token 仍不闭合、也不出 `\boxed`），于是每条样本都被截断、reward 恒为 −1、
`filter_groups` 连续 10 轮 kept 0，直接抛 `RuntimeError: filter_groups collected no valid groups`。
Base 版能正常输出 `Answer:`。

**flydsl 0.1.8**：镜像自带的 0.1.4.2 会让 `from aiter import flash_attn_varlen_func` 报版本不兼容，
训练**前向直接挂**。这一步必做：

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install "flydsl==0.1.8" && python3 -c "
import flydsl, transformers, vllm
print(\"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__, \"vllm\", vllm.__version__)"'
```
> 期望 `flydsl 0.1.8 transformers 5.12.0 vllm 0.23.0`（`pip list` 里显示为 `0.23.0+rocm723`）。
> transformers **5.x 是必需的**，它把 Qwen3-MoE 的专家融合成 3D 张量，仓库的权重同步按这个布局写。

### 13.2 下载模型（数据不用重下）

```bash
sudo docker exec "$CONTAINER" bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
du -sh "$DATA_ROOT/models/Qwen3-30B-A3B-Base"   # 约 57G
```
中国内网走 ModelScope（ID 同名 `Qwen/Qwen3-30B-A3B-Base`），用法照抄 §6.1 的 modelscope 片段。

**数据直接复用 §6.2 的产物**，不需要重新过滤：Qwen3-8B-Base 与 Qwen3-30B-A3B-Base 的
`tokenizer.json` / `vocab.json` / `merges.txt` 三个文件 md5 完全相同（vocab 151936），
所以按 8B tokenizer 过滤出的 prompt ≤1024 对 MoE 同样成立。

### 13.3 config 与规模（对齐 verl 的 FP8 参考实验）

只有两个 config，都是 verl `recipe/low_precision/run_dapo_qwen3_moe_30b_megatron_fp8e2e.sh`
的逐项移植（对应 verl `docs/low_precision/fp8.md` 的 "FP8 End-to-End → Qwen3-30B-A3B MoE Model"）。
verl 那张图有三条曲线（BF16 / FP8 E2E / FP8-rollout-only），**我们复现的是 BF16 那条**，
也就是 FP8 结果所对标的基线。

| 用途 | config | 规模 |
|---|---|---|
| **4k smoke** | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | prompt=2048, resp=4096, 8 prompt × 16, gen_batch=24, 3 步 |
| 长跑 | `dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml` | prompt=2048, resp=20480, **128 prompt × 16 = 2048 序列**, gen_batch=384, 1000 步 |

verl 脚本里的参数原样搬过来的部分：

| verl | 值 | LumenRL |
|---|---|---|
| `max_prompt_length` / `max_response_length` | 2048 / 20480 | `max_total_sequence_length: 22528` / `max_response_length: 20480` |
| `train_prompt_bsz` × `n_resp_per_prompt` | 128 × 16 | `train_global_batch_size: 2048`（**是序列数**） |
| `gen_prompt_bsz` | 384（=128×3） | `gen_batch_size: 384`（**是 prompt 数**） |
| `train_prompt_mini_bsz` | 128（= bsz，单次更新） | 无 mini-batch 切分，语义相同 |
| lr / wd / clip_grad / entropy_coeff | 1e-6 / 0.1 / 1.0 / 0 | 同 |
| **lr warmup** | **0**（`lr_warmup_steps_ratio=0.0`） | `lr_warmup_steps: 0` |
| clip 0.2/0.28/10 · token-mean · grpo | 同 | 同 |
| overlong_buffer 512 / 1.0 | 同 | 同 |
| filter_groups acc / 10 | 同 | 同 |
| `rollout_is` / C | token / 2.0 | 同 |
| temperature/top_p/top_k | 1.0/1.0/−1 | 同 |
| `max_num_batched_tokens` / `max_num_seqs` | 32768 / 256 | 同 |
| `test_freq` | 10 | `val_steps: 10` |

⚠️ **注意单位**：`train_global_batch_size` 是**序列数**，`gen_batch_size` 是 **prompt 数**。
LumenRL 用 `train_prompts = train_global_batch_size // num_generations` 反推 prompt 数，
所以 verl 的 "prompt batch size 128" 对应的是 `2048`，不是 `128`。

**MoE router 用 BF16，这是刻意的**。verl 那个脚本把 fp32 那行注释掉了，理由写在注释里：
vLLM 的 Qwen3-MoE router 本来就是 BF16，两侧对齐比单侧提精度更重要，而且他们两侧都试过 fp32、
没有精度收益。我们自己 15k token 的 A/B 也是同一结论。所以跑这两个 config **必须带
`LUMENRL_FP32_MOE_ROUTER=0`**（LumenRL 默认是 fp32），日志里应看到
`[lumenrl] MoE router patched on 48 gates (fp32=False)`。
顺带一提这在 verl 里是特例——其它 18 个 MoE 脚本都显式设了 `moe_router_dtype=fp32`。

**R3（Rollout Routing Replay）默认关**，因为 verl 这个 recipe 没有它。R3 让训练侧复放 rollout
实际选中的 top-k 专家，实测（3 步 smoke，R3 是唯一变量）`rollout_corr/kl` 0.00160→0.00044、
`mismatch/chi2_seq` 1.33e6→1.35、step 时间 +4.2%。要开来做 A/B：
`EXTRA_OVERRIDE='moe.r3.rollout_replay=true'`。

因为环境/硬件差异而**没能对齐**的四项（都只影响吞吐，不改变优化问题）：

| 项 | verl | 这里 | 原因 |
|---|---|---|---|
| 训练后端 | Megatron + Megatron-Bridge | FSDP2 | 这条线就是 FSDP2；verl 另有一个 FSDP 版但那是 FP8-rollout-only |
| GPU | 2×8 H100 | 1×8 MI350X | 同样的 global batch，一半的卡，约 2 倍 step 时间 |
| rollout 并行 | `gen_tp=2` | TP=1 每卡一个 colocated replica | 架构差异 |
| `gpu_memory_utilization` | 0.5 | **0.30** | verl 生成时把参数/优化器 offload 到 CPU，LumenRL 的 Ray 路径不支持，训练显存常驻。**0.40 试过，撑不住**，见 §13.8 |
| 验证采样 | temperature 0.6, top_p 1.0, n=1 | greedy | LumenRL val 是 greedy，仅诊断用，不影响训练 |

**数据集不用另做**：verl 用 `max_prompt_length=2048`，而 §6.2 产出的缓存是按 1024 过滤的——
实测按 2048 重新过滤 DAPO-Math-17k **一行都不删**（1,791,700 → 1,791,700，AIME 960 → 960），
这个数据集里没有超过 1024 token 的 prompt，两个切法是同一份数据。

### 13.4 融合专家权重同步（这条线的关键修复，已在仓库里）

transformers 5.x 把 MoE 专家存成融合 3D 张量，训练侧 `state_dict()` 发的是
`model.layers.N.mlp.experts.gate_up_proj` / `down_proj`，而 vLLM 把同样的 buffer 叫
`experts.w13_weight` / `w2_weight`，其 `expert_params_mapping` 只认 per-expert 名。
**融合名匹配不上任何 mapping，会走 vLLM 的静默 `continue` 分支**：不报错、不加载，
96 个融合专家张量（约 57GB、**93% 的参数**）每次同步全被丢弃，rollout 引擎的专家永远停在磁盘加载值。

`lumenrl/engine/inference/vllm_moe_weight_sync.py`（commit `3aab539`）修掉了它，并加了覆盖率断言。
**`git pull` 到 `dev/vllm-fsdp-dapo` 最新即可，不需要手动做任何事。**先跑纯 CPU 测试确认代码完整：

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/Lumen-RL" &&
  python3 -m lumenrl.tests.test_moe_weight_sync &&      # 11 项，融合专家同步
  python3 -m lumenrl.tests.test_rollout_routing &&      #  9 项，R3
  python3 -m lumenrl.tests.test_dataproto_ragged &&     # 10 项
  python3 -m lumenrl.tests.test_mismatch_metrics'       #  4 项
```

两个自检开关：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `LUMENRL_WEIGHT_SYNC_CHECK` | `error` | 每次同步断言 `loaded` 覆盖 `named_parameters()`，漏一个就抛。可设 `warn` / `off` |
| `LUMENRL_WEIGHT_SYNC_VERIFY` | `0` | `1` = 加载后把 vLLM buffer 读回来和发送张量 `torch.equal` 比对，不等就抛（约 +0.1s/步） |

覆盖率断言默认开着，**新机器第一次跑 MoE 建议再加一次 `VERIFY=1` 的短跑**做端到端确认：

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"
sudo docker exec "$CONTAINER" bash -lc "$ENVX LUMENRL_WEIGHT_SYNC_VERIFY=1 LUMENRL_FP32_MOE_ROUTER=0 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-verify.log bash '$S'; tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""
```
> 通过的判据是**没有异常**：跑完 exit 0 就说明 96 个融合张量 × 8 replica × 3 次同步全部逐位一致。
> 失败会抛 `weight sync verify failed for ... shard w1/w3/w2` 或
> `weight sync (colocate-ipc) left N/M rollout parameters untouched: ...`。

### 13.5 Smoke（4k，3 步，前台等结果）

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"

sudo docker exec "$CONTAINER" bash -lc "$ENVX LUMENRL_FP32_MOE_ROUTER=0 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-a3b-4k-smoke.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""
```
> `MODEL_PATH` 必须显式给 —— `run_dapo.sh` 的默认值是 8B。
> `LUMENRL_FP32_MOE_ROUTER=0` 也必须给，否则 router 会走 LumenRL 默认的 fp32，和 verl 对不上。

实测（8×MI350X，3 步约 10 分钟，其中约 5 分钟是 8 个 actor 各自加载 57GB 模型）：

| step | 1 | 2 | 3 |
|---|---|---|---|
| `rollout_corr/kl` | 0.00150 | 0.00181 | 0.00149 |
| `mismatch/abs_diff` | 0.0201 | 0.0218 | 0.0234 |
| `reward/accuracy` | 0.102 | 0.102 | 0.133 |

Smoke 期望证据：
- `RLTrainer.setup (ray-controller) complete: ... actor_workers=8`
- **`[lumenrl] MoE router patched on 48 gates (fp32=False)`** —— `False` 才是对的，`True` 说明忘了传 `LUMENRL_FP32_MOE_ROUTER=0`
- 第 1 步的 `lr` 就是 `9.99998e-07`（满值），说明 warmup 确实是 0；如果看到 `2e-07` 说明用错了配置
- `rollout_corr/kl` 在 **1.5e-3 量级**，且不随步数单调爬升。这个水平比我们自己那条线（R3 + fp32 router）高约 3 倍，
  **是预期内的** —— verl 文档原话就是 "rollout & training distribution mismatch is in general higher for MoE,
  rollout correction required even for BF16"，这个 recipe 两个 mismatch 抑制手段都没有。
  `mismatch/chi2_seq` 会到 e6 量级，同样是 R3 关闭时的已知水平
- 修复权重同步之前，`rollout_corr/kl` 会从 0.0005 一路爬到 0.0166（55 步，与 step 相关 +0.98）
- `timing/weight_sync_s` 约 1.1–1.7s；`mem/actor_max_reserved_gb` 约 75–115GB
- 不应出现 `left N/M rollout parameters untouched`（同步漏了）或 `filter_groups collected no valid groups`（用错了 instruct 版模型）

### 13.6 长跑

**先看磁盘**。MoE 的 FSDP2 checkpoint 是 **~342GB**（fp32 权重 + optimizer state × 8 分片，
= 8B 那份 90GB 按参数量线性放大）。`save_steps=10` / `save_total_limit=1` 时峰值就是一份
（`CheckpointCallback` 会**先删旧的再写新的**，commit `1e01aef`），但仍然需要 342GB 可用空间。

```bash
df -h "$DATA_ROOT"     # 至少 400G 可用才谈得上存 checkpoint
```

长跑配置的 `policy.model_name` / `checkpoint_dir` 用的是 `$SCRATCH_ROOT`（本地大容量盘）。
**这个变量必须导出**，否则 omegaconf 解析 `${oc.env:SCRATCH_ROOT}` 直接失败：

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
LUMENRL_FP32_MOE_ROUTER=0 PYTORCH_CUDA_ALLOC_CONF=;"

# 空间够（≥400G）：正常存 checkpoint
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe-a3b.log bash '$S'"

# 空间不够：关掉落盘（崩了只能从头跑，先想清楚）
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  EXTRA_OVERRIDE='checkpointing.save_steps=1000000 checkpointing.resume=false' \
  LOG=\$DATA_ROOT/logs/longrun-moe-a3b.log bash '$S'"
```
> `SCRATCH_ROOT` 即使关掉落盘也要给 —— config 里 `checkpoint_dir` 引用了它，omegaconf 解析不到直接退出。

> ⚠️ 关落盘**不要写 `checkpointing.checkpoint_dir=`**。Hydra 会把空值解析成 `None`，
> omegaconf 立刻报 `Incompatible value 'None' for field of type 'str'` 并退出。
> 用 `save_steps` 设成一个跑不到的大数才是干净的做法。

W&B（可选，`$RL_ROOT/wandb.key` 里放 `WANDB_API_KEY=xxxx`，脚本自动读）。MoE 长跑默认进
`LUMEN-RL-MOE` 项目，换 run 名：`EXTRA_OVERRIDE='logger.wandb.name=<your-name>'`。

### 13.7 健康判据（verlref 配置，8×MI350X 实测 101 步 / 21.6 小时）

训练指标：

| 指标 | step 1 | step 50 | step 101 | 与步数相关 |
|---|---|---|---|---|
| `reward/accuracy` | 0.136 | 0.494 | **0.581** | **+0.85** |
| `rollout_corr/kl` | 0.00148 | 0.00052 | 0.00069 | −0.68 |
| `mismatch/abs_diff` | 0.0213 | 0.0064 | 0.0077 | −0.75 |
| `entropy` | 0.844 | 0.082 | 0.094 | −0.75 |
| `grad_norm` | 0.197 | 0.056 | 0.031 | −0.56 |

AIME-2024 在线验证（每 10 步，greedy）：

| step | 10 | 30 | 50 | 70 | 90 | 100 |
|---|---|---|---|---|---|---|
| `val-core/acc/mean@1` | 0.041 | 0.183 | 0.167 | 0.272 | **0.361** | 0.348 |
| `val/response_length_mean` | 2407 | 4094 | 5792 | 5838 | 8331 | **10389** |

**这就是这条线跑通的证据**：AIME 100 步从 4% 到约 35%，回答长度从 2.4k 涨到 10.4k
（模型学会想更久），训练 accuracy 0.136 → 0.581。

- **step 时间中位数约 11 分钟**（672s，resp=20480、batch 2048 序列）。前一版 batch 512 的配置是 8–11 分钟
- `timing/weight_sync_s` 全程 1.1–1.4s，不随步数增长
- **最关键的判据是 `rollout_corr/kl` 不随步数单调爬升。**它降下去是正常的（策略收敛变确定，
  log 空间分歧自然缩小）；**爬上去说明权重同步又漏了**，回到 §13.4 用 `VERIFY=1` 复查
- **已知问题：熵坍缩。**entropy 101 步从 0.844 掉到 0.094。verl 的 recipe 就是 `entropy_coeff=0`，
  所以这是照搬 recipe 的必然结果，不是我们移植出的偏差。前一版配置（R3 开、batch 512）也是同样形状：
  127 步 0.998 → 0.069。这是在权重同步正确的前提下测到的，是**真实的策略坍缩**，不是同步 bug 的假象。
  要治得先加 `entropy_coeff`，但那样就偏离 verl 的对照组了
- **这一跑在 step 101 死于显存耗尽**（当时 `gpu_memory_utilization` 还是 0.40），见 §13.8。
  上表的数据本身有效，但 101 步之后的行为还没有观测

### 13.8 MoE 专属排障

**`weight sync (colocate-ipc) left N/M rollout parameters untouched: ...`**
同步漏了参数。异常信息会列出前 8 个名字：若是 `...experts.w13_weight` / `w2_weight`，说明融合 MoE
路由没生效——要么代码没拉到最新，要么 vLLM/transformers 升级后布局假设失效了。
后者用下面这个探针重新验证：它把同一份 checkpoint 分别用 HF 和 vLLM 加载，逐位比对
`gate_up_proj`/`w13_weight`、`down_proj`/`w2_weight`，并打印 vLLM 的 `unquantized_backend`
（必须是 `TRITON`；若是 `AITER` 说明 kernel 做了 shuffle，整块加载的前提不再成立，
`vllm_moe_weight_sync.py` 需要跟着改）。

```bash
cat > "$RL_ROOT/moe_layout_probe.py" <<'PYEOF'
"""Cross-check the transformers fused MoE expert layout against vLLM's."""
import gc, os, torch

MODEL = os.environ.get("PROBE_MODEL", f"{os.environ['DATA_ROOT']}/models/Qwen3-30B-A3B-Base")
LAYERS = (0, 1, 23, 47)


def vllm_side():
    from vllm import LLM
    llm = LLM(model=MODEL, tensor_parallel_size=1, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.30, max_model_len=2048)

    def extract(self):
        m = self.model_runner.model
        out = {}
        for i in LAYERS:
            e = m.model.layers[i].mlp.experts
            out[f"{i}.w13"] = e.w13_weight.data.detach().cpu().clone()
            out[f"{i}.w2"] = e.w2_weight.data.detach().cpu().clone()
        out["_backend"] = str(getattr(e.quant_method, "unquantized_backend", None))
        out["_contig"] = bool(e.w13_weight.data.is_contiguous())
        return out

    (r,) = llm.collective_rpc(extract)
    del llm; gc.collect(); torch.cuda.empty_cache()
    return r


def hf_side():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
    out = {}
    for i in LAYERS:
        ex = m.model.layers[i].mlp.experts
        out[f"{i}.gate_up"] = ex.gate_up_proj.data.detach().clone()
        out[f"{i}.down"] = ex.down_proj.data.detach().clone()
    del m; gc.collect()
    return out


if __name__ == "__main__":
    v = vllm_side()
    print("backend:", v["_backend"], "w13 contiguous:", v["_contig"])
    h = hf_side()
    ok = True
    for i in LAYERS:
        for tag, a, b in ((f"L{i} gate_up/w13", h[f"{i}.gate_up"], v[f"{i}.w13"]),
                          (f"L{i} down/w2", h[f"{i}.down"], v[f"{i}.w2"])):
            same = a.shape == b.shape and torch.equal(a, b)
            ok &= same
            print(f"  {tag:<22} {tuple(a.shape)} {'EXACT' if same else 'DIFFERS'}")
    print("VERDICT:", "element-wise identical" if ok else "LAYOUTS DIFFER")
    raise SystemExit(0 if ok else 1)
PYEOF
sudo docker exec "$CONTAINER" bash -lc '
  cd "$RL_ROOT" && VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ROCM_USE_AITER=0 \
  python3 moe_layout_probe.py'
```
> 期望 `backend: UnquantizedMoeBackend.TRITON`、8 行全 `EXACT`、`VERDICT: element-wise identical`。
> 实测约 50 秒，占一张卡 + 约 60GB 主机内存；退出码 0 = 通过。
> 临时绕过报错可用 `LUMENRL_WEIGHT_SYNC_CHECK=warn`，但那等于回到 bug 能藏 54 步的状态。

**`filter_groups collected no valid groups`**
基本都是用了 instruct/thinking 版模型而不是 Base 版，见 §13.1。

**`from aiter import flash_attn_varlen_func` 版本不兼容 / 训练前向挂掉**
`flydsl` 没升到 0.1.8，见 §13.1。

**`Incompatible value 'None' for field of type 'str'`（`full_key: checkpointing.checkpoint_dir`）**
用了 `checkpointing.checkpoint_dir=` 这种空值覆盖，见 §13.6。

**`InterpolationResolutionError` / `SCRATCH_ROOT` 未定义**
长跑配置引用 `${oc.env:SCRATCH_ROOT}`，`ENVX` 里必须带上它，见 §13.6。

**`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `chunk_cat_cuda_kernel`**

**复现过两次**，都在反向的 reduce-scatter copy-in，整个 Ray job 随出事的 actor 一起死：

| | 第一次 | 第二次 |
|---|---|---|
| 配置 | 旧配置（batch 512、R3 开） | verlref（batch 2048、`gpu_memory_utilization` 0.4） |
| 崩溃步数 | step 128（21.5 小时） | **step 11（1.9 小时）** |
| 出事 rank | 3 | **7** |
| kernel | `chunk_cat_cuda_kernel<float, BFloat16>` | **同一个** |
| grid | `[19472896, 8, 1]` | `[19447936, 8, 1]` |

**先把"崩在哪"换算成"多久崩一次"**，规律才出来。梯度累积下每个 micro-batch 每层都触发一次
reduce-scatter copy-in：旧配置每步 `512/8 × 48 = 3,072` 次，verlref 每步 `2048/8 × 48 = 12,288` 次。
两次崩溃分别在约 **39 万次和 14 万次调用**之后——同一个数量级。
所以这不是"第 128 步有什么特殊"，而是**热路径上约十万分之一概率的偶发故障**。
这也解释了为什么隔离复现几十次全是干净的：样本量差了四个数量级。

**这不是 OOM**，别按显存不足去处理：

- OOM 在这套栈上长这样：`torch.OutOfMemoryError: HIP out of memory. Tried to allocate ...` ——
  分配阶段抛的 Python 异常。aperture violation 是 kernel **执行**阶段的硬件故障，
  含义是"访问了不属于任何合法地址空间的地址"，即野指针，两回事
- 崩之前显存很宽裕且平稳：`mem/actor_allocated_gb` 恒为 42.80、`max_reserved` 84.6–94.6GB
  （252GiB 的卡，vLLM 另占 ~86GB），10 步内无上升趋势；日志里 0 条 OOM
- `dmesg` 9 天 23738 行里 0 条 `oom-kill`；进程是 HSA runtime 在自己进程内抛的 SIGABRT
  （带 Python traceback），不是内核 OOM killer 的 SIGKILL。主机还剩 2.4TB 内存，容器无内存上限
- 0 条 amdgpu/kfd/RAS/ECC 内核消息，无 retired HBM page

定位到的信息：faulting kernel 在 FSDP2 里**只有一个调用点** `foreach_reduce_scatter_copy_in` ——
反向把 bf16 梯度打包进 fp32 的 reduce-scatter 缓冲区。grid 对得上：grid.y = 8 = world_size，
19472896 × 4 elem/work-item = 77,891,584 ≈ 一个 MoE 层的每卡份额（623,120,640 / 8）。

查过但**已排除**的假设：

| 假设 | 结论 |
|---|---|
| 显存不足 / OOM | **排除**，见上面四条 |
| fp32 缓冲区 2.32 GiB 超 2^31 字节导致 int32 溢出 | **排除**。`chunk_cat_repro.py` 用完全相同的形状与切片参考实现逐位比对，128 专家（2.32 GiB）及以下全部 exact；kernel 用的是 64 位索引 |
| 某块卡的 HBM 坏了 | **排除**。两次死在不同 rank（3、7）；GPU 3 单独做 220 GiB 写入回读 ×3 + 复现该形状 ×3 全部干净 |
| [pytorch#122026](https://github.com/pytorch/pytorch/issues/122026) 的元数据越界 | **排除**。用它的复现方式 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 重跑仍然干净（该缺陷 2024 年已由 PR #122076 修掉） |
| 通信库的特殊内存 aperture | **排除**。FSDP2 默认是 `DefaultReduceScatter`，目标缓冲区就是普通 `torch.empty`；只有调了 `set_allocate_memory_from_process_group` 才走 RCCL 注册内存，LumenRL 没调 |
| kernel/形状本身有确定性 bug | **不成立**。8 个 rank 每步跑同样形状的同一个 kernel，每次只死一个 |

**尚未解释**：为什么这个 kernel 会偶发访问非法地址。它在隔离环境下怎么测都是对的，
所以问题多半在并发/内存状态这类隔离测不出来的地方。另外 ROCm 把 abort 归给队列读指针处的 packet，
当时有 ~88 个 packet 在飞（rptr 50676415 / wptr 50676503），**真凶也可能是相邻的 kernel**。

#### 规避手段：`LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1`

`lumenrl/engine/training/fsdp_chunk_cat_fallback.py` 把 `foreach_reduce_scatter_copy_in`
换成普通切片拷贝：同样的字节、同样的布局，只是不走那个融合 kernel。
每个 layer group 从 1 次 launch 变成 88 次（11 个张量 × 8 个 chunk），
但每次都是大块连续拷贝，带宽不变，相对 9 分钟的 step 可忽略。加进 `ENVX` 即可：

```bash
LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1
```
生效时每个 actor 打印 `[lumenrl] FSDP2 reduce-scatter copy-in: slicing fallback installed`。
正确性有 6 项单测（真实 MoE 层形状、需要补零的行数、整块都是 padding 的 chunk、非连续梯度、
不同 world_size、无 dtype 提升），全部与 `torch._chunk_cat` 逐位一致：

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/Lumen-RL" &&
  python3 -m lumenrl.tests.test_fsdp_chunk_cat_fallback'   # 需要 1 张卡
```

**实测效果**（这个开关当初是作为可证伪的实验加的，结果如下）：

| | 无 fallback | 有 fallback |
|---|---|---|
| 崩溃步数 | step 11 | step 101 |
| 撑过的 copy-in 次数 | ~13.5 万 | **~124 万** |
| 故障 | `APERTURE_VIOLATION` in `chunk_cat` | 换成了别的原因（见下） |

**非法地址故障没有再出现**，在 9 倍的调用量下都没复现。这是支持 fallback 有效的证据，
但原故障是概率性的，两三个数据点还不能算定论。**它仍然是规避而不是修复** ——
为什么那个 kernel 会偶发访问非法地址，至今没有解释。

> 如果开着 fallback 还是撞 `APERTURE_VIOLATION`，说明 ROCm 的归属误导了我们、真凶在别处，
> 那时上 `AMD_SERIALIZE_KERNEL=3` 串行化 kernel 拿准确归属（很慢，只在复现时开）。
> 目前 GPU coredump 抓不到（`GPU coredump: execvp failed: No such file or directory`），
> 要拿 dump 得先把 handler 装好。

**`HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 0 MB`（这个是真 OOM）**

和上面那个非法地址故障**不是一回事**，别搞混。实测在 `gpu_memory_utilization=0.40` 下
跑到 step 101（21.6 小时）出现，faulting kernel 是 `ncclDevKernel_Generic`：驱动侧没内存了，
RCCL 分配不到通信资源。

账是这么算的（252GiB 的卡）：

| | 0.40 | 0.30 |
|---|---|---|
| vLLM | ~101GB | ~76GB |
| actor `max_reserved` 峰值 | 117.6GB | 85–95GB |
| **余量** | **~33GB** | **~59GB** |

余量要留给 HSA runtime、RCCL 缓冲和驱动开销，33GB 不够。而且分配器水位会随步数爬
（+0.086 GB/步，101 步涨了 8.7GB）——注意**这不是泄漏**：`mem/actor_allocated_gb` 恒为 42.8、
`max_allocated` 恒为 75.6，存活内存完全不动，涨的是 caching allocator 的碎片。

**所以 verlref 配置定在 `gpu_memory_utilization: 0.30`。**要往上调就得同时压 batch 或序列长度。

**实践上更重要的是：存 checkpoint。**这三次崩溃分别丢了 128、11、101 步 ——
第一次是 21.5 小时，第三次是 21.6 小时。这是 §13.6 关落盘那条路的真实代价。

### 13.9 换成 Megatron-Native 后端（EP=8，同一配方）

同一个 MoE policy、同一份数据、同一个 colocated vLLM rollout，只把训练后端从 FSDP2 换成
Megatron-Native + Expert Parallel。**这一节存在的意义是"只有一个变量"**：两个 config 除了
`policy.training_backend` 和 `megatron_cfg` 之外逐字段相同，所以两条线的任何差异都只能来自训练后端。

| | §13.1–13.8（FSDP2） | 本节（Megatron-Native） |
|---|---|---|
| `policy.training_backend` | `fsdp2`（默认） | `megatron_native` |
| 并行 | FSDP2 全分片，DP=8 | **TP=1, PP=1, CP=1, EP=8 → DP=8** |
| 专家 | 融合 3D 张量（transformers 5.x 布局） | Megatron `TopKRouter` + grouped-GEMM，128 专家 / 8 卡 = 每卡 16 个 |
| 优化器 | fp32 master + bf16 compute | Megatron distributed optimizer（fp32 master 分片到 DP） |
| 额外依赖 | 无 | **megatron-core 0.18.2 + ROCm Apex + ROCm TransformerEngine**（§13.10） |
| checkpoint | FSDP2 分片，约 342GB | Megatron dist-checkpoint，**约 400GB** |
| step 时间（resp=20480, batch 2048） | 约 11 分钟 | **约 9.3 分钟**（首步约 14 分钟，含 vLLM 加载） |

**拓扑为什么选 EP=8 而不是 TP/PP 混合**：`DP = 8 / (TP × PP × CP) = 8`，和 FSDP2 的 DP8 一致，
每个 rank 仍然看到 2048/8 = 256 条序列，梯度是同一个全局 batch 在同一种切分下的平均。
任何缩小 DP 的改动都会让 distributed optimizer 的 state 每卡翻倍（DP 8→4 多约 8.5GB），
把激活上省下来的又吃回去 —— **CP=2 实测当场 OOM**，比 CP=1 更早死。TP、PP 同理，
它们不是这条线上的显存解法。

> **两个后端不能共用 checkpoint 目录**（格式不同），也**不能同时占卡**。起之前确认无残留进程、
> 显存回到空闲基线（每卡约 298MB）。

### 13.10 额外环境：megatron-core / Apex / TransformerEngine

§2–§5 的环境是 FSDP2 用的，Megatron 还需要三样东西。**完整的从零构建步骤见
`dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md` §6**（含逐步验证、编译期坑和 manifest），
这里只给已验证的 revision 和最短路径：

| 组件 | 版本 / revision | 装法 |
|---|---|---|
| megatron-core | `0.18.2` | `pip install --no-deps "megatron-core==0.18.2"` |
| ROCm Apex | `daed85255d51476425080e7e6203f0bee6d7e4cc` | 源码 `setup.py install --cpp_ext --cuda_ext`，`PYTORCH_ROCM_ARCH=gfx950` |
| ROCm TransformerEngine | `6e541a10419a6e31bdc98b1516db04eb81a463b6` → `2.15.0.dev0+6e541a1` | 源码 `pip install -v . --no-build-isolation`，约 9 分钟 |

```bash
# megatron-core：不要装 megatron-bridge，Qwen3 HF<->Megatron 转换由 LumenRL
# lumenrl/engine/training/qwen3_megatron_bridge.py 负责。
sudo docker exec "$CONTAINER" bash -lc 'pip install --no-deps "megatron-core==0.18.2"'
```

TE 编译要点（细节见那份 runbook §6.3）：必须用 **ROCm fork**、必须递归拉全部 submodule
（约 5.1GiB，含 AOTriton / CK JIT / Composable Kernel），编译前先卸掉可能存在的 NVIDIA TE 包，
并且要带 `TORCH_DONT_CHECK_COMPILER_ABI=1` —— ROCm 7.2.3 的 `hipcc -v` 在没有输入文件时返回 1，
CK-JIT 的编译器 ABI 探测会把它误判成"编译器不可用"。

> ⚠️ **绝对不要 `pip install transformer_engine`**，那会装成 NVIDIA 版，导入即 undefined symbol。

一站式验证（TE / Apex / megatron-core / 双 engine 注册）：

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 megatron_verify.py'
```
> 期望看到 `megatron.core 0.18.2`、`transformer_engine 2.15.0.dev0+6e541a1`、Apex
> `FusedLayerNorm/FusedAdam OK`，以及 `fsdp2` 和 `megatron_native` 两个 engine 都注册成功。
> 这个脚本在 `$RL_ROOT/`，是那份 runbook 生成的产物。

**flydsl 有个坑值得单独记**：`run_dapo.sh` 把本地 `$AITER_DIR` 放在 `PYTHONPATH` 最前，
运行时用的是**仓库里的 aiter 源码**，它要求 flydsl ≥ `0.1.5.dev515`，所以必须是 §13.1 那个
`flydsl==0.1.8`。而镜像自带的 wheel `amd-aiter 0.1.13.post1` 反过来 pin `flydsl<0.1.5`，
升级后它会报 `cannot import name 'fly_values'`。两者互斥，**只有走 `run_dapo.sh` 的 PYTHONPATH
才是对的**，别按 wheel 的 pin 回退。

### 13.11 config、拓扑与 router 精度

两个 config，都是 §13.3 那两个 FSDP2 verlref 文件的逐字段拷贝：

| 用途 | config | 规模 |
|---|---|---|
| **4k smoke** | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | prompt=2048, resp=4096, 8 prompt × 16, gen_batch=24, 3 步 |
| **长跑** | `dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml` | prompt=2048, resp=**20480**, **128 prompt × 16 = 2048 序列**, gen_batch=384, 1000 步 |

`megatron_cfg` 的全部内容（长跑）：

```yaml
      use_distributed_optimizer: true
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      expert_model_parallel_size: 8       # 128 专家分到 8 卡
      sequence_parallel: false
      moe_grouped_gemm: true
      moe_permute_fusion: true            # 融合 MoE permute，少一批 scratch buffer
      moe_aux_loss_coeff: 0.0             # RL loss 不变，和 FSDP2 一致
      moe_router_dtype: fp32              # 见下方"router 精度"
      recompute_granularity: full         # resp=20480 必需
      recompute_method: uniform
      recompute_num_layers: 1
      log_probs_chunk_size: 1024
      enable_dynamic_batch: true
      max_tokens_per_gpu: 8192            # 不是 22528，见下方"显存"
```

相对 §13.3 那个 FSDP2 长跑，**只有三处偏离，都只影响显存/吞吐，不改变优化问题**：

| 项 | FSDP2 | Megatron | 原因 |
|---|---|---|---|
| `gpu_memory_utilization` | 0.30 | **0.25** | Megatron-Native 每 actor 常驻显存比 FSDP2 高约 68%（4k smoke 实测 72.0 vs 42.8 GB allocated、137 vs 114 GB 峰值 reserved）。0.25 已接近地板：每个 colocated replica 要装整个 30.5B BF16 权重（约 61GB），0.25×288GB 只剩约 11GB 给 KV cache，再低就起不来 |
| `max_tokens_per_gpu` | —（FSDP2 用 `max_token_len_per_gpu`） | **8192**，不是 22528 | **这不降低最坏 bin** —— `_build_bins` 会给任何超过预算的行单独一个 bin，每步那条约 20.7k token 的序列照样独占一个。但它能阻止**每个打包 bin** 也都涨到 22.5k：从每步约 7 次峰值级的 allocate/free 变成 1 次。见下方"显存" |
| `recompute_granularity` | 未开 | `full` | 把 resp=20480 的激活峰值压住 |

**router 精度：这里是 fp32，和 §13.3 的 FSDP2 刻意不同。**
FSDP2 那条线用 BF16 是对的 —— FSDP2 和 vLLM 跑的是**同一个 PyTorch router op、同一种布局**，
BF16 舍入会让两边落到同一组 top-8 专家。但 Megatron 走的是它自己的 `TopKRouter` 喂 grouped-GEMM，
BF16 下两种实现会在"近乎平票"的 token 上选出不同专家，而翻一个专家会让那个 token 的 log-prob 变很多。
实测：`moe_router_dtype: null` 时 `rollout_corr/kl` 到 step 77 一直平在 6.5e-4，然后每步约 +16%
爬到 step 110 的 2.4e-2，而同期 FSDP2 稳在 6.6e-4；换成 fp32 之后到 step 185 都还是 7e-4。

所以**长跑必须带 `LUMENRL_FP32_MOE_ROUTER=1`**，smoke（`moe_router_dtype: null`）必须带 `=0`。
这个环境变量只作用于 **vLLM worker**（Megatron engine 读的是 `megatron_cfg.moe_router_dtype`），
**两处必须一起翻**，否则变成反向不匹配。日志里对应
`MoE+EP spec: ... EP=8 ... router_dtype=fp32 pre_softmax=False aux_loss_coeff=0.0`。

> `pre_softmax=False` 是对的：HF 的 `norm_topk_prob=True` 和 Megatron 的 `pre_softmax=False`
> 在"选哪些专家"和"权重是多少"上严格等价。`score_function` / `topk_scaling_factor` 两边都没设。

**R3（Rollout Routing Replay）默认关**，和 §13.3 同理（verl 这个 recipe 没有它）。

### 13.12 必须带上的代码修复（拉到最新就有）

这条线上有三个已修的 bug，全部在 `dev/vllm-fsdp-dapo`。**`git pull` 到最新即可，不需要手动改任何东西**，
但值得知道它们是什么，因为前两个会静默地改变训练结果：

| commit | 修的东西 |
|---|---|
| `2bbdfde` | **`_row_policy_loss` 里 `response_mask` 对齐差一位**（Megatron 独有） |
| `a09b702` | Ray 路径从未真正应用 rollout IS 权重（两个后端都中） |
| `f3d20e2` | `rollout_corr/kl` 背后三个 per-token 张量的落盘探针（诊断用，默认关） |
| `c9d75ca` | 本节这两个 config |

**`2bbdfde` 是这条线上最重要的一个，说明一下为什么。**
`rl_trainer._build_response_mask` 返回的是 `mask[:, 1:]` —— 宽度 `S-1`、第 i 项标记 token i+1，
和 `old_log_probs`、训练侧 `token_lp` 是同一个坐标系。`megatron_base_engine._row_policy_loss` 里
原来写的是 `_col("response_mask", shift=True)`，**又 shift 了一次**，整个 loss 窗口往前滑一格：

| | 应该覆盖 | 实际覆盖 |
|---|---|---|
| `token_lp` 下标 | `plen-1 .. L-2` | `plen-2 .. L-3` |
| 含义 | 首个 response token 的预测 … **EOS 的预测** | 最后一个 **prompt** token 的预测 … EOS 前一个 |

后果是**每条序列的 EOS 位置拿不到任何策略梯度**，而 EOS 恰好是决定响应长度的那个 token；
同时最后一个 prompt 位置被算进了 loss（这一半危害小：GRPO 组内 advantage 和为 0、同组 prompt 完全相同、
单步单 epoch 下 ratio 恒为 1，这项梯度精确抵消）。

FSDP2 那条路径（`actor_worker._policy_loss_fn`）用的是宽度判断
`if mask.shape[-1] == L + 1: mask = mask[:, 1:]`，不会重复 shift，**所以只有 Megatron 错**。
这是两个后端在 loss 层面唯一的行为差异，也就是"同一配方、FSDP2 正常而 Megatron 长度崩塌"的头号嫌疑。

它还顺带解释了一个当时看不懂的现象：窗口前移后被纳入 loss 的那个 prompt 位置，rollout 引擎从来没为它
上报过 logprob，于是 `rollout_lp` 在那里精确为 `0.0` —— **每条序列恰好一个**，2048 条序列就是
2048 个坏 token，它们贡献了 `rollout_corr/kl` 的 **97.4%**（那些位置上 `old_lp` 均值 −52.3，
`log p = 0` 意味着概率 1.0）。剔掉它们之后 `mean(rollout − train)` 从 +0.0318 掉到 +0.00082，
和 FSDP2 同期的 +0.00066 是同一量级。**换句话说，修之前那条 kl 曲线本身不可信。**

不占 GPU 的对齐自检。它重建 trainer 交给引擎的那套张量坐标系，再回放 `_row_policy_loss` 的切片，
断言选中的位置正好是 response token 集合（含 EOS）、且每个位置都有真实的 rollout logprob：

```bash
cat > "$RL_ROOT/test_response_mask_alignment.py" <<'PYEOF'
"""Check that the Megatron per-row loss masks exactly the response tokens."""
import os, sys
import torch

_ROOT = os.environ.get("RL_ROOT") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Lumen-RL"))
from lumenrl.trainer.rl_trainer import RLTrainer

# Set to 1 to replay the pre-2bbdfde behaviour (unconditional shift).
BUGGY = os.environ.get("REPLAY_BUG") == "1"


def build_batch(prompt_lens, resp_lens, pad_id=0):
    """Left-padded sequences plus the tensors the Ray controller derives."""
    lens = [p + r for p, r in zip(prompt_lens, resp_lens)]
    S, B = max(lens), len(lens)
    ids = torch.full((B, S), pad_id, dtype=torch.long)
    am = torch.zeros((B, S), dtype=torch.long)
    for i, L in enumerate(lens):
        ids[i, S - L:] = torch.arange(1, L + 1)
        am[i, S - L:] = 1
    # engine_compute_log_probs writes lp[r, start : start+L-1]; entry i scores token i+1
    old_lp = torch.zeros((B, S), dtype=torch.float32)
    rollout_lp = torch.zeros((B, S - 1), dtype=torch.float32)
    for i, L in enumerate(lens):
        off = S - L
        old_lp[i, off:off + L - 1] = -1.0
        rs = off + prompt_lens[i] - 1          # one log-prob per generated token
        rollout_lp[i, rs:rs + resp_lens[i]] = -2.0
    return {
        "input_ids": ids, "attention_mask": am,
        "old_log_probs": old_lp, "rollout_log_probs": rollout_lp,
        "response_mask": RLTrainer._build_response_mask(None, ids, am, prompt_lens),
        "advantages": torch.ones(B, dtype=torch.float32),
    }


def replay_row(t, r):
    """The slicing MegatronBaseEngine._row_policy_loss performs for one row."""
    idx = t["attention_mask"][r].nonzero(as_tuple=False).squeeze(-1)
    start, L = int(idx[0]), int(idx.numel())
    token_lp = torch.zeros(1, L - 1)           # scores tokens start+1 .. start+L-1

    def col(name, shift):
        return t[name][r][start + (1 if shift else 0):].reshape(1, -1)

    rm = t["response_mask"]
    shift = True if BUGGY else rm.shape[-1] >= t["input_ids"].shape[-1]
    mask, old_lp, rlp = col("response_mask", shift), col("old_log_probs", False), \
        col("rollout_log_probs", False)
    Le = min(v.shape[-1] for v in (token_lp, old_lp, mask, rlp))
    return start, L, mask[0, :Le], rlp[0, :Le]


def main():
    prompt_lens, resp_lens = [7, 5, 9], [6, 11, 4]
    t = build_batch(prompt_lens, resp_lens)
    failures = 0
    for r, (plen, rlen) in enumerate(zip(prompt_lens, resp_lens)):
        start, L, mask, rlp = replay_row(t, r)
        # token_lp index j scores token start+1+j, so response tokens
        # start+plen .. start+L-1 are indices plen-1 .. L-2
        want = torch.zeros(mask.shape[-1])
        want[plen - 1:L - 1] = 1.0
        sel = mask.nonzero(as_tuple=True)[0].tolist()
        zeros_in_mask = int(((rlp == 0.0) & (mask > 0)).sum())
        ok = torch.equal(mask, want) and zeros_in_mask == 0 and len(sel) == rlen
        print(f"row {r}: prompt={plen} resp={rlen} start={start} L={L} selected={sel} "
              f"n={len(sel)} (want {rlen}) mask_ok={torch.equal(mask, want)} "
              f"rollout_lp_zero_in_mask={zeros_in_mask}")
        failures += not ok
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 test_response_mask_alignment.py'
```
> 期望 `PASS`，每行 `n=<resp_len> (want <resp_len>) mask_ok=True rollout_lp_zero_in_mask=0`。
> 加 `REPLAY_BUG=1` 可以看到修复前的行为：窗口整体前移一格（`selected=[5..10]` 而不是 `[6..11]`）、
> 丢掉 EOS 位置，并且 `rollout_lp_zero_in_mask=1` —— **每行恰好一个**，这就是那 2048 个坏 token 的来源。

**`a09b702`（TIS 接线）**：`compute_rollout_is_weights` 原来只在 `rl_trainer.train()` 里被调用，
而 Ray 控制器在 `train()` 开头就分流去 `_train_with_ray_controller()` 了，所以
`rollout_is_weights` 从未写进 batch，引擎查不到、`asymmetric_clip_loss` 什么修正都没做。
config 里 verl 对齐的 `rollout_is: token` / `rollout_is_threshold: 2.0` **是空转的，两个后端都一样**。
现在两条路径共用 `_apply_rollout_is_weights()`，并且多报四个 `rollout_correction/*` 指标。

⚠️ 这**同时改变了 FSDP2 基线的行为**（§13.7 那组数是 TIS 空转时测的）。4k 实测影响很小
（`is_weight_mean=0.9997`、`is_weight_max=2` 打到 clip 上限），但要和 §13.7 严格对齐做 A/B 时，
把 config 里 `quantization.rollout_correction.rollout_is` 置空即可关掉。

### 13.13 Smoke 与长跑

**Smoke（4k，3 步）。**注意 `LUMENRL_FP32_MOE_ROUTER=0` —— smoke config 的
`moe_router_dtype` 是 `null`，两处必须一致：

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
PYTORCH_CUDA_ALLOC_CONF=;"

sudo docker exec "$CONTAINER" bash -lc "$ENVX LUMENRL_FP32_MOE_ROUTER=0 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-a3b-4k-smoke-megatron.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""
```

实测（8×MI355X，修复后，3 步约 11 分钟，其中约 6 分钟是加载）：

| step | 1 | 2 | 3 |
|---|---|---|---|
| `rollout_corr/kl` | 0.00184 | 0.00180 | 0.00163 |
| `rollout_correction/kl` | 0.00173 | 0.00164 | 0.00176 |
| `rollout_correction/is_weight_mean` | 0.9997 | 1.00002 | 0.9999 |
| `ppo_kl` | 1.07e-4 | 1.59e-4 | −1.26e-4 |
| `seq/mean_response_len` | 871 | 714 | 971 |
| `timing/step_s` | 111 | 93 | 103 |
| `mem/actor_max_reserved_gb` | 128 | 139 | 140 |

Smoke 期望证据：
- `MoE+EP spec: num_experts=128 topk=8 moe_ffn=768 | tp=1 pp=1 cp=1 EP=8 etp=1 -> local_experts/rank=16 | grouped_gemm=True router_dtype=fp32 pre_softmax=False aux_loss_coeff=0.0`
  （smoke 用 `null` 时这里是 `router_dtype=None`）
- **`rollout_correction/*` 这一族指标必须出现** —— 它们在就说明 `a09b702` 的 TIS 真的接上了；没有就是代码没拉到最新
- `rollout_corr/kl` 在 **1.5e-3 量级**且不随步数爬升
- 无 `Traceback` / `HSA_STATUS`

**长跑（resp=20480）。先看磁盘**：Megatron dist-checkpoint 约 **400GB**，
`CheckpointCallback` 会先删旧的再写新的（commit `1e01aef`），稳态一份，但仍需 400GB 可用。
长跑 config 的 `save_steps: 5` 很激进（9.3 min/步 → 约 46 分钟一次 400GB 落盘），
按容错需求调大：`EXTRA_OVERRIDE='checkpointing.save_steps=20'`。

```bash
df -h "$DATA_ROOT"     # 至少 400G

S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
# 注意是 =1，和 config 的 moe_router_dtype: fp32 配对
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
LUMENRL_FP32_MOE_ROUTER=1 PYTORCH_CUDA_ALLOC_CONF=;"

sudo docker exec -d "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe-a3b-megatron.log bash '$S'"
```
> `PYTORCH_CUDA_ALLOC_CONF=` 置空是刻意的（§8）：本平台的 ROCm 没有 `expandable_segments`。
> `SCRATCH_ROOT` 必须给，config 的 `checkpoint_dir` 引用了它，omegaconf 解析不到直接退出。
> config 里 `resume: true`，新机器目录为空时就是从 step 0 开始；**checkpoint 目录和 FSDP2 那条线必须分开**。

启动后先确认三件事，再放手：`MoE+EP spec ... EP=8 ... router_dtype=fp32`、
无 `Traceback` / `HSA_STATUS`、首步在约 14 分钟内出 `callbacks: step=1`。
监控 / 停止 / 续跑照 §11，停止时记得连 Ray actor 一起清：

```bash
sudo docker exec "$CONTAINER" bash -lc \
  'ray stop --force; pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"; \
   pkill -9 -f "[E]ngineCore"; sleep 10'
```

### 13.14 健康判据与当前已知状态

**先说清楚状态**：`2bbdfde` 修完之后，**resp=20480 的长跑还没有跑过完整的观测窗口**。
下面把"修之前测到什么"和"修之后测到什么"分开列，别把两组数混着用。

**修之前（有 EOS 对齐 bug）：长度会崩，可复现、跨配置、跨拓扑。**

| run | 配置 | 形状 |
|---|---|---|
| `lepges5o` | EP=8, batch 2048, resp 20480 | step 1–100 健康（acc 0.295→0.516、resp_len 844→2150），step 101–120 `seq/max_len` 掉到 11074、`resp_len` 回落到 1884、acc 0.503 |
| `p819dsox` | TP2·PP2·CP2·EP2·ETP2, batch 512, fp32 router | step 1–140 健康（acc 0.15→0.54、长度增长），step 142 起 `max_len` 单调收缩 18099→10014→5629→3723→1896，acc 掉到 0.28 |
| 同期 FSDP2 `z8aeo3cf` | 同配方 | step 101–120 `resp_len` 3204 →（121–140）4253，**在涨**；`seq/max_len` 打满 20480 预算；acc 0.539→0.560 |

崩塌的特征是 **`seq/max_len` 单调收缩**，转折点在 step ~140 附近，而且两次 batch 差 4 倍
（2048 / 512）转折点都在 140 —— **它跟的是 optimizer step 数，不是样本数**。

**修之后**：在一个把响应预算压到 4096、batch 压到 256 的配方上连跑 91 步（config 也在仓库里，
`..._verlref_4k_longrun.yaml`，专门用来把这段观测从一天缩到几小时），三个信号都反向：

| 区块 | `reward/accuracy` | `seq/mean_response_len` | `rollout_corr/kl` |
|---|---|---|---|
| 1–20 | 0.168 | 773 | 0.00136 |
| 21–40 | 0.366 | 800 | 0.00083 |
| 61–80 | 0.42 | 880 | 0.00065 |
| 81–91 | 0.42 | 925 | 0.00060 |

AIME-2024 在线验证（greedy，独立于训练 batch）：

| step | 20 | 40 | 80 |
|---|---|---|---|
| `val-core/acc/mean@1` | 0.086 | 0.159 | 0.199 |
| `val/response_length_mean` | 1319 | 1513 | 1514 |

**关键判据，按重要性排序：**

1. **`seq/max_len` 不收缩。**这是长度崩塌的直接指标。它在预算上限附近波动是健康的
   （说明每步都有序列打满），单调往下走就是崩了。
2. **`rollout_corr/kl` 不随步数单调爬升。**降下去正常（策略变确定，log 空间分歧自然缩小）；
   爬上去有三种可能，按概率排：router 精度两侧不匹配（§13.11）、权重同步漏参数（§13.4 的
   `VERIFY=1` 复查）、或者又出现了对齐类 bug（§13.16 的落盘探针）。
   健康水平是 **6e-4 ~ 1.8e-3**，和 FSDP2 同期同量级。
3. `reward/accuracy` 与 `val-core/acc/mean@1` 单调改善；`val/response_length_mean` 增长
   （模型学会想更久，和 §13.7 的 FSDP2 一样）。
4. `mem/actor_allocated_gb` 恒定（Megatron 4k 下约 72GB，20k 下约 130GB）。
   `max_reserved` 随每步 batch 波动是正常的，**存活内存在动才是泄漏**。
5. **已知会有熵坍缩**，和 §13.7 同理（`entropy_coeff=0` 是 verl recipe 本身的选择）。
   4k 那 91 步 entropy 从 0.71 掉到约 0.19。只有在"entropy 掉到 0.05 以下 **且** 长度开始缩"
   同时出现时才需要警惕。

### 13.15 Megatron 专属排障

**显存：`HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `torch.OutOfMemoryError` in `loss_func`**

resp=20480 这个配置**在 `max_tokens_per_gpu: 22528` 下会在 step 14 死在 actor backward**。
关键结论是：**崩溃不是 allocated 峰值的问题**（改前后都是约 130GB），而是**碎片** ——
ROCm 没有 `expandable_segments`，原来每步约 7 个打满 22.5k 的 bin 反复申请释放巨块，
reserved 比 allocated 多出 42GB。把 bin 压到 8192 之后碎片间隙塌到 4–11GB，
**峰值 reserved 从 177GB 降到 134GB**。所以长跑 config 里那个 8192 不能随手调回去。

典型报错点是 `_pp_update_policy` 的 `loss_func` 里
`logits = lt.reshape(-1, lt.shape[-1]).float() / temperature` —— packed logits 的 fp32 上采，
一个 6k token 的 bin 就是 3.44GiB。看到这一行 OOM，先确认是不是**别的进程占了卡**
（`rocm-smi --showmeminfo vram`；`GPU% = 0` 但 `VRAM%` 不为 0 就是有人占着不算），
再考虑调 `max_tokens_per_gpu` / `gpu_memory_utilization` / `recompute_granularity`。

**往引擎里加探针，要加在 `_pp_update_policy`，不是基类**

```python
# megatron_native_engine.py
if self._pp == 1 and self._cp == 1 and not getattr(self, "_is_moe", False):
    return super().engine_update_policy(batch)
return self._pp_update_policy(batch)
```
**MoE 永远走 `_pp_update_policy`**，即使 PP=1、CP=1。基类的 `engine_update_policy` 对这条线是死代码。
`_row_policy_loss` 是两条路共用的。

**往 trainer 里加逻辑，先确认在哪个函数里**

`train()` 在开头就 `if self._use_ray_controller: self._train_with_ray_controller(); return`。
Ray 路径不经过 `train()` 的主体 —— `a09b702` 之前那个 TIS 补丁就是打在了 `train()` 里，
成了死代码，这个坑踩两次就够了。

**env 变量传不到 Ray actor**

actor 创建时没带 `runtime_env`，driver 侧 export 的变量到不了 actor。
诊断探针因此用 sentinel 文件绕开（`megatron_base_engine._gap_dump_dir()`）：

```bash
# 打开 rollout_corr/kl 的落盘探针（容器内）
sudo docker exec "$CONTAINER" bash -lc 'echo /path/to/dump > /tmp/lumenrl_gap_dump_dir'
# 跑完记得删掉，否则每步都写
sudo docker exec "$CONTAINER" bash -lc 'rm -f /tmp/lumenrl_gap_dump_dir'
```
落盘的是引擎侧三张对齐并 mask 过的 per-token 张量（train / old / rollout）加上这个 rank
上报的标量，所以可以从原始张量把指标算术复现出来（恒等式 `rc − ppo = mean(rollout_lp − old_lp)`
是精确成立的，实测 +0.031807 vs 报告 +0.031779，可以用它自检 dump 有没有取错）。
当时用的四个分析脚本（`analyze_engine_gap.py` / `analyze_logprob_gap.py` /
`wandb_fetch_run.py` / `wandb_analyze.py`）不在仓库里，只在那台机器的 run area，
用途和判读见 `megatron-moe-length-collapse-handoff.md` §8。

**vLLM worker 里的 `logger.info` 不进 driver 日志**

所以 §13.4 那行 `weight sync coverage` 在这条线上看不到。**不能**据此认为断言没跑 ——
判断断言是否触发，看它有没有抛异常。

**`rollout_log_probs` 里出现精确的 `0.0`，是不是坏了？**

看日志那行 INFO 里的均值。`a09b702` 之后 trainer 会打：

```
rollout_log_probs: 522 of 206171 response positions are exactly 0.0;
substituted old_log_probs (mean -2.7e-07 there -- ...)
```
均值**接近 0** 说明这些 token 真的极度确信（float32 下 `log p` 下溢到 0，需要 p > 1−6e-8），
回填成 `old_log_probs` 是无害的 no-op。均值**很负**（历史上测到 −52.3）才是真的上报缺口，
那说明有位置对齐问题，回 §13.12。

---

## 14. FP8 底层 patch 清单（供排查，正常无需手动）

- **训练侧**（Lumen `cfg.enable`，monkey-patch，`_apply_lumen_fp8`）：FP8 blockwise2d 线性 GEMM
  (`nn.Linear.forward`) + model-sensitive RMSNorm(模块替换) + **lm_head 反 patch 回 BF16**
  (CK blockscale INT32 溢出)。**关掉**:FP8ParamManager(fp32-master 冲突)、hf_attn_patch(用 native
  纯 SDPA packed 前向)、fp8_attn(=none)。可选 `LUMENRL_FP8_VIA_VERL=1` 直接委托 verl `_maybe_apply_lumen`。
- **rollout 侧**（本仓库 `lumenrl/engine/inference/vllm_fp8_utils.py`，非 verl）：vLLM 在线 per-block
  ROCm-safe 权重处理 + layerwise reload;worker-ext IPC 权重更新时对 bucket tensor **`.clone()`**
  (否则在线 reload 持有被覆盖的共享 IPC buffer 视图 → 权重更新后塌缩) + `monkey_patch_model`(OOV logit mask)。
- **vLLM wheel**（第 5 节）：AITER RMSNorm `use_model_sensitive_rmsnorm=1`，**新容器需重打**。
- **ATOM rollout 侧**（`MODE=atomfp8`，仓库 `ATOM/atom/`，已随代码提交）：
  - `atom/model_ops/layernorm.py` 的 `rmsnorm2d_fwd_` / `rmsnorm2d_fwd_with_add_` 传
    `use_model_sensitive_rmsnorm=1`(与 vLLM RMSNorm patch 等价的 T5-like 对齐;否则 `rollout_corr/kl` ~0.007)。
  - ATOM 使用本地 `aiter` 源码 JIT；`aiter/3rdparty/composable_kernel` submodule 必须拉取。
    首次运行前建议按 §8.2 预编译 `module_rmsnorm`。首次 ATOM FP8 rollout 还可能继续编译
    `module_gemm_a8w8_blockscale_bpreshuffle`、`module_activation`、`module_sample`、`module_rope_*`
    / `module_cache` 等小模块；这些是环境预热成本。
  - `atom/rollout/weight_updater.py`：每次 ZMQ CUDA-IPC 权重更新把 BF16 训练权重**在线重量化**成 128×128
    block FP8(`quantize_weight_to_fp8_128x128_blockscale`),与 vLLM `fp8_per_block` 同理。
  - **no-eager level=3 正式方案**：`policy.generation.vllm_cfg.enforce_eager=false`、
    `policy.generation.atom_cfg.engine_kwargs.enforce_eager=false`、
    `policy.generation.atom_cfg.engine_kwargs.compilation_config.level=3`；同时开启
    `ATOM_ISOLATE_TORCH_COMPILE_CACHE=1`（每 replica 独立 compile cache）与
    `policy.generation.vllm_cfg.enable_sleep_mode=true`、`sleep_level=2`（训练前释放 rollout 显存）。
  - `lumenrl/engine/inference/atom_ray_server.py` 的 `ATOMReplicaManager.create()`：给 rollout actor 的
    `runtime_env.env_vars` 注入 `TORCHDYNAMO_DISABLE=0`，判据 `_torch_compile_enabled()`
    （`compilation_config.level>0` 或 `enforce_eager=false`，与 compile-cache 隔离同一判据）。
    这样 no-eager level=3 需要的 dynamo 只落在 rollout 进程，FSDP2 训练 actor 保持
    `TORCHDYNAMO_DISABLE=1`；**不需要（也不应）在启动脚本里全局 export**。
  - aiter shim `examples/DAPO/atom_aiter_shim/sitecustomize.py`:补齐 ATOM 所需的 aiter FP8/MLA 子模块
    (`run_dapo.sh` 自动挂到 PYTHONPATH,`LUMENRL_ATOM_AITER_SRC` 默认取 `$AITER_DIR`)。

---

## 15. verl 运行（复用同一环境：BF16 基线 + FP8 rollout）

> 本节让**同一台机器、同一个容器**在原生 LumenRL 之外，再跑通 **verl `recipe/dapo`**（DAPO 动态采样）的
> 两条路线，**完全复用前面第 3–6 节**的容器 (`$CONTAINER`)、模型 (`Qwen3-8B-Base`) 与过滤后的数据
> (`data_cached/qwen3-8b-maxprompt1024/*.filtered.parquet`)，**无需重建容器、无需重下模型/数据**：
>
> - **BF16 基线**：`recipe.dapo.main_dapo` + BF16 FSDP 训练 + vLLM BF16 async rollout（AITER off，`quantization=null`）。
> - **FP8 rollout + ATOM 对齐训练**：`lumen.rl.verl.verl_entry`（`LUMEN_VERL_MAIN=dapo` 委托 dapo，保留 filter_groups）
>   + FSDP2 actor 的 ATOM 对齐前向（norm/sdpa/BF16 TunedGemm linear/mlp）
>   + vLLM async rollout `fp8_per_block` + AITER unified attention。
>   **⚠️ 训练侧是 BF16 不是 FP8**：`LUMEN_ROLLOUT=ATOM` 让 `LumenConfig.enable()` 在
>   `lumen/config.py:253-260` 提前 `return None, model`，FP8 量化（第 262 行之后）走不到。
>   要训练侧真 FP8 就删掉 `LUMEN_ROLLOUT=ATOM`，并用 §15.5 的指纹判据核实。
>
> 关键点：verl 与 Lumen/aiter 一样放在 `$RL_ROOT` 下（`$RL_ROOT/verl`），第 4 节 `-v "$RL_ROOT":"$RL_ROOT"`
> 已把它挂进容器，因此**不用改容器**。`Lumen`(`amd-atom-rollout`)、`aiter`(`lumen/triton_kernels`) 已在第 3 节
> clone，分支正好是 FP8 路线所需，直接复用；FP8 rollout 复用第 5 节的 vLLM AITER RMSNorm patch。
> **已验证**：8×MI350X, Qwen3-8B-Base, 两条路线 smoke 各 1 步全过、exit 0、无 Traceback。

### 15.1 追加拉取 verl（含 recipe 子模块 = DAPO 动态采样 trainer）

| 仓库 | 分支/pin | 用途 |
|---|---|---|
| `verl` | `amd-v0.8.0` | unified engine + rollout `fp8_per_block` + worker-side Lumen hook (`8a8a9a8`) |
| `recipe`（verl 子模块） | pin `e7f8895` | DAPO 动态采样 trainer（`recipe/dapo`），随子模块拉取 |

非国内网络直接从 GitHub 拉取：

```bash
cd "$RL_ROOT"
git clone -b amd-v0.8.0 https://github.com/xysheng-AMD/verl.git
git -C verl submodule update --init --depth 1 recipe
test -f verl/recipe/dapo/main_dapo.py && echo "recipe/dapo OK" || echo "MISSING recipe/dapo"
```

中国内网机器同第 3 节用代理镜像（不写死进 remote）：

```bash
cd "$RL_ROOT"
GHP=https://gh-proxy.com/https://github.com
git -c http.version=HTTP/1.1 clone --single-branch -b amd-v0.8.0 "$GHP/xysheng-AMD/verl.git"
cd "$RL_ROOT/verl"
git -c http.version=HTTP/1.1 \
  -c url."$GHP/".insteadOf=https://github.com/ \
  submodule update --init --depth 1 recipe
test -f recipe/dapo/main_dapo.py && echo "recipe/dapo OK" || echo "MISSING recipe/dapo"
```

> `recipe/dapo`（动态采样 DAPO trainer）只在 verl 的 git 子模块里，不初始化就没有动态采样。
> `verl amd-v0.8.0` HEAD 应含 `8a8a9a8 _build_model_optimizer hook + lm_head BF16 workaround + .clone() fix`
> （FP8 E2E 必需：把 Lumen FP8 注入移进每个 Ray worker 的 `_build_model_optimizer`，并加 logits `.clone()`
> 与 lm_head 保 BF16）。用 `git -C "$RL_ROOT/verl" log --oneline -1` 确认。

### 15.2 在同一容器装 verl（不覆盖镜像内 vLLM/torch）

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
git config --global --add safe.directory "$RL_ROOT/verl" || true
cd "$RL_ROOT/verl"
python3 -m pip install -e . --no-deps
python3 -m pip install -r requirements.txt --upgrade-strategy only-if-needed
'
# 验证：verl / recipe.dapo 导入 + FP8 依赖 Lumen/aiter 导入 + vLLM patch 生效
sudo docker exec "$CONTAINER" bash -lc '
export PYTHONPATH=$RL_ROOT/verl:$RL_ROOT/aiter:$RL_ROOT/Lumen
python3 - <<PY
import verl, torch, vllm, ray, importlib, inspect
print("verl", verl.__file__, "| torch", torch.__version__, "vllm", vllm.__version__, "ray", ray.__version__, "gpus", torch.cuda.device_count())
importlib.import_module("recipe.dapo.main_dapo"); print("recipe.dapo OK")
import lumen.quantize            # 先经 lumen.quantize 进入，避免 ops.quantize 直入触发循环 import
from lumen.ops.quantize.ops import quant_fp8_blockwise_impl
importlib.import_module("lumen.rl.verl.verl_entry"); print("lumen verl_entry OK")
from vllm.kernels import aiter_ops as k
ms = all("use_model_sensitive_rmsnorm=1" in inspect.getsource(getattr(k,a)) for a in ["_rms_norm_impl","_rocm_aiter_rmsnorm2d_fwd_with_add_impl"])
print("vLLM RMSNorm model-sensitive patch:", ms, "(FP8 需为 True)")
PY
'
```
> - `requirements.txt` 会把 **numpy 降到 <2.0.0**（verl 硬性要求，例如 1.26.4）。这对 verl 与 LumenRL 都能正常
>   跑；只是 `opencv-python-headless` 会有一条 `numpy>=2` 的告警，可忽略。若想彻底隔离，也可单独给 verl
>   建一个同镜像的容器（`CONTAINER=rl-vllm-verl`，其余同第 4 节），但通常不必。
> - FP8 E2E 依赖 Lumen/aiter 与第 5 节 vLLM RMSNorm patch。若第 5 节已在本容器打过，则 `patch: True`；否则先跑
>   第 5 节的 `patch_vllm_aiter_rmsnorm.py`（BF16 路线不需要，AITER off）。
> - `lumen.ops.quantize.ops` 若被**直接**首个导入会报“partially initialized … QuantizedLinearFunction”循环
>   import；这是导入顺序假象。真实运行经 `lumen.quantize` 先进入即可，`verl_entry` 导入正常。

### 15.3 BF16 smoke 脚本 `$RL_ROOT/run_smoke_dapo.sh`

```bash
cat > "$RL_ROOT/run_smoke_dapo.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
WORKDIR=$RL_ROOT/verl
MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base
DC=$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024
EXP_NAME="smoke-bf16-dapo-$(date +%Y%m%d-%H%M%S)"
cd "$WORKDIR"; export PYTHONPATH=$WORKDIR
export HF_HOME=$DATA_ROOT/hf_home HF_HUB_CACHE=$DATA_ROOT/cache/hub HF_DATASETS_CACHE=$DATA_ROOT/cache/datasets
export RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 VLLM_USE_V1=1
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_LOGGING_LEVEL=WARN
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_LINEAR=0 VERL_VLLM_ROCM_USE_AITER=0 VERL_VLLM_ROCM_USE_AITER_LINEAR=0
export VERL_EMPTY_CACHE_PER_MICRO_BATCH=1
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF 2>/dev/null || true
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
python3 -m recipe.dapo.main_dapo \
  data.train_files="$DC/dapo-math-17k.filtered.parquet" data.val_files="$DC/aime-2024.filtered.parquet" \
  data.prompt_key=prompt data.truncation=left data.return_raw_chat=True data.filter_overlong_prompts=False \
  data.max_prompt_length=1024 data.max_response_length=2048 data.train_batch_size=8 data.gen_batch_size=24 data.seed=10086 \
  actor_rollout_ref.rollout.n=8 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=True algorithm.filter_groups.metric=acc algorithm.filter_groups.max_num_gen_batches=10 \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_is_batch_normalize=false algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.rollout_rs_threshold=null \
  actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=10 actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.seed=10086 actor_rollout_ref.actor.data_loader_seed=10086 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.quantization=null actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.use_torch_compile=False actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true actor_rollout_ref.ref.fsdp_config.seed=10086 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  reward.reward_manager.name=dapo reward.reward_kwargs.overlong_buffer_cfg.enable=True reward.reward_kwargs.overlong_buffer_cfg.len=512 \
  reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 reward.reward_kwargs.overlong_buffer_cfg.log=False reward.reward_kwargs.max_resp_len=2048 \
  trainer.logger="['console']" trainer.project_name=AMD-BF16-VERL trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.resume_mode=disable trainer.log_val_generations=0
EOF
chmod +x "$RL_ROOT/run_smoke_dapo.sh"
```

### 15.4 FP8 rollout smoke 脚本 `$RL_ROOT/run_smoke_dapo_fp8.sh`（训练侧 BF16，见脚本内注释）

相对 14.3：入口换成 `lumen.rl.verl.verl_entry`（委托 dapo），actor `fsdp2` + Lumen ATOM 对齐前向
（**训练侧仍是 BF16**，`LUMEN_FP8=1` 被 `LUMEN_ROLLOUT=ATOM` 的提前 return 跳过），rollout
`fp8_per_block` + AITER unified attention。依赖第 3 节的 `Lumen`/`aiter` 与第 5 节 vLLM patch。

```bash
cat > "$RL_ROOT/run_smoke_dapo_fp8.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
WORKDIR=$RL_ROOT/verl; LUMEN_DIR=$RL_ROOT/Lumen; AITER_DIR=$RL_ROOT/aiter
MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base
DC=$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024
SEED=${SEED:-10086}; EXP_NAME="smoke-lumen-fp8pb-atom-mha-dapo-$(date +%Y%m%d-%H%M%S)"
CKPTS_DIR="$DATA_ROOT/ckpts/$EXP_NAME"; mkdir -p "$CKPTS_DIR"
cd "$WORKDIR"; touch recipe/dapo/config/__init__.py 2>/dev/null || true
export PYTHONPATH=$WORKDIR:$AITER_DIR:$LUMEN_DIR MODEL_NAME=$MODEL_PATH
export HF_HOME=$DATA_ROOT/hf_home HF_HUB_CACHE=$DATA_ROOT/cache/hub HF_DATASETS_CACHE=$DATA_ROOT/cache/datasets
export RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 VLLM_USE_V1=1
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_LOGGING_LEVEL=WARN
export VERL_EMPTY_CACHE_PER_MICRO_BATCH=1
# Lumen actor ATOM 对齐 patch + 委托 dapo。
#
# ⚠️ 训练侧是 BF16，不是 FP8。LUMEN_ROLLOUT=ATOM 让 LumenConfig.enable()
#    (lumen/config.py:253-260) 走 ATOM 分支并 `return None, model`，而 FP8 量化在
#    第 262 行之后的「3. FP8 linear quantization」，永远走不到。所以这里的
#    LUMEN_FP8=1 / LUMEN_FP8_SCALING=blockwise2d 对 actor 无效，只有
#    norm/sdpa/linear/mlp 四个 ATOM 对齐 patch 生效（linear 是 BF16 TunedGemm）。
#    本脚本的准确描述是「FP8 rollout（vLLM fp8_per_block）+ BF16 ATOM 对齐训练」。
#    要训练侧真 FP8：删掉 LUMEN_ROLLOUT=ATOM（其余不动），并用 §15.5 的指纹判据核实。
export LUMEN_VERL_MAIN=dapo LUMEN_FP8=1 LUMEN_ROLLOUT=ATOM LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_BLOCK_SIZE=128
export LUMEN_FP8_ATTN=mha LUMEN_FP8_QUANT_TYPE=blockwise LUMEN_ATTN_BACKEND=auto LUMEN_FORCE_FSDP=1
export LUMEN_ACTOR_PATCH_NORM=1 LUMEN_ACTOR_PATCH_SDPA=1 LUMEN_ACTOR_PATCH_LINEAR=1 LUMEN_ACTOR_PATCH_MLP=1
# vLLM rollout FP8 + AITER unified attention
export ROLLOUT_QUANTIZATION=fp8_per_block KV_CACHE_DTYPE=auto CALCULATE_KV_SCALES=False VLLM_ROLLOUT_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VERL_VLLM_ROCM_USE_AITER=1 VERL_VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_FP8_PADDING=1 VLLM_FP8_ACT_PADDING=1 VLLM_FP8_WEIGHT_PADDING=1 VLLM_FP8_REDUCE_CONV=1
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF 2>/dev/null || true
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
python3 -m lumen.rl.verl.verl_entry \
  data.train_files="$DC/dapo-math-17k.filtered.parquet" data.val_files="$DC/aime-2024.filtered.parquet" \
  data.prompt_key=prompt data.truncation=left data.return_raw_chat=True data.filter_overlong_prompts=False \
  data.max_prompt_length=1024 data.max_response_length=2048 data.train_batch_size=8 data.gen_batch_size=24 data.seed="$SEED" \
  actor_rollout_ref.rollout.n=8 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=True algorithm.filter_groups.metric=acc algorithm.filter_groups.max_num_gen_batches=10 \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_is_batch_normalize=false algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.rollout_rs_threshold=null \
  actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=10 actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.seed="$SEED" actor_rollout_ref.actor.data_loader_seed="$SEED" \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.quantization="$ROLLOUT_QUANTIZATION" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype="$KV_CACHE_DTYPE" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.calculate_kv_scales="$CALCULATE_KV_SCALES" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config="{backend: $VLLM_ROLLOUT_ATTENTION_BACKEND}" \
  actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.use_torch_compile=False actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true actor_rollout_ref.ref.fsdp_config.seed="$SEED" actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  reward.reward_manager.name=dapo reward.reward_kwargs.overlong_buffer_cfg.enable=True reward.reward_kwargs.overlong_buffer_cfg.len=512 \
  reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 reward.reward_kwargs.overlong_buffer_cfg.log=False reward.reward_kwargs.max_resp_len=2048 \
  trainer.logger="['console']" trainer.project_name=AMD-BF16-VERL trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.default_local_dir="$CKPTS_DIR" trainer.resume_mode=disable trainer.log_val_generations=0
EOF
chmod +x "$RL_ROOT/run_smoke_dapo_fp8.sh"
```

### 15.5 跑 Smoke（先各 1 step 验证整链路）

```bash
sudo docker restart "$CONTAINER"; sleep 6            # 清残留 ray 进程
# BF16 smoke
sudo docker exec -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" "$CONTAINER" \
  bash -lc "bash $RL_ROOT/run_smoke_dapo.sh" 2>&1 | tee $DATA_ROOT/logs/verl_smoke_bf16_$(date +%Y%m%d-%H%M%S).log

# FP8 rollout smoke（训练侧 BF16；要真 FP8 见 §15.4 注释）
sudo docker restart "$CONTAINER"; sleep 6
sudo docker exec -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" "$CONTAINER" \
  bash -lc "bash $RL_ROOT/run_smoke_dapo_fp8.sh" 2>&1 | tee $DATA_ROOT/logs/verl_smoke_fp8_$(date +%Y%m%d-%H%M%S).log
```

期望证据（已验证：8×MI350X, Qwen3-8B-Base, 各 1 步, exit 0, 无 Traceback）：

> 下表第三列是 §15.4 脚本的原样行为，即 **FP8 rollout + BF16 ATOM 对齐训练**（`LUMEN_ROLLOUT=ATOM`
> 使 actor FP8 量化被跳过，见 §15.4 脚本内注释）。第四列是删掉 `LUMEN_ROLLOUT=ATOM` 后的
> **训练侧真 FP8**，2026-07-29 在 8×MI355X 实测。

| 证据 | BF16 | ATOM 对齐（actor BF16） | 训练侧真 FP8 |
|---|---|---|---|
| `step:1 ... actor/loss ... actor/grad_norm` | loss≈-0.093, grad_norm≈1.09 | loss≈0.028, grad_norm≈0.89 | loss≈0.084, grad_norm≈1.04 |
| `rollout_corr/rollout_is_mean` | ≈1.0（0.9999） | ≈1.0（1.0000） | ≈1.0（0.9994） |
| `rollout_corr/kl` | ≈0.001 | **≈0.005**（rollout FP8 gap；逼近 2.0 才警惕） | ≈0.004（长跑稳态降到 0.0010–0.0012） |
| `train/num_gen_batches` | ≥1（filter_groups 动态采样生效） | ≥1（同） | ≥1（同） |
| Lumen 注入生效 | — | 8 worker `[verl] Lumen optimizations applied (actor/full) before FSDP2 wrapping` | 同左 |
| 训练侧 FP8 判据 | 无 | **无**（见下方指纹检查） | `Restored lm_head` ×8 + 指纹检查为真 |
| rollout FP8 | — | vLLM `quantization=fp8_per_block`；`kernel_unified_attention_3d` JIT | 同左 |
| 收尾 | `Final validation metrics: None`（未测评） | 同左；首次会 JIT 编译 aiter `module_rmsnorm` / per-block GEMM（较慢，二次快） | 同左 |

> ⚠️ `[verl] Lumen optimizations applied (actor/full)` **不能**用来判断训练侧是否 FP8。它由 verl
> worker patch 打印，与 FP8 是否启用无关，两种情况都出现 8 次。历史版本把它列为「FP8 特有」是错的。

**判断训练侧到底跑没跑 FP8（唯一可靠的方法）**——看训练 actor 加载了哪些 aiter 模块。
注意 verl 的训练进程名是 `WorkerDict`（LumenRL 侧是 `LumenActorWorker`，见 §9）：

```bash
LOG=<你的 verl 日志>
grep -a "WorkerDict" "$LOG" | grep -aoE "module_[a-z0-9_]+" | sort | uniq -c
```

- 出现 `module_gemm_a8w8_blockscale` / `module_quant` → 训练侧是**真 FP8**
- 只有 `module_rmsnorm` + `module_activation` → **ATOM 分支，训练侧 BF16**

更强的一条是运行时调用（不只是启动期导入），并且能证明 FP8 在训练侧而非 rollout 侧：

```bash
grep -a "a8w8_blockscale_tuned_gemm" "$LOG" | grep -c WorkerDict        # 训练侧 FP8 GEMM 调用数
grep -a "a8w8_blockscale_tuned_gemm" "$LOG" | grep -c vLLMHttpServer    # 应为 0
```

2026-07-29 实测真 FP8 长跑：前者 9426、后者 0；GEMM shape 为 `N:4096,K:12288`(down_proj)、
`N:12288,K:4096`(gate/up)、`N:1024,K:4096`(k/v)，M 随动态批变化——即 Qwen3-8B 的真实训练层。

### 15.6 长跑规模（相对 smoke 只放大规模/落盘）

把 14.3 / 14.4 复制为 `run_longrun_dapo.sh` / `run_longrun_dapo_fp8.sh`，按下面替换规模参数：
`max_response_length=20480`、`train_batch_size=32`、`gen_batch_size=96`、`rollout.n=16`、`ppo_mini_batch_size=32`、
`ppo_max_token_len_per_gpu=21504`、`log_prob_max_token_len_per_gpu=21504`、`gpu_memory_utilization=0.6`(FP8)/`0.30`(BF16)、
`max_num_batched_tokens=32768`、`reward.reward_kwargs.overlong_buffer_cfg.len=4096`、`reward.reward_kwargs.max_resp_len=20480`，
并在 trainer 段加：

```bash
  trainer.save_freq=20 trainer.max_actor_ckpt_to_keep=5 trainer.test_freq=10 trainer.total_training_steps=500 \
  trainer.default_local_dir="$CKPTS_DIR" trainer.log_val_generations=1
```

长跑开 wandb（脚本头部 `cd`/PYTHONPATH 之后加，复用第 3 节 `$RL_ROOT/wandb.key`，格式 `WANDB_API_KEY=xxxx`）：

```bash
WANDB_KEY_FILE=${WANDB_KEY_FILE:-$RL_ROOT/wandb.key}
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_KEY_FILE" ]; then export WANDB_API_KEY="$(cut -d= -f2- "$WANDB_KEY_FILE" | tr -d '[:space:]')"; fi
LOGGER=$([ -n "${WANDB_API_KEY:-}" ] && echo "['console','wandb']" || echo "['console']")
export WANDB_DIR=$DATA_ROOT/wandb
```

并把 `trainer.logger` 改为 `trainer.logger="$LOGGER"`（project `AMD-BF16-VERL`）。后台起长跑：

```bash
sudo docker restart "$CONTAINER"; sleep 6
LOG=$DATA_ROOT/logs/verl_longrun_fp8_$(date +%Y%m%d-%H%M%S).log
sudo docker exec -d -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" "$CONTAINER" \
  bash -lc "bash $RL_ROOT/run_longrun_dapo_fp8.sh > $LOG 2>&1"
echo "LOG=$LOG"; sleep 200
grep -aE "step:[0-9]+ -" "$LOG" | tail -1      # 最新 step 指标
# 停止：sudo docker exec "$CONTAINER" bash -lc "pkill -9 -f 'main_[d]apo|verl_entry'; ray stop --force"
```

### 15.7 verl vs LumenRL / BF16 vs FP8 一览

> ⚠️ 第三列（§15.4 脚本原样）历史上被标为「FP8 E2E」，但**训练侧其实是 BF16**：
> `LUMEN_ROLLOUT=ATOM` 让 `LumenConfig.enable()` 在 `lumen/config.py:253-260` 提前
> `return None, model`，FP8 量化（第 262 行之后）走不到。要真 FP8 训练见 §15.4 注释与
> §15.5 指纹判据。

| 维度 | verl BF16 | verl ATOM 对齐（actor BF16） | LumenRL 原生（§1） |
|---|---|---|---|
| 入口 | `recipe.dapo.main_dapo` | `lumen.rl.verl.verl_entry`（→ dapo） | `lumenrl.trainer.main` |
| actor 策略 | FSDP | FSDP2 + Lumen ATOM(norm/sdpa/linear/mlp) | Lumen FSDP2 |
| actor 量化 | 无（BF16） | **无（BF16）**——ATOM 对齐前向：norm/sdpa/**BF16 TunedGemm** linear/mlp；`LUMEN_FP8=1` 被提前 return 跳过 | 可选 FP8 blockwise2d |
| rollout | vLLM async BF16，`quantization=null`，AITER off | vLLM async，`quantization=fp8_per_block`，AITER unified attention | vLLM/ATOM AsyncLLM |
| 动态采样 | filter_groups (acc) | filter_groups (acc)（保留） | `algorithm.dapo.filter_groups` |
| 需手动 patch | 无 | 复用第 5 节 vLLM AITER RMSNorm patch | 第 5 节同 patch |
| 复用资源 | §3 clone(+verl)、§4 容器、§6 模型/数据 | 同 + §3 的 Lumen/aiter | §3–§6 |

> 排障速查：FP8 若 `rollout_corr/kl` 异常偏大或发散，多半是第 5 节 vLLM RMSNorm patch 未打（新容器需重打），
> 或 Lumen/aiter 未在 `PYTHONPATH`。BF16 路线 `VLLM_ROCM_USE_AITER=0`，不依赖该 patch。
