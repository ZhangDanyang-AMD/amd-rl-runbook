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
git -C Lumen-RL checkout 523e92329d312a3265e0a031dd7982b0529c3ef5
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
git -C Lumen-RL checkout 523e92329d312a3265e0a031dd7982b0529c3ef5
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
| `Lumen-RL` | `dev/vllm-fsdp-dapo` @ `523e92329d312a3265e0a031dd7982b0529c3ef5` | RL 主框架 |
| `Lumen` | `amd-atom-rollout` | FSDP2 训练后端（FP8） |
| `aiter` | `lumen/triton_kernels` | AMD kernel |
| `ATOM` | `lumen-rl` | **仅 `MODE=atomfp8`**：ATOM rollout 引擎；BF16/vLLM-FP8 路线不需要 |

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

启动脚本 **已在 Lumen-RL 仓库内**:`examples/DAPO/run_dapo.sh`(`git clone` 即得)。用 `MODE`(bf16|fp8|atomfp8)、
`TRAIN_FP8`(0|1)、`STEPS` 控制,**没有重复的 env 块**;所有路径走 `$RL_ROOT`/`$DATA_ROOT` 变量+仓库标准布局
自动定位,**无任何机器专属路径**。内容如下(若仓库缺失可用此 heredoc 重建):

```bash
cat > "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh" <<'EOF'
#!/usr/bin/env bash
# 统一 DAPO 启动：MODE=bf16|fp8|atomfp8, TRAIN_FP8=0|1, STEPS=N。路径取容器内 $RL_ROOT/$DATA_ROOT。
set -uo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
MODE="${MODE:-bf16}"; TRAIN_FP8="${TRAIN_FP8:-0}"; STEPS="${STEPS:-1000}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-8B-Base}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet}"
RUN_ID="${RUN_ID:-${MODE}$([ "$TRAIN_FP8" = 1 ] && echo -e2e)-ray-vllm-8b-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
USER_PYTHONPATH="${PYTHONPATH:-}"
# 仓库定位：标准 clone 布局 $RL_ROOT/<repo>，回退到 Lumen-RL/third_party/<repo>（无机器专属路径）。
LUMEN_DIR="${LUMEN_DIR:-$RL_ROOT/Lumen}"
if [ ! -f "$LUMEN_DIR/lumen/config.py" ]; then
  LUMEN_DIR="$RL_ROOT/Lumen-RL/third_party/Lumen"
fi
AITER_DIR="${AITER_DIR:-$RL_ROOT/aiter}"
if [ ! -d "$AITER_DIR/aiter" ]; then
  AITER_DIR="$RL_ROOT/Lumen-RL/third_party/aiter"
fi
ATOM_DIR="${ATOM_DIR:-$RL_ROOT/ATOM}"
if [ ! -f "$ATOM_DIR/atom/rollout/async_engine.py" ]; then
  ATOM_DIR="$RL_ROOT/Lumen-RL/third_party/ATOM"
fi
# Checkpoint dir is STABLE per-mode (no timestamp) so resume works across relaunches.
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/ckpts/lumenrl-dapo/${MODE}$([ "$TRAIN_FP8" = 1 ] && echo -e2e)-ray-vllm-8b}"
cd "$RL_ROOT/Lumen-RL"

# ---- 通用 env（BF16/FP8/ATOM 共用）----
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN ATOM_DISABLE_VLLM_PLUGIN=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LUMEN_DISABLE_HF_ATTN_PATCH=1 MODEL_NAME="$MODEL_PATH"
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" LUMENRL_LOG_LEVEL=INFO
export PYTHONPATH="$RL_ROOT/Lumen-RL:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:${PYTHONPATH:-}"
for _wandb_key in "$RL_ROOT/wandb.key" "$RL_ROOT/../wandb.key"; do
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_wandb_key" ]; then
    export WANDB_API_KEY="$(cut -d= -f2- "$_wandb_key" | tr -d '[:space:]')"
  fi
done

EXTRA_ARGS=()
if [ "$MODE" = "atomfp8" ] || [ "$MODE" = "atom_fp8" ]; then
  if [ "${ATOM_DEBUG:-0}" = "1" ]; then
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_debug.yaml
  else
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_longrun.yaml
  fi
  ATOM_ONLINE_QUANT="${ATOM_ONLINE_QUANT:-per_block_fp8}"
  unset ATOM_DISABLE_VLLM_PLUGIN
  export LUMENRL_ATOM_AITER_SRC="${LUMENRL_ATOM_AITER_SRC:-$AITER_DIR}"
  export PYTHONPATH="$RL_ROOT/Lumen-RL/examples/DAPO/atom_aiter_shim:$RL_ROOT/Lumen-RL:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:$USER_PYTHONPATH"
  # ATOM FP8 正式方案：no-eager + level=3。每个 colocated replica 独立
  # torch compile cache，避免 8 个 rank0 同时写同一路径导致 Inductor rename race。
  export TORCHDYNAMO_DISABLE=0 ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
  export ATOM_TORCH_COMPILE_CACHE_ROOT="${ATOM_TORCH_COMPILE_CACHE_ROOT:-/tmp/atom_torch_compile_cache}"
  export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
  # 训练侧与 vLLM fp8 完全一致（Lumen FP8 blockwise2d + norm）；rollout 换成 ATOM per-block FP8。
  unset LUMEN_ROLLOUT
  export LUMEN_DISABLE_HF_ATTN_PATCH=1 LUMEN_NORM=1
  if [ "$TRAIN_FP8" = "1" ]; then
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_BLOCK_SIZE=128
    export LUMEN_FP8_ATTN=none LUMEN_FP8_QUANT_TYPE=blockwise LUMEN_ATTN_BACKEND=auto
    export LUMEN_FP8_WGRAD="${LUMEN_FP8_WGRAD:-0}"
  fi
  EXTRA_ARGS+=(
    policy.generation.atom_cfg.online_quant_config.global_quant_config="$ATOM_ONLINE_QUANT"
    policy.generation.vllm_cfg.enforce_eager=false
    policy.generation.atom_cfg.engine_kwargs.enforce_eager=false
    policy.generation.atom_cfg.engine_kwargs.compilation_config.level=3
    policy.generation.vllm_cfg.enable_sleep_mode=true
    policy.generation.vllm_cfg.sleep_level=2
  )
elif [ "$MODE" = "fp8" ]; then
  CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml
  # rollout per_block_fp8 + AITER unified attention
  export LUMENRL_FP8_PER_BLOCK=1
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_LINEAR=0
  if [ "$TRAIN_FP8" = "1" ]; then    # FP8 E2E 训练（blockwise2d，param manager 必须关）
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0 LUMEN_NORM=1
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_BLOCK_SIZE=128 LUMEN_FP8_ATTN=none
  fi
else
  CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_longrun.yaml
  export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
fi
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

echo "$LOG" > /tmp/run_dapo_log.txt
echo "=== MODE=$MODE TRAIN_FP8=$TRAIN_FP8 STEPS=$STEPS  CONFIG=$CONFIG  CKPT=$CKPT_DIR  LOG=$LOG ==="

# 清理旧进程（含 ATOM/vLLM ray server 与孤儿 EngineCore）
ray stop --force >/dev/null 2>&1 || true
python3 - <<'PY'
import os
import signal
import subprocess

patterns = (
    "lumenrl.trainer.main",
    "VLLMRayServer",
    "ATOMRayServer",
    "VLLM::EngineCore",
    "EngineCore",
    "spawn_main",
    "torch/_inductor/compile_worker",
    "multiprocessing.resource_tracker",
)
skip = {os.getpid(), os.getppid()}
out = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,cmd"], text=True)
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid = int(parts[0])
    stat = parts[2]
    cmd = parts[3]
    if pid in skip or "Z" in stat:
        continue
    if any(p in cmd for p in patterns):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
sleep 8

python3 -u -m lumenrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" reward.dataset="$TRAIN_FILE" val_dataset="$VAL_FILE" \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  num_training_steps="$STEPS" seed=10086 "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
exit_code=$?
echo "=== exit=$exit_code ==="
exit "$exit_code"
EOF
chmod +x "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
```

### 8.1 已验证 helper scripts（可选，本机 run area）

如果使用本 run area（`lumenrl_native_vllm_fsdp_run/`）而不是直接在 `$RL_ROOT/Lumen-RL` 内手敲
`docker exec`，可用 `scripts/` 下的两个 wrapper；它们只是把本节 `run_dapo.sh` 的 ATOM FP8
no-eager level=3 方案固化成可重复命令：

```bash
# 4k smoke：默认 3 step、batch=64、gen_batch=24、num_generations=8、no eval/no save。
# 打开 no-eager level=3 时需同时打开 sleep2。
SMOKE_STEPS=1 \
TORCHDYNAMO_DISABLE=0 \
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
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT';"   # 显式注入，避免容器内 RL_ROOT 为空

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
sudo docker exec "$CONTAINER" bash -lc '
L=$(cat /tmp/run_dapo_log.txt)
grep -aE "enforce_eager=false|compilation_config|level=3|TORCHDYNAMO_DISABLE=0|Ray ATOM rollout ready" "$L" | tail -20
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
    或等价的 ATOM engine kwargs；`TORCHDYNAMO_DISABLE=0`。
- 小配置可能关闭 `filter_groups`；若开启，`kept` 应 >0。若 `kept 0/...` + `Rollout reward: accuracy=0.0000` + 大量 `finished with reason max`,是 rollout 退化,见 §12。
- `callbacks: step=1 ... entropy=... grad_norm~0.85 ppo_kl=0`,exit 0
  - BF16:`rollout_corr/kl≈0.001`;vLLM-FP8:`≈0.003–0.004`;ATOM-FP8:`≈0.004`
- FP8 训练额外:`[verl] Restored lm_head to BF16` / `Lumen optimizations applied`、`online fp8 reload: ...weights=399`
- **不应**出现 `materialized on CPU`、`has no attribute`、entropy≈0.04 / grad_norm 1e4+(见第 12 节排障)

---

## 10. 启动长跑（`docker exec -d` 分离，防中断）

```bash
# 三选一：
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
# 显式把 RL_ROOT/DATA_ROOT 注入容器命令（从宿主 §2 变量展开），detached exec 不再依赖 §4 的 -e 注入。
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT';"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=bf16                bash '$S'"   # BF16
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8                bash '$S'"   # vLLM FP8 rollout + BF16 训练
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8     TRAIN_FP8=1 bash '$S'"   # vLLM FP8 rollout + FP8 训练
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8 TRAIN_FP8=1 bash '$S'"   # ATOM FP8 rollout + FP8 训练（no-eager level=3 + sleep2）
# W&B(可选):把 key 放 $RL_ROOT/wandb.key,格式 WANDB_API_KEY=xxxx(脚本自动读)
```

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
- 若失败后显存仍高,清理 `spawn_main` / `torch/_inductor/compile_worker` 孤儿进程,或直接
  `docker restart "$CONTAINER"`；容器 stop/start 不丢依赖。

**ATOM rollout 退化**(`MODE=atomfp8` 时 `filter_groups: kept 0/96` + `Rollout reward: accuracy=0.0000`
+ 日志大量 `finished with reason max`、无 `eos`):rollout 生成崩坏(全部答错、打满 max length)。优先检查
ATOM `layernorm.py` 的 model-sensitive RMSNorm 是否生效(见 §13);未对齐会表现为
`rollout_corr/kl` 偏大(~0.007 而非 ~0.004),严重时可能导致生成质量退化。

**ATOM `rollout_corr/kl` 偏大**(~0.007 而非 ~0.004):ATOM `atom/model_ops/layernorm.py` 的 plain
RMSNorm 未传 `use_model_sensitive_rmsnorm=1`,与训练侧 T5-like RMSNorm 不一致(见 §13 修复)。

---

## 13. FP8 底层 patch 清单（供排查，正常无需手动）

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
  - aiter shim `examples/DAPO/atom_aiter_shim/sitecustomize.py`:补齐 ATOM 所需的 aiter FP8/MLA 子模块
    (`run_dapo.sh` 自动挂到 PYTHONPATH,`LUMENRL_ATOM_AITER_SRC` 默认取 `$AITER_DIR`)。

---

## 14. verl 运行（复用同一环境：BF16 基线 + Lumen FP8 E2E）

> 本节让**同一台机器、同一个容器**在原生 LumenRL 之外，再跑通 **verl `recipe/dapo`**（DAPO 动态采样）的
> 两条路线，**完全复用前面第 3–6 节**的容器 (`$CONTAINER`)、模型 (`Qwen3-8B-Base`) 与过滤后的数据
> (`data_cached/qwen3-8b-maxprompt1024/*.filtered.parquet`)，**无需重建容器、无需重下模型/数据**：
>
> - **BF16 基线**：`recipe.dapo.main_dapo` + BF16 FSDP 训练 + vLLM BF16 async rollout（AITER off，`quantization=null`）。
> - **Lumen FP8 E2E**：`lumen.rl.verl.verl_entry`（`LUMEN_VERL_MAIN=dapo` 委托 dapo，保留 filter_groups）
>   + FSDP2 actor Lumen FP8（blockwise2d linear + mha attn + model-sensitive RMSNorm）
>   + vLLM async rollout `fp8_per_block` + AITER unified attention。
>
> 关键点：verl 与 Lumen/aiter 一样放在 `$RL_ROOT` 下（`$RL_ROOT/verl`），第 4 节 `-v "$RL_ROOT":"$RL_ROOT"`
> 已把它挂进容器，因此**不用改容器**。`Lumen`(`amd-atom-rollout`)、`aiter`(`lumen/triton_kernels`) 已在第 3 节
> clone，分支正好是 FP8 路线所需，直接复用；FP8 rollout 复用第 5 节的 vLLM AITER RMSNorm patch。
> **已验证**：8×MI350X, Qwen3-8B-Base, 两条路线 smoke 各 1 步全过、exit 0、无 Traceback。

### 14.1 追加拉取 verl（含 recipe 子模块 = DAPO 动态采样 trainer）

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

### 14.2 在同一容器装 verl（不覆盖镜像内 vLLM/torch）

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

### 14.3 BF16 smoke 脚本 `$RL_ROOT/run_smoke_dapo.sh`

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

### 14.4 Lumen FP8 E2E smoke 脚本 `$RL_ROOT/run_smoke_dapo_fp8.sh`

相对 14.3：入口换成 `lumen.rl.verl.verl_entry`（委托 dapo），actor `fsdp2` + Lumen FP8 env，rollout
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
# Lumen actor FP8 + ATOM 对齐 patch + 委托 dapo
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

### 14.5 跑 Smoke（先各 1 step 验证整链路）

```bash
sudo docker restart "$CONTAINER"; sleep 6            # 清残留 ray 进程
# BF16 smoke
sudo docker exec -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" "$CONTAINER" \
  bash -lc "bash $RL_ROOT/run_smoke_dapo.sh" 2>&1 | tee $DATA_ROOT/logs/verl_smoke_bf16_$(date +%Y%m%d-%H%M%S).log

# FP8 E2E smoke
sudo docker restart "$CONTAINER"; sleep 6
sudo docker exec -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" "$CONTAINER" \
  bash -lc "bash $RL_ROOT/run_smoke_dapo_fp8.sh" 2>&1 | tee $DATA_ROOT/logs/verl_smoke_fp8_$(date +%Y%m%d-%H%M%S).log
```

期望证据（已验证：8×MI350X, Qwen3-8B-Base, 各 1 步, exit 0, 无 Traceback）：

| 证据 | BF16 | Lumen FP8 E2E |
|---|---|---|
| `step:1 ... actor/loss ... actor/grad_norm` | loss≈-0.093, grad_norm≈1.09 | loss≈0.028, grad_norm≈0.89 |
| `rollout_corr/rollout_is_mean` | ≈1.0（0.9999） | ≈1.0（1.0000） |
| `rollout_corr/kl` | ≈0.001 | **≈0.005**（FP8 gap，正常；逼近 2.0 才警惕） |
| `train/num_gen_batches` | ≥1（filter_groups 动态采样生效） | ≥1（同） |
| FP8 特有 | — | 8 worker `[verl] Lumen optimizations applied (actor/full) before FSDP2 wrapping`；vLLM `quantization=fp8_per_block`；`kernel_unified_attention_3d` JIT（AITER unified attention 生效） |
| 收尾 | `Final validation metrics: None`（未测评） | 同左；首次会 JIT 编译 aiter `module_rmsnorm` / per-block GEMM（较慢，二次快） |

### 14.6 长跑规模（相对 smoke 只放大规模/落盘）

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

### 14.7 verl vs LumenRL / BF16 vs FP8 一览

| 维度 | verl BF16 | verl Lumen FP8 E2E | LumenRL 原生（§1） |
|---|---|---|---|
| 入口 | `recipe.dapo.main_dapo` | `lumen.rl.verl.verl_entry`（→ dapo） | `lumenrl.trainer.main` |
| actor 策略 | FSDP | FSDP2 + Lumen ATOM(norm/sdpa/linear/mlp) | Lumen FSDP2 |
| actor 量化 | 无（BF16） | ATOM 对齐 + blockwise2d linear + mha attn + model-sensitive RMSNorm | 可选 FP8 blockwise2d |
| rollout | vLLM async BF16，`quantization=null`，AITER off | vLLM async，`quantization=fp8_per_block`，AITER unified attention | vLLM/ATOM AsyncLLM |
| 动态采样 | filter_groups (acc) | filter_groups (acc)（保留） | `algorithm.dapo.filter_groups` |
| 需手动 patch | 无 | 复用第 5 节 vLLM AITER RMSNorm patch | 第 5 节同 patch |
| 复用资源 | §3 clone(+verl)、§4 容器、§6 模型/数据 | 同 + §3 的 Lumen/aiter | §3–§6 |

> 排障速查：FP8 若 `rollout_corr/kl` 异常偏大或发散，多半是第 5 节 vLLM RMSNorm patch 未打（新容器需重打），
> 或 Lumen/aiter 未在 `PYTHONPATH`。BF16 路线 `VLLM_ROCM_USE_AITER=0`，不依赖该 patch。
