# LumenRL DAPO Megatron-Native BF16 新机 Runbook

> 目标：在一台全新的 8×AMD Instinct MI355X / `gfx950` 机器上，从零构建
> LumenRL Megatron-Native BF16 环境，并运行
> `Qwen3-30B-A3B-Base` MoE + DAPO。
>
> 本文是独立 runbook，不依赖旧机器文件；大型源码构建、模型、数据、日志和
> checkpoint 全部放在本地大盘 `DATA_ROOT`。换机器时只修改路径变量。
>
> 本文只覆盖 BF16：训练端 Megatron-Native，rollout 端原生 vLLM。
> 不覆盖 FSDP、FP8、ATOM、verl 或多节点。

---

## 0. 先读：已验证范围与默认决策

### 0.1 硬件与默认拓扑

目标硬件：

- 8×AMD Instinct MI355X；
- GPU 架构 `gfx950`；
- ROCm 容器内可见 8 张卡；
- `rocm-smi` 可能显示每卡 `270566162432 B`，即约 `252 GiB`。

两个运行 profile：

```text
smoke:
  TP=1, PP=1, CP=1, EP=8, ETP=1
  2K 总序列，2 steps
  目的：验证 MoE 建模、EP、权重转换、rollout 权重同步和训练闭环

longrun-safe（本文默认正式配置）:
  TP=2, PP=2, CP=2, EP=2, ETP=2, DP=1
  20K response / 21.5K 总序列，1000 steps
  目的：在 252 GiB/卡机器上分摊层、dense 权重、expert 权重和长序列激活
```

为什么正式长跑不用纯 `EP8`：

- `TP1·PP1·CP1·EP8` 在 288 GB 十进制显存的机器上可完成 smoke；
- 在本次 `252 GiB/卡` 实机上，20K EP8 longrun 曾在 step 25 的 MoE
  all-to-all 处报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`；
- 将 actor token budget 从 21.5K 降到 16K 后，EP8 跑到 step 74，
  仍在 step 75 old-logprob MoE all-to-all 处 OOM；
- 改用仓库已验证的 `TP2·PP2·CP2·EP2·ETP2` 后，在同一机器、同一 20K
  上下文下已运行超过 step 145，actor 峰值 reserved 常见约 130–139 GiB/卡，
  未再触发前述 OOM 点。

这只证明了环境与显存拓扑可持续越过已有失败点；1000-step 收敛仍需最终实跑确认。

### 0.2 强制约束

1. `moe.r3.enabled=false`。真实 DAPO 中 old-logprob 与 update 的动态分箱不同，
   跨调用 R3 replay 会错配路由，曾导致 `ppo_kl≈0.9`。
2. vLLM 保持 `enable_sleep_mode=false`。ROCm 7.2.3 的 cumem sleep/wake
   不能可靠释放显存，且可能在 wake 时 OOM 或破坏权重。
3. 不要替换基础镜像中的 torch、triton、vLLM、flash-attn。
4. 不要执行 `pip install transformer_engine`；必须构建 ROCm fork。
5. TE/Apex build、模型、日志、checkpoint 不要放网络 NFS `/home`。
6. 训练前确认没有另一任务占 GPU；同一组卡只能跑一个后端。
7. 不修改 git config；需要处理 safe-directory 时使用命令级环境变量。
8. 默认关闭 validation 和 checkpoint：
   - validation 会生成长响应，耗时很长；
   - Megatron dist-checkpoint 单个可能超过 100 GiB；
   - 如需开启，先按本文容量要求准备大盘。

### 0.3 磁盘容量

建议：

```text
不保存 checkpoint：DATA_ROOT 至少 250 GiB 可用
保存 1 个 checkpoint：DATA_ROOT 至少 500 GiB 可用
每 50 步保存并保留多个：DATA_ROOT 建议至少 1 TiB 可用
```

根文件系统使用率不要超过 95%。Ray 在低磁盘空间下会持续报
`Object creation will fail if spilling is required`。

---

## 1. 固定版本与 revision

已验证运行栈：

```text
Docker tag           vllm/vllm-openai-rocm:v0.23.0
Docker RepoDigest    sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4
Docker image ID      sha256:648be227ec3ee60b566f9def3485d29713f3d76464081e10a5d9ac56d25732cb
Python               3.12.13
PyTorch              2.10.0+git8514f05
PyTorch HIP          7.2.53211
ROCm userspace       7.2.3
vLLM                 0.23.0
Ray                  2.56.0
flash_attn           2.8.3
Megatron-Core        0.18.2
TransformerEngine    2.15.0.dev0+6e541a10
Apex                 1.14.0a0
```

源码 revision：

```text
Lumen-RL
  repo    https://github.com/ZhangDanyang-AMD/Lumen-RL.git
  branch  dev/vllm-fsdp-dapo
  commit  84c72648e9b7501fd7b1b5c744ffbb92149ff917

Lumen
  repo    https://github.com/ZhangDanyang-AMD/Lumen.git
  branch  amd-atom-rollout
  commit  ee5efbaefcf9400ace124e2afbcd50288eb4aafc

aiter
  repo    https://github.com/ZhangDanyang-AMD/aiter.git
  branch  lumen/triton_kernels
  commit  3fb3ec0f1d703c94a361c447bef352fb9dc235db

ROCm TransformerEngine
  repo    https://github.com/ROCm/TransformerEngine.git
  commit  6e541a10419a6e31bdc98b1516db04eb81a463b6

ROCm Apex
  repo    https://github.com/ROCm/apex.git
  commit  daed85255d51476425080e7e6203f0bee6d7e4cc
```

如主动升级任一 revision，必须重新跑完整 smoke；不能把未验证的新 HEAD 当作本文环境。

---

## 2. Host 前置检查与路径变量

Host 需要 Docker、Git、8 张 gfx950 GPU，并能访问 `/dev/kfd`、`/dev/dri`：

```bash
hostname
whoami
docker --version
git --version
command -v rg
ls -l /dev/kfd /dev/dri
rocm-smi --showproductname
rocm-smi --showmeminfo vram
df -h / /tmp
```

如果没有 `rg`，先安装 ripgrep（例如 Ubuntu/Debian：`sudo apt-get install ripgrep`）。

设置变量。路径中不要有空格：

```bash
export RL_ROOT=/ABSOLUTE/PATH/lumen-rl
export DATA_ROOT=/ABSOLUTE/PATH/lumen-data
export CONTAINER=rl-lumenrl-megatron-bf16
export IMAGE='vllm/vllm-openai-rocm@sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4'
```

创建目录：

```bash
mkdir -p \
  "$RL_ROOT" \
  "$DATA_ROOT/logs" \
  "$DATA_ROOT/tmp" \
  "$DATA_ROOT/hf_home" \
  "$DATA_ROOT/pip_cache" \
  "$DATA_ROOT/wheels" \
  "$DATA_ROOT/models" \
  "$DATA_ROOT/raw" \
  "$DATA_ROOT/data_cached" \
  "$DATA_ROOT/ckpts" \
  "$DATA_ROOT/wandb"

df -h "$DATA_ROOT"
```

后续每次新 shell 都重新 export 这四个变量。

---

## 3. 在线 clone 并固定源码

```bash
git clone -b dev/vllm-fsdp-dapo \
  https://github.com/ZhangDanyang-AMD/Lumen-RL.git \
  "$RL_ROOT/Lumen-RL"
git -C "$RL_ROOT/Lumen-RL" checkout --detach \
  84c72648e9b7501fd7b1b5c744ffbb92149ff917

git clone -b amd-atom-rollout \
  https://github.com/ZhangDanyang-AMD/Lumen.git \
  "$RL_ROOT/Lumen"
git -C "$RL_ROOT/Lumen" checkout --detach \
  ee5efbaefcf9400ace124e2afbcd50288eb4aafc

git clone -b lumen/triton_kernels \
  https://github.com/ZhangDanyang-AMD/aiter.git \
  "$RL_ROOT/aiter"
git -C "$RL_ROOT/aiter" checkout --detach \
  3fb3ec0f1d703c94a361c447bef352fb9dc235db
git -C "$RL_ROOT/aiter" submodule sync --recursive
git -C "$RL_ROOT/aiter" submodule update --init --recursive --jobs 16
```

验证关键提交和文件：

```bash
git -C "$RL_ROOT/Lumen-RL" rev-parse HEAD
git -C "$RL_ROOT/Lumen" rev-parse HEAD
git -C "$RL_ROOT/aiter" rev-parse HEAD

test -f "$RL_ROOT/Lumen-RL/lumenrl/engine/training/megatron_native_engine.py"
test -f "$RL_ROOT/Lumen-RL/lumenrl/engine/training/qwen3moe_megatron_bridge.py"
test -f "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
test -f "$RL_ROOT/Lumen-RL/examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_smoke.yaml"
test -f "$RL_ROOT/Lumen-RL/examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_longrun_e_full.yaml"
```

预期 Lumen-RL HEAD 为 `84c7264...`，其中包含：

```text
ddb3b4c feat(megatron): add MoE + Expert Parallel to native engine
84c7264 feat(megatron): R3 router record/replay building blocks + PP/SP-eff/parallel combos
```

---

## 4. 拉取固定镜像并启动持久容器

```bash
docker pull "$IMAGE"

docker image inspect "$IMAGE" \
  --format 'id={{.Id}} repo_digests={{json .RepoDigests}}'
```

必须看到：

```text
id=sha256:648be227ec3ee60b566f9def3485d29713f3d76464081e10a5d9ac56d25732cb
repo_digests 包含 sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4
```

确认容器名未被占用：

```bash
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ERROR: container $CONTAINER already exists; inspect it first" >&2
  exit 1
fi
```

读取 host 上 video/render 设备的实际 GID，避免不同发行版组名不一致：

```bash
export VIDEO_GID="$(stat -c '%g' /dev/dri/card0)"
export RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
```

启动：

```bash
docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --network=host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add="$VIDEO_GID" \
  --group-add="$RENDER_GID" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --shm-size=64G \
  -v "$RL_ROOT":"$RL_ROOT" \
  -v "$DATA_ROOT":"$DATA_ROOT" \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  -e HF_HOME="$DATA_ROOT/hf_home" \
  -e PIP_CACHE_DIR="$DATA_ROOT/pip_cache" \
  -e TMPDIR="$DATA_ROOT/tmp" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" \
  -e AITER_DIR="$RL_ROOT/aiter" \
  "$IMAGE" -lc 'sleep infinity'
```

基础环境验证：

```bash
docker exec "$CONTAINER" bash -lc '
python - <<PY
import torch
import vllm
import transformers

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, getattr(p, "gcnArchName", None), round(p.total_memory / 2**30, 1))
PY
hipcc --version
'
```

预期：

```text
torch 2.10.0+git8514f05
hip 7.2.53211
vllm 0.23.0
gpus 8
每张卡 gcnArchName 含 gfx950
```

版本不匹配时停止，不要继续编译 TE/Apex。

---

## 5. 安装 build/runtime 依赖

不要在这一步安装或升级 torch、triton、vLLM、flash-attn。

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail

python -m pip install \
  "pip==26.1.2" \
  "setuptools==79.0.1" \
  "wheel==0.47.0" \
  "packaging==26.2" \
  "cmake==3.31.10" \
  "ninja==1.13.0" \
  "pybind11==3.0.4" \
  "build==1.5.1" \
  "cxxfilt==0.3.0"

python -m pip install \
  "ray[default]==2.56.0" \
  "omegaconf==2.3.1" \
  "safetensors==0.8.0" \
  "accelerate==1.14.0" \
  "datasets==5.0.0" \
  "math-verify[antlr4_13_2]==0.9.0" \
  "huggingface_hub==1.19.0" \
  "wandb==0.28.0" \
  "pyzmq==27.1.0" \
  "absl-py==2.5.0" \
  "transformers==5.12.0" \
  "tokenizers==0.22.2" \
  "fsspec==2026.4.0" \
  "pyarrow==24.0.0" \
  "numpy==1.26.4" \
  "pandas==3.0.3"
' 2>&1 | tee "$DATA_ROOT/logs/common-deps-install.log"
```

检查编译工具：

```bash
docker exec "$CONTAINER" bash -lc '
command -v gcc
command -v g++
command -v make
command -v cmake
command -v ninja
command -v hipcc
'
```

如缺 `gcc/g++/make`：

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential git ca-certificates pkg-config
rm -rf /var/lib/apt/lists/*
'
```

---

## 6. 安装 Megatron-Core 0.18.2

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail
python -m pip install --no-deps "megatron-core==0.18.2"

python - <<PY
import megatron.core as mc
import megatron.training
import megatron.core.dist_checkpointing
from megatron.core.pipeline_parallel import get_forward_backward_func

print("megatron.core", mc.__version__)
print("megatron.training", megatron.training.__file__)
print("dist-checkpoint + pipeline schedule OK")
PY
'
```

此时可能警告 TE/Apex 未安装，属于预期；下一节完成后警告应消失。

不要安装 `megatron-bridge`。Qwen3-MoE HF↔Megatron 转换由 LumenRL 自己完成。

---

## 7. 构建 ROCm Apex

源码和 build 均放在 `DATA_ROOT`：

```bash
git clone https://github.com/ROCm/apex.git "$DATA_ROOT/apex_src"
git -C "$DATA_ROOT/apex_src" checkout --detach \
  daed85255d51476425080e7e6203f0bee6d7e4cc
git -C "$DATA_ROOT/apex_src" submodule sync --recursive
git -C "$DATA_ROOT/apex_src" submodule update --init --recursive --jobs 16
git -C "$DATA_ROOT/apex_src" rev-parse HEAD
```

编译安装：

```bash
docker exec \
  -e APEX_SRC="$DATA_ROOT/apex_src" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
set -euxo pipefail
cd "$APEX_SRC"
rm -rf build dist
export PYTORCH_ROCM_ARCH=gfx950
export MAX_JOBS="${MAX_JOBS:-32}"
python setup.py install --cpp_ext --cuda_ext \
  2>&1 | tee "$DATA_ROOT/logs/apex-build.log"
'
```

验证：

```bash
docker exec "$CONTAINER" bash -lc '
python - <<PY
import torch
from apex.normalization import FusedLayerNorm
from apex.optimizers import FusedAdam

x = torch.randn(4, 16, 128, device="cuda", requires_grad=True)
ln = FusedLayerNorm(128).cuda()
y = ln(x)
y.square().mean().backward()
opt = FusedAdam(ln.parameters(), lr=1e-3)
opt.step()
assert torch.isfinite(y).all()
print("Apex FusedLayerNorm/FusedAdam OK", tuple(y.shape))
PY
'
```

首次调用可能触发 JIT，等待其完成。

---

## 8. 构建 ROCm TransformerEngine 2.15

必须使用 ROCm fork，并递归拉取 submodule：

```bash
git clone https://github.com/ROCm/TransformerEngine.git "$DATA_ROOT/te_src"
git -C "$DATA_ROOT/te_src" checkout --detach \
  6e541a10419a6e31bdc98b1516db04eb81a463b6
git -C "$DATA_ROOT/te_src" submodule sync --recursive
git -C "$DATA_ROOT/te_src" submodule update --init --recursive --jobs 16

git -C "$DATA_ROOT/te_src" rev-parse HEAD
git -C "$DATA_ROOT/te_src" submodule status --recursive
```

先移除可能误装的 NVIDIA TE：

```bash
docker exec "$CONTAINER" bash -lc '
python -m pip uninstall -y \
  transformer-engine transformer_engine \
  transformer-engine-torch transformer_engine_torch || true
'
```

源码编译：

```bash
docker exec \
  -e TE_SRC="$DATA_ROOT/te_src" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
set -euxo pipefail
cd "$TE_SRC"
rm -rf build dist transformer_engine.egg-info
mkdir -p "$DATA_ROOT/tmp"

export TMPDIR="$DATA_ROOT/tmp"
export NVTE_FRAMEWORK=pytorch
export NVTE_USE_ROCM=1
export NVTE_ROCM_ARCH=gfx950
export PYTORCH_ROCM_ARCH=gfx950
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export MAX_JOBS="${MAX_JOBS:-32}"

# ROCm 7.2.3 的 hipcc -v 在无输入时可能返回 1。
# 跳过保护性 ABI 探测，不会跳过实际 kernel 编译。
export TORCH_DONT_CHECK_COMPILER_ABI=1

python -m pip install -v . --no-build-isolation \
  2>&1 | tee "$DATA_ROOT/logs/te-build.log"
'
```

成功日志应包含：

```text
Successfully installed ... transformer_engine-2.15.0.dev0+6e541a10
```

验证 Linear、RMSNorm 和 attention 前反向：

```bash
docker exec "$CONTAINER" bash -lc '
python - <<PY
import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention import DotProductAttention

print("TE", transformer_engine.__version__, transformer_engine.__file__)

x = torch.randn(
    4, 16, 128,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
norm = te.RMSNorm(128).to("cuda", dtype=torch.bfloat16)
linear = te.Linear(128, 256, bias=False).to("cuda", dtype=torch.bfloat16)
y = linear(norm(x))
y.float().square().mean().backward()
assert torch.isfinite(y).all()
print("Linear/RMSNorm OK", tuple(y.shape))

q = torch.randn(32, 1, 8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(32, 1, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(32, 1, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
attn = DotProductAttention(
    num_attention_heads=8,
    kv_channels=16,
    num_gqa_groups=2,
    attention_dropout=0.0,
    attn_mask_type="causal",
    qkv_format="sbhd",
).cuda()
out = attn(q, k, v)
out.float().sum().backward()
assert torch.isfinite(out).all()
print("DotProductAttention OK", tuple(out.shape))
PY
'
```

---

## 9. 安装 aiter、Lumen 与 LumenRL

不修改 git config。对需要 git introspection 的安装进程临时注入 safe-directory：

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="*"

cd "$AITER_DIR"
git submodule update --init --recursive --jobs 16
AITER_USE_SYSTEM_TRITON=1 python setup.py develop

python -m pip install -e "$LUMEN_DIR" --no-deps
python -m pip install -e "$RL_ROOT/Lumen-RL" --no-deps
' 2>&1 | tee "$DATA_ROOT/logs/lumenrl-install.log"
```

完整 import/engine 注册验证：

```bash
docker exec "$CONTAINER" bash -lc '
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen:${PYTHONPATH:-}"
cd "$RL_ROOT/Lumen-RL"

python - <<PY
import torch
import vllm
import ray
import apex
import transformer_engine
import transformer_engine.pytorch
import megatron.core
import megatron.training
import megatron.core.dist_checkpointing
from megatron.core.pipeline_parallel import get_forward_backward_func
from lumenrl.engine.training.base_engine import EngineRegistry

cls = EngineRegistry.get_engine_cls(
    model_type="language_model",
    backend="megatron_native",
)
print("engine megatron_native", cls.__name__)
print("torch", torch.__version__, "hip", torch.version.hip)
print("vllm", vllm.__version__, "ray", ray.__version__)
print("megatron", megatron.core.__version__)
print("TE", transformer_engine.__version__)
print("gpus", torch.cuda.device_count())
print("IMPORT STACK OK")
PY
'
```

必须看到：

```text
engine megatron_native MegatronNativeEngineWithLMHead
megatron 0.18.2
TE 2.15.0.dev0+6e541a10
gpus 8
IMPORT STACK OK
```

---

## 10. 下载 Qwen3-30B-A3B-Base 与数据集

公开仓库通常不需要 token。受限网络可先：

```bash
export HF_TOKEN=your_token_if_needed
```

下载模型和数据：

```bash
docker exec \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$CONTAINER" bash -lc '
python - <<PY
import os
from huggingface_hub import snapshot_download

D = os.environ["DATA_ROOT"]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    "Qwen/Qwen3-30B-A3B-Base",
    local_dir=f"{D}/models/Qwen3-30B-A3B-Base",
    allow_patterns=[
        "*.json",
        "*.txt",
        "*.safetensors",
        "*.model",
        "tokenizer*",
        "*.tiktoken",
        "*.py",
        "*.jinja",
    ],
    token=token,
)

snapshot_download(
    "BytedTsinghua-SIA/DAPO-Math-17k",
    repo_type="dataset",
    local_dir=f"{D}/raw/DAPO-Math-17k",
    token=token,
)

snapshot_download(
    "BytedTsinghua-SIA/AIME-2024",
    repo_type="dataset",
    local_dir=f"{D}/raw/AIME-2024",
    token=token,
)
PY
'
```

模型约 57 GiB，通常有 16 个 safetensors 分片。

### 10.1 过滤 prompt 长度 ≤1024

创建过滤脚本：

```bash
cat > "$RL_ROOT/filter_moe_prompts.py" <<'PYEOF'
import glob
import os

import datasets
from transformers import AutoTokenizer

DATA_ROOT = os.environ["DATA_ROOT"]
MODEL_PATH = f"{DATA_ROOT}/models/Qwen3-30B-A3B-Base"

# 历史目录名包含 qwen3-8b，但 Qwen3 tokenizer 通用；
# 仓库内 MoE YAML 默认读取这个路径。
OUT_DIR = f"{DATA_ROOT}/data_cached/qwen3-8b-maxprompt1024"
MAX_PROMPT_LENGTH = 1024


def parquet_files(root: str) -> list[str]:
    files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    return files


def load_parquet(root: str):
    files = parquet_files(root)
    print(f"Loading {len(files)} parquet file(s) from {root}")
    return datasets.load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
    )


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def prompt_length(example) -> int:
    return len(
        tokenizer.apply_chat_template(
            example["prompt"],
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def process(src_root: str, dst_name: str) -> None:
    ds = load_parquet(src_root)
    if "prompt" not in ds.column_names:
        raise KeyError(f"{src_root} has no prompt column: {ds.column_names}")
    before = len(ds)
    workers = max(1, min(64, (os.cpu_count() or 8) // 4))
    ds = ds.filter(
        lambda row: prompt_length(row) <= MAX_PROMPT_LENGTH,
        num_proc=workers,
        desc=f"Filtering prompts > {MAX_PROMPT_LENGTH} tokens",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = os.path.join(OUT_DIR, dst_name)
    ds.to_parquet(dst)
    print(f"{src_root} -> {dst}: {before} -> {len(ds)}")


if __name__ == "__main__":
    process(
        f"{DATA_ROOT}/raw/DAPO-Math-17k",
        "dapo-math-17k.filtered.parquet",
    )
    process(
        f"{DATA_ROOT}/raw/AIME-2024",
        "aime-2024.filtered.parquet",
    )
PYEOF
```

该脚本假定模型 tokenizer 带 Qwen3 chat template，且两个数据集包含 verl/DAPO
格式的 `prompt` 列；脚本会在列缺失时立即报错，避免生成错误缓存。

执行：

```bash
docker exec \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc \
  'python "$RL_ROOT/filter_moe_prompts.py"' \
  2>&1 | tee "$DATA_ROOT/logs/filter-moe-prompts.log"
```

检查：

```bash
docker exec "$CONTAINER" bash -lc '
set -e
test -f "$DATA_ROOT/models/Qwen3-30B-A3B-Base/config.json"
test -f "$DATA_ROOT/models/Qwen3-30B-A3B-Base/model.safetensors.index.json"
test -f "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet"
test -f "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet"

python - <<PY
import os
import pyarrow.parquet as pq

D = os.environ["DATA_ROOT"]
base = f"{D}/data_cached/qwen3-8b-maxprompt1024"
for name in (
    "dapo-math-17k.filtered.parquet",
    "aime-2024.filtered.parquet",
):
    path = f"{base}/{name}"
    pf = pq.ParquetFile(path)
    print(name, pf.metadata.num_rows, pf.schema.names)
PY
'
```

---

## 11. 生成统一 Megatron BF16 启动器

启动器支持：

```text
PROFILE=smoke          EP8，2K，默认 2 steps
PROFILE=longrun-safe   TP2·PP2·CP2·EP2·ETP2，20K，默认 1000 steps
PROFILE=longrun-ep8    TP1·PP1·CP1·EP8，20K，仅建议显存更大的已验证机器
```

创建 `$RL_ROOT/run_megatron_bf16.sh`：

```bash
cat > "$RL_ROOT/run_megatron_bf16.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${RL_ROOT:?RL_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"

PROFILE="${PROFILE:-smoke}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-30B-A3B-Base}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.25}"
ENABLE_CHECKPOINTS="${ENABLE_CHECKPOINTS:-false}"
VAL_STEPS="${VAL_STEPS:-1000000}"
VAL_NUM_SAMPLES="${VAL_NUM_SAMPLES:-30}"

case "$PROFILE" in
  smoke)
    CONFIG="examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_smoke.yaml"
    STEPS="${STEPS:-2}"
    WANDB_ENABLED="${WANDB_ENABLED:-false}"
    ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-2048}"
    LOGPROB_CHUNK="${LOGPROB_CHUNK:-512}"
    VLLM_BATCHED_TOKENS="${VLLM_BATCHED_TOKENS:-4096}"
    VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
    TOPOLOGY=(
      policy.training.megatron_cfg.tensor_model_parallel_size=1
      policy.training.megatron_cfg.pipeline_model_parallel_size=1
      policy.training.megatron_cfg.context_parallel_size=1
      policy.training.megatron_cfg.expert_model_parallel_size=8
      policy.training.megatron_cfg.expert_tensor_parallel_size=1
    )
    ;;

  longrun-safe)
    CONFIG="examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_longrun_e_full.yaml"
    STEPS="${STEPS:-1000}"
    WANDB_ENABLED="${WANDB_ENABLED:-true}"
    ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-16384}"
    LOGPROB_CHUNK="${LOGPROB_CHUNK:-512}"
    VLLM_BATCHED_TOKENS="${VLLM_BATCHED_TOKENS:-16384}"
    VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
    TOPOLOGY=(
      policy.training.megatron_cfg.tensor_model_parallel_size=2
      policy.training.megatron_cfg.pipeline_model_parallel_size=2
      policy.training.megatron_cfg.context_parallel_size=2
      policy.training.megatron_cfg.expert_model_parallel_size=2
      policy.training.megatron_cfg.expert_tensor_parallel_size=2
    )
    ;;

  longrun-ep8)
    CONFIG="examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_longrun.yaml"
    STEPS="${STEPS:-1000}"
    WANDB_ENABLED="${WANDB_ENABLED:-true}"
    ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-16384}"
    LOGPROB_CHUNK="${LOGPROB_CHUNK:-512}"
    VLLM_BATCHED_TOKENS="${VLLM_BATCHED_TOKENS:-16384}"
    VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
    TOPOLOGY=(
      policy.training.megatron_cfg.tensor_model_parallel_size=1
      policy.training.megatron_cfg.pipeline_model_parallel_size=1
      policy.training.megatron_cfg.context_parallel_size=1
      policy.training.megatron_cfg.expert_model_parallel_size=8
      policy.training.megatron_cfg.expert_tensor_parallel_size=1
    )
    ;;

  *)
    echo "ERROR: PROFILE must be smoke, longrun-safe, or longrun-ep8" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-megatron-bf16-${PROFILE}-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/ckpts/lumenrl-dapo/${PROFILE}}"

mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/wandb" "$(dirname "$CKPT_DIR")"

if [ "$ENABLE_CHECKPOINTS" = true ]; then
  RESUME="${RESUME:-true}"
  SAVE_STEPS="${SAVE_STEPS:-50}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
else
  RESUME=false
  SAVE_STEPS=1000000
  SAVE_TOTAL_LIMIT=1
fi

# run_dapo.sh 会清理旧 Ray/vLLM/trainer。默认先拒绝启动，防止误杀一个
# 没有 checkpoint 的长跑；只有明确设置 FORCE_RESTART=true 才允许覆盖。
if [ "${FORCE_RESTART:-false}" != true ] \
  && pgrep -af "[l]umenrl.trainer.main|[V]LLMRayServer|[E]ngineCore" >/dev/null; then
  echo "ERROR: another LumenRL/vLLM task is running in this container." >&2
  pgrep -af "[l]umenrl.trainer.main|[V]LLMRayServer|[E]ngineCore" >&2 || true
  echo "Stop it explicitly first, or set FORCE_RESTART=true." >&2
  exit 2
fi

if [ "$WANDB_ENABLED" = true ] \
  && [ -z "${WANDB_API_KEY:-}" ] \
  && [ ! -s "$RL_ROOT/wandb.key" ] \
  && [ ! -s "$RL_ROOT/../wandb.key" ]; then
  echo "ERROR: W&B enabled but no WANDB_API_KEY or wandb.key found" >&2
  exit 2
fi

OVERRIDES=(
  checkpointing.resume="$RESUME"
  checkpointing.save_steps="$SAVE_STEPS"
  checkpointing.save_total_limit="$SAVE_TOTAL_LIMIT"
  checkpointing.checkpoint_dir="$CKPT_DIR"
  logger.wandb_enabled="$WANDB_ENABLED"
  logger.wandb.project=LumenRL
  logger.wandb.name="$RUN_ID"
  val_steps="$VAL_STEPS"
  eval.num_samples="$VAL_NUM_SAMPLES"
  policy.max_token_len_per_gpu="$ACTOR_TOKEN_BUDGET"
  policy.training.megatron_cfg.max_tokens_per_gpu="$ACTOR_TOKEN_BUDGET"
  policy.training.megatron_cfg.log_probs_chunk_size="$LOGPROB_CHUNK"
  policy.generation.vllm_cfg.enable_sleep_mode=false
  policy.generation.vllm_cfg.gpu_memory_utilization="$VLLM_GPU_UTIL"
  policy.generation.vllm_cfg.max_num_batched_tokens="$VLLM_BATCHED_TOKENS"
  policy.generation.vllm_cfg.max_num_seqs="$VLLM_MAX_SEQS"
  moe.r3.enabled=false
  "${TOPOLOGY[@]}"
)

if [ -n "${USER_EXTRA_OVERRIDE:-}" ]; then
  OVERRIDES+=("$USER_EXTRA_OVERRIDE")
fi

EXTRA_OVERRIDE="${OVERRIDES[*]}"
S="$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"

printf "%s\n" "$LOG" > "$DATA_ROOT/logs/megatron-bf16-${PROFILE}.latest"

echo "PROFILE=$PROFILE"
echo "CONFIG=$CONFIG"
echo "STEPS=$STEPS"
echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG"
echo "CKPT_DIR=$CKPT_DIR"
echo "ENABLE_CHECKPOINTS=$ENABLE_CHECKPOINTS"
echo "VAL_STEPS=$VAL_STEPS"
echo "EXTRA_OVERRIDE=$EXTRA_OVERRIDE"

CONFIG_OVERRIDE="$CONFIG" \
MODE=bf16 \
STEPS="$STEPS" \
MODEL_PATH="$MODEL_PATH" \
RUN_ID="$RUN_ID" \
LOG="$LOG" \
EXTRA_OVERRIDE="$EXTRA_OVERRIDE" \
bash "$S"
EOF

chmod +x "$RL_ROOT/run_megatron_bf16.sh"
```

默认 longrun 开启 W&B，因此启动正式长跑前必须二选一：

```bash
# 方案 A：写入挂载目录，推荐
printf 'WANDB_API_KEY=%s\n' 'YOUR_KEY' > "$RL_ROOT/wandb.key"
chmod 600 "$RL_ROOT/wandb.key"

# 方案 B：仅当前 host shell
# export WANDB_API_KEY=YOUR_KEY
```

不使用 W&B 时显式设置 `WANDB_ENABLED=false`。不要把 `wandb.key` 提交到 git。

---

## 12. 运行 EP8 smoke

> 安全提示：底层 `run_dapo.sh` 会清理同一容器中的 Ray/vLLM/trainer。
> 本文 wrapper 默认在发现已有任务时拒绝启动。不要在 longrun 运行期间启动 smoke；
> `FORCE_RESTART=true` 只用于你明确要终止旧任务的场景。

### 12.1 起跑前确认 GPU 空闲

```bash
docker exec "$CONTAINER" bash -lc '
pgrep -af "lumenrl.trainer.main|VLLMRayServer|EngineCore" || true
'
rocm-smi --showmeminfo vram
```

空卡基线通常每卡只有数百 MiB。

### 12.2 前台运行 2 steps

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
PROFILE=smoke \
STEPS=2 \
RUN_ID=smoke-moe-a3b-ep8 \
WANDB_ENABLED=false \
ENABLE_CHECKPOINTS=false \
VAL_STEPS=1000000 \
bash "$RL_ROOT/run_megatron_bf16.sh"
'
```

查看日志：

```bash
RUN_LOG="$DATA_ROOT/logs/smoke-moe-a3b-ep8.log"

rg -a \
  "MoE\\+EP spec|rollout ready|callbacks: step=|LumenRL finished|Traceback|OutOfMemory|HSA_STATUS_ERROR|NaN" \
  "$RUN_LOG"
```

必须看到：

```text
MoE+EP spec: num_experts=128 topk=8 ... EP=8 ... local_experts/rank=16
callbacks: step=1
callbacks: step=2
LumenRL finished.
```

健康参考：

```text
ppo_kl            约 0
rollout_corr/kl   约 0.002
entropy           有限
grad_norm         有限
NaN               0
```

252 GiB/卡机器必须覆盖 `VLLM_GPU_UTIL=0.25`。原 YAML 的 0.40 在该机器上可能
在第二步 MoE all-to-all 时耗尽显存。

---

## 13. 启动正式 20K longrun

> 启动前先执行 §12.1 的进程/GPU 检查。不要重复执行 longrun 启动命令；
> 默认无 checkpoint，误杀后不能恢复。

### 13.1 默认：252 GiB/卡安全拓扑

默认关闭 checkpoint 与 validation：

```bash
docker exec -d \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "$CONTAINER" bash -lc '
PROFILE=longrun-safe \
STEPS=1000 \
WANDB_ENABLED=true \
ENABLE_CHECKPOINTS=false \
VAL_STEPS=1000000 \
VLLM_GPU_UTIL=0.25 \
bash "$RL_ROOT/run_megatron_bf16.sh"
'
```

该命令使用：

```text
训练拓扑         TP2·PP2·CP2·EP2·ETP2，DP1
rollout          8×vLLM TP1 colocated replicas
max response     20480
max total seq    21504
actor token      16384/GPU（CP/TP 后本地激活进一步分摊）
vLLM util        0.25
vLLM batch token 16384
vLLM max seqs    32
R3               false
validation       off
checkpoint       off
```

### 13.2 有大盘时启用 checkpoint

先确认至少数百 GiB 可用：

```bash
df -h "$DATA_ROOT"
```

然后：

```bash
docker exec -d \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "$CONTAINER" bash -lc '
PROFILE=longrun-safe \
STEPS=1000 \
WANDB_ENABLED=true \
ENABLE_CHECKPOINTS=true \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=1 \
RESUME=true \
VAL_STEPS=1000000 \
VLLM_GPU_UTIL=0.25 \
bash "$RL_ROOT/run_megatron_bf16.sh"
'
```

关闭 checkpoint 时无法续跑；任何 OOM/节点故障都会从 step 0 重来。

### 13.3 可选 validation

`val-core/acc/mean@1` 只在 validation 执行时产生。默认 `VAL_STEPS=1000000`
等价于本次 1000-step run 不做验证，因此 W&B 只有训练 rollout 的
`reward/accuracy`。

如磁盘、显存与运行时间充足，可低频、小样本启用：

```bash
VAL_STEPS=100 VAL_NUM_SAMPLES=30
```

例如把这两个变量加入 §13.1 的启动命令。

注意：

- 验证使用 greedy、最长 20480-token 生成；
- 全量 AIME parquet 可能含数百样本，长验证可能耗时数小时；
- 验证本身不必然 OOM，但会增加 KV 压力和显存碎片风险；
- 当前运行中不能热修改 `VAL_STEPS`，必须重启；无 checkpoint 时会丢进度。

### 13.4 可选 EP8 longrun

只在显存更大或已单独验证的机器使用：

```bash
PROFILE=longrun-ep8
```

不要在 252 GiB/卡机器上把它当默认方案。

---

## 14. 监控 longrun

获取当前日志：

```bash
PROFILE=longrun-safe
RUN_LOG="$(<"$DATA_ROOT/logs/megatron-bf16-${PROFILE}.latest")"
echo "$RUN_LOG"
```

确认进程：

```bash
docker exec "$CONTAINER" bash -lc '
pgrep -af "python3 -u -m lumenrl.trainer.main"
'
```

确认拓扑和 W&B：

```bash
rg -a \
  "MoE\\+EP spec|model-parallel=|mesh_mapping|rollout ready|View run|Syncing run" \
  "$RUN_LOG"
```

`longrun-safe` 必须看到：

```text
tp=2 pp=2 cp=2 EP=2 etp=2
Megatron model-parallel=8: actor DP=1
mesh_mapping=[0, 0, 0, 0, 0, 0, 0, 0]
```

查看训练步与异常：

```bash
rg -a "callbacks: step=" "$RUN_LOG" | tail -5

rg -ai \
  "Traceback|OutOfMemory|HSA_STATUS_ERROR|Fatal Python|ActorDied|NaN|collective|timeout" \
  "$RUN_LOG" | tail -20

rocm-smi --showmeminfo vram
```

健康判据：

- 无 OOM、NaN、Traceback、ActorDiedError、collective timeout；
- `ppo_kl≈0`；
- `rollout_corr/kl` 通常约 `0.001–0.002`；
- grad_norm 有限；
- actor max reserved 不持续逐步上涨；
- W&B 出现 `View run at https://wandb.ai/...`。

全并行 20K 实机前几步参考：

```text
actor max reserved  约 130–139 GiB
ppo_kl              约 0
rollout_corr/kl     约 0.002
step time           约 530–700 秒（rollout 主导）
```

指标是参考区间，不是必须逐点相等。

另外监控 entropy：若持续跌到 `0.3` 以下，说明策略可能变得过于确定；
这是训练质量问题，不是显存问题，应结合 reward/accuracy、响应长度和样本检查后决定是否停止。

---

## 15. 停止、清理与续跑

停止：

```bash
docker exec "$CONTAINER" bash -lc '
ray stop --force 2>/dev/null || true
pkill -9 -f "[l]umenrl.trainer.main" 2>/dev/null || true
pkill -9 -f "[V]LLMRayServer" 2>/dev/null || true
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
pkill -9 -f "[E]ngineCore" 2>/dev/null || true
sleep 8
'

rocm-smi --showmeminfo vram
```

如启用了 checkpoint，续跑时必须保持：

- 同一个 `PROFILE`；
- 相同 TP/PP/CP/EP/ETP；
- 相同 `CKPT_DIR`；
- 相同 optimizer、lr、warmup、batch、token budget；
- `ENABLE_CHECKPOINTS=true`；
- `RESUME=true`。

重新执行 §13.2 即可。

恢复证据：

```bash
rg -a "resume_step=|Resume:|callbacks: step=" "$RUN_LOG" | tail -10
```

未启用 checkpoint 时不能恢复训练权重。

---

## 16. 常见问题

### 16.1 `ModuleNotFoundError: megatron` / `transformer_engine`

容器只有 FSDP 环境，没有 Megatron 栈。完整执行 §6–§9。

### 16.2 TE 被装成 NVIDIA 版本

症状：import 时出现 NVIDIA/CUDA library 错误，或版本不是
`2.15.0.dev0+6e541a10`。

处理：

```bash
docker exec "$CONTAINER" bash -lc '
python -m pip uninstall -y \
  transformer-engine transformer_engine \
  transformer-engine-torch transformer_engine_torch || true
'
```

清理 `$DATA_ROOT/te_src/build`，重新执行 §8。

### 16.3 TE CK-JIT 报 `cxx_interceptor -v`

确认构建环境包含：

```bash
export TORCH_DONT_CHECK_COMPILER_ABI=1
```

### 16.4 `HSA_STATUS_ERROR_OUT_OF_RESOURCES: Available Free mem : 0 MB`

这是 GPU 显存耗尽，不是 host RAM。

按顺序处理：

1. 确认使用 `PROFILE=longrun-safe`，不是 EP8 longrun；
2. 确认 `VLLM_GPU_UTIL=0.25`；
3. 确认 `ACTOR_TOKEN_BUDGET=16384`；
4. 必要时降到 `ACTOR_TOKEN_BUDGET=12288` 或 `8192`；
5. 确认没有 validation 与其他 GPU 进程；
6. 保留日志中的失败 step、`seq/max_len` 和 actor memory 指标。

不要打开 vLLM sleep mode 作为 workaround。

### 16.5 EP8 smoke 第二步 OOM

原 smoke YAML 的 vLLM util 是 0.40。本文 wrapper 强制默认 0.25；
不要直接绕过 wrapper 调 YAML。

### 16.6 `ppo_kl` 突然约 0.9

检查：

```text
moe.r3.enabled=false
```

真实 DAPO 流程不要打开跨调用 R3。

### 16.7 W&B 没有 `val-core/acc/mean@1`

默认 validation 关闭。训练准确率看：

```text
reward/accuracy
```

需要 validation 时设置：

```bash
VAL_STEPS=100 VAL_NUM_SAMPLES=30
```

### 16.8 Ray 每 10 秒警告磁盘超过 95%

这是文件系统空间不足。把 `DATA_ROOT`、Ray temp 和日志放到容量更大的本地盘，
或清理明确可删除的旧文件。不要对共享目录执行模糊通配删除。

### 16.9 aiter submodule 缺失

重新执行：

```bash
git -C "$RL_ROOT/aiter" submodule sync --recursive
git -C "$RL_ROOT/aiter" submodule update --init --recursive --jobs 16
```

### 16.10 `expandable_segments not supported on this platform`

这是当前 ROCm allocator 的 warning；不是训练失败。真正 OOM 会出现
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` 或 worker SIGABRT。

### 16.11 vLLM 提示未知 `VLLM_USE_V1`

vLLM 0.23.0 已默认使用 V1，该 warning 可忽略。

### 16.12 初始化或第一步很慢

首次启动包括：

- 8 个 actor 初始化 Megatron process groups；
- 8 个 vLLM replica 加载 30B MoE；
- HF→Megatron 权重转换与分片；
- EP/ETP/PP 权重同步；
- 20K 长响应 rollout。

判断 hang 前至少检查 GPU 利用率、日志更新时间和进程是否仍存在。

---

## 17. 保存环境 manifest

```bash
{
  date -Is
  docker image inspect "$IMAGE" \
    --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}}'

  docker exec "$CONTAINER" bash -lc '
    python -V
    hipcc --version
    python - <<PY
import torch
import vllm
import ray
import megatron.core
import transformer_engine
import apex
from lumenrl.engine.training.base_engine import EngineRegistry

print("torch", torch.__version__, "hip", torch.version.hip)
print("vllm", vllm.__version__)
print("ray", ray.__version__)
print("megatron-core", megatron.core.__version__)
print("transformer-engine", transformer_engine.__version__)
print("apex", getattr(apex, "__version__", "1.14.0a0"))
print(
    "engine",
    EngineRegistry.get_engine_cls(
        model_type="language_model",
        backend="megatron_native",
    ).__name__,
)
print("gpus", torch.cuda.device_count())
PY
  '

  echo "Lumen-RL $(git -C "$RL_ROOT/Lumen-RL" rev-parse HEAD)"
  echo "Lumen $(git -C "$RL_ROOT/Lumen" rev-parse HEAD)"
  echo "aiter $(git -C "$RL_ROOT/aiter" rev-parse HEAD)"
  echo "TE $(git -C "$DATA_ROOT/te_src" rev-parse HEAD)"
  echo "Apex $(git -C "$DATA_ROOT/apex_src" rev-parse HEAD)"
  df -h "$DATA_ROOT"
  rocm-smi --showproductname
} 2>&1 | tee "$DATA_ROOT/logs/megatron-bf16-env-manifest.txt"
```

---

## 18. 完成标准

以下全部满足才算新机环境可用：

1. 镜像 RepoDigest、torch、HIP、vLLM 版本匹配；
2. 容器内可见 8×gfx950；
3. 三个仓库与 TE/Apex revision 匹配；
4. `megatron-core==0.18.2`；
5. Apex FusedLayerNorm/FusedAdam 前反向通过；
6. TE 版本为 `2.15.0.dev0+6e541a10`；
7. TE Linear/RMSNorm/DPA 前反向通过；
8. `megatron_native` engine 注册成功；
9. 模型、训练 parquet、验证 parquet 齐全；
10. EP8 smoke 完成 2 steps，无 OOM/NaN/Traceback；
11. longrun 日志出现 `TP2·PP2·CP2·EP2·ETP2` 与 `actor DP=1`；
12. longrun 前两步 `ppo_kl≈0`、`rollout_corr/kl≈0.002`、grad_norm 有限；
13. W&B 正常同步；
14. checkpoint/validation 的启停与实际磁盘容量一致；
15. 环境 manifest 已保存。
