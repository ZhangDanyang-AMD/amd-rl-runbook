# LumenRL DAPO 双后端新机 Runbook（FSDP2 + Megatron-Native）

> 目标：在一台全新的 8 卡 AMD `gfx950` 机器上，仅通过在线拉取固定 revision 的代码、
> 固定 digest 的 Docker 镜像以及 Hugging Face 数据，构建一个可同时支持以下两种训练后端的
> LumenRL DAPO BF16 环境：
>
> - `policy.training_backend=fsdp2`：8 卡 DP8；
> - `policy.training_backend=megatron_native`：8 卡 DP2×TP2×PP2×CP1。
>
> 两条路线共用 Qwen3-8B-Base、DAPO-Math-17k、AIME-2024、vLLM 在线 rollout、
> Ray controller、ZMQ CUDA-IPC 权重同步和正式 DAPO 超参数。
>
> 本文档完全独立，不需要旧机器文件，不使用 `rsync`，也不依赖任何机器专属绝对路径。
> 新机器只需修改 `RL_ROOT` 与 `DATA_ROOT`。大型源码构建（TE/Apex）、模型、数据、日志和
> checkpoint 均写入 `DATA_ROOT`；`RL_ROOT` 只放 LumenRL/Lumen/aiter 代码与小型启动脚本。
>
> 本文档只覆盖两种后端共同验证过的 **BF16 基线**。FSDP FP8、ATOM rollout 和 verl
> 不在本流程内。

---

## 0. 适用范围与关键约束

本流程的已验证目标硬件是：

- 8×AMD Instinct MI355X；
- GPU 架构 `gfx950`；
- 288 GiB VRAM/卡；
- Host 驱动可运行 ROCm 7.2.x 容器。

其他 GPU 或更小显存机器可能需要降低序列长度、batch 或 vLLM 显存比例，不能直接视为已验证。

两种后端的拓扑：

```text
FSDP2:
  actor workers = 8
  data parallel = 8

Megatron-Native:
  actor workers = 8
  tensor model parallel = 2
  pipeline model parallel = 2
  context parallel = 1
  data parallel = 8 / (2 × 2 × 1) = 2
```

关键约束：

1. 两种后端共用 GPU，**不能同时运行**。启动另一条路线前先停止当前训练。
2. vLLM 必须保持 `enable_sleep_mode=false`。ROCm 7.2.3 上 cumem sleep/wake
   可能释放失败并在 wake 时 OOM。
3. Megatron-Native 必须使用 ROCm TransformerEngine 2.15 源码构建版本。
   **不要执行 `pip install transformer_engine`**，否则可能安装 NVIDIA 版本。
4. 不要替换基础镜像中的 torch、triton、vLLM 或 flash-attn。
5. 不要把 TE/Apex build、模型、日志或 checkpoint 放在网络 `/home`。
6. Megatron dist-checkpoint 单个约 107 GiB；正式配置最多保留 5 个 checkpoint，
   建议 `DATA_ROOT` 至少预留 1 TiB。
7. 容器内安装的 TE/Apex 位于容器 overlay：
   `docker stop/start` 会保留，`docker rm` 会丢失。

---

## 1. 固定版本与 revision

严格使用以下版本，不要混用其他 ROCm/PyTorch/TE wheel：

```text
Docker base          vllm/vllm-openai-rocm:v0.23.0
Docker image digest  sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4
Python               3.12.13
PyTorch              2.10.0+git8514f05
PyTorch HIP          7.2.53211
ROCm userspace       7.2.3
vLLM                 0.23.0
Ray                  2.56.1
flash_attn           2.8.3
Megatron-Core        0.18.2
TransformerEngine    2.15.0.dev0+6e541a1
Apex                 1.14.0a0
```

固定源码 revision：

```text
Lumen-RL
  repo    https://github.com/ZhangDanyang-AMD/Lumen-RL.git
  branch  dev/vllm-fsdp-dapo
  commit  523e92329d312a3265e0a031dd7982b0529c3ef5

Lumen
  repo    https://github.com/ZhangDanyang-AMD/Lumen.git
  commit  e6379cbd9057b03c18213fbf65a4d891160545ca

aiter
  repo    https://github.com/ZhangDanyang-AMD/aiter.git
  commit  ff1006d03b53a693424c30e192c6e700e632bef8

ROCm TransformerEngine
  repo    https://github.com/ROCm/TransformerEngine.git
  commit  6e541a10419a6e31bdc98b1516db04eb81a463b6

ROCm Apex
  repo    https://github.com/ROCm/apex.git
  commit  daed85255d51476425080e7e6203f0bee6d7e4cc
```

---

## 2. Host 前置检查与路径变量

Host 需要安装 Docker、Git，并能访问 `/dev/kfd` 与 `/dev/dri`：

```bash
docker --version
git --version
ls -l /dev/kfd /dev/dri
rocm-smi --showproductname
df -h / /tmp /path/to/large-local-disk
```

期望看到 8 张 GPU，GFX version 为 `gfx950`。

设置路径。下面两个路径必须替换为新机器上的绝对路径：

路径中不要包含空格；`run_dapo.sh` 的 Hydra override 使用 shell 分词，带空格路径不受支持。
建议 `RL_ROOT` 与 `DATA_ROOT` 都位于本地 SSD/NVMe；其中 `DATA_ROOT` 必须是大容量盘。

```bash
export RL_ROOT=/ABSOLUTE/PATH/lumen_rl
export DATA_ROOT=/ABSOLUTE/PATH/large-local-data
export CONTAINER=rl-vllm-fsdp-megatron
export IMAGE='vllm/vllm-openai-rocm@sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4'

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
```

后续每次登录新 shell 都要重新 export 这四个变量。

如果当前用户没有 Docker 权限，可给本文所有 `docker` 命令加 `sudo`；不要只给部分命令加，
否则文件 ownership 容易混乱。

---

## 3. 在线 clone 固定代码 revision

所有源码均从远端仓库获取，不复制旧机器文件：

```bash
git clone -b dev/vllm-fsdp-dapo \
  https://github.com/ZhangDanyang-AMD/Lumen-RL.git \
  "$RL_ROOT/Lumen-RL"
git -C "$RL_ROOT/Lumen-RL" checkout \
  523e92329d312a3265e0a031dd7982b0529c3ef5

git clone \
  https://github.com/ZhangDanyang-AMD/Lumen.git \
  "$RL_ROOT/Lumen"
git -C "$RL_ROOT/Lumen" checkout \
  e6379cbd9057b03c18213fbf65a4d891160545ca

git clone \
  https://github.com/ZhangDanyang-AMD/aiter.git \
  "$RL_ROOT/aiter"
git -C "$RL_ROOT/aiter" checkout \
  ff1006d03b53a693424c30e192c6e700e632bef8
git -C "$RL_ROOT/aiter" submodule sync --recursive
git -C "$RL_ROOT/aiter" submodule update --init --recursive --jobs 16
```

验证：

```bash
git -C "$RL_ROOT/Lumen-RL" rev-parse HEAD
git -C "$RL_ROOT/Lumen" rev-parse HEAD
git -C "$RL_ROOT/aiter" rev-parse HEAD
git -C "$RL_ROOT/Lumen-RL" status --short

test -f "$RL_ROOT/Lumen-RL/lumenrl/engine/training/megatron_native_engine.py"
test -f "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
grep -q 'EXTRA_OVERRIDE' "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
```

期望三个 HEAD 与第 1 节完全一致，Lumen-RL `status --short` 为空。

---

## 4. 拉取固定镜像并启动共用容器

拉取并核对 digest：

```bash
docker pull "$IMAGE"
docker image inspect "$IMAGE" \
  --format '{{.Id}} {{json .RepoDigests}}'
```

输出中的 image ID 必须为：

```text
sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4
```

为避免误删已有容器，先检查：

```bash
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ERROR: container $CONTAINER already exists; inspect it before continuing"
  exit 1
fi
```

启动持久容器：

```bash
docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --network=host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
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

验证基础环境。版本不匹配时停止，不要继续编译 TE：

```bash
docker exec "$CONTAINER" bash -lc '
python - <<PY
import torch, vllm, transformers
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        i,
        p.name,
        getattr(p, "gcnArchName", None),
        round(p.total_memory / 2**30, 1),
    )
PY
hipcc --version | grep -E "HIP version|clang version"
'
```

期望：

```text
torch 2.10.0+git8514f05
hip 7.2.53211
vllm 0.23.0
gpus 8
每张卡的 gcnArchName 含 gfx950
```

---

## 5. 安装共用 build/runtime 依赖

基础镜像已经带 torch、vLLM、triton 和 flash-attn。以下命令不得升级或替换它们：

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail
git config --global --add safe.directory "*"

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
  "ray[default]==2.56.1" \
  "omegaconf==2.3.1" \
  "safetensors==0.8.0" \
  "accelerate>=0.28" \
  datasets \
  "math_verify[antlr4_13_2]" \
  huggingface_hub \
  wandb \
  pyzmq \
  absl-py
' 2>&1 | tee "$DATA_ROOT/logs/common-deps-install.log"
```

检查编译工具；若缺失再安装：

```bash
docker exec "$CONTAINER" bash -lc '
command -v gcc
command -v g++
command -v make
' || docker exec "$CONTAINER" bash -lc '
set -euxo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential git ca-certificates pkg-config
rm -rf /var/lib/apt/lists/*
'
```

---

## 6. 构建 Megatron-Native 运行栈

FSDP2 本身不依赖 TE/Apex，但共用容器要能运行 Megatron-Native，因此仍需完整执行本节。

### 6.1 安装 Megatron-Core 0.18.2

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

不要安装 `megatron-bridge`。Qwen3 HF↔Megatron 转换由 LumenRL
`qwen3_megatron_bridge.py` 完成。

### 6.2 在线获取并构建 ROCm Apex

源码和 build 目录都放在 `DATA_ROOT`：

```bash
git clone https://github.com/ROCm/apex.git "$DATA_ROOT/apex_src"
git -C "$DATA_ROOT/apex_src" checkout \
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
export MAX_JOBS="${MAX_JOBS:-64}"
python setup.py install --cpp_ext --cuda_ext \
  2>&1 | tee "$DATA_ROOT/logs/apex-build.log"
'
```

CPU/RAM 较少时把 `MAX_JOBS` 降为 16 或 32。

验证 Apex。首次调用可能触发一次 JIT 编译：

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

### 6.3 在线获取 ROCm TransformerEngine 2.15

必须使用 ROCm fork，并递归拉取全部 submodule：

```bash
git clone https://github.com/ROCm/TransformerEngine.git \
  "$DATA_ROOT/te_src"
git -C "$DATA_ROOT/te_src" checkout \
  6e541a10419a6e31bdc98b1516db04eb81a463b6
git -C "$DATA_ROOT/te_src" submodule sync --recursive
git -C "$DATA_ROOT/te_src" submodule update --init --recursive --jobs 16

git -C "$DATA_ROOT/te_src" rev-parse HEAD
git -C "$DATA_ROOT/te_src" status --short
git -C "$DATA_ROOT/te_src" submodule status --recursive
```

TE 源码及 submodule 约 5.1 GiB，必须完整包含 AOTriton、CK JIT 和
Composable Kernel。

先清理可能存在的 NVIDIA TE 包：

```bash
docker exec "$CONTAINER" bash -lc '
python -m pip uninstall -y \
  transformer-engine transformer_engine \
  transformer-engine-torch transformer_engine_torch || true
'
```

源码编译安装：

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
export MAX_JOBS="${MAX_JOBS:-64}"

# ROCm 7.2.3 的 `hipcc -v` 在无输入文件时返回 1。aiter CK-JIT 的
# 编译器 ABI 探测会把该返回码误判为编译器不可用；跳过该保护性探测，
# 不会跳过实际 kernel 编译。
export TORCH_DONT_CHECK_COMPILER_ABI=1

python -m pip install -v . --no-build-isolation \
  2>&1 | tee "$DATA_ROOT/logs/te-build.log"
'
```

期望日志末尾：

```text
Successfully installed ... transformer_engine-2.15.0.dev0+6e541a1
```

如果日志出现 `CK-JIT ... cxx_interceptor -v returned non-zero exit status 1`，
确认上面命令包含 `TORCH_DONT_CHECK_COMPILER_ABI=1`，清理 build 后重跑。

### 6.4 验证 TE Linear、RMSNorm 和 DPA 前反向

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

分别强制 CK 与 AOTriton backend：

```bash
for backend in ck aotriton; do
  if [ "$backend" = ck ]; then
    ENV_ARGS=(-e NVTE_FUSED_ATTN_CK=1 -e NVTE_FUSED_ATTN_AOTRITON=0)
  else
    ENV_ARGS=(-e NVTE_FUSED_ATTN_CK=0 -e NVTE_FUSED_ATTN_AOTRITON=1)
  fi

  echo "=== $backend ==="
  docker exec "${ENV_ARGS[@]}" "$CONTAINER" bash -lc '
python - <<PY
import torch
from transformer_engine.pytorch.attention import DotProductAttention
q = torch.randn(32, 1, 8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(32, 1, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(32, 1, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
a = DotProductAttention(
    num_attention_heads=8,
    kv_channels=16,
    num_gqa_groups=2,
    attention_dropout=0.0,
    attn_mask_type="causal",
    qkv_format="sbhd",
).cuda()
o = a(q, k, v)
o.float().sum().backward()
assert torch.isfinite(o).all()
print(tuple(o.shape), "finite=True")
PY
'
done
```

两个 backend 都必须输出 `finite=True`。

---

## 7. 安装 aiter、Lumen 与 LumenRL

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail
git config --global --add safe.directory "*"

cd "$AITER_DIR"
git submodule update --init --recursive --jobs 16
AITER_USE_SYSTEM_TRITON=1 python setup.py develop

python -m pip install -e "$LUMEN_DIR" --no-deps
python -m pip install -e "$RL_ROOT/Lumen-RL" --no-deps
' 2>&1 | tee "$DATA_ROOT/logs/lumenrl-install.log"
```

验证 FSDP2 与 Megatron-Native engine 都已注册：

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

for backend in ("fsdp2", "megatron_native"):
    cls = EngineRegistry.get_engine_cls(
        model_type="language_model",
        backend=backend,
    )
    print(backend, cls.__name__)

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
fsdp2 <engine class>
megatron_native <engine class>
megatron 0.18.2
TE 2.15.0.dev0+6e541a1
gpus 8
IMPORT STACK OK
```

---

## 8. 在线下载模型和数据

模型与数据全部从 Hugging Face 下载。公开仓库通常不需要 token；受限网络可先设置：

```bash
export HF_TOKEN=your_token_if_needed
```

下载：

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
    "Qwen/Qwen3-8B-Base",
    local_dir=f"{D}/models/Qwen3-8B-Base",
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

网络需要镜像时，可在容器创建前或 `docker exec` 时设置兼容的 `HF_ENDPOINT`；
后续本地目录不变。

### 8.1 过滤 prompt 长度 ≤1024

在新机器上生成过滤脚本：

```bash
cat > "$RL_ROOT/filter_prompts.py" <<'PYEOF'
import glob
import os

import datasets
from transformers import AutoTokenizer

DATA_ROOT = os.environ["DATA_ROOT"]
MODEL_PATH = f"{DATA_ROOT}/models/Qwen3-8B-Base"
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
    prompt = example["prompt"]
    return len(
        tokenizer.apply_chat_template(
            prompt,
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

docker exec \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc \
  'python "$RL_ROOT/filter_prompts.py"' \
  2>&1 | tee "$DATA_ROOT/logs/filter-prompts.log"
```

检查模型与过滤数据：

```bash
docker exec "$CONTAINER" bash -lc '
set -e
test -f "$DATA_ROOT/models/Qwen3-8B-Base/config.json"
test -f "$DATA_ROOT/models/Qwen3-8B-Base/model.safetensors.index.json"
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

## 9. 生成双后端统一启动器

固定 commit 中已有 `examples/DAPO/run_dapo.sh`。下面只增加一个轻量 wrapper，
根据 `BACKEND=fsdp2|megatron_native` 与 `RUN_KIND=smoke|longrun` 选择配置和 override。

所有路径都从 `RL_ROOT`/`DATA_ROOT` 获取，不包含机器专属路径：

> 固定 revision 中的 DAPO YAML 已统一使用 `${oc.env:DATA_ROOT}`，Megatron YAML 默认 backend
> 已是 `megatron_native`。wrapper 仍显式覆盖 backend、独立 checkpoint 路径和 Megatron
> TP/PP/CP，避免不同路线误共享状态。

```bash
cat > "$RL_ROOT/run_lumenrl_dapo_backend.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${RL_ROOT:?RL_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"

BACKEND="${BACKEND:-fsdp2}"
RUN_KIND="${RUN_KIND:-smoke}"

case "${BACKEND}:${RUN_KIND}" in
  fsdp2:smoke)
    CONFIG="examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml"
    ;;
  fsdp2:longrun)
    CONFIG="examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_longrun.yaml"
    ;;
  megatron_native:smoke)
    CONFIG="examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml"
    ;;
  megatron_native:longrun)
    CONFIG="examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_longrun.yaml"
    ;;
  *)
    echo "ERROR: use BACKEND=fsdp2|megatron_native and RUN_KIND=smoke|longrun" >&2
    exit 2
    ;;
esac

if [ "$RUN_KIND" = "smoke" ]; then
  STEPS="${STEPS:-2}"
  RESUME="${RESUME:-false}"
  SAVE_STEPS="${SAVE_STEPS:-1000}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
  WANDB_ENABLED="${WANDB_ENABLED:-false}"
else
  STEPS="${STEPS:-1000}"
  RESUME="${RESUME:-true}"
  SAVE_STEPS="${SAVE_STEPS:-50}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
  WANDB_ENABLED="${WANDB_ENABLED:-true}"
fi

CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/ckpts/lumenrl-dapo/${BACKEND}-${RUN_KIND}}"
RUN_ID="${RUN_ID:-dapo-qwen3-8b-${BACKEND}-${RUN_KIND}-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-$RUN_ID}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"

mkdir -p "$DATA_ROOT/logs" "$(dirname "$CKPT_DIR")" "$DATA_ROOT/wandb"

OVERRIDES=(
  "policy.training_backend=$BACKEND"
  "policy.generation.vllm_cfg.enable_sleep_mode=false"
  "checkpointing.resume=$RESUME"
  "checkpointing.save_steps=$SAVE_STEPS"
  "checkpointing.save_total_limit=$SAVE_TOTAL_LIMIT"
  "checkpointing.checkpoint_dir=$CKPT_DIR"
  "logger.wandb_enabled=$WANDB_ENABLED"
)

if [ "$BACKEND" = "fsdp2" ]; then
  OVERRIDES+=(
    "policy.training.fsdp_cfg.param_offload=false"
    "policy.training.fsdp_cfg.optimizer_offload=false"
  )
else
  OVERRIDES+=(
    "policy.training.megatron_cfg.tensor_model_parallel_size=2"
    "policy.training.megatron_cfg.pipeline_model_parallel_size=2"
    "policy.training.megatron_cfg.context_parallel_size=1"
    "policy.training.megatron_cfg.sequence_parallel=false"
  )
fi

if [ "$RUN_KIND" = "longrun" ]; then
  OVERRIDES+=(
    "logger.wandb.project=LumenRL"
    "logger.wandb.name=$RUN_NAME"
  )
  if [ "$WANDB_ENABLED" = "true" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_ENABLED=true but WANDB_API_KEY is empty" >&2
    exit 2
  fi
fi

if [ -n "${USER_EXTRA_OVERRIDE:-}" ]; then
  OVERRIDES+=("$USER_EXTRA_OVERRIDE")
fi

EXTRA_OVERRIDE="${OVERRIDES[*]}"
S="$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"

printf "%s\n" "$LOG" > "$DATA_ROOT/logs/${BACKEND}-${RUN_KIND}.latest"

echo "BACKEND=$BACKEND"
echo "RUN_KIND=$RUN_KIND"
echo "CONFIG=$CONFIG"
echo "STEPS=$STEPS"
echo "CKPT_DIR=$CKPT_DIR"
echo "LOG=$LOG"
echo "EXTRA_OVERRIDE=$EXTRA_OVERRIDE"

set +e
CONFIG_OVERRIDE="$CONFIG" \
STEPS="$STEPS" \
MODE=bf16 \
RUN_ID="$RUN_ID" \
LOG="$LOG" \
EXTRA_OVERRIDE="$EXTRA_OVERRIDE" \
bash "$S"
exit_code=$?
set -e

if [ "$exit_code" -ne 0 ]; then
  echo "ERROR: run_dapo.sh exited with $exit_code" >&2
  grep -aE \
    "Traceback|OOM|OutOfMemory|CUDA error|NaN|Error:" \
    "$LOG" | tail -50 >&2 || true
  exit "$exit_code"
fi

# 固定 revision 的 run_dapo.sh 会传播 trainer 退出码；同时要求成功收尾日志，
# 防止异常退出被外层调度器误判为完成。
if ! grep -q "LumenRL finished." "$LOG"; then
  echo "ERROR: training did not reach 'LumenRL finished.'" >&2
  grep -aE \
    "Traceback|OOM|OutOfMemory|CUDA error|NaN|Error:" \
    "$LOG" | tail -50 >&2 || true
  exit 1
fi
EOF

chmod +x "$RL_ROOT/run_lumenrl_dapo_backend.sh"
```

说明：

- smoke 默认 2 steps、`resume=false`、不保存 checkpoint、不启用 W&B；
- longrun 默认 1000 steps、`resume=true`、每 50 步保存、最多保留 5 个 checkpoint；
- FSDP2 固定 DP8；
- Megatron-Native 固定 TP2、PP2、CP1，因此 8 卡自动得到 DP2；
- 两种路线都强制 `enable_sleep_mode=false`；
- `CKPT_DIR` 是稳定路径，重启相同命令可 resume；
- `LOG` 默认带时间戳，避免覆盖历史日志。
- `$DATA_ROOT/logs/<backend>-<kind>.latest` 保存该路线最近一次启动的实际日志路径；
- `run_dapo.sh` 返回 trainer 的真实退出码，wrapper 还会检查 `LumenRL finished.`。

---

## 10. Smoke：先 FSDP2，再 Megatron-Native

先确认没有旧训练占卡：

```bash
docker exec "$CONTAINER" bash -lc '
pgrep -af "lumenrl.trainer.main|VLLMRayServer|EngineCore" || true
'
rocm-smi --showmeminfo vram
```

### 10.1 FSDP2 DP8 smoke

前台运行并等待完成：

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
BACKEND=fsdp2 \
RUN_KIND=smoke \
RUN_ID=smoke-fsdp2-dp8 \
CKPT_DIR="$DATA_ROOT/ckpts/lumenrl-dapo/smoke-fsdp2-dp8" \
LOG="$DATA_ROOT/logs/smoke-fsdp2-dp8.log" \
bash "$RL_ROOT/run_lumenrl_dapo_backend.sh"
'
```

检查：

```bash
grep -aE \
  "RLTrainer.setup|callbacks: step=|finished after|LumenRL finished|Traceback|OOM|OutOfMemory|NaN" \
  "$DATA_ROOT/logs/smoke-fsdp2-dp8.log"
```

健康判据：

- `actor_workers=8`；
- 完成 2 steps；
- entropy 有限且通常约 0.5–0.9，不应快速塌到接近 0；
- grad_norm 有限，通常约 0.7–1.1；
- `ppo_kl≈0`；
- BF16 `rollout_corr/kl` 通常约 0.001；
- 无 Traceback、OOM、NaN。

### 10.2 Megatron-Native DP2×TP2×PP2 smoke

FSDP smoke 完成后再运行：

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
BACKEND=megatron_native \
RUN_KIND=smoke \
RUN_ID=smoke-megatron-native-dp2-tp2-pp2 \
CKPT_DIR="$DATA_ROOT/ckpts/lumenrl-dapo/smoke-megatron-native-dp2-tp2-pp2" \
LOG="$DATA_ROOT/logs/smoke-megatron-native-dp2-tp2-pp2.log" \
bash "$RL_ROOT/run_lumenrl_dapo_backend.sh"
'
```

检查：

```bash
grep -aE \
  "model-parallel=|mesh_mapping|callbacks: step=|finished after|LumenRL finished|Traceback|OOM|OutOfMemory|NaN" \
  "$DATA_ROOT/logs/smoke-megatron-native-dp2-tp2-pp2.log"
```

8 卡期望拓扑证据：

```text
Megatron model-parallel=4: actor DP=2, mesh_mapping=[0, 0, 1, 1, 0, 0, 1, 1]
```

以实际 Megatron DP rank 排列为准，但必须有 2 个 DP shard，每个 shard 含 4 个模型并行 rank。

健康判据：

- 完成 2 steps；
- entropy 通常约 0.6–0.75；
- grad_norm 有限，通常约 0.8–1.0；
- `ppo_kl≈0`；
- `rollout_corr/kl` 通常约 0.002；
- 无 Traceback、OOM、NaN 或 collective hang。

smoke 默认不会到保存步数。若目录被创建，可在确认日志后清理：

```bash
docker exec "$CONTAINER" rm -rf \
  "$DATA_ROOT/ckpts/lumenrl-dapo/smoke-fsdp2-dp8" \
  "$DATA_ROOT/ckpts/lumenrl-dapo/smoke-megatron-native-dp2-tp2-pp2"
```

---

## 11. 可选：Megatron dist-checkpoint save/resume 验证

正式 longrun 前建议验证一次 Megatron optimizer/scheduler/dist-checkpoint 恢复。
该测试会临时占用约 107 GiB。

Run A：2 steps，在 step 2 保存：

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
BACKEND=megatron_native \
RUN_KIND=smoke \
STEPS=2 \
SAVE_STEPS=2 \
RESUME=false \
RUN_ID=megatron-ckpt-save \
CKPT_DIR="$DATA_ROOT/ckpts/lumenrl-dapo/megatron-native-ckpttest" \
LOG="$DATA_ROOT/logs/megatron-native-ckpt-save.log" \
bash "$RL_ROOT/run_lumenrl_dapo_backend.sh"
'
```

Run B：恢复到 step 2，再完成 step 3/4，不再保存：

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
BACKEND=megatron_native \
RUN_KIND=smoke \
STEPS=4 \
SAVE_STEPS=100 \
RESUME=true \
RUN_ID=megatron-ckpt-resume \
CKPT_DIR="$DATA_ROOT/ckpts/lumenrl-dapo/megatron-native-ckpttest" \
LOG="$DATA_ROOT/logs/megatron-native-ckpt-resume.log" \
bash "$RL_ROOT/run_lumenrl_dapo_backend.sh"
'
```

检查：

```bash
grep -aE \
  "resume_step=|Resume:|callbacks: step=3|callbacks: step=4|finished after|Traceback|OOM|NaN" \
  "$DATA_ROOT/logs/megatron-native-ckpt-resume.log"
```

必须看到 `resume_step=2`，并完成 step 3/4。lr 应从 step 2 的 `2e-7`
继续到 step 3/4 的 `3e-7`、`4e-7`，grad_norm 与训练指标必须有限。

测试完成后删除测试 checkpoint：

```bash
du -sh "$DATA_ROOT/ckpts/lumenrl-dapo/megatron-native-ckpttest"
docker exec "$CONTAINER" rm -rf \
  "$DATA_ROOT/ckpts/lumenrl-dapo/megatron-native-ckpttest"
```

---

## 12. 正式 longrun 配置

两种后端使用各自仓库内 longrun YAML，但正式超参数一致：

```text
num_training_steps           1000
train_global_batch_size      512（32 prompts × n=16）
gen_batch_size               96
max prompt                   1024
max response                 20480
max total sequence           21504
max token budget / GPU       21504
learning rate                1e-6
warmup steps                 10
weight decay                 0.1
max grad norm                1.0
GRPO / DAPO / token-mean     enabled
clip low/high/c              0.2 / 0.28 / 10.0
TIS rollout_is/threshold     token / 2.0
validation interval          10
checkpoint interval          50
checkpoint retain count      5
vLLM gpu memory utilization  0.30
vLLM max model len           21504
vLLM max batched tokens      32768
vLLM max sequences           64
vLLM sleep mode              false
```

W&B 在线同步前设置：

```bash
export WANDB_API_KEY=your_wandb_api_key
```

没有 W&B key 时，把启动命令中的 `WANDB_ENABLED=true` 改为
`WANDB_ENABLED=false`，并删除 `-e WANDB_API_KEY=...`。

### 12.1 FSDP2 DP8 longrun

独立日志和 checkpoint：

```text
$DATA_ROOT/logs/dapo-qwen3-8b-fsdp2-longrun-<timestamp>.log
$DATA_ROOT/logs/fsdp2-longrun.latest
$DATA_ROOT/ckpts/lumenrl-dapo/longrun-fsdp2-dp8
```

后台启动：

```bash
docker exec -d \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  -e WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is required}" \
  "$CONTAINER" bash -lc '
BACKEND=fsdp2 \
RUN_KIND=longrun \
STEPS=1000 \
RESUME=true \
WANDB_ENABLED=true \
RUN_NAME="dapo-qwen3-8b-fsdp2-dp8-$(date +%Y%m%d-%H%M%S)" \
CKPT_DIR="$DATA_ROOT/ckpts/lumenrl-dapo/longrun-fsdp2-dp8" \
bash "$RL_ROOT/run_lumenrl_dapo_backend.sh"
'
```

### 12.2 Megatron-Native DP2×TP2×PP2 longrun

必须先停止 FSDP longrun；两条路线不能同时运行。

独立日志和 checkpoint：

```text
$DATA_ROOT/logs/dapo-qwen3-8b-megatron_native-longrun-<timestamp>.log
$DATA_ROOT/logs/megatron_native-longrun.latest
$DATA_ROOT/ckpts/lumenrl-dapo/longrun-megatron-native-dp2-tp2-pp2
```

后台启动：

```bash
docker exec -d \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  -e WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is required}" \
  "$CONTAINER" bash -lc '
BACKEND=megatron_native \
RUN_KIND=longrun \
STEPS=1000 \
RESUME=true \
WANDB_ENABLED=true \
RUN_NAME="dapo-qwen3-8b-megatron-native-dp2-tp2-pp2-$(date +%Y%m%d-%H%M%S)" \
CKPT_DIR="$DATA_ROOT/ckpts/lumenrl-dapo/longrun-megatron-native-dp2-tp2-pp2" \
bash "$RL_ROOT/run_lumenrl_dapo_backend.sh"
'
```

---

## 13. Longrun 启动后验证

获取主进程 PID：

```bash
docker exec "$CONTAINER" bash -lc '
pgrep -af "python3 -u -m lumenrl.trainer.main"
'
```

选择当前路线的日志：

```bash
# 二选一
export RUN_LOG="$(cat "$DATA_ROOT/logs/fsdp2-longrun.latest")"
# export RUN_LOG="$(cat "$DATA_ROOT/logs/megatron_native-longrun.latest")"

echo "$RUN_LOG"
```

检查 W&B、拓扑与初始化：

```bash
grep -aE \
  "View run|Syncing run|model-parallel=|mesh_mapping|RLTrainer.setup|rollout ready" \
  "$RUN_LOG" | tail -20
```

至少监控前两个训练 step：

```bash
grep -aE "callbacks: step=" "$RUN_LOG" | tail -5
grep -aiE \
  "Traceback|OOM|OutOfMemory|CUDA error|NaN|collective|timeout" \
  "$RUN_LOG" | tail -20
rocm-smi --showmeminfo vram
```

健康判据：

- 无 OOM、NaN、Traceback、collective hang；
- `ppo_kl≈0`；
- grad_norm 有限；
- entropy 未持续跌到 0.3 以下；
- `rollout_corr/kl` 远低于 TIS 阈值 2.0；
- 显存峰值稳定，没有每步持续增长；
- W&B 输出 `View run at https://wandb.ai/...`。

已验证的 Megatron-Native DP2×TP2×PP2 正式长跑前几步常见范围：

```text
entropy          约 0.5–0.7
grad_norm        约 0.2–0.4
ppo_kl           约 0
rollout_corr/kl  约 0.001–0.002
```

这些是参考范围，不是逐步必须精确匹配的常数。

如果 entropy 持续低于 0.3、出现 NaN/OOM、collective timeout 或显存持续增长，
停止盲目长跑并保留完整日志。

---

## 14. 停止、续跑与 checkpoint

### 14.1 停止当前训练

```bash
docker exec "$CONTAINER" bash -lc '
ray stop --force 2>/dev/null || true
pkill -9 -f "[l]umenrl.trainer.main" 2>/dev/null || true
pkill -9 -f "[V]LLMRayServer" 2>/dev/null || true
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
pkill -9 -f "[E]ngineCore" 2>/dev/null || true
sleep 8
'
```

确认 GPU 显存回落：

```bash
rocm-smi --showmeminfo vram
```

### 14.2 续跑

保持以下内容完全一致：

- `BACKEND`；
- Megatron TP/PP/CP 拓扑；
- `CKPT_DIR`；
- 正式 longrun config；
- optimizer、lr、warmup、batch 等超参数。

重新执行对应的第 12 节命令即可。wrapper 已设置 `RESUME=true`。

检查恢复证据：

```bash
grep -aE "resume_step=|Resume:|callbacks: step=" "$RUN_LOG" | tail -10
```

Megatron-Native 使用 dist-checkpoint；FSDP2 使用其原生 checkpoint 路径。
不要让两个后端共享 checkpoint 目录。

### 14.3 checkpoint ownership

容器内创建的 checkpoint 可能属于 root。只在明确确认目录后使用容器删除：

```bash
docker exec "$CONTAINER" rm -rf /absolute/checkpoint/path
```

不要对 `DATA_ROOT` 上层目录执行模糊通配删除。

---

## 15. 保存环境 manifest

环境完成后保存一份可跨机比对的清单：

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
for backend in ("fsdp2", "megatron_native"):
    cls = EngineRegistry.get_engine_cls(
        model_type="language_model",
        backend=backend,
    )
    print("engine", backend, cls.__name__)
PY
    python -m pip freeze | grep -Ei \
      "megatron|transformer.engine|apex|vllm|ray|flash|torch|triton|aiter|safetensors|omegaconf"
    echo "Lumen-RL: $(git -C "$RL_ROOT/Lumen-RL" rev-parse HEAD)"
    echo "Lumen: $(git -C "$RL_ROOT/Lumen" rev-parse HEAD)"
    echo "aiter: $(git -C "$RL_ROOT/aiter" rev-parse HEAD)"
    echo "TE source: $(git -C "$DATA_ROOT/te_src" rev-parse HEAD)"
    echo "Apex source: $(git -C "$DATA_ROOT/apex_src" rev-parse HEAD)"
    echo "--- Lumen-RL status --short ---"
    git -C "$RL_ROOT/Lumen-RL" status --short
  '
} 2>&1 | tee "$DATA_ROOT/logs/lumenrl-fsdp-megatron-env-manifest.txt"
```

---

## 16. 常见问题

### 16.1 TE 被装成 NVIDIA 版本

症状：import 时出现 CUDA/NVIDIA library 错误，或版本不是
`2.15.0.dev0+6e541a1`。

处理：

```bash
docker exec "$CONTAINER" bash -lc '
python -m pip uninstall -y \
  transformer-engine transformer_engine \
  transformer-engine-torch transformer_engine_torch || true
'
```

然后清理 `$DATA_ROOT/te_src/build` 并重新执行第 6.3 节。

### 16.2 TE CK-JIT 在 `cxx_interceptor -v` 失败

确认构建环境包含：

```bash
export TORCH_DONT_CHECK_COMPILER_ABI=1
```

这是绕过 ROCm 7.2.3 `hipcc -v` 无输入时返回 1 的 ABI 探测误判；
实际 CK/AOTriton kernel 仍会正常编译并由第 6.4 节验证。

### 16.3 TE undefined symbol 或 segfault

通常是镜像、torch、ROCm 或旧 build 目录不匹配。重新核对固定 image digest，
删除以下目录后重编：

```bash
docker exec "$CONTAINER" bash -lc '
cd "$DATA_ROOT/te_src"
rm -rf build dist transformer_engine.egg-info
'
```

### 16.4 FSDP2 报 CPU materialization/offload 错误

Ray worker 路径不要开启：

```text
policy.training.fsdp_cfg.param_offload=true
policy.training.fsdp_cfg.optimizer_offload=true
```

统一 wrapper 已强制两者为 false。

### 16.5 Megatron 拓扑未生效

字段必须使用：

```text
tensor_model_parallel_size
pipeline_model_parallel_size
context_parallel_size
```

不要使用旧字段 `tensor_parallel_size` / `pipeline_parallel_size`。
日志必须出现 `model-parallel=4` 和 `actor DP=2`。

### 16.6 Megatron TP2 不要开启 sequence parallel

RL 输入为变长序列，长度不一定能被 TP 整除。本流程固定：

```text
policy.training.megatron_cfg.sequence_parallel=false
```

### 16.7 vLLM sleep/wake OOM

确认日志和 override 中始终为：

```text
policy.generation.vllm_cfg.enable_sleep_mode=false
```

不要为了释放显存改回 true。

### 16.8 checkpoint 写到了错误目录

`run_dapo.sh` 的 `EXTRA_OVERRIDE` 不会再次展开其中的字面量 `$DATA_ROOT`。
本文 wrapper 先在 shell 中构造绝对 `CKPT_DIR`，因此不会触发此问题。
手工调用时也必须传已经展开的绝对路径。

### 16.9 启动另一后端后前一后端消失

这是预期行为：`run_dapo.sh` 启动前会清理旧 Ray/vLLM/trainer 进程。
两种后端只能串行运行。

### 16.10 首步明显较慢

首次启动包括：

- 8 个 vLLM replica 加载模型；
- Apex/TE/aiter 首次 JIT；
- 长响应首次 rollout；
- 首次权重同步。

判断健康应至少看到前两个完整训练 step，不要只依据初始化耗时。

---

## 17. 完成标准

以下全部满足才算新机双后端环境构建完成：

1. 固定 image digest、torch、HIP、vLLM 版本完全匹配。
2. 容器内可见 8×`gfx950`。
3. Lumen-RL、Lumen、aiter、ROCm TE、ROCm Apex revision 完全匹配。
4. `megatron-core==0.18.2`，pipeline schedule 与 dist-checkpoint 可 import。
5. Apex FusedLayerNorm/FusedAdam 前反向通过。
6. TE 版本为 `2.15.0.dev0+6e541a1`。
7. TE Linear/RMSNorm 前反向通过。
8. TE DPA 在 CK 和 AOTriton backend 都通过。
9. `fsdp2` 与 `megatron_native` engine 都注册成功。
10. 模型与两份过滤 parquet 均由新机器在线下载/生成。
11. FSDP2 DP8 smoke 完成 2 steps，指标有限且无异常。
12. Megatron-Native DP2×TP2×PP2 smoke 完成 2 steps，mesh 与指标健康。
13. 正式 longrun 使用独立日志和 checkpoint 目录。
14. longrun 前两个 step 无 OOM/NaN/Traceback/hang，W&B 正常同步。
15. `$DATA_ROOT/logs/lumenrl-fsdp-megatron-env-manifest.txt` 已保存。
