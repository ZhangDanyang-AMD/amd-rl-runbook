# LumenRL Megatron-Native 新机环境构建 Runbook（ROCm / gfx950 / TE 2.15）

> 目标：在一台新的 AMD gfx950 机器上，从
> `vllm/vllm-openai-rocm:v0.23.0` 基础镜像构建与当前
> `MegatronNativeEngine` 验证环境一致的运行环境，包括：
>
> - PyTorch ROCm 7.2.3
> - vLLM 0.23.0
> - Megatron-Core 0.18.2（含 `megatron.training`）
> - ROCm TransformerEngine 2.15（源码编译，CK + AOTriton）
> - ROCm Apex 1.14（源码编译）
> - LumenRL / Lumen / aiter
>
> 本文档适用于 `policy.training_backend=megatron_native`。旧
> `dapo-lumenrl-native-vllm-megatron-runbook.md` 的 local-spec 路线写着
> “无需 TE/Apex”，**不适用于 native TE 后端**。

---

## 0. 当前已验证的环境锁

新机尽量严格复现以下版本。不要混用其他 PyTorch/ROCm/TE wheel：

```text
GPU                         gfx950，当前机器 8×MI355X，288 GiB/卡
Docker base                 vllm/vllm-openai-rocm:v0.23.0
Docker image digest         sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4
Python                      3.12.13
PyTorch                     2.10.0+git8514f05
PyTorch HIP                 7.2.53211
ROCm userspace              7.2.3
vLLM                        0.23.0
Ray                         2.56.1
flash_attn                  2.8.3
Megatron-Core               0.18.2
TransformerEngine           2.15.0.dev0+6e541a1
Apex                        1.14.0a0
```

源码 revision：

```text
Lumen-RL  branch dev/vllm-fsdp-dapo
          HEAD 77c874bd08ffbbd11d62288e06700a19d7cb0622
          native TP/PP/CP 改动已提交并推到 origin，工作区干净；
          新机可直接 clone + checkout 此 commit。
Lumen     branch amd-atom-rollout
          e6379cbd9057b03c18213fbf65a4d891160545ca
aiter     branch lumen/triton_kernels
          ff1006d03b53a693424c30e192c6e700e632bef8
ROCm TE   https://github.com/ROCm/TransformerEngine.git
          6e541a10419a6e31bdc98b1516db04eb81a463b6
ROCm Apex https://github.com/ROCm/apex.git
          daed85255d51476425080e7e6203f0bee6d7e4cc
```

已验证拓扑：DP8、DP4×TP2、DP4×PP2、DP4×CP2、DP2×TP2×PP2、
DP1×TP2×PP2×CP2；dist-checkpoint save/resume（含 CP2）均通过。
CP2 thd zigzag、完整 logprob 重构、loss/梯度归一均已实现。详细状态见：

```text
/home/xysheng/working/amd-rl-runbook/megatron-native-refactor-handoff.md
```

---

## 1. 新机前置条件

要求：

- AMD GPU 架构为 `gfx950`。
- Host 驱动能运行 ROCm 7.2.x 容器。
- Docker 可访问 `/dev/kfd`、`/dev/dri`。
- `DATA_ROOT` 使用大容量本地盘；TE 源码约 5.1 GiB，构建/缓存/模型/ckpt
  需要更多空间，建议至少预留 100 GiB（训练数据和 checkpoint 另算）。
- 不要把 TE build、模型、checkpoint 放在 `/home` NFS。

Host 检查：

```bash
docker --version
ls -l /dev/kfd /dev/dri
df -h / /tmp /mnt
```

定义路径。新机可以换路径，但后续必须始终使用同一组变量：

```bash
export RL_ROOT=/path/to/lumen_rl
export DATA_ROOT=/path/to/large-local-data
export CONTAINER=rl-vllm-megatron
export IMAGE='vllm/vllm-openai-rocm@sha256:3813e31cc3ab56f71ba83a72da77a84f3c75030e54c9c316951a99317b818ce4'

mkdir -p \
  "$RL_ROOT" \
  "$DATA_ROOT/logs" \
  "$DATA_ROOT/tmp" \
  "$DATA_ROOT/hf_home" \
  "$DATA_ROOT/wheels"
```

---

## 2. 拉取固定代码 revision

Native TP/PP/CP 改动已经提交并推到 origin；新机直接 clone：

```bash
cd "$RL_ROOT"
git clone -b dev/vllm-fsdp-dapo \
  https://github.com/ZhangDanyang-AMD/Lumen-RL.git Lumen-RL
git -C Lumen-RL checkout 77c874bd08ffbbd11d62288e06700a19d7cb0622

git clone https://github.com/ZhangDanyang-AMD/Lumen.git Lumen
git -C Lumen checkout e6379cbd9057b03c18213fbf65a4d891160545ca

git clone https://github.com/ZhangDanyang-AMD/aiter.git aiter
git -C aiter checkout ff1006d03b53a693424c30e192c6e700e632bef8
git -C aiter submodule update --init --recursive --jobs 16
```

验证：

```bash
git -C "$RL_ROOT/Lumen-RL" branch --show-current
git -C "$RL_ROOT/Lumen-RL" rev-parse HEAD
git -C "$RL_ROOT/Lumen-RL" status --short
git -C "$RL_ROOT/Lumen" rev-parse HEAD
git -C "$RL_ROOT/aiter" rev-parse HEAD
test -f "$RL_ROOT/Lumen-RL/lumenrl/engine/training/megatron_native_engine.py"
```

期望：

- Lumen-RL branch 为 `dev/vllm-fsdp-dapo`（detached HEAD 也可，只要 commit 正确）。
- HEAD 为 `77c874bd...`。
- Lumen-RL `status --short` 为空。
- `megatron_native_engine.py` 存在。

如果未来旧机又出现未提交改动，再用 rsync 搬完整 worktree：

```bash
rsync -aH --info=progress2 \
  OLD_HOST:/home/xysheng/working/11/lumen_rl/ \
  "$RL_ROOT/"
```

不要加 `--delete`，避免误删新机已有内容。

---

## 3. 拉取固定 Docker 镜像并启动容器

```bash
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{.Id}} {{json .RepoDigests}}'
```

仅在新机没有同名容器时创建。已有有价值的容器不要直接 `docker rm`：

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
  -e LUMEN_DIR="$RL_ROOT/Lumen" \
  -e AITER_DIR="$RL_ROOT/aiter" \
  "$IMAGE" -lc 'sleep infinity'
```

检查 base image。若版本不匹配，先停止，不要继续编译 TE：

```bash
docker exec "$CONTAINER" bash -lc '
python - <<PY
import torch, vllm
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("vllm", vllm.__version__)
print("gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, getattr(p, "gcnArchName", None), round(p.total_memory / 2**30, 1))
PY
hipcc --version | grep -E "HIP version|clang version" | head -2
'
```

期望：

```text
torch 2.10.0+git8514f05
hip 7.2.53211
vllm 0.23.0
8 GPUs（若新机也是 8 卡）
gcnArchName 含 gfx950
```

---

## 4. 安装通用 build/runtime 依赖

基础镜像已经带 ROCm、PyTorch、vLLM、flash_attn。不要替换 torch、triton 或 vLLM。

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
  pyzmq
'
```

如果 `gcc`、`g++` 或 `make` 缺失，再执行：

```bash
docker exec "$CONTAINER" bash -lc '
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential git ca-certificates pkg-config
rm -rf /var/lib/apt/lists/*
'
```

---

## 5. 安装 Megatron-Core 0.18.2

当前验证环境使用 PyPI `megatron-core==0.18.2`；该包同时包含
`megatron.core` 和 `megatron.training`：

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

不要安装 `megatron-bridge`；当前 Qwen3 HF↔Megatron 转换由
`qwen3_megatron_bridge.py` 自己完成。

---

## 6. 构建 ROCm Apex（已验证 revision）

Native TE spec 的 norm 来自 TE；Apex 主要给 Megatron 优化器/融合算子使用。为精确复现
当前环境仍建议安装。Qwen RMSNorm **不能**走 Apex `FusedLayerNorm`；native spec 不会这样做。

### 6.1 获取源码

推荐在大盘上构建：

```bash
export APEX_SRC="$DATA_ROOT/apex_src"

if [ ! -d "$APEX_SRC/.git" ]; then
  git clone https://github.com/ROCm/apex.git "$APEX_SRC"
fi

git -C "$APEX_SRC" fetch origin
git -C "$APEX_SRC" checkout daed85255d51476425080e7e6203f0bee6d7e4cc
git -C "$APEX_SRC" submodule sync --recursive
git -C "$APEX_SRC" submodule update --init --recursive --jobs 16
git -C "$APEX_SRC" rev-parse HEAD
```

也可从旧机复制已验证源码以绕开 GitHub/submodule 网络问题：

```bash
rsync -aH --info=progress2 \
  OLD_HOST:/mnt/m2m_nobackup/xysheng/data/apex_src/ \
  "$APEX_SRC/"
```

### 6.2 编译并安装

这是当前机器实际成功使用的命令：

```bash
docker exec \
  -e APEX_SRC="$APEX_SRC" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
set -euxo pipefail
cd "$APEX_SRC"
rm -rf build dist

export PYTORCH_ROCM_ARCH=gfx950
export MAX_JOBS=64

python setup.py install --cpp_ext --cuda_ext \
  2>&1 | tee "$DATA_ROOT/logs/apex_build.log"
'
```

若新机 CPU/RAM 较少，将 `MAX_JOBS` 降到 16 或 32。

验证（首次调用融合 op 可能再打印一次 hipify/JIT 编译日志，属正常）：

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
print("Apex FusedLayerNorm/FusedAdam OK", tuple(y.shape), torch.isfinite(y).all().item())
PY
'
```

---

## 7. 构建 ROCm TransformerEngine 2.15（核心步骤）

### 7.1 获取完整递归源码

TE 源码及 submodules 约 5.1 GiB。必须使用 ROCm fork，不要执行
`pip install transformer_engine`（可能装到 NVIDIA 版本）。

```bash
export TE_SRC="$DATA_ROOT/te_src"

if [ ! -d "$TE_SRC/.git" ]; then
  git clone https://github.com/ROCm/TransformerEngine.git "$TE_SRC"
fi

git -C "$TE_SRC" fetch origin
git -C "$TE_SRC" checkout 6e541a10419a6e31bdc98b1516db04eb81a463b6
git -C "$TE_SRC" submodule sync --recursive
git -C "$TE_SRC" submodule update --init --recursive --jobs 16
git -C "$TE_SRC" rev-parse HEAD
git -C "$TE_SRC" status --short
```

期望 `status --short` 为空。若 GitHub/submodule 很慢，推荐复制旧机已经完整拉好的 5.1 GiB：

```bash
rsync -aH --info=progress2 \
  OLD_HOST:/mnt/m2m_nobackup/xysheng/data/te_src/ \
  "$TE_SRC/"
```

复制后仍需检查：

```bash
git -C "$TE_SRC" rev-parse HEAD
git -C "$TE_SRC" submodule status --recursive
```

关键 submodule 应包括 AOTriton、CK JIT、Composable Kernel 等。

### 7.2 源码编译与安装

当前成功构建耗时约 18.6 分钟（完整 pip 步骤约 22 分钟），生成约 387 MiB wheel，
安装目录约 935 MiB。CK kernel 默认以 JIT 源码方式安装；AOTriton gfx950 images 一并安装。

```bash
docker exec \
  -e TE_SRC="$TE_SRC" \
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
export MAX_JOBS=64

python -m pip install -v . --no-build-isolation \
  2>&1 | tee "$DATA_ROOT/logs/te_build.log"
'
```

成功日志末尾应类似：

```text
Total time for bdist_wheel: ...
Successfully built transformer_engine
Successfully installed ... transformer_engine-2.15.0.dev0+6e541a1
```

不要在构建中设置 `NVTE_FUSED_ATTN_CK=0` 或
`NVTE_FUSED_ATTN_AOTRITON=0`；TP/PP/CP 验证环境需要融合注意力，CP probe 已在该环境通过。

### 7.3 TE import + Linear/RMSNorm/DPA 前反向验证

```bash
docker exec "$CONTAINER" bash -lc '
python - <<PY
import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention import DotProductAttention

print("TE", transformer_engine.__version__, transformer_engine.__file__)

x = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
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

分别强制 CK / AOTriton 再跑一次 DPA，确认两个 backend 均可用：

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
q = torch.randn(32,1,8,16,device="cuda",dtype=torch.bfloat16,requires_grad=True)
k = torch.randn(32,1,2,16,device="cuda",dtype=torch.bfloat16,requires_grad=True)
v = torch.randn(32,1,2,16,device="cuda",dtype=torch.bfloat16,requires_grad=True)
a = DotProductAttention(
    num_attention_heads=8, kv_channels=16, num_gqa_groups=2,
    attention_dropout=0.0, attn_mask_type="causal", qkv_format="sbhd",
).cuda()
o = a(q,k,v)
o.float().sum().backward()
assert torch.isfinite(o).all()
print(tuple(o.shape), "finite=True")
PY
'
done
```

---

## 8. 安装 aiter、Lumen 和 LumenRL

```bash
docker exec "$CONTAINER" bash -lc '
set -euxo pipefail
export LUMEN_DIR="$RL_ROOT/Lumen"
export AITER_DIR="$RL_ROOT/aiter"

git config --global --add safe.directory "*"

cd "$AITER_DIR"
git submodule update --init --recursive --jobs 16
AITER_USE_SYSTEM_TRITON=1 python setup.py develop

python -m pip install -e "$LUMEN_DIR" --no-deps
python -m pip install -e "$RL_ROOT/Lumen-RL" --no-deps
'
```

检查 native engine 注册和完整依赖：

```bash
docker exec "$CONTAINER" bash -lc '
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen:${PYTHONPATH:-}"
cd "$RL_ROOT/Lumen-RL"
python - <<PY
import torch, vllm, ray
import transformer_engine.pytorch
import apex
import megatron.core
import megatron.training
import megatron.core.dist_checkpointing
from megatron.core.pipeline_parallel import get_forward_backward_func
from lumenrl.engine.training.base_engine import EngineRegistry

cls = EngineRegistry.get_engine_cls(
    model_type="language_model",
    backend="megatron_native",
)
print("torch", torch.__version__, "hip", torch.version.hip)
print("vllm", vllm.__version__, "ray", ray.__version__)
print("megatron", megatron.core.__version__)
print("native engine", cls.__name__)
print("gpus", torch.cuda.device_count())
print("IMPORT STACK OK")
PY
'
```

---

## 9. 模型和数据

最快方式是从旧机复制已过滤数据：

```bash
mkdir -p \
  "$DATA_ROOT/models/Qwen3-8B-Base" \
  "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024"

rsync -aH --info=progress2 \
  OLD_HOST:/mnt/m2m_nobackup/xysheng/data/models/Qwen3-8B-Base/ \
  "$DATA_ROOT/models/Qwen3-8B-Base/"

rsync -aH --info=progress2 \
  OLD_HOST:/mnt/m2m_nobackup/xysheng/data/data_cached/qwen3-8b-maxprompt1024/ \
  "$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/"
```

最低需要：

```text
$DATA_ROOT/models/Qwen3-8B-Base
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet
```

若需重新下载和过滤，执行
`dapo-lumenrl-native-vllm-megatron-runbook.md` 的“下载模型 / 数据 + 过滤 prompt”章节。

---

## 10. 最终 DP8 native smoke

先确认没有其他任务占 GPU：

```bash
docker exec "$CONTAINER" bash -lc '
pgrep -af "lumenrl.trainer.main|VLLMRayServer|EngineCore" || true
'
```

运行 2-step smoke。checkpoint 路径必须是绝对路径；不要在
`EXTRA_OVERRIDE` 中留下字面量 `$DATA_ROOT`：

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
S="$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
STEPS=2 \
MODE=bf16 \
EXTRA_OVERRIDE="policy.training_backend=megatron_native \
checkpointing.resume=false \
checkpointing.checkpoint_dir=$DATA_ROOT/ckpts/lumenrl-dapo/native-env-smoke" \
LOG="$DATA_ROOT/logs/native-env-smoke.log" \
bash "$S"
'
```

检查：

```bash
grep -E \
  "MegatronNativeEngine|callbacks: step=|finished after|Traceback|OOM|NaN" \
  "$DATA_ROOT/logs/native-env-smoke.log"
```

健康判据：

- `MegatronNativeEngine` 加载 TE spec 成功。
- 2 steps 结束、exit 0。
- entropy 约 0.6–0.75。
- grad_norm 有限，通常约 0.8。
- ppo_kl 约 0。
- rollout_corr/kl 约 0.002。
- 无 Traceback / OOM / NaN。

清理 smoke checkpoint（容器内通常由 root 创建）：

```bash
docker exec "$CONTAINER" rm -rf \
  "$DATA_ROOT/ckpts/lumenrl-dapo/native-env-smoke"
```

### 10.1 CP2 smoke（验证新机 TE context-parallel）

DP8 通过后，再跑 DP4×CP2，确认新机的 TE thd zigzag context-parallel
前反向和 LumenRL CP 重构/归一均正常：

```bash
docker exec \
  -e RL_ROOT="$RL_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  "$CONTAINER" bash -lc '
S="$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
STEPS=2 \
MODE=bf16 \
EXTRA_OVERRIDE="policy.training_backend=megatron_native \
policy.training.megatron_cfg.context_parallel_size=2 \
checkpointing.resume=false \
checkpointing.checkpoint_dir=$DATA_ROOT/ckpts/lumenrl-dapo/native-cp2-env-smoke" \
LOG="$DATA_ROOT/logs/native-cp2-env-smoke.log" \
bash "$S"
'
```

检查：

```bash
grep -E \
  "model-parallel=|mesh_mapping|callbacks: step=|finished after|Traceback|OOM|NaN" \
  "$DATA_ROOT/logs/native-cp2-env-smoke.log"
```

8 卡期望 mesh 为 `[0,0,1,1,2,2,3,3]`、DP=4；entropy/grad_norm/ppo_kl
应与 DP8 同量级。通过后清理：

```bash
docker exec "$CONTAINER" rm -rf \
  "$DATA_ROOT/ckpts/lumenrl-dapo/native-cp2-env-smoke"
```

之后可按 handoff 文档继续跑 TP2、PP2、TP2×PP2 或 TP2×PP2×CP2。

---

## 11. 保存环境清单

构建成功后记录版本，便于以后对比：

```bash
docker exec "$CONTAINER" bash -lc '
{
  date -Is
  python -V
  hipcc --version
  python - <<PY
import torch, vllm, ray, megatron.core, transformer_engine
print("torch", torch.__version__, "hip", torch.version.hip)
print("vllm", vllm.__version__)
print("ray", ray.__version__)
print("megatron-core", megatron.core.__version__)
print("transformer-engine", transformer_engine.__version__)
PY
  python -m pip freeze | grep -Ei \
    "megatron|transformer.engine|apex|vllm|ray|flash|torch|triton|aiter|safetensors|omegaconf"
  echo "TE source: $(git -C "$DATA_ROOT/te_src" rev-parse HEAD)"
  echo "Apex source: $(git -C "$DATA_ROOT/apex_src" rev-parse HEAD)"
  echo "Lumen-RL: $(git -C "$RL_ROOT/Lumen-RL" rev-parse HEAD)"
  git -C "$RL_ROOT/Lumen-RL" status --short
} | tee "$DATA_ROOT/logs/native-env-manifest.txt"
'
```

---

## 12. 生命周期和已知坑

### 容器不要删除

TE/Apex 安装在容器 overlay 的 site-packages：

- `docker stop/start`：保留。
- `docker rm`：安装丢失，需重编 TE（约 22 分钟）。
- 默认不要 `docker commit`；如要制作持久镜像，先获得用户明确许可。

### TE import 到了 NVIDIA 版本

症状包括找不到 ROCm library 或 CUDA 相关错误。处理：

```bash
docker exec "$CONTAINER" bash -lc '
python -m pip uninstall -y transformer-engine transformer_engine \
  transformer-engine-torch transformer_engine_torch || true
'
```

然后严格从 ROCm fork revision 重新执行第 7 节。

### TE undefined symbol / segfault

最常见原因是：

- base image、PyTorch 或 ROCm 版本变化；
- 用旧 build 目录链接了不同 torch；
- TE 不是针对 gfx950 构建。

处理：

```bash
docker exec "$CONTAINER" bash -lc '
cd "$DATA_ROOT/te_src"
rm -rf build dist transformer_engine.egg-info
'
```

确认 base image digest和 `torch 2.10.0+git8514f05 / HIP 7.2.53211` 后重编。

### TE submodule 下载失败

TE 的 AOTriton / CK / composable-kernel submodule 较多。优先 rsync 旧机完整
`te_src`；不要用缺 submodule 的浅拷贝直接编译。

### Apex 与 local-spec RMSNorm

Apex `FusedLayerNorm` 不支持 Qwen RMSNorm。`megatron_native` 使用 TE RMSNorm，
不受影响。旧 `megatron` local spec 必须继续强制 `WrappedTorchNorm`，不要删该逻辑。

### ROCm vLLM sleep

该机器的 ROCm 7.2.3 上 cumem sleep/wake 会出现释放无效和 wake OOM。
所有 native config 必须保持：

```yaml
enable_sleep_mode: false
```

### `/home` 空间

所有源码 build、pip 临时目录、模型、日志、checkpoint 放 `DATA_ROOT`。特别注意
`run_dapo.sh` 的 `EXTRA_OVERRIDE` 不会再次展开字面量 `$DATA_ROOT`；checkpoint_dir
使用绝对路径或在当前 shell 中先展开。

### root-owned checkpoint

通过容器训练创建的 checkpoint 可能归 root。清理使用：

```bash
docker exec "$CONTAINER" rm -rf /absolute/checkpoint/path
```

---

## 13. 完成标准

以下全部满足才算新机环境构建完成：

1. Base image digest正确，torch/ROCm/vLLM 版本正确。
2. 8 张 gfx950 GPU 在容器内可见（或符合新机实际 GPU 数）。
3. `megatron.core==0.18.2`，且 `megatron.training`、pipeline schedule、
   dist-checkpoint 都能 import。
4. Apex FusedLayerNorm/FusedAdam 前反向通过。
5. TE 版本为 `2.15.0.dev0+6e541a1`。
6. TE Linear + RMSNorm 前反向通过。
7. TE DPA 在 CK 和 AOTriton 两个 backend 下前反向都通过。
8. `megatron_native` engine 注册成功。
9. DP8 2-step smoke exit 0 且指标健康。
10. DP4×CP2 2-step smoke exit 0 且指标健康。
11. `native-env-manifest.txt` 已保存。

