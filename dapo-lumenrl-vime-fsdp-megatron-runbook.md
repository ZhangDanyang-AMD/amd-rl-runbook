# LumenRL 原生 DAPO Runbook（vime-rocm 镜像 · FSDP2 + Megatron-Native · sleep 开启）

> 在一台**全新 8 卡 AMD GPU 机器**上，只用 **LumenRL**（不依赖 verl）复现 verl `recipe/dapo` 的
> DAPO 数学 RL 训练，**两个训练后端都跑**：Lumen **FSDP2** 与 **Megatron-Native**（TE spec）。
> rollout 一律是同卡 colocated vLLM `AsyncLLM`（TP=1，8 replica），权重经 **ZMQ CUDA-IPC** 同步。
>
> ---
>
> ## ⚠️ 边界：只借镜像，不跑 vime 代码
>
> `vllm/vime-rocm` 这个镜像里**预装了一份 vime 框架**（`/root/vime`），本 runbook **完全不使用它**。
> 借的只是镜像里的 ROCm 依赖栈：torch+rocm、vLLM、Ray、**Megatron-LM**、TransformerEngine、Apex。
> 训练/rollout 的每一行代码都来自你 clone 的 **LumenRL**。
>
> 实测核查（容器内，`PYTHONPATH` 与 `run_dapo.sh` 一致）：
>
> ```
> lumenrl        -> $RL_ROOT/Lumen-RL/lumenrl/__init__.py     ← 你的代码
> lumen          -> $RL_ROOT/Lumen/lumen/__init__.py          ← 你的代码
> aiter          -> $RL_ROOT/aiter/aiter/__init__.py          ← 你的代码（不是镜像的 /workspace/aiter）
> megatron.core  -> /root/Megatron-LM/megatron/core/__init__.py   ← 镜像提供的上游依赖
> vllm           -> /workspace/vllm/vllm/__init__.py              ← 镜像提供
> vime           -> /root/vime/vime/__init__.py（可 import，但全程没被加载）
> sys.path 中含 "vime" 的条目：无
> import LumenRL 之后已加载的 vime 模块：无
> ```
>
> 四次真实运行的日志里，框架模块名只有 `lumenrl.trainer.rl_trainer` / `lumenrl.workers.actor_worker` /
> `lumenrl.main`，Ray actor 只有 `LumenActorWorker` 和 `VLLMRayServer`，**`vime` 出现 0 处**。
> 复核脚本见 §6.1。
>
> > `megatron.core` 来自 `/root/Megatron-LM` —— 那是上游 Megatron-LM，不是 vime 的东西，
> > 两个框架恰好都依赖它。旧 runbook 是自己源码编译 megatron-core，这里直接用镜像的。
>
> ---
>
> **和旧 runbook（`dapo-lumenrl-native-vllm-fsdp-runbook.md`）的三处关键差异：**
>
> 1. **镜像换成 `vllm/vime-rocm`**，不是 `vllm/vllm-openai-rocm:v0.23.0`。这个镜像自带
>    `megatron-core` + ROCm `TransformerEngine` + Apex fused ops，**省掉旧 runbook §13.10 的三个源码编译**
>    （megatron-core / ROCm Apex / ROCm TE，其中 TE 一次约 9 分钟、submodule 约 5.1GiB）。
> 2. **`enable_sleep_mode` 默认开启**（level=2），且**`gpu_memory_utilization` 默认抬到 0.6**（config 里是 0.30）。
>    两个 config 家族里 sleep 都写着 `false` 并附注释说「cumem sleep 在 ROCm 上没用且会损坏权重」，
>    **这条注释在本平台不成立**，实测每引擎释放 82.9–85.7 GiB 且指标无漂移，见 §11。
>    sleep 一开，0.30 这个为「引擎全程常驻」而选的保守值就没必要了 —— 但注意这两个开关是绑定的，见 §8。
> 3. **BF16 单条路线**，不含 FP8 / ATOM。要 FP8 或 ATOM rollout 回旧 runbook。
> 4. **`LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1` 默认开启**（FSDP2），规避 `chunk_cat` 偶发非法地址故障，见 §8。
> 5. **含 Qwen3-30B-A3B MoE**（vLLM + FSDP2 BF16），见 **§15** —— 那一节记了两个必看的坑：
>    MoE 上 `GPU_UTIL` 不能用 0.6（且 sleep 救不了），以及 vLLM ≥0.22 需要一处融合专家权重同步的代码修复。
>
> **一句话复现**：设路径变量 → clone 3 仓库 → 拉 vime 镜像起容器 → 装 3 个包（含两处必须的版本修正）
> → 校验环境 → 下模型/数据 + 过滤 → 写 1 个启动脚本 → smoke（两个后端）→ `docker exec -d` 起长跑。
>
> ✅ **已验证**（2026-08-04，8×MI355X gfx950 / 288GB per card，Qwen3-8B-Base）：
> **四种组合全部 `exit=0`、无 Traceback、无 OOM** —— {FSDP2, Megatron-Native} × {smoke 512, longrun 20480 首步}，
> 全部开着 sleep level=2。实测数据见 §14。
> ⚠️ 这批数据是在 **`gpu_memory_utilization=0.30`** 下取的；wrapper 现在默认 **0.6**，
> **0.6 本身没有实测过**。撞 OOM 就回 `GPU_UTIL=0.30`，见 §8。

---

## 1. 架构与两个训练后端

| 维度 | verl 正式长跑 | 本 runbook |
|---|---|---|
| 入口 | `python -m recipe.dapo.main_dapo` | `python -m lumenrl.trainer.main`（Ray 控制器，非 torchrun） |
| 编排 | Ray + HybridEngine | **单 Ray-driver：8 训练 actor + 8 colocated vLLM replica** |
| 训练后端 | FSDP | **①Lumen FSDP2** ②**Megatron-Native**（TE spec + distributed optimizer） |
| 推理后端 | vLLM `mode=async` | vLLM `AsyncLLM`（`transport=ray_http`，逐请求并发） |
| 权重同步 | ZMQ CUDA-IPC 分桶 | 同（`update_weights_ipc_send`，`NCCL_CUMEM_ENABLE=0`） |
| rollout 显存 | 生成后 offload | **vLLM cumem sleep level=2**（本 runbook 开启，见 §11） |
| rollout 输入 | token-in | token-in（HF tokenize，`add_special_tokens=False`） |
| 动态采样 | `algorithm.filter_groups` | `algorithm.dapo.filter_groups`（同语义，metric=acc，cap 10 轮） |
| TIS 修正 | `rollout_correction.rollout_is=token` thr=2.0 | 同 + vLLM `calculate_log_probs=true` |
| overlong 奖励 | `overlong_buffer_cfg` | `algorithm.dapo.overlong_buffer`（同公式） |
| 策略损失 | clip-higher + dual-clip + token-mean | 同（`asymmetric_clip_loss` + `loss_agg_mode=token-mean`） |
| 优势估计 | grpo（按 uid 组归一化） | grpo（同） |

两个训练后端的差别（rollout 侧完全相同，所以任何差异都只来自训练后端）：

| | FSDP2 | Megatron-Native |
|---|---|---|
| `policy.training_backend` | `fsdp2` | `megatron_native` |
| 并行 | FSDP2 全分片，DP=8 | TP=1, PP=1, CP=1 → **DP=8**（8B dense 不需要 TP/EP） |
| 优化器 | fp32 master + bf16 compute | Megatron distributed optimizer（fp32 master 分片到 DP） |
| 长序列激活 | 无显式 recompute | `recompute_granularity: full` + `method: uniform` + `num_layers: 1` |
| 动态批 | `max_token_len_per_gpu` | `enable_dynamic_batch` + `max_tokens_per_gpu` |
| log-prob 显存 | — | `log_probs_chunk_size: 1024`（单个 `[chunk,V]` softmax 缓冲） |
| 常驻显存（实测） | `mem/actor_allocated_gb` **11.59** | **57.43**（约 5 倍） |
| `mismatch/*` 诊断指标 | 有 | **没有**（只有 `rollout_corr/*` 和 `rollout_correction/*`） |
| 额外依赖 | 无 | **镜像自带**，不需要装（旧 runbook 要编三个包） |

---

## 2. 路径变量（换机只改这里）

```bash
export RL_ROOT=/home/$USER/lumen_rl                      # 代码根（Lumen-RL / Lumen / aiter + helper 脚本）
export DATA_ROOT=/mnt/<big_disk>/$USER/lumenrl_data      # 模型 / 数据 / 日志 / ckpt 根，放大容量盘
export CONTAINER=rl-vllm-fsdp
mkdir -p "$RL_ROOT" "$DATA_ROOT"/{logs,models,data_cached,raw,hf_home,ckpts,cache}
```

> ⚠️ **`DATA_ROOT` 一定要放大盘**。8B FSDP2 单个 checkpoint 约 90GB，`save_total_limit: 5` 就是 450GB；
> MoE 是约 342GB/份。本次验证用的是 `/mnt/m2m_nobackup`（28T，node-local nobackup 盘），
> docker 的 data-root 也在同一块盘上（`docker info | grep "Docker Root Dir"` 确认），所以 81.8GB 的镜像不会占系统盘。
>
> ⚠️ **但 node-local 盘有个代价，本次实测吃到了教训。** 那块 nobackup 盘**只在该计算节点上挂载**，
> 登录节点看不到（`ls /mnt/m2m_nobackup` → `No such file or directory`）。一次 MoE 长跑跑到约 4 小时 50 分钟时
> 节点故障（`JobState=NODE_FAIL Reason=NodeDown`），于是盘上的 57G 模型、过滤后数据、**全部训练日志和 checkpoint
> 一起失联** —— 想取回必须重新分配到**同一个节点**，而它很可能已经被别人占用。
>
> 建议的分工是：**模型与 `data_cached/` 放共享 NFS**（如 `$HOME` 下，跨节点可见、只需下载一次，
> 换节点重建环境时能省一个多小时），**checkpoint 放 node-local 大盘**（体积大、IO 敏感，
> 丢了还能从上一个 checkpoint 或从头恢复）。config 里 checkpoint 走
> `${oc.env:SCRATCH_ROOT}`，所以这两者本来就可以分开：
>
> ```bash
> export DATA_ROOT=$HOME/lumenrl_data                      # 模型 / 数据，共享 NFS
> export SCRATCH_ROOT=/mnt/<node_local_disk>/$USER/scratch  # checkpoint，本地大盘
> ```
>
> wrapper 在 `SCRATCH_ROOT` 未设时会默认取 `DATA_ROOT`，所以想分开就显式导出它。
> **代码和 runbook 务必留在 NFS 上** —— 这次节点故障里它们是唯一无损的东西。
>
> ⚠️ **`$DATA_ROOT/models/Qwen3-8B-Base` 必须是容器挂载范围内的真实目录**（或同盘硬链接），
> **不能是指向挂载范围之外的软链**。否则容器里链接是断的，transformers 会把绝对路径当成 HF repo id
> 报 `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`。见 §13。

---

## 3. 拉代码（三个仓库，各自分支）

```bash
cd "$RL_ROOT"
git clone -b dev/vllm-fsdp-dapo   https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git clone -b amd-atom-rollout     https://github.com/ZhangDanyang-AMD/Lumen.git
git clone -b lumen/triton_kernels https://github.com/ZhangDanyang-AMD/aiter.git

# aiter 的 JIT 依赖 composable_kernel，补齐 submodule（BF16 路线用不到 AITER kernel，
# 但补上成本很低，且避免以后开 FP8 时才发现缺 generate.py）。
cd "$RL_ROOT/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel
```

| 仓库 | 分支 | 用途 |
|---|---|---|
| `Lumen-RL` | `dev/vllm-fsdp-dapo` | RL 主框架（含两个后端的 config 与 `run_dapo.sh`） |
| `Lumen` | `amd-atom-rollout` | FSDP2 训练后端 |
| `aiter` | `lumen/triton_kernels` | AMD kernel（训练前向的 `flash_attn_varlen_func` 从这里来） |

> **ATOM 不需要**（那是 `MODE=atomfp8` 才用的），本 runbook 是 BF16 单线。
>
> 已有 checkout 更新代码：
> ```bash
> git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
> git -C "$RL_ROOT/Lumen"    pull --ff-only origin amd-atom-rollout
> git -C "$RL_ROOT/aiter"    pull --ff-only origin lumen/triton_kernels
> ```
>
> ⚠️ **首次运行会 JIT 编译 aiter kernel**（`module_aiter_core` 等，落到 `$RL_ROOT/aiter/aiter/jit/*.so`，
> 这些 `.so` 被 `.gitignore` 忽略，所以新机器上一定是空的）。第一次 smoke 会比后续慢，这是环境预热成本，
> 不要算进训练性能。

---

## 4. 拉镜像并启动容器

```bash
docker pull vllm/vime-rocm        # 约 82GB，首次较久
```

> ⚠️ **必须是 `vllm/vime-rocm`（ROCm 版）**。`vllm/vime:latest` 是 CUDA 版（torch `+cu130`），
> 在 AMD 上 `torch.cuda.device_count()=0`、`rocm-smi` 看不到卡。

镜像 ENTRYPOINT 是 `sleep`，所以必须 `--entrypoint /bin/bash` 覆盖，否则会变成 `sleep sleep infinity` 直接退出：

```bash
docker rm -f "$CONTAINER" 2>/dev/null
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --ulimit nofile=1048576:1048576 \
  --device=/dev/kfd --device=/dev/dri --group-add=video --privileged \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$RL_ROOT":"$RL_ROOT" -v "$DATA_ROOT":"$DATA_ROOT" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" \
  vllm/vime-rocm -lc "sleep infinity"
```

> - `-v "$RL_ROOT":"$RL_ROOT"` 用**同路径挂载**（不是挂到 `/root/...`），这样宿主和容器里的绝对路径一致，
>   脚本、日志、config 里的 `${oc.env:DATA_ROOT}` 全都不用改。
> - `--privileged` + `--group-add=video` 足够访问 GPU；避免 `--group-add render`（该组在容器内可能不存在）。
> - 容器 `stop/start` 不丢依赖，只有 `docker rm` 才丢（丢了重跑 §5）。
> - **如果这台机器上还跑着 vime 自己的容器（`vime-dapo` 之类），请务必另起一个容器，不要复用。**
>   §5 会把 `antlr4-python3-runtime` 和 `flydsl` 改版本，这两个包 vime 也在用。同镜像不同容器文件系统互相隔离，
>   PID namespace 也不共享（没给 `--pid=host`），所以 `run_dapo.sh` 开头那段 `pkill EngineCore` 不会误杀邻居容器的进程。
>   **代价是两个容器不能同时占卡。**

---

## 5. 装依赖（三个包 + 两处必须的版本修正）

这个镜像已经自带了 `torch 2.10.0+rocm7.0`、`vllm 0.22.1rc1`、`ray 2.56.0`、`transformers 5.13.1`、
`datasets 5.0.0`、`accelerate 1.14.0`、`omegaconf 2.3.1`、`safetensors 0.8.0`、`wandb 0.28.0`、`numpy 1.26.4`，
以及 Megatron 那一套。**所以只需要装 `Lumen`、`Lumen-RL`、`math_verify` 三个，其余一律 `--no-deps` 保护**，
否则 pip 的 resolver 会把 torch/vllm 换掉。

```bash
cat > "$RL_ROOT/install_deps.sh" <<'EOF'
#!/bin/bash
# Install LumenRL + Lumen into the vime-rocm container.
# --no-deps everywhere that matters: the image already ships torch 2.10+rocm7,
# vllm, ray 2.56, transformers 5.x, datasets, accelerate, omegaconf, wandb, numpy 1.26.4,
# and a pip resolver run would happily replace them.
set -ex
: "${RL_ROOT:?}"

git config --global --add safe.directory "$RL_ROOT/Lumen-RL" || true
git config --global --add safe.directory "$RL_ROOT/Lumen" || true
git config --global --add safe.directory "$RL_ROOT/aiter" || true

pip install -e "$RL_ROOT/Lumen" --no-deps
pip install -e "$RL_ROOT/Lumen-RL" --no-deps

# FIX 1: the old runbook uses math_verify[antlr4_13_2], but this image ships
# omegaconf 2.3.1, whose generated grammar only deserializes under
# antlr4-python3-runtime 4.9.*. Pairing them makes `import omegaconf` raise
# "Could not deserialize ATN with version 3 (expected 4)", which kills the run
# in lumenrl/core/config.py before anything starts. Take the 4.9 extra instead.
pip install --no-cache-dir "math_verify[antlr4_9_3]"

# FIX 2: run_dapo.sh puts the repo aiter first on PYTHONPATH, and that aiter
# requires flydsl >= 0.1.5.dev515 while the image ships 0.1.4.2. Without this the
# training forward dies at packing.py's `from aiter import flash_attn_varlen_func`.
# pip will warn that the image's amd-aiter wheel pins flydsl<0.1.5 -- ignore it,
# the wheel is not what run_dapo.sh imports.
pip install --no-cache-dir "flydsl==0.1.8"
EOF

docker exec "$CONTAINER" bash -lc 'bash "$RL_ROOT/install_deps.sh"'
```

**两处修正的报错长什么样**（新机器上如果照旧 runbook 装，一定会撞上这两个）：

| 症状 | 原因 | 修法 |
|---|---|---|
| `Exception: Could not deserialize ATN with version 3 (expected 4)`，栈在 `omegaconf/grammar/gen/OmegaConfGrammarLexer.py` | `math_verify[antlr4_13_2]` 把 antlr4 runtime 升到 4.13.2，omegaconf 2.3.1 要求 4.9.\* | `pip install "math_verify[antlr4_9_3]"`；已经装错了就 `pip install "antlr4-python3-runtime==4.9.3"` 回退 |
| `ImportError: Unsupported flydsl version: expected >=0.1.5.dev515, got 0.1.4.2`，栈在 `lumenrl/engine/training/packing.py` → `from aiter import flash_attn_varlen_func` | 镜像自带 flydsl 0.1.4.2，仓库 aiter 要求更新 | `pip install "flydsl==0.1.8"`。之后 `amd-aiter 0.1.13.post1 requires flydsl<0.1.5` 的 pin 冲突是**预期的**，不要按它回退 |

---

## 6. 校验环境（FSDP2 与 Megatron 一起验）

```bash
cat > "$RL_ROOT/env_verify.sh" <<'EOF'
#!/bin/bash
# One-shot check: base stack, LumenRL imports, omegaconf interpolation (the antlr
# canary), and everything the megatron_native backend needs.
#
# Mirror the PYTHONPATH that run_dapo.sh exports. This is not cosmetic: `import apex`
# only survives when the repo aiter is importable, because apex's fused_rope picks the
# aiter RoPE backend and never reaches its own fused_rotary_positional_embedding .so,
# which is ABI-broken in this image. Checking apex without this PYTHONPATH reports a
# failure that never happens in a real run.
: "${RL_ROOT:?}"
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen"
python3 - <<'PY'
import importlib.util
import os

import ray
import torch
import transformers
import vllm

print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("vllm", vllm.__version__, "ray", ray.__version__, "transformers", transformers.__version__)
print("gpus", torch.cuda.device_count())

import lumen
import lumenrl
from lumenrl.engine.inference.vllm_ray_server import VLLMRayServer

print("lumen", lumen.__file__)
print("lumenrl", lumenrl.__file__)
print("VLLMRayServer", VLLMRayServer.__name__)

# antlr canary: omegaconf interpolation is what every config resolution depends on.
from omegaconf import OmegaConf

os.environ["PROBE_ROOT"] = "/tmp/probe"
cfg = OmegaConf.create({"a": 3, "b": "${a}", "c": "${oc.env:PROBE_ROOT}"})
print("omegaconf interpolation:", cfg.b, cfg.c)

from math_verify import parse, verify

print("math_verify:", verify(parse("$42$"), parse("$42$")))

import flydsl

print("flydsl", flydsl.__version__, "(must be >= 0.1.5.dev515)")

# --- megatron_native backend ---
import megatron.core

print("megatron.core from", megatron.core.__file__)
import transformer_engine

print("transformer_engine", transformer_engine.__version__)
from apex.normalization import FusedLayerNorm  # noqa: F401
from apex.optimizers import FusedAdam  # noqa: F401

print("apex FusedLayerNorm/FusedAdam OK")
for mod in ("lumenrl.engine.training.megatron_native_engine",
            "lumenrl.engine.training.qwen3_megatron_bridge"):
    assert importlib.util.find_spec(mod), mod
    print(f"{mod}: OK")
print("ENV OK")
PY
EOF

docker exec "$CONTAINER" bash -lc 'bash "$RL_ROOT/env_verify.sh"'
```

**本次实测的期望输出**（版本号可能随镜像更新漂移，`ENV OK` 是硬判据）：

```
torch 2.10.0+rocm7.0 hip 7.0.51831
vllm 0.22.1rc1.dev392+g43914dd74 ray 2.56.0 transformers 5.13.1
gpus 8
lumen    $RL_ROOT/Lumen/lumen/__init__.py
lumenrl  $RL_ROOT/Lumen-RL/lumenrl/__init__.py
omegaconf interpolation: 3 /tmp/probe
math_verify: True
flydsl 0.1.8 (must be >= 0.1.5.dev515)
megatron.core from /root/Megatron-LM/megatron/core/__init__.py
transformer_engine 2.12.0.dev0+86438dc3
apex FusedLayerNorm/FusedAdam OK
ENV OK
```

> - `megatron.core` 解析到 **`/root/Megatron-LM`**（镜像里的 editable 安装，`megatron-core 0.16.0rc0`）。
>   旧 runbook 指定的是 `0.18.2`；**0.16.0rc0 实测能跑通** LumenRL 的 `megatron_native`，不要为了对齐版本号去动它。
> - **apex 能不能 import 取决于 PYTHONPATH 里有没有仓库 aiter**，这是本次踩到的一个陷阱。
>   裸 `python3 -c "import apex"` 会挂在 `apex/transformer/functional/fused_rope.py:53` 的
>   `import fused_rotary_positional_embedding`（那个 `.so` 报
>   `undefined symbol: _ZN3c103hip28c10_hip_check_implementationEiPKcS2_ib`，与当前 torch ABI 不匹配）。
>   但只要仓库 aiter 可导入，apex 在**第 49 行**就选中 aiter RoPE 后端（打一条
>   `UserWarning: Aiter backend is selected for fused RoPE`），**第 53 行永远走不到**，于是 import 成功。
>   `run_dapo.sh` 正是这样设 PYTHONPATH 的，所以**真实运行里 apex 是好的** —— 实测 8 个
>   `LumenActorWorker` 都打了那条第 49 行的告警。
>   **所以：看到裸 `import apex` 失败不要去重编 Apex，先确认 PYTHONPATH。**
> - **不要装 `megatron-bridge`**：Qwen3 的 HF↔Megatron 转换由 `lumenrl/engine/training/qwen3_megatron_bridge.py`
>   负责。**也因此不需要任何离线权重转换步骤** —— 两个后端都直接吃 HF 目录。

### 6.1 代码来源复核（确认没有跑成镜像里的 vime）

镜像自带一份 vime（`/root/vime`）。这个脚本证明它全程没被加载，且 `lumenrl`/`lumen`/`aiter`
都解析到你 clone 的仓库而不是镜像内的副本：

```bash
cat > "$RL_ROOT/provenance_check.sh" <<'EOF'
#!/bin/bash
# Prove the run uses LumenRL code only, and that nothing resolves to the image's
# preinstalled vime tree. The vime-rocm image is reused for its ROCm/vLLM/Megatron
# stack only; /root/vime must never end up on the import path of a LumenRL run.
: "${RL_ROOT:?}"
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen"
python3 - <<'PY'
import importlib.util
import sys

print("--- where does each package resolve to")
for mod in ("lumenrl", "lumen", "aiter", "megatron.core", "vllm", "torch"):
    spec = importlib.util.find_spec(mod)
    print(f"{mod:14s} -> {spec.origin if spec else 'NOT FOUND'}")

print("--- is the image's vime importable at all?")
spec = importlib.util.find_spec("vime")
print("vime spec:", spec.origin if spec else "NOT FOUND")

print("--- any /root/vime on sys.path?")
hits = [p for p in sys.path if "vime" in p]
print("sys.path entries containing 'vime':", hits or "none")

print("--- does LumenRL import vime anywhere at runtime?")
import lumenrl.trainer.rl_trainer  # noqa: F401
import lumenrl.engine.inference.vllm_ray_server  # noqa: F401
import lumenrl.engine.training.fsdp_backend  # noqa: F401

leaked = sorted(m for m in sys.modules if m == "vime" or m.startswith("vime."))
print("vime modules loaded after importing LumenRL:", leaked or "none")
PY
EOF

docker exec "$CONTAINER" bash -lc 'bash "$RL_ROOT/provenance_check.sh"'
```

判据：`aiter` 必须指向 `$RL_ROOT/aiter`（**不是** 镜像的 `/workspace/aiter`），
最后两行必须是 `none` / `none`。

跑起来之后还可以从日志侧再确认一次 —— 应该只看到 `lumenrl.*` 模块和两种 Ray actor：

```bash
L=$(cat /tmp/run_dapo_log.txt)
grep -aoE "(lumenrl|vime)[a-z._]*(rl_trainer|actor_worker|main)" "$L" | sort | uniq -c
grep -aoE "\((LumenActorWorker|VLLMRayServer) pid=" "$L" | sort -u
grep -acE "/root/vime|import vime|from vime" "$L"        # 必须是 0
```

---

## 7. 下模型 / 数据 + 过滤 prompt ≤1024

**7.1 模型 + 原始 parquet 数据**（模型 Qwen3-8B-Base；数据 = verl DAPO 同款 train/val）：

```bash
docker exec "$CONTAINER" bash -lc '
set -e
# HF_HUB_DISABLE_XET=1 是必要的：xet 后端在 NFS / 某些共享盘上会报
# "OSError: I/O error: Permission denied (os error 13)"（实测踩到）。
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3-8B-Base --local-dir "$DATA_ROOT/models/Qwen3-8B-Base" --max-workers 8
hf download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset --local-dir "$DATA_ROOT/raw/DAPO-Math-17k"
hf download BytedTsinghua-SIA/AIME-2024     --repo-type dataset --local-dir "$DATA_ROOT/raw/AIME-2024"
du -sh "$DATA_ROOT/models/Qwen3-8B-Base"    # 约 16G
'
```

**7.2 过滤 prompt ≤1024**（复刻 verl `RLHFDataset.maybe_filter_out_long_prompts`；不预过滤则每次启动都要做一遍
耗时的 overlong-prompt 扫描）：

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
    return datasets.load_dataset("parquet", data_files={"train": files}, split="train")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def prompt_length(example) -> int:
    return len(
        tokenizer.apply_chat_template(
            example["prompt"], add_generation_prompt=True, tokenize=True
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
    process(f"{DATA_ROOT}/raw/DAPO-Math-17k", "dapo-math-17k.filtered.parquet")
    process(f"{DATA_ROOT}/raw/AIME-2024", "aime-2024.filtered.parquet")
PYEOF

docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 filter_prompts.py'
```

产出（即 config 与 `run_dapo.sh` 的默认 `TRAIN_FILE` / `VAL_FILE`）：

```
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train，约 1.0GB
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val，约 0.9MB
```

**7.3 W&B key**（可选；长跑 config 默认 `wandb_enabled: true`）：

```bash
echo "WANDB_API_KEY=<你的key>" > "$RL_ROOT/wandb.key"    # run_dapo.sh 自动读这个文件
```

---

## 8. 统一启动脚本

真正的启动逻辑在 **Lumen-RL 仓库自带的 `examples/DAPO/run_dapo.sh`**（`git clone` 即得），它是环境设置的唯一来源。
下面这个 wrapper 只做四件事：选 config（后端 × 规模）、打开 sleep、把 `PYTORCH_CUDA_ALLOC_CONF` 置空、给日志起名。

> ⚠️ **不要 fork `run_dapo.sh`**。旧 runbook 记过一次教训：内嵌副本和仓库 HEAD 漂移了 117 行。
> 脚本被误改就还原：`git -C "$RL_ROOT/Lumen-RL" checkout -- examples/DAPO/run_dapo.sh`。

```bash
cat > "$RL_ROOT/run_lumenrl_dapo_vime.sh" <<'EOF'
#!/bin/bash
# LumenRL native DAPO launcher for the vllm/vime-rocm image.
#
#   MODEL=8b|moe             Qwen3-8B-Base (dense) or Qwen3-30B-A3B-Base (MoE)
#   BACKEND=fsdp2|megatron   training backend (picks the config pair)
#   SCALE=smoke|longrun      short chain check vs verl-aligned long run
#   SLEEP=1|0                vLLM cumem sleep level 2 between rollout and training
#   GPU_UTIL=0.6             vLLM gpu_memory_utilization (configs ship 0.30)
#   LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1
#                            FSDP2 only; on by default, see the block below
#   STEPS=N                  overrides num_training_steps
#   LOG=/path/to.log         defaults to $DATA_ROOT/logs/<model>-<backend>-<scale>-sleep<n>.log
#
# Everything else is delegated to Lumen-RL's own examples/DAPO/run_dapo.sh, which
# stays the single source of truth for env setup.
set -uo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
# The MoE longrun config resolves checkpoint_dir through ${oc.env:SCRATCH_ROOT};
# omegaconf fails outright if it is unset, so default it to DATA_ROOT.
export SCRATCH_ROOT="${SCRATCH_ROOT:-$DATA_ROOT}"
# ROCm/HIP has no expandable_segments. run_dapo.sh treats an explicitly empty
# value as "caller asked for none" and unsets it, which silences 25 warnings/run.
export PYTORCH_CUDA_ALLOC_CONF=

MODEL="${MODEL:-8b}"
BACKEND="${BACKEND:-fsdp2}"
SCALE="${SCALE:-smoke}"
SLEEP="${SLEEP:-1}"
GPU_UTIL="${GPU_UTIL:-0.6}"

case "$MODEL/$BACKEND/$SCALE" in
  8b/fsdp2/smoke)      CFG=dapo_qwen3_8b_ray_vllm_smoke.yaml ;;
  8b/fsdp2/longrun)    CFG=dapo_qwen3_8b_ray_vllm_longrun.yaml ;;
  8b/megatron/smoke)   CFG=dapo_qwen3_8b_ray_megatron_smoke.yaml ;;
  8b/megatron/longrun) CFG=dapo_qwen3_8b_ray_megatron_longrun.yaml ;;
  # MoE: the verlref pair ports verl's FP8 reference recipe's BF16 baseline.
  # Its "smoke" is a 3-step 4k-response chain check, not a 512-token one.
  moe/fsdp2/smoke)     CFG=dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml ;;
  moe/fsdp2/longrun)   CFG=dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml ;;
  moe/megatron/*)      CFG=dapo_qwen3moe_a3b_ray_megatron_verlref_$([ "$SCALE" = smoke ] && echo 4k_smoke || echo longrun).yaml ;;
  *) echo "MODEL=8b|moe, BACKEND=fsdp2|megatron, SCALE=smoke|longrun" >&2; exit 2 ;;
esac

if [ "$BACKEND" = "fsdp2" ]; then
  # Swap FSDP2's fused foreach_reduce_scatter_copy_in for plain slice copies. The
  # fused torch._chunk_cat kernel intermittently faults with
  # HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on this stack -- roughly one in 1e5
  # calls, so it surfaces as a long run dying at a random step (observed at step 11
  # and step 128). The fallback moved that from ~135k copy-ins survived to ~1.24M
  # with no recurrence, at negligible cost (same bytes, 88 launches per layer group
  # instead of 1, all large contiguous copies). Correctness has 6 unit tests that
  # compare bit-for-bit against torch._chunk_cat.
  # It is a workaround, not a fix: nobody has explained the faulting address yet.
  export LUMENRL_FSDP_CHUNK_CAT_FALLBACK="${LUMENRL_FSDP_CHUNK_CAT_FALLBACK:-1}"
fi

if [ "$MODEL" = "moe" ]; then
  # run_dapo.sh defaults MODEL_PATH to the 8B model, so it must be given here.
  export MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-30B-A3B-Base}"
  # The router dtype has to match on BOTH sides or the mismatch just flips sign.
  # This env var only reaches the vLLM workers; the training side reads the config
  # (fsdp2 verlref configs leave the router in BF16 to match vLLM, which is verl's
  # deliberate choice, so LumenRL's fp32 default must be turned off).
  export LUMENRL_FP32_MOE_ROUTER="${LUMENRL_FP32_MOE_ROUTER:-0}"
fi

# Both config families ship enable_sleep_mode: false with a stale comment claiming
# cumem sleep frees nothing and corrupts weights on ROCm. That was measured on
# MI300X + vLLM 0.23; on MI355X + the vime image's vLLM 0.22.1 sleep level 2
# frees 82.9-85.7 GiB per engine with no metric drift, so turn it on by default.
if [ "$SLEEP" = "1" ]; then
  export EXTRA_OVERRIDE="policy.generation.vllm_cfg.enable_sleep_mode=true policy.generation.vllm_cfg.sleep_level=2 ${EXTRA_OVERRIDE:-}"
fi

# gpu_memory_utilization: configs ship 0.30, which the verl-aligned recipe chose for
# 256GB cards without sleep. On a 288GB card with sleep on, the engine hands its whole
# budget back before every training pass, so a larger KV cache is affordable and 0.30
# leaves most of the card idle during rollout.
#
# This only holds together with SLEEP=1. At 0.6 the engine reserves ~173GB of a 288GB
# card; the Megatron backend alone peaked at 145GB reserved during a 20480-token
# training pass, so if the engine stays resident the two do not fit.
#
# The 30B MoE is the tightest case, since every replica keeps the whole 30.5B BF16
# model (~61GB) inside that budget. The verlref config ships 0.30 and its comment
# reports an OOM at 0.40 -- but that reference run had the engine resident through
# training, which is exactly what sleep removes, so the ceiling does not carry over.
export EXTRA_OVERRIDE="policy.generation.vllm_cfg.gpu_memory_utilization=$GPU_UTIL ${EXTRA_OVERRIDE:-}"
if [ "$SLEEP" != "1" ]; then
  echo "WARNING: GPU_UTIL=$GPU_UTIL with SLEEP=0 keeps the engine resident through" >&2
  echo "         training and is expected to OOM at longrun scale. Use GPU_UTIL=0.30." >&2
fi

if [ "$SCALE" = smoke ]; then
  # The MoE chain check wants all 3 of its steps; 1 step tells you much less.
  DEFAULT_STEPS=$([ "$MODEL" = moe ] && echo 3 || echo 1)
else
  DEFAULT_STEPS=1000
fi

CONFIG_OVERRIDE="examples/DAPO/configs/$CFG" \
STEPS="${STEPS:-$DEFAULT_STEPS}" \
MODE=bf16 \
LOG="${LOG:-$DATA_ROOT/logs/$MODEL-$BACKEND-$SCALE-sleep$SLEEP.log}" \
  bash "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh"
EOF
chmod +x "$RL_ROOT/run_lumenrl_dapo_vime.sh"
```

**wrapper 的四个开关：**

| 变量 | 默认 | 作用 |
|---|---|---|
| `BACKEND` | `fsdp2` | `fsdp2` / `megatron`，选训练后端 |
| `SCALE` | `smoke` | `smoke`（resp=512，1 步）/ `longrun`（resp=20480，1000 步） |
| `MODEL` | `8b` | `8b`（Qwen3-8B-Base）/ `moe`（Qwen3-30B-A3B-Base，见 §15） |
| `SLEEP` | **`1`** | vLLM cumem sleep level=2。见 §11 |
| `GPU_UTIL` | **`0.6`** | 覆盖 `vllm_cfg.gpu_memory_utilization`（config 里写的是 0.30）。见下。**MoE 必须降到 0.45**，见 §15.2 |
| `LUMENRL_FSDP_CHUNK_CAT_FALLBACK` | **`1`**（仅 FSDP2） | 规避 `chunk_cat` 偶发非法地址故障。见下 |
| `STEPS` | smoke 1（MoE 3）/ longrun 1000 | 覆盖 `num_training_steps` |
| `LOG` | `$DATA_ROOT/logs/<model>-<backend>-<scale>-sleep<n>.log` | 日志路径 |

**为什么默认开 `LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1`。** FSDP2 反向的
`foreach_reduce_scatter_copy_in` 用融合 kernel `torch._chunk_cat` 把 bf16 梯度打包进 fp32 的
reduce-scatter 缓冲区，这个 kernel 在这套栈上会**偶发**访问非法地址
（`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`），概率约十万分之一 —— 表现就是长跑在某个随机步数
整个 Ray job 猝死（历史上分别死在 step 11 和 step 128，对应约 13.5 万和 39 万次 copy-in）。
`lumenrl/engine/training/fsdp_chunk_cat_fallback.py` 把它换成普通切片拷贝：同样的字节、同样的布局，
每个 layer group 从 1 次 launch 变成 88 次，但都是大块连续拷贝，相对分钟级的 step 可忽略。
实测撑过的 copy-in 从约 13.5 万提到约 **124 万**且未复现。

> **这是规避而不是修复** —— 为什么那个 kernel 会偶发访问非法地址至今没有解释。
> **它不是 OOM**：OOM 在这套栈上是 `torch.OutOfMemoryError: HIP out of memory`（分配阶段的 Python 异常），
> 而 aperture violation 是 kernel **执行**阶段的硬件故障，别按显存不足去处理。
> 正确性有 6 项单测与 `torch._chunk_cat` 逐位比对（真实 MoE 层形状、需补零的行数、整块 padding、
> 非连续梯度、不同 world_size、无 dtype 提升），本镜像上实测全过：
>
> ```bash
> docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/Lumen-RL" &&
>   PYTHONPATH=$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen \
>   python3 -m lumenrl.tests.test_fsdp_chunk_cat_fallback'   # 需要 1 张卡
> ```
>
> 生效判据：每个 actor 打一行 `[lumenrl] FSDP2 reduce-scatter copy-in: slicing fallback installed`，
> 8 卡应有 8 条。要关掉做 A/B 就传 `LUMENRL_FSDP_CHUNK_CAT_FALLBACK=0`。
> Megatron 后端不涉及这条路径，wrapper 只对 `BACKEND=fsdp2` 设置它。

**为什么默认 `GPU_UTIL=0.6` 而不是 config 里的 0.30。** config 那个 0.30 是 verl-aligned recipe
为**没有 sleep 的 256GB 卡**选的：引擎全程常驻，必须给训练留出余量。开了 sleep 之后前提变了 ——
引擎在每次训练前把整份预算还给驱动，所以 KV cache 可以给得更大，而 0.30 在 288GB 卡上
rollout 阶段会让大半张卡闲着（实测 0.30 时 KV cache 只有 67.6–70.4 GiB）。

> ⚠️ **`GPU_UTIL=0.6` 与 `SLEEP=1` 是绑定的，不能只开一个。** 0.6 时引擎要占 288GB 的约 173GB，
> 而 Megatron 后端自己在 resp=20480 的训练 pass 上实测 `max_reserved` 就有 145GB —— 引擎不睡，两者装不下。
> wrapper 在 `SLEEP=0` + 高 `GPU_UTIL` 时会打一条 warning 提醒你改回 `GPU_UTIL=0.30`。
>
> ⚠️ **`0.6` 这个默认值本身没有实测过**（§14 的数据全部是在 0.30 下取的，那是四种组合都验证过的配置）。
> 撞 OOM 就先回 `GPU_UTIL=0.30`，那是已知能跑的值。参照数据：在同一台机器上用 vime 框架（不是 LumenRL）
> 测过 util 0.3→0.6，KV cache 从 70.0 GiB 涨到 156.4 GiB、sleep 释放量从 85.3 GiB 涨到 171.7 GiB、
> 峰值 236.8 GiB / 288 GiB（82%），没有 OOM —— 但那是 resp=512 的 smoke 规模，**不能外推到 resp=20480**。

四个开关组合对应的 config 与规模：

| `BACKEND` | `SCALE` | config | 规模 |
|---|---|---|---|
| `fsdp2` | `smoke` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | resp=512, batch 128(=8×16), gen_batch 24 |
| `megatron` | `smoke` | `dapo_qwen3_8b_ray_megatron_smoke.yaml` | 同上 + `max_tokens_per_gpu: 2048` |
| `fsdp2` | `longrun` | `dapo_qwen3_8b_ray_vllm_longrun.yaml` | resp=20480, batch 512(=32×16), gen_batch 96, 1000 步 |
| `megatron` | `longrun` | `dapo_qwen3_8b_ray_megatron_longrun.yaml` | 同上 + recompute full + `max_tokens_per_gpu: 21504` |

**长跑规模/超参（两个后端逐字段相同，对齐 verl BF16 formal long run）：**

| 项 | 值 |
|---|---|
| `num_training_steps` | 1000 |
| `train_global_batch_size` / `dapo.num_generations` | 512（=32 prompt × 16 生成，**是序列数**） |
| `gen_batch_size` | 96（**是 prompt 数**，over-sample 供 filter_groups） |
| `max_response_length` / `max_total_sequence_length` | 20480 / 21504 |
| `max_token_len_per_gpu`（FSDP2）/ `max_tokens_per_gpu`（Megatron） | 21504 / 21504 |
| `learning_rate` / `lr_warmup_steps` / `weight_decay` / `max_grad_norm` | 1e-6 / 10 / 0.1 / 1.0 |
| `lr_decay_style` | constant（warmup 后恒定） |
| clip low/high/c，`loss_agg_mode` | 0.2 / 0.28 / 10.0，token-mean |
| `overlong_buffer.len` / `penalty_factor` | 512 / 1.0 |
| `filter_groups` | enable，metric=acc，`max_num_gen_batches=10` |
| `rollout_correction.rollout_is` / threshold | token / 2.0 |
| `optimizer_dtype` | bf16（compute；master 权重 engine 内部 fp32） |
| `fsdp_cfg.param_offload` / `optimizer_offload` | **false**（Ray 路径不支持 offload，开了会报 `parameters should be materialized on CPU`） |
| vllm_cfg | `max_model_len=21504`, `enforce_eager=true`, `max_num_batched_tokens=32768`, `max_num_seqs=64`, `calculate_log_probs=true`；**两项由 wrapper 覆盖：`enable_sleep_mode=true` / `sleep_level=2`，以及 `gpu_memory_utilization=0.6`（config 里是 0.30）** |
| `val_steps` / `save_steps` / `save_total_limit` / `resume` / `seed` | 10（AIME greedy）/ 50 / 5 / true / 10086 |

> ⚠️ **注意单位**：`train_global_batch_size` 是**序列数**，`gen_batch_size` 是 **prompt 数**。
> LumenRL 用 `train_prompts = train_global_batch_size // num_generations` 反推 prompt 数。
>
> ⚠️ 两个长跑 config 的 W&B project **名字不一样**（FSDP2 是 `LUMEN_RL`，Megatron 是 `LumenRL`），
> 想放到一起对比就用 `EXTRA_OVERRIDE='logger.wandb.project=<name>'` 统一。
>
> ⚠️ **两个后端的 checkpoint 目录本来就是分开的**（`longrun-ray-vllm-8b` vs `longrun-ray-megatron-8b`），
> 格式不兼容，**不要让它们共用目录**。

---

## 9. Smoke（两个后端各 1 步，前台等结果）

```bash
# FSDP2 BF16 smoke（sleep 开）
docker exec "$CONTAINER" bash -lc \
  'BACKEND=fsdp2 SCALE=smoke SLEEP=1 STEPS=1 bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"; \
   tail -30 "$(cat /tmp/run_dapo_log.txt)"'

# Megatron-Native BF16 smoke（sleep 开）
docker exec "$CONTAINER" bash -lc \
  'BACKEND=megatron SCALE=smoke SLEEP=1 STEPS=1 bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"; \
   tail -30 "$(cat /tmp/run_dapo_log.txt)"'
```

**Smoke 期望证据**（全部满足即链路 OK）：

- `RLTrainer.setup (ray-controller) complete: algo=dapo, model=.../Qwen3-8B-Base, actor_workers=8, ref=False, resume_step=0`
- 8 个 `LumenActorWorker` + 8 个 `VLLMRayServer` actor：
  ```bash
  L=$(cat /tmp/run_dapo_log.txt)
  grep -aoE "VLLMRayServer pid=[0-9]+" "$L" | sort -u | wc -l    # 应为 8
  grep -aoE "LumenActorWorker pid=[0-9]+" "$L" | sort -u | wc -l  # 应为 8
  ```
- **后端判据**：`setup ... complete` 这行两个后端**完全一样**，分不出来。可靠的判法有两个 ——
  Megatron 会打 `LumenActorWorker: initialized megatron_native engine.`（每 rank 一条）；
  更省事的是看 `mem/actor_allocated_gb`：**FSDP2 ≈ 11.6、Megatron ≈ 57.4**，差 5 倍不会认错。
- `filter_groups round N: kept M/24 prompt groups (total .../8)`，**`kept` 必须 >0**，且累计要够 8 个组。
  实测 smoke 是两轮凑满（`kept 6/24` → `kept 9/24`，total 15/8）。
- `callbacks: step=1 entropy=... grad_norm≈0.83 ppo_kl≈0 rollout_corr/kl≈0.001`，最后 `=== exit=0 ===`
- **sleep 生效**（见 §11 的取证方法）：`CuMemAllocator: sleep freed 85.73 GiB`、`fall asleep` × 8。
- **不应**出现：`materialized on CPU`、`has no attribute`、`OutOfMemory`、`HSA_STATUS`、
  `Could not deserialize ATN`、`Unsupported flydsl version`、`Repo id must be in the form`。
- Megatron 侧的两条**无害告警**：`transformer_config.py:1734 UserWarning: full scope is deprecated`
  （每 rank 一条）和顶层 `import apex` 的 undefined symbol（只要 `env_verify.sh` 里
  `FusedLayerNorm/FusedAdam OK` 就不用管）。

> `logger.wandb_enabled` 在两个 smoke config 里都是 `false`，**所以 smoke 不会产生 W&B run，这是设计如此**，
> 不是配置漏了。

---

## 10. 长跑（`docker exec -d` 分离，防中断）

**先看磁盘**：8B FSDP2 单个 checkpoint 约 **90GB**，`save_total_limit: 5` 即约 450GB。
Megatron dist-checkpoint 的 8B 体积本次**没有实测**（1 步跑不到 `save_steps=50`），按不小于 FSDP2 预留。

```bash
df -h "$DATA_ROOT"      # 至少 500G 可用才谈得上按默认 save_total_limit=5 存
```

```bash
# FSDP2 长跑（1000 步）
docker exec -d "$CONTAINER" bash -lc \
  'BACKEND=fsdp2 SCALE=longrun SLEEP=1 STEPS=1000 bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"'

# Megatron-Native 长跑（1000 步）
docker exec -d "$CONTAINER" bash -lc \
  'BACKEND=megatron SCALE=longrun SLEEP=1 STEPS=1000 bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"'
```

> ⚠️ **两个后端不能同时跑**（同一批卡）。起之前确认显存回到空闲基线（每卡约 298MB）。
>
> 建议先 `STEPS=30` 起一版确认显存/指标健康，再上 1000 步。落盘更密或换位置用
> `EXTRA_OVERRIDE='checkpointing.save_steps=10 checkpointing.save_total_limit=2'`。
>
> ⚠️ 关落盘**不要写 `checkpointing.checkpoint_dir=`**（Hydra 会解析成 `None`，omegaconf 报
> `Incompatible value 'None' for field of type 'str'` 直接退出）。用 `save_steps` 设成一个跑不到的大数。

**确认已在跑：**

```bash
docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 240
  grep -aE "setup .ray-controller. complete|filter_groups round|callbacks: step=" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|HSA_STATUS" "$L" | tail'
```

首步实测约 **3 分钟**（`timing/step_s` FSDP2 172.8s / Megatron 180.1s），加上 8 个 actor 各自加载模型
和首次 rollout，从启动到看到 `callbacks: step=1` 约 5–6 分钟。

---

## 11. sleep：为什么开、实测数据、为什么 config 里的注释过时了

### 11.1 config 里那两条注释

四个 config 都带 `enable_sleep_mode: false`，理由写在注释里，**两个后端的说法还不一样**：

```yaml
# FSDP2 config：
#   vLLM cumem sleep is broken on this ROCm box (frees no memory + corrupts
#   weights) and the old KV-only patch is incompatible with vLLM 0.23. With
#   256GB MI300X the engine (gpu_mem_util=0.30 => ~76GB) stays resident.
# Megatron config：
#   cumem sleep broken on ROCm 7.2.3 (wake_up OOMs); vLLM stays resident
```

**三个说法在本平台全部不复现**：既释放了显存，也没损坏权重，Megatron 侧也没有 wake_up OOM。
注释里的前提是 **256GB MI300X + vLLM 0.23**，而这里是 **288GB MI355X (gfx950) + 镜像自带的 vLLM 0.22.1rc1**。

> ⚠️ **这个结论是绑定镜像的。** 如果换回旧 runbook 的 `vllm/vllm-openai-rocm:v0.23.0`，
> 注释里说的「旧 KV-only patch 与 vLLM 0.23 不兼容」可能重新成立。**换镜像后请重新按 §11.3 取证，
> 别直接信这份 runbook 的 `SLEEP=1` 默认值。**

### 11.2 sleep 在这条链路上的位置

`rl_trainer.py` 里 rollout 收集完就 `self._ray_vllm_engine.sleep()`；权重同步时先
`wake(tags=["weights"])` → IPC 推权重 → 再 `wake(tags=["kv_cache"])`。
用的是 **level=2**，即权重和 KV 全部丢弃、**不往 CPU 备份**（`0.00 GiB is backed up in CPU`），
因为每步都会经 ZMQ CUDA-IPC 重新推权重进引擎。两处都由 `enable_sleep` 单一开关控制，
`sleep_level` 只在开关打开时才有意义。

### 11.3 怎么取证（有个坑）

`run_dapo.sh` **把 `VLLM_LOGGING_LEVEL` 硬编码成 `WARN`**，而 cumem 的 sleep 日志是 INFO 级，
所以默认跑完**日志里一条 sleep 记录都没有**，很容易误判成「sleep 没生效」。
用 vLLM 的 `VLLM_LOGGING_CONFIG_PATH` 绕过（不用改 `run_dapo.sh`）：

```bash
cat > "$RL_ROOT/vllm_logging_info.json" <<'EOF'
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "vllm": {
      "class": "vllm.logging_utils.NewLineFormatter",
      "datefmt": "%m-%d %H:%M:%S",
      "format": "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "vllm": {
      "class": "logging.StreamHandler",
      "formatter": "vllm",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}}
}
EOF

docker exec -e VLLM_LOGGING_CONFIG_PATH="$RL_ROOT/vllm_logging_info.json" "$CONTAINER" bash -lc \
  'BACKEND=fsdp2 SCALE=smoke SLEEP=1 STEPS=1 bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"'

# 硬证据
docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt)
  grep -aoE "Enabling cumem allocator[^\"]*" "$L" | sort | uniq -c
  grep -aoE "sleep freed [0-9.]+ GiB" "$L" | sort -u
  grep -ac "fall asleep" "$L"                                   # 应为 8
  grep -aoE "It took [0-9.]+ seconds to wake up tags .*" "$L" | head -4'
```

实测输出：

```
      8 Enabling cumem allocator because sleep mode requires it.
sleep freed 85.73 GiB          （smoke；longrun 是 82.92 GiB）
8                              （fall asleep 次数 = 8 个引擎）
It took 0.0116-0.0128 seconds to wake up tags ['weights'].
It took 0.0055-0.0064 seconds to wake up tags ['kv_cache'].
```

Megatron 侧还会多打一行 vLLM `gpu_worker` 的设备级账：
`Sleep mode freed 83.34 GiB memory, 51.49 GiB memory is still in use` ——
**「still in use」不是漏了没释放**，那是同一张卡上 Megatron actor 的常驻训练显存。

### 11.4 实测收益：峰值不变，省的是训练窗口

用 `rocm-smi --showmeminfo vram` 每 5 秒采样 GPU0（`sleep OFF` 与 `sleep ON` 各跑一遍 smoke，
同样的 INFO 日志配置）：

```
sleep OFF   47.1 -> 117.8 -> 91.5 -> 91.8 -> 91.9 -> 102.1 -> 106.8 -> 0.5   GiB
sleep ON    47.1 -> 115.4 -> 91.6 -> 91.7 -> 40.3 ->  16.7 -> 102.7 -> 0.5   GiB
                                              ^^^^^^^^^^^^^ 引擎睡着，只剩训练态
```

**峰值几乎不变（117.8 vs 115.4 GiB）**，因为峰值出现在 rollout 阶段、那时引擎必须醒着。
sleep 省的是**训练阶段**那个窗口：从 91.9–106.8 GiB 掉到 16.7 GiB，差值和日志里的 85.73 GiB 对得上。

长跑规模下这个窗口的形状（GPU0，5s 采样）：

```
FSDP2 longrun      89.0（rollout 稳态，持续约 2 分钟）-> 23.4 -> 24.5 -> 30.6 -> 38.6 -> 95.1（唤醒）
Megatron longrun  134.6（rollout 稳态）-> 130.6 -> 111.3 -> 111.4 -> 131.3（唤醒）-> 57.9
```

**结论：smoke 规模上 sleep 没有实际收益**（`step_s` 24.15 vs 24.05，纯噪声；`max_reserved` 只有 31GB，
288GB 的卡怎么都撑得住）。**它的价值在长跑**：resp=20480 时训练侧激活大得多，Megatron 的
`mem/actor_max_reserved_gb` 实测已经到 **145.2GB**，那 83GB 常驻的引擎显存正好压在 backward 头上。
唤醒开销可以忽略（weights + kv_cache 合计约 18ms/步）。

**还有一层收益是 sleep 让你敢把 `gpu_memory_utilization` 调高。** 上面那些数字都是 0.30 下测的，
KV cache 只有 67.6–70.4 GiB；引擎既然每步都会把预算整份还回去，rollout 阶段就没理由只用三成卡。
所以 §8 的 wrapper 默认 `GPU_UTIL=0.6`。**这两个开关必须一起开**：0.6 下引擎约占 173GB，
Megatron 训练 pass 峰值 145GB，引擎不睡就装不下 288GB。同一台机器上用 vime 框架（不是 LumenRL）
在 resp=512 规模测过 0.3→0.6 的效果，可作参照：

| | util 0.3 | util 0.6 |
|---|---|---|
| Available KV cache memory | 70.02 GiB | 156.41 GiB |
| GPU KV cache size | 509,840 tokens | 1,138,960 tokens |
| sleep 每引擎释放 | 85.31 GiB | 171.73 GiB |
| 实测 VRAM 峰值 | — | 236.8 GiB / 288（82%） |
| `step_s` | 23.85 | 20.11（**主要是 aiter/JIT 缓存变热，不是 util 的功劳**） |

> 那次对比里 rollout 等待时间几乎没变（11.54 → 10.77s），降的是训练时间，而训练阶段引擎是睡着的、
> 与 util 无关。**所以不要把 util 提升当成吞吐优化** —— resp=512 时 0.30 给的 KV 本来就富余。
> util 真正有意义的场合是 resp=20480 的长跑，那里 KV 才可能成为瓶颈。
> **但 LumenRL 这条线在 0.6 下没有实测过，见 §8 的告示。**

### 11.5 权重没被损坏的判据

四次 smoke（sleep 开/关各两次）的 `rollout_corr/kl` 分别是 **0.00086 / 0.00096（sleep）/ 0.00103 / 0.00066（sleep）**,
全部落在旧 runbook 给的 BF16 期望 ≈0.001 区间内，**run-to-run 的采样噪声比 sleep 带来的差异还大**。
`reward/accuracy` 0.070–0.102、`grad_norm` 0.77–0.84、`ppo_kl` ~1e-5 量级，都在同一水平。
真的权重损坏会表现成 `rollout_corr/kl` 爆掉 + `reward/accuracy` 塌到 0，这里完全没有。

---

## 12. 监控 / 停止 / 续跑

**监控**（W&B `core/` 面板，或直接看日志）：

```bash
docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); grep -aE "callbacks: step=" "$L" | tail -5'
```

健康判据：

| 指标 | 期望 |
|---|---|
| `entropy` | ~0.8 起、缓降（趋势约 25 步后显现）；**已知会熵坍缩**，因为 `entropy_coeff=0` 是 verl recipe 本身的选择 |
| `grad_norm` | 无持续尖峰。smoke ~0.83；longrun 首步实测 ~0.25 |
| `ppo_kl` | ≈0（单步优化、old==train） |
| `rollout_corr/kl` | **6e-4 ~ 1.1e-3，且不随步数单调爬升**。爬上去优先查权重同步；逼近 TIS 阈值 2.0 才警惕 |
| `rollout_correction/is_weight_mean` | ≈1.0（实测 0.99997–1.0001） |
| `seq/max_len` | 在 20480 预算附近波动是健康的；**单调收缩**是长度崩塌 |
| `mem/actor_allocated_gb` | **恒定**（FSDP2 11.59 / Megatron 57.43）。`max_reserved` 随每步 batch 波动正常，**存活内存在动才是泄漏** |
| step 时间 | 首步约 3 分钟（resp=20480）；后续步随响应变长而增加 |

**停止**：

```bash
docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"
  pkill -9 -f "[E]ngineCore"; pkill -9 -f "spawn[_]main"; pkill -9 -f "compile_[w]orker"
  sleep 10; rocm-smi --showmeminfo vram | grep -i "Total Used" | head -3'   # 应回到约 298MB/卡
```

> `pkill -f` 的模式要写成 `[l]umenrl` 这种，否则会匹配到自己的命令行把自己杀掉。
> `run_dapo.sh` 只在**启动前**清理进程、收尾不清，所以中断后可能留下占显存的孤儿进程；
> 下次启动会被清掉，但期间这台机器等于没卡。

**续跑**：两个长跑 config 都是 `resume: true`，重跑 §10 同一命令即从最近 checkpoint 恢复（`save_steps: 50`）。

---

## 13. 排障（本次实测踩到的坑，按遇到顺序）

| 症状 | 原因 | 处理 |
|---|---|---|
| `Exception: Could not deserialize ATN with version 3 (expected 4)`（栈在 `omegaconf/grammar/gen/...Lexer.py`） | `math_verify[antlr4_13_2]` 把 antlr4 runtime 升到 4.13.2，镜像的 omegaconf 2.3.1 要求 4.9.\* | 用 `math_verify[antlr4_9_3]`；已装错则 `pip install "antlr4-python3-runtime==4.9.3"` |
| `ImportError: Unsupported flydsl version: expected >=0.1.5.dev515, got 0.1.4.2`（栈在 `packing.py` → `from aiter import flash_attn_varlen_func`） | `run_dapo.sh` 把仓库 aiter 放 PYTHONPATH 最前，它要求更新的 flydsl | `pip install "flydsl==0.1.8"`。之后 `amd-aiter ... requires flydsl<0.1.5` 的 pin 冲突是预期的，**不要回退** |
| `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/abs/path/models/Qwen3-8B-Base'` | 模型目录是**指向容器挂载范围之外**的软链，容器里链接断了，transformers 把绝对路径当成 HF repo id | 换成真实目录或**同盘硬链接**（`cp -al src dst`，不额外占空间） |
| `hf download` 报 `OSError: I/O error: Permission denied (os error 13)`（栈在 `xet_get`） | HF 的 xet 后端在这套共享盘上写缓存失败 | `export HF_HUB_DISABLE_XET=1`，并把 `HF_HOME` 指到大盘 |
| 日志里一条 sleep 记录都没有，但 `SLEEP=1` | `run_dapo.sh` 硬编码 `VLLM_LOGGING_LEVEL=WARN`，cumem 日志是 INFO 级 | 按 §11.3 用 `VLLM_LOGGING_CONFIG_PATH` |
| `import apex` 报 `undefined symbol: _ZN3c103hip28c10_hip_check_implementationEiPKcS2_ib`（栈在 `fused_rope.py:53`） | **PYTHONPATH 里没有仓库 aiter**。有 aiter 时 apex 在第 49 行就切到 aiter RoPE 后端，第 53 行那个 ABI 不匹配的 `.so` 根本不会 import | **不要重编 Apex**。把 `$RL_ROOT/aiter` 加到 PYTHONPATH（`run_dapo.sh` 已经这么做，`env_verify.sh` 也照抄了同一份 PYTHONPATH）。真实运行里 8 个 actor 都能正常 import |
| `pip install` 刷一堆 `megatron-bridge 0.5.0 requires ...` 未满足 | 镜像里预装了 megatron-bridge 的元数据，但本 runbook 不用它 | 忽略。**不要**去装 megatron-bridge 或按它的 pin 降 transformers |
| 每个 actor 刷 `expandable_segments not supported on this platform`（一次 smoke 约 25 条） | ROCm/HIP allocator 不支持 `expandable_segments`，而 `run_dapo.sh` 的默认值带它 | wrapper 已经 `export PYTORCH_CUDA_ALLOC_CONF=`（显式空值 → `run_dapo.sh` 会 `unset`） |
| `run_dapo.sh` 直接报错退出，提示 `RL_ROOT` | 脚本开头有 `: "${RL_ROOT:?}"`，`docker exec -d` 时容器环境没注入 | §4 的 `docker run -e RL_ROOT=... -e DATA_ROOT=...` 已经注入；自建 wrapper 记得**无条件** `export`（`${VAR:-default}` 对已存在的变量不生效，会静默写回旧盘） |
| `parameters should be materialized on CPU` | Ray 路径开了 `fsdp_cfg.param_offload` / `optimizer_offload` | 保持 `false`（config 默认就是 false） |
| `Incompatible value 'None' for field of type 'str'`（`full_key: checkpointing.checkpoint_dir`） | 用了 `checkpointing.checkpoint_dir=` 空值覆盖 | 想关落盘就把 `save_steps` 设成跑不到的大数 |
| `filter_groups collected no valid groups` | 用了 instruct/thinking 版模型而不是 **Base** 版 | 必须 `Qwen/Qwen3-8B-Base` |
| 长跑忽然全没了：`spur exec` 报 `job <id> is not running (state: NODE_FAIL)`，`squeue -u $USER` 空 | **计算节点故障**，不是训练崩了 | `scontrol show job <id>` 看 `Reason=NodeDown`。**先分清是节点挂了还是训练挂了再排查** —— 训练自己崩会在日志里留 Traceback/HSA_STATUS，节点挂了则连日志都取不到（日志在 node-local 盘上，见 §2）。需要真人重新 `spur alloc` 拿节点 |

**显存回退（OOM）**：先确认不是别的进程占着卡（`rocm-smi --showmeminfo vram`；`GPU% = 0` 但 `VRAM%` 不为 0
就是有人占着）。然后按顺序：**先把 `GPU_UTIL` 从默认 0.6 降回 0.30**（那是 §14 四种组合都验证过的值，
也是最可能的原因，因为 0.6 没有实测过）→ 确认 `SLEEP=1` 已开 → 降 `policy.max_response_length`（如 8192，同时
`max_total_sequence_length`/`max_token_len_per_gpu` 降到 9216）→ 降 `train_global_batch_size` / `gen_batch_size`
→ Megatron 侧降 `megatron_cfg.max_tokens_per_gpu`（旧 runbook 在 MoE resp=20480 上实测：22528 会因**碎片**
在 step 14 OOM，压到 8192 后峰值 reserved 从 177GB 降到 134GB。8B dense 在 21504 下实测没问题，
但如果撞 OOM，这是第一个该调的旋钮）→ 降 `gpu_memory_utilization`。

---

## 14. 实测数据汇总（8×MI355X gfx950 · 288GB/卡 · Qwen3-8B-Base · sleep level=2 开启 · **`gpu_memory_utilization=0.30`**）

四种组合**全部 `exit=0`、无 Traceback、无 OOM**。longrun 两列是**首步**（`STEPS=1` 跑长跑 config）。

> ⚠️ **这张表是在 `GPU_UTIL=0.30` 下测的**（当时 wrapper 还没有这个开关，用的是 config 原值）。
> wrapper 现在默认 `GPU_UTIL=0.6`，会让 `Available KV cache memory` 和 `sleep 每引擎释放` 这两行
> 大约翻倍、`实测 VRAM 峰值` 显著上升，**其余指标（kl / grad_norm / accuracy / step_s）不应受影响**
> —— 改的只是 KV cache 预算，不改优化问题。要拿 0.6 的准确数字请自己复测一遍。

| | FSDP2 smoke | Megatron smoke | FSDP2 longrun 首步 | Megatron longrun 首步 |
|---|---|---|---|---|
| `timing/step_s` | 24.05 | 20.68 | **172.8** | **180.1** |
| `timing/gen_s` | 13.81 | 14.73 | 153.4 | 164.3 |
| `timing/train_s` | 3.94 | 4.93 | 10.06 | 12.48 |
| `timing/ref_s` | 6.03 | 0.73 | 9.11 | 2.99 |
| `mem/actor_allocated_gb` | 11.59 | 57.43 | 11.59 | 57.43 |
| `mem/actor_max_reserved_gb` | 31.19 | 78.64 | 36.01 | **145.21** |
| 实测 VRAM 峰值（rocm-smi） | 115.4 | 149.9 | 115.0 | 144.9 |
| sleep 每引擎释放 | 85.73 GiB | 85.73 GiB | 82.92 GiB | 82.92 GiB |
| `Available KV cache memory` | 70.42 GiB | 70.42 GiB | 67.58 GiB（492,096 tok） | 67.58 GiB |
| `entropy` | 0.573 | 0.689 | 0.557 | 0.598 |
| `grad_norm` | 0.844 | 0.828 | 0.250 | 0.248 |
| `ppo_kl` | −5.5e-5 | −9.4e-5 | −1.7e-5 | −1.2e-5 |
| `rollout_corr/kl` | 0.000657 | 0.001078 | 0.000941 | 0.000874 |
| `rollout_correction/is_weight_mean` | 0.99998 | 1.00004 | 0.99997 | 1.00008 |
| `reward/accuracy` | 0.078 | 0.102 | 0.148 | 0.148 |
| `seq/mean_response_len` | 403 | 403 | 865 | 808 |
| `seq/max_len` | 774 | 774 | 15913 | 18602 |
| `lr`（step 1） | 2e-07 | 1e-07 | 2e-07 | 1e-07 |

几点读数说明：

- **`lr` 两个后端差一格**（2e-07 vs 1e-07）：warmup=10、lr=1e-6，FSDP2 报的是第 2 格、Megatron 报的是第 1 格，
  是调度器 step 索引起点的差异，**不是配置不一致**，对训练无影响。
- **`timing/ref_s` 差很多**（6.03 vs 0.73）：`ref=False`，这一项是 actor 自己重算 log-prob 的时间。
  Megatron 侧有 `log_probs_chunk_size: 1024` 的分块融合实现，所以快得多。
- **Megatron 的常驻显存是 FSDP2 的约 5 倍**（57.43 vs 11.59 GB allocated），长跑首步 `max_reserved` 到
  145.2GB，**这是 Megatron 这条线最需要盯的数**。它在 288GB 卡上还有余量，但换成 256GB 的 MI300X
  就要重新算账。
- **`mismatch/*` 那一族指标只有 FSDP2 有**（`mismatch/abs_diff`、`chi2_seq`、`chi2_token`、`k3_kl`、
  `frac_abs_gt_0.1`）。Megatron 侧只报 `rollout_corr/*` 和 `rollout_correction/*`，
  **不要以为是 Megatron 跑坏了。**
- **`rollout_correction/*` 这一族必须出现**，它们在就说明 TIS 真的接上了（Ray 控制器路径上
  `compute_rollout_is_weights` 曾经是死代码，config 里的 `rollout_is: token` 空转过一段时间）。
- 两条长跑首步的 `reward/accuracy` 都是 0.148、`seq/mean_response_len` 865 vs 808 —— 同一个
  `seed: 10086` 和同一份数据，两个后端拿到的是同一批 prompt，所以这个量级一致是预期的。

**尚未验证的部分（诚实说明）：**

- **wrapper 默认的 `GPU_UTIL=0.6` 没有在 LumenRL 这条线上跑过**，上表全部是 0.30 的数据。
  最需要担心的是 **Megatron + longrun**：0.6 时引擎约 173GB，而 Megatron 训练 pass 的
  `max_reserved` 已经 145GB，靠 sleep 在训练阶段把引擎腾空才装得下，rollout 阶段则是
  173GB 引擎 + 约 58GB Megatron 常驻 ≈ 231GB / 288GB，账面能过但余量不多。
  第一次上长跑建议先 `GPU_UTIL=0.6 STEPS=2` 探一版并采 VRAM，撞 OOM 就回 0.30。
- 两条长跑都只跑了**首步**，没有跑完整的观测窗口。所以「1000 步能不能跑完」「后期会不会长度崩塌 /
  熵坍缩到不可用」在这个镜像上**没有数据**。旧 runbook 在别的镜像上记录过 resp=20480 长序列在
  step ~278 的 `grad_norm` 尖峰、以及 MoE 那条线的长度崩塌，值得作为盯盘重点。
- **checkpoint 没有实际落过盘**（`save_steps: 50`，1 步跑不到），所以 Megatron 8B dist-checkpoint 的
  体积、以及 `resume` 是否正常，**都没验证**。第一次上长跑建议先用
  `EXTRA_OVERRIDE='checkpointing.save_steps=2'` 跑 3 步，确认落盘和恢复都正常再放手。
- AIME 在线验证（`val_steps: 10`）没有触发过。
- 只在 **8×MI355X (gfx950)** 上验证。MI300X（256GB）上 Megatron 长跑的 145GB `max_reserved`
  加上 83GB 的引擎会更紧张，且 §11.1 那两条 sleep 注释原本就是 MI300X 上的观测。

---

## 15. Qwen3-30B-A3B MoE（vLLM + FSDP2 BF16）

同一套环境换一个 MoE policy：**Qwen3-30B-A3B-Base**（30.5B 总参 / 3.3B 激活，48 层、128 专家、top-8）。
训练侧仍是 FSDP2，rollout 仍是同卡 colocated vLLM（TP=1）+ ZMQ CUDA-IPC 权重同步。
**§2–§7 的环境完全复用**，新机器只需再做两件事：下模型、以及注意下面两个坑。

> 规模/超参对齐 verl `recipe/low_precision/run_dapo_qwen3_moe_30b_megatron_fp8e2e.sh` 的 **BF16 基线**
> （verl 那张图三条曲线里 FP8 所对标的那条）。细节见旧 runbook §13.3。

### 16.1 相对 8B 路线的差异

| 项 | 8B dense | Qwen3-30B-A3B MoE |
|---|---|---|
| 模型 | `Qwen/Qwen3-8B-Base` | **`Qwen/Qwen3-30B-A3B-Base`，必须 Base 版** |
| 数据 | `data_cached/qwen3-8b-maxprompt1024/` | **同一份直接复用**（两个模型 `tokenizer.json`/`vocab.json`/`merges.txt` 的 md5 相同） |
| `flydsl` | 0.1.8（§5 已装） | 同 |
| router 精度 | — | **必须 `LUMENRL_FP32_MOE_ROUTER=0`**（wrapper 自动传） |
| `MODEL_PATH` | wrapper 默认 | **必须显式给**（`run_dapo.sh` 默认值是 8B；wrapper 自动传） |
| `GPU_UTIL` | 0.6 | **0.45**，见 §15.2 |
| actor 常驻显存 | 11.6 GB | **114.5 GB** |

**为什么必须是 Base 版**：instruct/thinking 版在 `max_response_length` 内永远不闭合 `</think>`，
于是每条样本都被截断、reward 恒为 −1、`filter_groups` 连续 10 轮 kept 0，直接抛
`RuntimeError: filter_groups collected no valid groups`。

下模型（约 57G）：

```bash
docker exec "$CONTAINER" bash -lc '
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3-30B-A3B-Base --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
du -sh "$DATA_ROOT/models/Qwen3-30B-A3B-Base"    # 约 57G，16 个 safetensors 分片
```

### 16.2 ⚠️ `GPU_UTIL=0.6` 在 MoE 上必然 OOM，而 sleep 救不了

这是本次实测踩到的第一个坑，**结论与直觉相反，值得完整记下来**。

`GPU_UTIL=0.6` 时引擎在**初始化阶段**就 OOM：

```
GPU 0 has a total capacity of 287.98 GiB of which 1.71 GiB is free.
Of the allocated memory 172.74 GiB is allocated by PyTorch
```

显存时间线（`rocm-smi` 每 5s 采样 GPU0）把三个阶段分得很清楚：

```
11:37:15  114.5 GiB   FSDP2 MoE actor 加载完（30.5B 权重 + fp32 master + optimizer）
11:37:45  171.6 GiB   vLLM 引擎又装进自己那份约 57GB 权重
11:39:12  285.4 GiB   引擎分配 KV cache 去凑满 0.6x288=172.8 GiB -> 撞上 288 上限
```

根因是 **`gpu_memory_utilization` 是「占整卡的比例」，不是「占剩余显存的比例」**。vLLM 按
`总显存 x util` 给自己算预算，**完全不看训练 actor 已经占掉的 114.5 GiB**。

**而 sleep 在这里帮不上忙**，因为 OOM 发生在引擎初始化、早于任何 sleep/wake 循环 ——
vLLM 必须先一次性把 KV cache 分配出来做 profiling。sleep 只能省训练窗口。
（这也说明 §11 那句「sleep 让你敢把 util 调高」对 8B 成立、对 MoE 有上限。）

**实测可用值是 0.45**，按 actor 常驻 114.5 GiB 算天花板约 `(288 − 114.5 − 余量)/288`：

| `GPU_UTIL` | 引擎预算 | + actor 114.5 | 余量 | 结果 |
|---|---|---|---|---|
| 0.30（config 原值） | 86.4 GiB | 201 GiB | 87 GiB | 未实测（宽裕） |
| **0.45** | **129.6 GiB** | **244 GiB** | **44 GiB** | ✅ **跑通**，KV 72.73 GiB / 794,352 tokens，sleep 每引擎释放 129.64 GiB，峰值 274.9 GiB（95.5%） |
| 0.50 | 144 GiB | 258 GiB | 31 GiB | 不建议：旧 runbook 记录过 33 GiB 余量不够用 |
| 0.60 | 172.8 GiB | 287 GiB | ~0 | ❌ 引擎初始化 OOM |

### 16.3 ⚠️ vLLM ≥0.22 的融合专家权重同步需要一处代码修复

第一次跑 MoE 时被覆盖率断言拦住：

```
weight sync (colocate-ipc) left 96/435 rollout parameters untouched:
  model.layers.N.mlp.experts.routed_experts.w13_weight / w2_weight
```

注意参数名多了一层 **`routed_experts`**（旧 runbook 里是 `experts.w13_weight`）。
这个 vLLM 版本重构了 `FusedMoE`，把专家权重挪进嵌套的 `RoutedExperts` 子模块
（`fused_moe/layer.py`: `routed_experts_prefix: str = "routed_experts"`），
而 `lumenrl/engine/inference/vllm_moe_weight_sync.py` 是按「FusedMoE 自己持有 `w13_weight`」写的：
它用「拥有 `w13_weight` 的模块名」当路由索引键，训练侧发来的却是 `...mlp.experts.gate_up_proj`
（前缀 `...mlp.experts`），**差一层就整批匹配不上，96 个融合张量（约 93% 的参数）全部被跳过**。

修复是在 `FusedMoEWeightRouter.__init__` 的发现阶段，除了模块自身名字，再按训练侧使用的短前缀
注册一个别名（`_CONTAINER_SEGMENTS = ("routed_experts",)` + `setdefault`，精确名字先注册，
所以真正的 FusedMoE 不会被别名遮蔽）。`module` / `param_name` / shard 派发本来就对，不用动。

> **这个修复应当提交回 Lumen-RL 仓库** —— 任何用 vLLM ≥0.22 跑 MoE 的机器都会中，不是本机配置问题。
>
> 验证方式（两重）：11 项单测 `python3 -m lumenrl.tests.test_moe_weight_sync` 全过，
> 以及带 `LUMENRL_WEIGHT_SYNC_VERIFY=1` 跑一次 smoke —— 它把加载后的 vLLM buffer 读回来和发送张量
> `torch.equal` 比对，96 个融合张量 × 8 replica × 3 次同步全部逐位一致才算通过。
>
> ⚠️ **不要用 `LUMENRL_WEIGHT_SYNC_CHECK=warn` 绕过这个断言。** 那等于回到 bug 能藏 54 步的状态：
> rollout 引擎会一直用磁盘加载的初始专家权重，`rollout_corr/kl` 会从 0.0005 一路爬到 0.0166。

### 16.4 Smoke（4k，3 步）与长跑

新机器第一次跑 MoE，**建议 smoke 带上 `LUMENRL_WEIGHT_SYNC_VERIFY=1`**（约 +0.1s/步）：

```bash
docker exec "$CONTAINER" bash -lc \
  'MODEL=moe BACKEND=fsdp2 SCALE=smoke SLEEP=1 STEPS=3 GPU_UTIL=0.45 \
   LUMENRL_WEIGHT_SYNC_VERIFY=1 bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"'
```

**长跑**。先看磁盘：MoE 的 FSDP2 checkpoint 约 **342GB/份**，`save_total_limit=N` 稳态就是 N 份。

```bash
df -h "$DATA_ROOT"      # save_total_limit=3 需要约 1.03TB

docker exec -d "$CONTAINER" bash -lc \
  'MODEL=moe BACKEND=fsdp2 SCALE=longrun SLEEP=1 STEPS=1000 GPU_UTIL=0.45 \
   EXTRA_OVERRIDE="checkpointing.save_steps=20 checkpointing.save_total_limit=3" \
   bash "$RL_ROOT/run_lumenrl_dapo_vime.sh"'
```

> W&B 走 config 自带的 project **`LUMEN-RL-MOE`**（不用覆盖），run 名
> `verlref-moe-a3b-bf16-B128-R20K-TIS`，key 从 `$RL_ROOT/wandb.key` 自动读。
> `checkpoint_dir` 走 `${oc.env:SCRATCH_ROOT}`，wrapper 默认把它指向 `$DATA_ROOT`，所以落在大盘上。
> 换 run 名：`EXTRA_OVERRIDE='logger.wandb.name=<your-name>'`。

**核实覆盖真的生效了**（比读日志可靠）：

```bash
docker exec "$CONTAINER" bash -lc \
  'tr "\0" " " < /proc/$(pgrep -f "[l]umenrl.trainer.main" | head -1)/cmdline' \
  | tr " " "\n" | grep -E "save_steps|save_total_limit|gpu_memory_utilization|enable_sleep|sleep_level"
```

### 16.5 MoE 实测数据（8×MI355X · `GPU_UTIL=0.45` · sleep level=2 · chunk_cat fallback 开）

**4k smoke（3 步，`VERIFY=1`，`exit=0`）：**

| step | 1 | 2 | 3 | 旧 runbook §13.5 参考（8×MI350X） |
|---|---|---|---|---|
| `rollout_corr/kl` | 0.00168 | 0.00178 | 0.00181 | 0.00150 / 0.00181 / 0.00149 |
| `mismatch/abs_diff` | 0.0202 | 0.0239 | 0.0222 | 0.0201 / 0.0218 / 0.0234 |
| `reward/accuracy` | 0.125 | 0.117 | 0.086 | 0.102 / 0.102 / 0.133 |
| `entropy` | 0.613 | 0.767 | 0.734 | — |
| `grad_norm` | 0.450 | 0.354 | 0.391 | — |
| `timing/step_s` | 105.5 | 98.0 | 101.6 | 约 200（3 步 10 分钟含加载） |
| `timing/weight_sync_s` | 2.61 | 1.20 | 1.18 | 1.1–1.7 |
| `mem/actor_max_reserved_gb` | 114.0 | 75.1 | 75.1 | 75–115 |

**长跑首步（resp=20480, batch 2048 序列, gen_batch 384）：**

| 指标 | 本次 | 旧 runbook §13.7 参考 |
|---|---|---|
| `rollout_corr/kl` | 0.00159 | 0.00148 |
| `reward/accuracy` | 0.131 | 0.136 |
| `entropy` | 0.882 | 0.844 |
| `grad_norm` | 0.0958 | 0.197 |
| `seq/max_len` | 20711（打满预算） | — |
| `timing/step_s` | 568.7（约 9.5 分钟） | 约 11 分钟中位数 |
| `timing/weight_sync_s` | 2.54 | 1.1–1.4 |
| `mem/actor_allocated_gb` / `max_reserved` | 42.80 / 114.0 | 42.8 / 75–115 |
| `filter_groups` | round 1 kept 163/384（一轮凑够 128） | — |

1000 步按 9.5 分钟/步约 **6.6 天**。

**盯盘判据（按重要性）：**

1. **`rollout_corr/kl` 不随步数单调爬升**，健康 6e-4 ~ 1.8e-3。爬升优先怀疑权重同步漏参数
   （回 §15.3 用 `VERIFY=1` 复查），其次是 router 精度两侧不匹配。
2. **`seq/max_len` 不收缩**。在 20480 附近波动健康，单调下滑是长度崩塌。
3. **`mem/actor_allocated_gb` 恒定**（42.80）。`max_reserved` 随 batch 波动正常，存活内存在动才是泄漏。
4. **已知会熵坍缩**：verl 这个 recipe 就是 `entropy_coeff=0`，旧 runbook 记录 101 步从 0.844 掉到 0.094。
   这是照搬 recipe 的必然结果。只有「entropy < 0.05 **且** 长度开始缩」同时出现才需要警惕。

**长跑实际结局：跑了约 4 小时 50 分钟后节点故障，不是训练本身出问题。**

```
JobId=36208  JobState=NODE_FAIL  Reason=NodeDown  ExitCode=-1:0
NodeList=crsuse2-m2m-090   EndTime=2026-08-04T17:45:12
```

起跑 12:44 UTC、step 1 在 12:53，节点 17:45 掉线。按约 10 分钟/步推算应该到了 30 步左右、
`save_steps=20` 应该落过一次盘，但**这两点都是推算、没有证据** —— 日志和 checkpoint 都在那块
node-local 盘上，节点没了就取不回来（见 §2 的盘位建议）。

**所以下列各项仍未验证：**

- **step 2 及之后的指标全部没有观测到**，只有 step 1。趋势类判据（`rollout_corr/kl` 是否爬升、
  `seq/max_len` 是否收缩、熵坍缩形状）在这个镜像上**一个都没测到**。
- **checkpoint 从未确认写成功**，342GB 这个体积、`save_total_limit=3` 的删旧逻辑、以及 `resume: true`
  能否正确恢复，**全都没有实测**。第一次上长跑建议先 `EXTRA_OVERRIDE='checkpointing.save_steps=2'`
  跑 3 步，确认落盘和恢复都正常再放手。
- AIME 在线验证（`val_steps: 10`）未触发过。
- `LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1` 在本镜像上只撑过约 4 小时 50 分钟的运行窗口，
  远未达到旧 runbook 记录的「约 124 万次 copy-in 未复现」那个量级，**不能算独立复现**。

---

## 16. 与其他 runbook 的关系

| 想做的事 | 去哪份 |
|---|---|
| BF16 + FSDP2/Megatron + MoE，vime 镜像（本文） | 就是这份 |
| FP8 rollout / FP8 E2E 训练 / ATOM rollout | `dapo-lumenrl-native-vllm-fsdp-runbook.md`（§7–§12、§14） |
| Qwen3-30B-A3B MoE（FSDP2 或 Megatron+EP=8） | 同上 §13（注意那条线要 `flydsl==0.1.8`、`LUMENRL_FP32_MOE_ROUTER` 两侧对齐） |
| verl 原生 `recipe/dapo`（复用同一环境） | 同上 §15 |
| Vime 框架（不是 LumenRL）的 DAPO | `dapo-vime-vllm-megatron-runbook.md` |
| 从零构建 Megatron 环境（非 vime 镜像） | `dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md` |
