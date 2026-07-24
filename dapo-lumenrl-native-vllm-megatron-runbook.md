# LumenRL 原生 DAPO Runbook（BF16，Megatron 训练后端 + vLLM，跨机快速复现）

> 在一台**全新 8 卡 AMD GPU 机器**上,用 **LumenRL 的 Megatron 训练后端**(而非 FSDP2)复现 verl `recipe/dapo`
> 的 DAPO 数学 RL 训练。训练侧换成 **Megatron-Core `GPTModel`（TP=1, DP=8, 分布式优化器）**,rollout 仍是
> **同卡 colocated vLLM `AsyncLLM`**。规模/超参 1:1 对齐 `dapo-lumenrl-native-vllm-fsdp-runbook.md`(BF16 路线),
> 只把训练后端从 `fsdp2` 换成 `megatron`。
>
> - **单 Ray-driver**:8 个 Megatron 训练 actor(TP=1, PP=1, CP=1, DP=8, **分布式优化器分片**)+ 8 个同卡
>   colocated vLLM 引擎(TP=1)在线 rollout;训练→rollout 权重经 **ZMQ CUDA-IPC** 同步(Megatron→HF 转换后发送)。
> - 训练后端 **Megatron-Core `GPTModel`**(local spec + torch RMSNorm;BF16 compute + FP32 master,
>   **distributed optimizer** 把 FP32 master + Adam 状态分片到 8 个 DP rank)。与 vime 的 Megatron 训练同源思路。
> - DAPO = GRPO + 动态采样(filter_groups)+ clip-higher/dual-clip + 零 KL + TIS,损失复用 LumenRL
>   `asymmetric_clip_loss`(与 FSDP 路线同一套 loss/编排/controller)。
> - 模型 **Qwen3-8B-Base**;数据 **DAPO-Math-17k**(train)/ **AIME-2024**(val)。镜像
>   `vllm/vllm-openai-rocm:v0.23.0`(含 vllm 0.23 + torch-ROCm)。
>
> **一句话复现**:设路径变量 → clone 3 仓库(Lumen-RL 用含 Megatron 后端的分支)→ 起容器 → 装 LumenRL 依赖
> **并额外装 `megatron-core`(仅此一个,无需编译 apex/TE)** → 下模型/数据 → 过滤 → smoke → 长跑。
>
> ✅ 已验证:8×MI355X(gfx950)、Qwen3-8B-Base。
> - **BF16 smoke** exit 0,指标健康(entropy≈0.6、grad_norm≈0.8、ppo_kl=0、rollout_corr/kl≈0.002),
>   **每卡 actor 显存 ≈60GB**(分布式优化器分片,和 vime 的 ~57GB 同量级),稳态 **train ≈1.9s/步**。
> - **BF16 longrun**(resp=20480,1000 步) 正常训练:开 **flash attention + 分块 log-prob + 动态打包**(§7.1)后
>   训练峰值 **~110GB/卡**(未开 FA 时 O(L²) attn 会 >230GB → OOM);训练算子 ~13s→**~9.7s/步**(打包);
>   指标健康(entropy≈0.6、grad_norm≈0.26、ppo_kl≈0)。
>
> ℹ️ 运行时用 **Megatron local spec + torch RMSNorm**(非 TE),**不依赖 apex / Transformer Engine**;只装
> `megatron-core` 即可(自动回退 torch norm / torch 优化器)。长序列注意力用容器自带 `flash_attn`(§7.1),同样**无需 TE**。
>
> ⚠️ 本文档保留的是 legacy `backend="megatron"` 路线。固定 revision 中的 Megatron YAML
> 已默认使用 `megatron_native`；本文所有启动命令都显式传
> `policy.training_backend=megatron`。新部署优先使用
> `dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md`。

---

## 1. 架构与对应关系（相对 FSDP 路线只换训练后端）

| 维度 | LumenRL FSDP2 路线 | 本 runbook（LumenRL Megatron） |
|---|---|---|
| 入口 | `python -m lumenrl.trainer.main`（Ray 控制器） | **同**（controller 后端无关,只换 engine） |
| 训练后端 | Lumen FSDP2（fp32 master + bf16 compute） | **Megatron-Core `GPTModel`**（local spec + torch RMSNorm;bf16 compute + fp32 master） |
| 并行 | FSDP2 参数分片,DP=8 | **TP=1 / PP=1 / CP=1 / DP=8**;`use_distributed_optimizer=True`(优化器状态分片到 DP) |
| 梯度同步 | FSDP2 reduce-scatter | **Megatron DDP grad buffer + `finalize_model_grads`**(DP mean) |
| logprob / 训练前向 | packed varlen（HF attn） | **每序列单条前向**（GPTModel,causal;engine-level `compute_log_probs`/`update_policy`） |
| 推理后端 | vLLM `AsyncLLM`（colocated,TP=1） | **同**（`enable_sleep_mode=false`，训练期间保持常驻） |
| 权重同步 | DTensor all-gather → ZMQ CUDA-IPC | **Megatron→HF 名称转换** → ZMQ CUDA-IPC（同 receiver） |
| 损失 / 优势 / TIS | `asymmetric_clip_loss` + grpo + token-mean + TIS | **同**（engine 复用同一套 loss/meta 契约） |

后端选择只由 config 的 `policy.training_backend` 决定：`fsdp2` → FSDP2Engine；
`megatron_native` → TE 原生引擎；`megatron` → 本 runbook 的 legacy `MegatronEngine`
(已注册 `backend="megatron"`,`actor_worker` 自动委托 engine-level 计算)。

> 代码全部在 `Lumen-RL` 仓库内(见 §3 分支)。Megatron 后端相关文件:
> `lumenrl/engine/training/megatron_engine.py`、`lumenrl/engine/training/qwen3_megatron_bridge.py`、
> `lumenrl/workers/actor_worker.py`(委托 shim)、`examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml`。

---

## 2. 路径变量（所有后续命令都用这三个变量，换机只改这里）

```bash
export RL_ROOT=/path/to/lumen_rl      # 代码根（内含 Lumen-RL / Lumen / aiter；可放持久盘）
export DATA_ROOT=/path/to/data        # 模型 / 数据 / 日志 / ckpt 根（可放大盘）
export CONTAINER=rl-vllm-megatron
mkdir -p "$RL_ROOT" "$DATA_ROOT/logs"
```

> 说明:代码路径与数据路径解耦。RL_ROOT 建议放持久/可备份盘(代码会改),DATA_ROOT 放大盘(模型/ckpt 大)。

---

## 3. 拉取代码（Lumen-RL 用含 Megatron 后端的分支）

```bash
cd "$RL_ROOT"
# Lumen-RL：dev/vllm-fsdp-dapo 分支已包含 Megatron 训练后端（VIME-style GPTModel）
git clone -b dev/vllm-fsdp-dapo   https://github.com/ZhangDanyang-AMD/Lumen-RL.git
git -C Lumen-RL checkout 523e92329d312a3265e0a031dd7982b0529c3ef5
git clone -b amd-atom-rollout     https://github.com/ZhangDanyang-AMD/Lumen.git
git clone -b lumen/triton_kernels https://github.com/ZhangDanyang-AMD/aiter.git

# aiter 的 JIT 依赖 composable_kernel；必须补齐
cd "$RL_ROOT/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel
```

中国内网机器可对本次命令使用代理镜像(不写死进 remote):

```bash
cd "$RL_ROOT"
GHP=https://gh-proxy.com/https://github.com
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b dev/vllm-fsdp-dapo   "$GHP/ZhangDanyang-AMD/Lumen-RL.git"
git -C Lumen-RL checkout 523e92329d312a3265e0a031dd7982b0529c3ef5
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b amd-atom-rollout     "$GHP/ZhangDanyang-AMD/Lumen.git"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch -b lumen/triton_kernels "$GHP/ZhangDanyang-AMD/aiter.git"
cd "$RL_ROOT/aiter"
git -c http.version=HTTP/1.1 -c url."$GHP/".insteadOf=https://github.com/ \
  submodule update --init --depth 1 3rdparty/composable_kernel
```

| 仓库 | 分支 | 用途 |
|---|---|---|
| `Lumen-RL` | `dev/vllm-fsdp-dapo` @ `523e92329d312a3265e0a031dd7982b0529c3ef5` | RL 主框架 **+ Megatron 训练后端** |
| `Lumen` | `amd-atom-rollout` | Lumen 库（rollout/工具依赖） |
| `aiter` | `lumen/triton_kernels` | AMD kernel（vLLM/训练用） |

> Megatron 后端只依赖 `megatron-core`(§5.2),不依赖 apex / Transformer Engine / verl / megatron-bridge / modelopt。

---

## 4. 启动 Docker（镜像同 FSDP 路线）

```bash
sudo docker pull vllm/vllm-openai-rocm:v0.23.0     # 已有可跳过；务必确认容器内 vllm 版本是 0.23.0
```

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
> 若 docker 可免 sudo(用户在 docker 组),去掉 `sudo` 即可。容器 `stop/start` 不丢依赖;`docker rm` 才丢
> (丢了要重跑第 5 节)。依赖很轻(只多一个 `megatron-core` wheel),无需编译,重装很快;也可
> `docker commit "$CONTAINER" rl-vllm-megatron:built` 保存镜像备用。

---

## 5. 装依赖（LumenRL 依赖 + Megatron 环境）

### 5.1 LumenRL 基础依赖（同 FSDP 路线）

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
git config --global --add safe.directory "$RL_ROOT/Lumen-RL" || true
git config --global --add safe.directory "$LUMEN_DIR" || true
git config --global --add safe.directory "$AITER_DIR" || true
cd "$AITER_DIR" && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop || pip install -e .
pip install -e "$LUMEN_DIR" --no-deps || true
cd "$RL_ROOT/Lumen-RL" && pip install -e . --no-deps
pip install "ray[default]>=2.9" "accelerate>=0.28" datasets \
  "math_verify[antlr4_13_2]" "omegaconf>=2.3,<2.4" safetensors wandb
'
```

### 5.2 Megatron-Core（**唯一额外依赖**）

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install --no-deps megatron-core==0.18.2'
# 验证：能 import，且自动回退到 Torch norm / Torch 优化器（无需 apex / TE）
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import megatron.core as mc
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.torch_norm import WrappedTorchNorm
print("megatron.core", mc.__version__, "OK (torch-norm / torch-optimizer fallback)")
PY'
```
> Megatron 引擎运行时用 **local spec + torch RMSNorm**(引擎里 `tb.LayerNormImpl = WrappedTorchNorm`)+
> Megatron 分布式优化器(无 apex 时走 torch 实现)。**因此只需 `megatron-core` 一个 wheel,不用编译 apex / TE**。
> 装 megatron-core 时看到 `Transformer Engine and Apex are not installed. Falling back to Torch ...` 属正常,
> 不影响 BF16 smoke。

### 5.3 验证 LumenRL + Megatron 引擎可导入

```bash
sudo docker exec "$CONTAINER" bash -lc '
cd "$RL_ROOT/Lumen-RL" && python3 - <<PY
import torch, vllm, ray, lumenrl
from lumenrl.engine.training.base_engine import EngineRegistry
cls = EngineRegistry.get_engine_cls(model_type="language_model", backend="megatron")
print("vllm", vllm.__version__, "gpus", torch.cuda.device_count(), "megatron engine:", cls.__name__)
print("import OK")
PY'
```
> 期望:`gpus 8`、`megatron engine: MegatronEngineWithLMHead`、`import OK`。BF16 路线走
> `VLLM_ROCM_USE_AITER=0`,无需第 FSDP-runbook §5 的 vLLM RMSNorm patch。

---

## 6. 下载模型 / 数据 + 过滤 prompt（与 FSDP 路线完全相同）

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D=os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset", local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset", local_dir=f"{D}/raw/AIME-2024")
PY'
```
中国内网可改用 ModelScope(ID 同 FSDP runbook §6)。

**过滤 prompt ≤1024**(与 FSDP runbook §6.2 同一脚本,产出同一路径,后续 config 默认引用):

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
    (first_parquet(f"{DATA}/raw/DAPO-Math-17k/**/*.parquet"), os.path.join(OUT_DIR, "dapo-math-17k.filtered.parquet")),
    (first_parquet(f"{DATA}/raw/AIME-2024/**/*.parquet"), os.path.join(OUT_DIR, "aime-2024.filtered.parquet")),
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
        ds = ds.filter(lambda d: doc2len(d) <= MAX_PROMPT_LENGTH, num_proc=nproc, desc="filter")
        ds.to_parquet(dst)
        print(f"[{src}] -> {dst}: {before} -> {len(ds)}")
if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 filter_prompts.py'
```

产出:
```
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val
```

---

## 7. 训练配置（已提交在 Lumen-RL 仓库）

| 路线 | config 文件 |
|---|---|
| **Megatron BF16 smoke** | `examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml` |
| **Megatron BF16 longrun** | `examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_longrun.yaml` |

smoke = FSDP 的 `dapo_qwen3_8b_ray_vllm_smoke.yaml` 复制版;longrun = FSDP `..._vllm_longrun.yaml`
复制版。相对 FSDP 路线的差异:

| 项 | 值 |
|---|---|
| `policy.training_backend` | YAML 默认 `megatron_native`；本文命令显式覆盖为 **`megatron`**（触发 legacy MegatronEngine） |
| `policy.generation.vllm_cfg.enable_sleep_mode` | **`false`**（ROCm 7.2.3 sleep/wake 不可靠） |
| `policy.training.megatron_cfg` | 见下（分布式优化器 + 长序列显存开关） |

smoke 规模:resp=512、`train_global_batch_size=128`(8×16)、`gen_batch_size=24`、`num_generations=16`、
lr 1e-6 / warmup 10、clip 0.2/0.28/c=10、filter_groups(acc)、TIS thr=2.0、seed 10086。
longrun 正式规模:resp=20480、`max_total_sequence_length=21504`、`max_token_len_per_gpu=21504`、
batch=512、gen_batch=96、`num_generations=16`、`num_training_steps=1000`、val 每 10 步、save 每 50 步、
wandb `project=LumenRL`。

### 7.1 `megatron_cfg` 显存开关（smoke 与 longrun 都已开启 FA + chunk）

```yaml
policy:
  training:
    megatron_cfg:
      use_distributed_optimizer: true    # 引擎也会强制开;FP32 master+Adam 分片到 DP
      recompute_granularity: full        # 激活重计算(逐层),bound 激活峰值(longrun)
      recompute_method: uniform
      recompute_num_layers: 1
      attention_backend: flash           # ★ flash attention:O(L) 显存(替代 local O(L^2))
      log_probs_chunk_size: 1024         # ★ 融合/分块 token log-prob:单个 [chunk,V] softmax 缓冲
      enable_dynamic_batch: true         # ★ 动态打包:多条序列 concat 成一次 varlen 前向
      max_tokens_per_gpu: 21504          #   每次前向 token 预算(smoke 用 2048)
```

> **为什么需要 FA + chunk**:Megatron local spec 的 `DotProductAttention` 会物化完整 O(L²) 注意力分数矩阵,
> 且 log-prob 走 `log_softmax([L,V])` 保留多份 [L,V] 张量。resp=20480 时训练峰值 >230GB/卡 → OOM。
> - `attention_backend: flash` → 把 spec 的 `core_attention` 换成 `FlashSelfAttentionCore`(用容器自带
>   `flash_attn`,**无需 TransformerEngine**),注意力显存降为 O(L)(对齐 vime 的 `--attention-backend flash`)。
> - `log_probs_chunk_size>0` → 用 `_FusedTokenLogProb`(单个 [L,V] softmax 缓冲,反向精确)+ 分块 no-grad
>   log-prob/entropy(对齐 vime 的 `--log-probs-chunk-size`)。
>
> 三个开关**默认关闭**(不写 `megatron_cfg` 即回到旧的逐条前向行为);smoke/longrun config 里都已显式开启。
> 开启后 longrun 训练峰值 **~230GB → ~110GB/卡**(FA+chunk 生效),vLLM 保持满 `gpu_memory_utilization=0.30`。
>
> **`enable_dynamic_batch`(动态打包,对齐 vime `--use-dynamic-batch-size`)**:把一个 micro-batch 的多条序列
> 按 `max_tokens_per_gpu` 贪心分箱、concat 成一条 `[1,T]` 流,用 **thd `PackedSeqParams` + `flash_attn_varlen_func`
> (cu_seqlens)** 做**一次** varlen 前向(每条 rotary/attention 由 cu_seqlens 隔离,数值与逐条前向完全一致,
> 已验证 diff=0)。短序列因此能打满 GEMM,减少 kernel launch。训练算子:逐条 ~13s → 打包 **~9.7s**(begin step,
> 含 20k 长尾;纯短序列收益更大)。显存不增(token 预算不变)。**需要 `attention_backend: flash`(引擎会自动置上)。**

> Megatron 并行度默认 TP=1/PP=1/CP=1/DP=8(可在 `megatron_cfg` 里覆盖 `tensor_model_parallel_size` 等)。

---

## 8. 统一启动脚本

复用 Lumen-RL 仓库内 `examples/DAPO/run_dapo.sh`(`git clone` 即得)。它用 `MODE`/`STEPS`/`CONFIG_OVERRIDE`
控制,路径走 `$RL_ROOT`/`$DATA_ROOT`。Megatron 路线用 `MODE=bf16`(rollout 走 vLLM BF16,`VLLM_ROCM_USE_AITER=0`)
+ `CONFIG_OVERRIDE=..._megatron_smoke.yaml` 指定 Megatron config 即可。

---

## 9. Smoke（小配置验证整链路，前台等结果）

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT';"   # 显式注入，避免容器内 RL_ROOT 为空

sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  EXTRA_OVERRIDE='policy.training_backend=megatron' \
  STEPS=1 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-megatron.log bash '$S'; \
  tail -60 \"\$(cat /tmp/run_dapo_log.txt)\""
```

**Smoke 期望证据**(全部满足即链路 OK):
- `RLTrainer.setup (ray-controller) complete: ... actor_workers=8`
- `MegatronEngine: model+distributed-optimizer ready, ... params, dp_size=8`(每 rank 一次)
- `[step 0] filter_groups round N: kept K/24 ...`(K>0)
- `callbacks: step=1 entropy≈0.6 grad_norm≈0.8 ppo_kl=0 rollout_corr/kl≈0.002 ...`,exit 0
- `mem/actor_max_allocated_gb ≈ 60`(分布式优化器分片后的显存;若 ~184GB 说明没开分布式优化器)
- **不应**出现:`RMSNorm ... is not supported in FusedLayerNorm`(引擎会强制 torch RMSNorm,正常不触发)、
  `OutOfMemory`、entropy≈4+(前向退化)。

> 首步慢:首次含 GPTModel 构建 + HF 权重加载(8B,每 rank)+ aiter/torch 编译;稳态 train ≈1.9s/步。

多跑几步看稳态(去掉首步 warmup):

```bash
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  EXTRA_OVERRIDE='policy.training_backend=megatron' \
  STEPS=8 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-megatron-8step.log bash '$S'"
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); grep -aE "callbacks: step=" "$L" | tail -8'
```

---

## 10. 启动长跑（`docker exec -d` 分离，防中断）

```bash
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT';"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_longrun.yaml \
  EXTRA_OVERRIDE='policy.training_backend=megatron' \
  STEPS=1000 MODE=bf16 bash '$S'"
# W&B（可选）：把 key 放 $RL_ROOT/wandb.key（格式 WANDB_API_KEY=xxxx），脚本自动读
```

确认已在跑:
```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 200
  grep -aE "setup .ray-controller. complete|distributed-optimizer ready|callbacks: step=" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|CUDA error" "$L" | tail'
```
> 建议先 `STEPS=30` 起一版确认显存/指标健康,再上 1000 步。

---

## 11. 监控 / 停止

**监控**:
```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); grep -aE "callbacks: step=" "$L" | tail -5'
```
健康判据(BF16,对齐 FSDP 路线):`entropy` ~0.6–0.8;`grad_norm` ~0.8–0.9 无持续尖峰;`ppo_kl≈0`;
`rollout_corr/kl` ~0.002(逼近 TIS 阈值 2.0 才警惕)。

**停止**:
```bash
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f lumenrl.trainer.main; pkill -9 -f VLLMRayServer; pkill -9 -f raylet; pkill -9 -f gcs_server
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -1'
```

---

## 12. 排障 / 内存 / 与 FSDP·vime 对比

**`(RMSNorm) is not supported in FusedLayerNorm`**:只在**你另外装了 apex** 时才可能触发(megatron 会默认选
apex FusedLayerNorm,而它不支持 RMSNorm)。引擎已在 `initialize()` 里 `tb.LayerNormImpl = WrappedTorchNorm`
并把 spec 的各 norm 替换为 `WrappedTorchNorm`,正常不会触发;不装 apex 时 megatron 默认就是 torch norm,更不会触发。
本 runbook 不装 apex,可忽略。

**显存**:开分布式优化器后每卡 actor ≈60GB(FP32 master + Adam 状态分片到 8 DP)。**未开**分布式优化器
(每卡全量复制 optimizer)会到 ~184GB。

**长序列 OOM(resp=20480)**:若 `megatron_cfg` 里**没开** `attention_backend: flash` + `log_probs_chunk_size`,
Megatron local attention 的 O(L²) 分数矩阵 + 完整 [L,V] `log_softmax` 会让训练峰值 >230GB/卡 → OOM
(报错 `Tried to allocate 3x.xx GiB ... update_policy`)。**解法就是 §7.1 的两个开关**(smoke/longrun 已默认开启),
开启后峰值 ~67GB/卡,vLLM 可保持 `gpu_memory_utilization=0.30`。仍 OOM 再降 `gpu_memory_utilization` /
`max_token_len_per_gpu` / `train_global_batch_size` / resp 长度。

**首步很慢 / 长时间无日志**:首次 GPTModel 构建 + 8B HF 权重加载(每 rank)+ aiter JIT,属环境预热,不算训练性能。

**对比(resp=512 smoke,8×MI355X,稳态 7 步平均)**:

| 指标 | LumenRL FSDP2 | LumenRL Megatron（本 runbook） | VIME Megatron |
|---|---|---|---|
| train（fwd+bwd+opt） | ~7.1 s | **~1.9 s** | ~2.1 s |
| ref logprob | ~1.8 s | ~0.5 s | ~0.14 s |
| rollout（gen） | ~14 s | ~16 s | ~24 s |
| 总 step | ~22.9 s | ~19 s | ~28.6 s |
| 每卡 actor 显存（max_allocated） | — | **~60 GB** | ~57 GB |
| grad_norm / ppo_kl / rollout_corr·kl | 0.77 / 0 / 0.001 | 0.8–0.9 / 0 / 0.002 | 0.86 / 0 / — |

> 结论:换成 Megatron 训练后端后,**训练算子时间 ~1.9s(≈vime),比 FSDP2 的 ~7s 快约 3.7×**;分布式优化器把每卡
> 显存降到 ~60GB(≈vime)。rollout/编排/损失/controller 全部复用 FSDP 路线,指标一致健康。

---

## 13. Megatron 后端代码路径（供排查，正常无需手改）

- `lumenrl/engine/training/megatron_engine.py`:`MegatronEngine`——构建 Qwen3 `GPTModel`(local spec + torch
  RMSNorm)、加载 HF 权重、`DistributedDataParallel` + `get_megatron_optimizer(use_distributed_optimizer=True)`
  + `OptimizerParamScheduler`;engine-level `engine_compute_log_probs` / `engine_update_policy`(DAPO/GRPO+TIS,
  `finalize_model_grads` → `optimizer.step()`);`get_per_tensor_param` = Megatron→HF 导出供 vLLM IPC 同步。
  注册 `backend="megatron"`。
  长序列显存两件套(§7.1,由 `megatron_cfg` 开关触发):
  - `FlashSelfAttentionCore`:`attention_backend=flash` 时换进 spec 的 `core_attention`,用 `flash_attn`(O(L),
    原生 GQA/causal,无需 TE)替代 local 的 O(L²) `DotProductAttention`。
  - `_FusedTokenLogProb`(autograd)+ `_token_logprob_train` / `_logprob_entropy_nograd`:`log_probs_chunk_size>0`
    时启用,单个 [L,V] softmax 缓冲 + 分块,消掉 `log_softmax([L,V])` 的多份 [L,V] 峰值。
  - `_forward_logits_packed` + `_build_bins` + `_row_policy_loss`:`enable_dynamic_batch=true` 时,把行按
    `max_tokens_per_gpu` 贪心分箱,concat 成 thd `PackedSeqParams` 做一次 `flash_attn_varlen` 前向,按 cu_seqlens
    切回每条算 loss、整箱一次 backward(`engine_update_policy` / `engine_compute_log_probs` 均支持)。
- `lumenrl/engine/training/qwen3_megatron_bridge.py`:HF Qwen3 ↔ Megatron `GPTModel` 权重双向转换
  (GQA 交织 `linear_qkv`、融合 `linear_fc1=[gate;up]`、per-head q/k norm)。
- `lumenrl/workers/actor_worker.py`:`compute_log_probs` / `update_policy` 检测到 engine 暴露 `engine_*` 时委托给
  engine(Megatron 走 GPTModel 前向/反向;FSDP 路线不受影响)。
- `examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml` 默认
  `training_backend: megatron_native`；本文启动命令显式覆盖为 `megatron`。
