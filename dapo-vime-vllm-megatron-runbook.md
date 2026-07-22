# Vime 原生 DAPO Runbook（BF16，Megatron + vLLM，跨机快速复现）

> 在一台**全新 8 卡 AMD GPU 机器**上,用 **Vime**(slime 衍生、以 vLLM 为 rollout 后端、Megatron 为训练后端)
> 复现 verl `recipe/dapo` 的 DAPO 数学 RL 训练。**规模/超参 1:1 对齐** LumenRL/verl 的
> `dapo-lumenrl-native-vllm-fsdp-runbook.md`(BF16 路线),只是把框架换成 Vime。
>
> - **单 Ray-driver**:8 个 Megatron 训练 actor(TP=1，DP=8)+ 8 个**同卡 colocated vLLM 引擎**(TP=1)在线 rollout;
>   训练→rollout 权重经 **CUDA-IPC** 同步(`--colocate`,同步 `train.py`,非异步)。
> - DAPO = **GRPO + 动态采样(filter_groups)+ token-level loss + clip-higher/dual-clip + 零 KL + TIS**。
> - 模型 **Qwen3-8B-Base**;训练数据 **DAPO-Math-17k**,验证集 **AIME-2024**。
> - 镜像 **`vllm/vime-rocm`**(ROCm7 + torch2.10 + vLLM,含 Megatron，vime 预装于 `/root/vime`)。
>
> **一句话复现**:设路径变量 → 拉 ROCm 镜像起容器 → 校验环境 → 下模型/数据 → 转 Megatron 权重 →
> heredoc 写 3 个脚本 → smoke(前台等结果)→ `docker exec -d` 起长跑。
>
> ✅ 已验证:8×MI355X(gfx950)、Qwen3-8B-Base、BF16 smoke 1 步 exit 0,指标健康
> (grad_norm≈0.97、ppo_kl=0、tis≈1.0、lr warmup 1e-7、global batch 128)。

---

## 1. 架构与对应关系

| 维度 | verl / LumenRL runbook | 本 runbook（Vime 原生） |
|---|---|---|
| 入口 | `recipe.dapo.main_dapo` / `lumenrl.trainer.main` | `python3 train.py`（Ray job，`--train-backend megatron`） |
| 编排 | Ray + HybridEngine / 单 driver | **单 Ray-driver：8 Megatron actor + 8 colocated vLLM 引擎** |
| 训练后端 | FSDP / Lumen FSDP2 | **Megatron**（TP=1, DP=8, PP=1, CP=1；BF16，grad allreduce fp32，分布式优化器分片） |
| 推理后端 | vLLM async | **vLLM**（每引擎 TP=1，router 调度，`enable_sleep_mode` 训练时休眠让出显存） |
| 权重同步 | ZMQ CUDA-IPC | **CUDA-IPC**（`--colocate`，`update_weight_mode=full`, transport=nccl/ipc） |
| rollout 输入 | token-in | chat 模板（`--apply-chat-template`，输入为 OpenAI messages） |
| seed | replica_rank + data.seed | `--rollout-seed 10086` |
| 动态采样 | `algorithm.filter_groups`(metric=acc) | `--dynamic-sampling-filter-path ...check_reward_nonzero_std` |
| TIS 修正 | `rollout_correction.rollout_is=token`, thr=2.0 | `--use-tis --tis-clip 2.0` |
| 策略损失 | clip-higher + dual-clip + token-mean | `--eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 --calculate-per-token-loss` |
| 优势估计 | grpo（按组归一化） | `--advantage-estimator grpo` |
| 奖励 | reward_manager=dapo + overlong_buffer | `--rm-type dapo --reward-key score`（**无 overlong 软惩罚**，见 §11） |

> ⚠️ 与 verl/LumenRL 的两处不可精确对齐(Vime 无原生等价，已在脚本注释标注)：
> 1. DAPO **overlong-buffer 奖励惩罚**（len=512）：Vime 规则奖励只给 ±1，无超长软惩罚。
> 2. filter_groups **max_num_gen_batches=10** 上限：Vime 会持续重采样到目标，无固定轮次开关。

---

## 2. 路径变量（所有命令都用这两个变量，换机只改这里）

```bash
export DATA_ROOT=/mnt/xxx/vime_data     # 模型 / 数据 / 日志 / ckpt 根（放大盘，容器重建不丢）
export CONTAINER=vime-dapo
mkdir -p "$DATA_ROOT/logs"
```

---

## 3. 拉取 ROCm 镜像并启动容器

> ⚠️ **必须用 `vllm/vime-rocm`(ROCm 版)**。`vllm/vime:latest` 是 **CUDA 版**(torch `+cu130`),
> 在 AMD 上 `torch.cuda.device_count()=0`、`rocm-smi` 看不到卡,**不可用**。

```bash
docker pull vllm/vime-rocm        # 约 82GB，首次较久
```

启动持久容器。镜像 **ENTRYPOINT 是 `sleep`**,所以必须 `--entrypoint /bin/bash` 覆盖,否则
`sleep infinity` 会变成 `sleep sleep infinity` 直接退出:

```bash
docker rm -f "$CONTAINER" 2>/dev/null
docker run -d --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --ulimit nofile=1048576:1048576 \
  --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --privileged \
  --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":/root/data \
  vllm/vime-rocm -c "sleep infinity"
```

> `--privileged` + `--group-add video` 已足够访问 GPU；避免 `--group-add render` 报
> `unable to find group render`（该组在容器内可能不存在，需数值 gid，用 `--privileged` 可完全绕开）。
> 容器 `stop/start` 不丢；只有 `docker rm` 才丢（丢了重跑 §3–§6）。

**校验环境(一次过)**:

```bash
docker exec "$CONTAINER" bash -lc '
python3 -c "import torch,vllm,ray;print(\"torch\",torch.__version__,\"hip\",getattr(torch.version,\"hip\",None),\"gpus\",torch.cuda.device_count(),\"| vllm\",vllm.__version__,\"| ray\",ray.__version__)"
cd /root/vime && python3 -c "import vime;print(\"vime OK\",vime.__file__)"
ls -d /root/Megatron-LM && which hf
'
```
> 期望:`hip 7.x`、`gpus 8`、`vime OK`、有 `/root/Megatron-LM` 与 `hf`。vime 预装于镜像 `/root/vime`。

---

## 4. 下载模型 / 数据（下载到持久盘 `/root/data`，再软链到 `/root/`）

脚本里用 `/root/Qwen3-8B-Base` 等路径；实际数据放在持久盘 `/root/data`(=宿主 `$DATA_ROOT`),用软链桥接。

```bash
docker exec "$CONTAINER" bash -lc '
set -e
# 模型（Qwen3-8B-Base）+ 训练数据（DAPO-Math-17k）+ 验证数据（AIME-2024）
hf download Qwen/Qwen3-8B-Base                --local-dir /root/data/Qwen3-8B-Base
hf download zhuzilin/dapo-math-17k --repo-type dataset --local-dir /root/data/dapo-math-17k
hf download zhuzilin/aime-2024     --repo-type dataset --local-dir /root/data/aime-2024

# 软链（脚本按 /root/... 引用；数据实际在持久盘）
# 注意：smoke 与 longrun 用【不同】的 ckpt 目录，避免 longrun 误 resume smoke 的 checkpoint
# 导致 OptimizerParamScheduler warmup 不一致报错（见 §11.8）。
mkdir -p /root/data/Qwen3-8B-Base_torch_dist /root/data/Qwen3-8B-Base_vime /root/data/Qwen3-8B-Base_longrun
ln -sfn /root/data/Qwen3-8B-Base            /root/Qwen3-8B-Base
ln -sfn /root/data/dapo-math-17k            /root/dapo-math-17k
ln -sfn /root/data/aime-2024                /root/aime-2024
ln -sfn /root/data/Qwen3-8B-Base_torch_dist /root/Qwen3-8B-Base_torch_dist
ln -sfn /root/data/Qwen3-8B-Base_vime       /root/Qwen3-8B-Base_vime        # smoke ckpt
ln -sfn /root/data/Qwen3-8B-Base_longrun    /root/Qwen3-8B-Base_longrun     # longrun ckpt
ls -l /root | grep -E "Qwen3-8B-Base|dapo|aime"
'
```

> `zhuzilin/dapo-math-17k`(jsonl)与 runbook 的 `BytedTsinghua-SIA/DAPO-Math-17k`(parquet)**同源内容**,
> 但已是 Vime 原生 jsonl 格式(`prompt` 为 OpenAI messages、`label` 为答案),无需再转换。

---

## 5. HF → Megatron torch_dist 权重转换

ROCm 上需要两个 flag:`--no-gradient-accumulation-fusion` 与 `--attention-backend flash`。

```bash
docker exec "$CONTAINER" bash -lc '
cd /root/vime && source scripts/models/qwen3-8B.sh   # Qwen3-8B-Base 与 Qwen3-8B 结构相同
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/root/vime:/root/Megatron-LM \
  torchrun --nproc-per-node=1 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion --attention-backend flash \
  --hf-checkpoint /root/Qwen3-8B-Base --save /root/Qwen3-8B-Base_torch_dist
'
# 期望日志：successfully saved checkpoint ... iteration 1 to /root/Qwen3-8B-Base_torch_dist
```

---

## 6. 写脚本（heredoc；smoke / longrun / ckpt 裁剪器）

三个脚本都写进 vime 仓库的 `scripts/` 下（因为它们 `source scripts/models/qwen3-8B.sh`、调用 `train.py`）。
DAPO = GRPO + 动态采样 + token-level loss + clip-higher/dual-clip + 零 KL + TIS。

### 6.1 Smoke 脚本（runbook §9 BF16 规模：resp=512, 8×16=128, over-sampling 24）

```bash
docker exec "$CONTAINER" bash -lc 'cat > /root/vime/scripts/run-dapo-smoke.sh' <<'EOF'
#!/bin/bash
# Vime BF16 DAPO smoke（runbook §9 BF16 规模；1 步验证整链路）
ray stop --force; pkill -9 -f "VLLM::"; pkill -9 -f EngineCore; pkill -9 -f "ray::"
pkill -9 -f "raylet|gcs_server|default_worker"; pkill -9 -f "trainer_entry"; sleep 3
set -ex
export PYTHONUNBUFFERED=1
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=${VISIBLE_GPUS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
IFS=',' read -r -a _g <<< "${CUDA_VISIBLE_DEVICES}"; NUM_GPUS=${NUM_GPUS:-${#_g[@]}}; HAS_NVLINK=0

NUM_ROLLOUT=${NUM_ROLLOUT:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-24}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=( --hf-checkpoint /root/Qwen3-8B-Base --ref-load /root/Qwen3-8B-Base_torch_dist
  --load /root/Qwen3-8B-Base_vime/ --save /root/Qwen3-8B-Base_vime/ --save-interval 100000 )
ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label
  --apply-chat-template --rollout-shuffle --rm-type dapo --reward-key score
  --num-rollout ${NUM_ROLLOUT} --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
  --rollout-max-response-len ${MAX_RESPONSE_LEN} --rollout-max-prompt-len 1024
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1 --rollout-seed 10086
  --global-batch-size ${GLOBAL_BATCH_SIZE} --num-steps-per-rollout 1 --balance-data )
EVAL_ARGS=()
PERF_ARGS=( --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu 4096 )
DAPO_ARGS=( --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
  --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 --calculate-per-token-loss
  --dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --use-tis --tis-clip 2.0 )
OPTIMIZER_ARGS=( --optimizer adam --lr 1e-6 --lr-decay-style constant --lr-warmup-iters 10
  --lr-decay-iters 1000 --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.98 )
WANDB_ARGS=( --use-wandb --wandb-project "${WANDB_PROJECT:-VimeRL}" --wandb-group "${WANDB_GROUP:-qwen3-8B-base-dapo-bf16-smoke}"
  --wandb-key "${WANDB_API_KEY:?export WANDB_API_KEY first}" --wandb-mode online )
VLLM_ARGS=( --rollout-num-gpus-per-engine 1 --vllm-gpu-memory-utilization 0.30
  --vllm-max-num-batched-tokens 8192 --vllm-max-num-seqs 64 --vllm-enforce-eager )
MISC_ARGS=( --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --train-memory-margin-bytes 2147483648
  --no-gradient-accumulation-fusion --no-offload-train )

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${VIME_ROOT}:/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\"}}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py --train-backend megatron --actor-num-nodes 1 --actor-num-gpus-per-node ${NUM_GPUS} --colocate \
  ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${DAPO_ARGS[@]} \
  ${WANDB_ARGS[@]} ${PERF_ARGS[@]} ${EVAL_ARGS[@]} ${VLLM_ARGS[@]} ${MISC_ARGS[@]}
EOF
docker exec "$CONTAINER" bash -lc 'chmod +x /root/vime/scripts/run-dapo-smoke.sh'
```

### 6.2 长跑脚本（runbook §7 正式规模：1000 步, 512=32×16, gen 96, resp=20480）

```bash
docker exec "$CONTAINER" bash -lc 'cat > /root/vime/scripts/run-dapo-longrun.sh' <<'EOF'
#!/bin/bash
# Vime BF16 DAPO 长跑（runbook §7 正式规模）
ray stop --force; pkill -9 -f "VLLM::"; pkill -9 -f EngineCore; pkill -9 -f "ray::"
pkill -9 -f "raylet|gcs_server|default_worker"; sleep 3
set -ex
export PYTHONUNBUFFERED=1
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=${VISIBLE_GPUS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
IFS=',' read -r -a _g <<< "${CUDA_VISIBLE_DEVICES}"; NUM_GPUS=${NUM_GPUS:-${#_g[@]}}; HAS_NVLINK=0

NUM_ROLLOUT=${NUM_ROLLOUT:-1000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-96}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-20480}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-21504}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=( --hf-checkpoint /root/Qwen3-8B-Base --ref-load /root/Qwen3-8B-Base_torch_dist
  --load /root/Qwen3-8B-Base_longrun/ --save /root/Qwen3-8B-Base_longrun/ --save-interval ${SAVE_INTERVAL} )
ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label
  --apply-chat-template --rollout-shuffle --rm-type dapo --reward-key score
  --num-rollout ${NUM_ROLLOUT} --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
  --rollout-max-response-len ${MAX_RESPONSE_LEN} --rollout-max-prompt-len 1024
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1 --rollout-seed 10086
  --global-batch-size ${GLOBAL_BATCH_SIZE} --num-steps-per-rollout 1 --balance-data )
EVAL_ARGS=( --eval-interval ${EVAL_INTERVAL} --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
  --n-samples-per-eval-prompt 16 --eval-max-response-len ${MAX_RESPONSE_LEN} --eval-top-p 1.0 )
PERF_ARGS=( --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} )
DAPO_ARGS=( --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
  --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 --calculate-per-token-loss
  --dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --use-tis --tis-clip 2.0 )
OPTIMIZER_ARGS=( --optimizer adam --lr 1e-6 --lr-decay-style constant --lr-warmup-iters 10
  --lr-decay-iters 1000 --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.98 )
WANDB_ARGS=( --use-wandb --wandb-project "${WANDB_PROJECT:-VimeRL}" --wandb-group "${WANDB_GROUP:-qwen3-8B-base-dapo-bf16-longrun}"
  --wandb-key "${WANDB_API_KEY:?export WANDB_API_KEY first}" --wandb-mode online )
VLLM_ARGS=( --rollout-num-gpus-per-engine 1 --vllm-gpu-memory-utilization 0.30 --vllm-max-model-len 21504
  --vllm-max-num-batched-tokens 32768 --vllm-max-num-seqs 64 --vllm-enforce-eager )
MISC_ARGS=( --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --train-memory-margin-bytes 2147483648
  --no-gradient-accumulation-fusion --no-offload-train )

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${VIME_ROOT}:/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\"}}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py --train-backend megatron --actor-num-nodes 1 --actor-num-gpus-per-node ${NUM_GPUS} --colocate \
  ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${DAPO_ARGS[@]} \
  ${WANDB_ARGS[@]} ${PERF_ARGS[@]} ${EVAL_ARGS[@]} ${VLLM_ARGS[@]} ${MISC_ARGS[@]}
EOF
docker exec "$CONTAINER" bash -lc 'chmod +x /root/vime/scripts/run-dapo-longrun.sh'
```

### 6.3 “只保留一个 checkpoint” 裁剪器（save_total_limit=1 等价）

Vime/Megatron **没有原生 `save_total_limit`**(默认保留每个 `iter_<n>`)。下面的裁剪器只删除
`latest_checkpointed_iteration.txt` 里记录迭代号**之前**的旧 ckpt,永不动当前/正在写入的,稳态只留 **1 个**。

```bash
docker exec "$CONTAINER" bash -lc 'cat > /root/vime/scripts/ckpt_keep_one.sh' <<'EOF'
#!/bin/bash
set -u
CKPT_DIR="${1:-/root/data/Qwen3-8B-Base_vime}"; INTERVAL="${2:-120}"
echo "[ckpt-keep-one] pruning $CKPT_DIR every ${INTERVAL}s"
while true; do
  T="$CKPT_DIR/latest_checkpointed_iteration.txt"
  if [ -f "$T" ]; then
    latest="$(tr -dc '0-9' < "$T")"
    if [ -n "$latest" ]; then
      for d in "$CKPT_DIR"/iter_*; do
        [ -d "$d" ] || continue
        n="$(basename "$d" | sed 's/^iter_0*//')"; [ -z "$n" ] && n=0
        if [ "$n" -lt "$latest" ] 2>/dev/null; then echo "[ckpt-keep-one] rm $d (latest=$latest)"; rm -rf "$d"; fi
      done
    fi
  fi
  sleep "$INTERVAL"
done
EOF
docker exec "$CONTAINER" bash -lc 'chmod +x /root/vime/scripts/ckpt_keep_one.sh'
```

### 6.4 W&B key

脚本要求 **`WANDB_API_KEY` 非空**(否则 argparse 会因 `--wandb-key` 空值报 `expected one argument`,见 §11)。
运行前 `export`,或放到文件里读取:

```bash
export WANDB_API_KEY=你的key            # 或从文件：WANDB_API_KEY=$(cut -d= -f2- /path/wandb.key | tr -d '[:space:]')
```

---

## 7. Smoke（前台等结果，1 步验证整链路）

```bash
docker exec "$CONTAINER" bash -lc '
cd /root/vime && rm -rf /root/data/Qwen3-8B-Base_vime/* 2>/dev/null
export WANDB_API_KEY="'"$WANDB_API_KEY"'"
bash scripts/run-dapo-smoke.sh 2>&1 | tail -60'
```

**Smoke 期望证据**(全部满足即链路 OK):
- 8 个 vLLM 引擎就绪:`Initializing a V1 LLM engine ... dtype=torch.bfloat16, tensor_parallel_size=1, enforce_eager=True`
- 数据过滤:`Filtered N samples longer than max_length=1024`
- checkpoint 加载:`successfully loaded checkpoint from /root/Qwen3-8B-Base_torch_dist`
- rollout 生成:`First rollout sample: ...` + vLLM `POST /inference/v1/generate 200 OK`(GPU 利用率飙升)
- 训练一步:`step 0: {train/pg_loss:..., train/grad_norm≈0.9x, train/ppo_kl:0.0, train/tis≈1.0, train/lr-pg_0:1e-07, train/global_batch_size:128}`
- ray job:`Job 'raysubmit_...' succeeded`,exit 0
- **不应**出现:`must be real number, not dict`(reward-key 未设)、`assert self.lr_warmup_steps < self.lr_decay_steps`、
  `expected one argument`(wandb key 空)、CUDA/HSA error。收尾的 wandb `atexit` Traceback 与
  `RotaryEmbedding: Failed to load weights` WARNING 均**无害**。

---

## 8. 长跑（`docker exec -d` 分离，防中断）+ 只留一个 checkpoint

```bash
# 起裁剪器（后台，稳态只留 1 个 checkpoint；注意指向 longrun 专用目录）
docker exec -d "$CONTAINER" bash -lc 'nohup bash /root/vime/scripts/ckpt_keep_one.sh /root/data/Qwen3-8B-Base_longrun 120 > /root/data/ckpt_pruner.log 2>&1'

# 起长跑（后台）。首次起跑清空 longrun ckpt 目录（续跑时【不要】清空）。
docker exec -d "$CONTAINER" bash -lc '
cd /root/vime && rm -rf /root/data/Qwen3-8B-Base_longrun/* 2>/dev/null
export WANDB_API_KEY="'"$WANDB_API_KEY"'"
bash scripts/run-dapo-longrun.sh > /root/data/dapo_longrun.log 2>&1'
```

> 建议先 `NUM_ROLLOUT=30` 起一版确认显存/指标健康,再上 1000 步。续跑:`--load == --save`,checkpoint 存在时
> 自动从最近一步恢复(`--save-interval 50`);重跑同一命令即可(**不要**删 `/root/data/Qwen3-8B-Base_vime`)。
> ⚠️ 首步较慢:主要是 resp=20480 的首轮 rollout;TP=1 与 torch_dist(TP=1 存)一致,加载**无需 reshard**,起步比 TP>1 快。实测约 2–3 分钟/步。

**确认已在跑**:
```bash
docker exec "$CONTAINER" bash -c 'sleep 200; grep -aE "train/step|perf/step_time" /root/data/dapo_longrun.log | tail -3; grep -aiE "Traceback|OutOfMemory|CUDA error" /root/data/dapo_longrun.log | tail'
```

---

## 9. 监控 / 停止 / 续跑

**监控**（W&B project `VimeRL`，或看日志）:
```bash
docker exec "$CONTAINER" bash -c 'grep -aE "train/step|perf/step_time|eval" /root/data/dapo_longrun.log | tail -5'
```

健康判据(BF16,对齐 runbook)：`entropy` ~0.8 缓降;`grad_norm` ~0.85 无持续尖峰;`ppo_kl≈0`;
`tis(ois)≈1.0`;`train/tis_abs`(≈rollout_corr/kl)较小(逼近 TIS 阈值 2.0 才警惕)。

**停止**:
```bash
docker exec "$CONTAINER" bash -c 'ray stop --force; pkill -9 -f train.py; pkill -9 -f "VLLM::"; pkill -9 -f ckpt_keep_one; sleep 5; rocm-smi --showuse | grep -i "GPU use" | head'
```

**续跑**:重跑 §8 长跑命令(不清空 save 目录)即从最近 checkpoint 恢复。

---

## 10. runbook 配置 → Vime 参数映射（速查）

| runbook（§7 正式 / §9 smoke） | Vime 参数 |
|---|---|
| steps 1000 / global batch 512(=32×16) | `--num-rollout 1000` / `--global-batch-size 512` |
| kept prompts 32 / n 16 / gen_batch 96 | `--rollout-batch-size 32 --n-samples-per-prompt 16 --over-sampling-batch-size 96` |
| resp 20480 / prompt≤1024 / token_len_per_gpu 21504 | `--rollout-max-response-len 20480 --rollout-max-prompt-len 1024 --max-tokens-per-gpu 21504` |
| clip 0.2/0.28/c=10 + token-mean | `--eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10.0 --calculate-per-token-loss` |
| grpo / 零 KL | `--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.0` |
| filter_groups(metric=acc) | `--dynamic-sampling-filter-path ...check_reward_nonzero_std` |
| rollout_is=token, thr=2.0 | `--use-tis --tis-clip 2.0` |
| reward_manager=dapo | `--rm-type dapo --reward-key score` |
| lr 1e-6 / warmup 10 / wd 0.1 / grad_clip 1.0 | `--lr 1e-6 --lr-warmup-iters 10 --lr-decay-iters 1000 --weight-decay 0.1 --clip-grad 1.0` |
| seed 10086 / temp 1.0 / top_p 1.0 / top_k -1 | `--rollout-seed 10086 --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1` |
| val_steps 10 / save_steps 50 | `--eval-interval 10`(AIME) / `--save-interval 50`(+ §6.3 裁剪器) |
| vllm: mem 0.30, max_len 21504, eager, batched 32768, seqs 64 | `--vllm-gpu-memory-utilization 0.30 --vllm-max-model-len 21504 --vllm-enforce-eager --vllm-max-num-batched-tokens 32768 --vllm-max-num-seqs 64` |
| 8 卡 colocate 同步 | `--colocate --actor-num-gpus-per-node 8`,rollout `--rollout-num-gpus-per-engine 1`(8 引擎 TP=1),train **TP=1**（DP=8，分布式优化器分片；与 TP=1 存的 torch_dist 一致，加载无需 reshard，起步更快）。TP>1 时才加 `--sequence-parallel` |

---

## 11. 排障（本流程实测遇到的坑）

1. **`torch.cuda.device_count()=0` / `rocm-smi` 无卡** → 用错镜像。必须 `vllm/vime-rocm`,不是 `vllm/vime:latest`(CUDA)。
2. **容器起来就退出（`sleep: invalid time interval 'sleep'`）** → 镜像 ENTRYPOINT 是 `sleep`,启动要加
   `--entrypoint /bin/bash ... -c "sleep infinity"`(见 §3)。
3. **`train.py: error: argument --wandb-key: expected one argument`** → `WANDB_API_KEY` 为空,脚本里
   `${WANDB_ARGS[@]}` 无引号展开会丢掉空的 key 值。运行前必须 `export WANDB_API_KEY=...`(§6.4)。
4. **`assert self.lr_warmup_steps < self.lr_decay_steps` AssertionError** → `--num-rollout` 小(如 smoke=1)时
   Megatron 推出的 decay 步数 < warmup(10)。已用 `--lr-decay-iters 1000` 固定 LR 计划区间(constant decay,
   warmup 后 LR 恒定)解决。
5. **`TypeError: must be real number, not dict`（动态采样 filter 处）** → `--rm-type dapo` 返回
   `{score, acc, pred}` 字典,非标量。加 `--reward-key score` 取数值 score(与 filter_groups metric=acc 等价,
   因 score=2*acc-1)。
6. **colocate 显存**：已用 `--no-offload-train`(ROCm gfx950 上 colocate offload 会漏 VRAM)+ vLLM
   `enable_sleep_mode`(训练时休眠让出 KV)。若 OOM:降 `--vllm-gpu-memory-utilization`、`--max-tokens-per-gpu`,
   或降 `--global-batch-size` / resp 长度。
7. **无害告警**:`RotaryEmbedding: Failed to load weights`、退出时 wandb `atexit` Traceback、
   `[aiter] ... not found tuned config, will use default`——均不影响正确性。
8. **`OptimizerParamScheduler: class input value 5120 and checkpoint value 1280 for warmup iterations do not match`**
   → longrun 误 resume 了 smoke 存下的 checkpoint(smoke warmup=10×128=1280,longrun=10×512=5120)。
   原因:两者共用同一 `--load/--save` 目录。修复:**smoke 与 longrun 用不同 ckpt 目录**
   (本 runbook:smoke=`/root/Qwen3-8B-Base_vime`,longrun=`/root/Qwen3-8B-Base_longrun`),
   且首次起长跑前清空 longrun 目录。`--save-interval` 存的 `iter_0000000` 也会触发,务必隔离。

---

## 12. Vime vs verl/LumenRL（一览）

| 维度 | verl / LumenRL runbook | 本 runbook（Vime） |
|---|---|---|
| 入口 | `recipe.dapo.main_dapo` / `lumenrl.trainer.main` | `train.py`（Ray + `--train-backend megatron`） |
| 训练后端 | FSDP / FSDP2 | Megatron（TP=1, DP=8） |
| rollout | vLLM async | vLLM（router，8×TP=1，colocate 休眠） |
| 权重同步 | ZMQ CUDA-IPC | CUDA-IPC（`--colocate`） |
| 动态采样 | filter_groups(acc) | `check_reward_nonzero_std`(score，等价) |
| overlong 惩罚 | 有(len=512) | **无**（Vime 规则奖励只 ±1） |
| save_total_limit | 原生支持 | 外部裁剪器 `ckpt_keep_one.sh`（§6.3） |
| 已验证 | 8×MI350X | 8×MI355X(gfx950)，BF16 smoke exit 0 |
