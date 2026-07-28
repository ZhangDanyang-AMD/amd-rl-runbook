# Qwen3-30B-A3B 两节点 RL Runbook

> Megatron 训练 + vLLM rollout + RCCL/RoCE GPU Direct RDMA 权重同步  
> 硬件：2 × 8 MI308X（gfx942，192 GiB HBM/GPU）  
> LumenRL：`origin/dev/moe-grpo`  
> 当前验证日期：2026-07-28

## 1. 当前方案

本 runbook 只描述当前已验证的两节点训推分离方案：

- 训练节点 `172.25.12.3`：8 个 Megatron actor，TP=4、EP=8，ETP=1。
- rollout 节点 `172.25.12.1`：4 个 vLLM replica，每个 TP=2，共 8 个 vLLM worker。
- 训练权重：BF16 compute + Megatron distributed optimizer 的 FP32 master/Adam state。
- rollout 权重：BF16；KV cache 使用 `auto`，即当前稳定 BF16 baseline。
- 权重同步：独立 9-rank `torch.distributed` process group。
  - rank 0：Megatron sender。
  - rank 1–8：vLLM TP receivers。
  - ROCm 的 `backend="nccl"` 实际由 RCCL 执行。
  - 网络固定为 `mlx5_0` / `ens11np0` / GID index 3。
  - 日志必须出现 `Using network IB` 和 `NET/IB/0/GDRDMA`。
- R3：vLLM 记录 top-k expert IDs，Megatron `RouterReplay` 执行 hard assignment replay。
- 算法：GRPO/DAPO 风格 32 prompts × 8 generations，全局 batch 256。
- 正式目标：200 steps。

本流程不使用 ATOM rollout、ZMQ CUDA-IPC 或跨节点 safetensors 作为主路径。镜像名仍包含
`atom-dev`，但 ATOM Python 包只是基础镜像附带组件，不参与当前训练数据流。

## 2. 已验证结果

### 2.1 RDMA smoke

`qwen3-30b-a3b-rdma-smoke3-v4` 连续 3 步通过：

- 每步广播 61.1 GB，58 buckets。
- 总同步耗时 2.51–3.90 秒。
- 有效吞吐 134–215 Gb/s。
- `rollout_corr/kl`：0.0008425、0.0008446、0.0007718。
- ESS：0.998361、0.998328、0.998397。
- 每个 TP worker 动态校验完整权重覆盖；当次模型观测为 18,867 个 HF tensor、
  435 个 vLLM internal parameter。

以上 tensor 数量是 Qwen3-30B-A3B 当前版本的运行时观测，不是源码中的硬编码常量。

### 2.2 正式 v2

当前正式任务：

```text
run:  qwen3-30b-a3b-rdma-longrun-200-v2
log:  /mnt/raid0/lumenrl_logs/qwen3-30b-a3b-rdma-longrun-200-v2.log
ckpt: /mnt/raid0/ckpts/qwen3-30b-a3b-rdma-longrun-200-v2
W&B:  https://wandb.ai/danyzhan-amd/LumenRL/runs/jq122ogj
```

step 1–5 已验证无 NaN：

```text
KL:   0.0008134, 0.0008426, 0.0007873, 0.0007740, 0.0007181
ESS:  0.998350,  0.998342,  0.998371,  0.998409,  0.998529
RDMA: 132–217 Gb/s
```

截至 2026-07-28 15:20（UTC+8）已完成 step 18，仍无 NaN；step 18
`rollout_corr/kl=0.0008406`、ESS=`0.998302`、RDMA=`213.96 Gb/s`。

step 5 checkpoint 约 402 GiB，包含：

- 8 个 BF16 model shard。
- 8 个 optimizer metadata shard。
- 8 个 extra-state shard（global step、scheduler、CPU/HIP RNG）。
- 8 个 41–45 GiB distributed optimizer parameter-state shard，包含 FP32 master 与 Adam moments。

## 3. 为什么 v1 不能续训

v1 在 step 25 保存 checkpoint 时磁盘写满。旧 `save_checkpoint()` 只保存了
`optimizer.state_dict()`，对于 Megatron distributed optimizer，该对象只有约 2 KiB 的
`param_groups`，不包含 FP32 master 或 Adam moments。

因此 v1 的 step 5/10/15/20 只有可用 model shard，没有可续训 optimizer state。尝试只加载
model 会导致下一次 optimizer update 后出现 NaN。v1 checkpoint 已清理，正式训练从 step 0
以 v2 重启。

当前代码必须同时调用：

```text
optimizer.state_dict()
optimizer.save_parameter_state(...)
optimizer.load_parameter_state(...)
optimizer.reload_model_params()
```

严禁把只有约 2 KiB optimizer 文件的 checkpoint 视为完整 checkpoint。

## 4. 节点和路径

### 4.1 节点角色

| 角色 | Ray node IP | RoCE IP | GPU | 容器镜像 |
|---|---|---|---|---|
| rollout / Ray head | `172.25.12.1` | `192.168.100.5` | 8 | `rocm/atom-dev:vllm-latest` |
| Megatron trainer | `172.25.12.3` | 同网段另一端 | 8 | `rocm/atom-dev:latest` |

配置使用 Ray node resource 固定角色：

```yaml
controller:
  ray:
    actor:
      process_on_nodes: [8]
      topology_tags:
        node_ip: 172.25.12.3
    rollout:
      process_on_nodes: [8]
      topology_tags:
        node_ip: 172.25.12.1
```

### 4.2 主机路径

```bash
export RL_ROOT=/home/snx
export LUMENRL_DIR=/home/snx/Lumen-RL
export DATA_ROOT=/mnt/raid0
export MODEL_HOST_PATH=/volumes/oss0/models
export SHARED_ROOT=/volumes/oss1
export CONTAINER=qwen3-30b-rl
```

容器内路径：

```text
/workspace/Lumen-RL       LumenRL 源码
/workspace/Lumen          Lumen 依赖源码
/workspace/ATOM           基础镜像附带依赖；当前 flow 不调用 ATOM rollout
/workspace/aiter          AITER 源码
/workspace/flash-attention
/root/models              模型
/root/data_cached         已过滤数据
/mnt/raid0                日志与 checkpoint
/volumes/oss1             共享盘/可选 shared_folder fallback
/dev/infiniband           RoCE verbs 设备
```

## 5. LumenRL 代码版本

唯一要求的 LumenRL 分支是：

```text
origin/dev/moe-grpo
```

当前已验证提交：

```text
d16906b feat(megatron): flash-attn core + chunked/fused log-prob for long-seq DAPO
```

准备代码：

```bash
cd /home/snx/Lumen-RL
git fetch origin dev/moe-grpo
git switch dev/moe-grpo 2>/dev/null \
  || git switch -c dev/moe-grpo --track origin/dev/moe-grpo
git merge --ff-only origin/dev/moe-grpo

test "$(git rev-parse HEAD)" = "$(git rev-parse origin/dev/moe-grpo)"
git log -1 --oneline
```

注意：

- 不要把本 runbook 写成“主干最新版本”；必须明确 `origin/dev/moe-grpo`。
- 当前工作区包含尚未提交的 RDMA/checkpoint 修复。部署前应确认这些文件在两个节点 checksum
  一致，或将其提交到该分支后再部署。
- RDMA 实现参考 MILES 架构思想，但没有复制或嵌入 MILES 源码。

## 6. 当前容器和软件版本

### 6.1 两节点公共版本

以下版本来自当前 `qwen3-30b-rl` 容器实测：

| 包 | 版本 |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.10.0+rocm7.2.4.git3d3aa833` |
| HIP (`torch.version.hip`) | `7.2.53211` |
| Ray | `2.56.1` |
| Megatron-Core | `0.18.2` |
| flash-attn | `2.8.4` |
| amd-aiter | `0.1.0` |
| Transformers | `5.2.0` |
| Datasets | `5.0.0` |
| Accelerate | `1.14.0` |
| Safetensors | `0.8.0` |
| OmegaConf | `2.3.1` |
| Tokenizers | `0.22.2` |
| math-verify | `0.3.3` |
| antlr4-python3-runtime | `4.9.3` |
| W&B | `0.28.1` |

### 6.2 rollout 节点特有版本

| 包 | 版本 |
|---|---|
| vLLM | `0.22.1.dev0+g0b3ba88f1.d20260629.rocm724` |
| NumPy | `2.1.3` |
| Triton | `3.7.0+amd.rocm7.2.0.gitd0d77a509` |
| triton_kernels | `1.0.0+amd.rocm7.2.0.gitd0d77a509` |
| 基础镜像附带 ATOM | `0.1.4.dev208+g96ad40621`（当前 flow 不使用） |

### 6.3 trainer 节点特有版本

| 包 | 版本 |
|---|---|
| vLLM | 未安装，这是预期状态 |
| NumPy | `2.4.6` |
| Triton | `3.7.0+amd.rocm7.2.0.git89002410` |
| triton_kernels | `1.0.0+amd.rocm7.2.0.git89002410` |
| 基础镜像附带 ATOM | `0.1.6rc1.dev117+g3321d0ff0`（当前 flow 不使用） |

LumenRL/Lumen 通过源码路径和 `PYTHONPATH` 使用，因此 `pip show lumenrl` / `pip show lumen`
可能显示未安装，这不代表运行环境缺失。

## 7. Docker 配置

当前容器实测配置：

```text
network=host
ipc=private
shm=64 GiB
privileged=false
group-add=video
ulimit memlock=-1
ulimit stack=64 MiB
devices=/dev/kfd,/dev/dri,/dev/infiniband
```

### 7.1 rollout 节点

```bash
docker rm -f qwen3-30b-rl 2>/dev/null || true
docker run -d --name qwen3-30b-rl --entrypoint /bin/bash \
  --network=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add=video \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/snx/Lumen-RL:/workspace/Lumen-RL \
  -v /home/snx/Lumen:/workspace/Lumen \
  -v /home/snx/ATOM-pr-1612:/workspace/ATOM \
  -v /home/snx/aiter:/workspace/aiter \
  -v /home/snx/flash-attention:/workspace/flash-attention \
  -v /home/snx/data_cached:/root/data_cached \
  -v /volumes/oss0/models:/root/models \
  -v /volumes/oss1:/volumes/oss1 \
  -v /mnt/raid0:/mnt/raid0 \
  rocm/atom-dev:vllm-latest -lc 'sleep infinity'
```

### 7.2 trainer 节点

```bash
docker rm -f qwen3-30b-rl 2>/dev/null || true
docker run -d --name qwen3-30b-rl --entrypoint /bin/bash \
  --network=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add=video \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/snx/Lumen-RL:/workspace/Lumen-RL \
  -v /home/snx/Lumen:/workspace/Lumen \
  -v /home/snx/ATOM-pr-1612:/workspace/ATOM \
  -v /home/snx/aiter:/workspace/aiter \
  -v /home/snx/flash-attention:/workspace/flash-attention \
  -v /home/snx/data_cached:/root/data_cached \
  -v /home/snx/wheels:/tmp/wheels \
  -v /volumes/oss0/models:/root/models \
  -v /volumes/oss1:/volumes/oss1 \
  -v /mnt/raid0:/mnt/raid0 \
  rocm/atom-dev:latest -lc 'sleep infinity'
```

关键约束：

- 必须映射整个 `/dev/infiniband`，只映射 `/dev/kfd` 与 `/dev/dri` 不足以使用 verbs/GDRDMA。
- rollout 节点必须使用包含当前 ROCm vLLM build 的镜像。
- trainer 节点不需要安装 vLLM。
- 两节点容器必须使用 `--network=host`，否则 Ray IP、RoCE IP 与 rendezvous 地址需要重新配置。

## 8. 安装和恢复依赖

当前两节点容器已经包含 §6 的实测版本。重建环境时优先使用本地 wheel cache，避免在线安装漂移。
trainer 节点已将 `/home/snx/wheels` 挂载到 `/tmp/wheels`；rollout 镜像中的 ROCm vLLM 必须保留，
不要被 PyPI wheel 覆盖。

trainer 节点按实际 wheel 文件名恢复 Megatron/flash-attn/AITER：

```bash
docker exec qwen3-30b-rl bash -lc '
set -e
P=/opt/venv/bin/pip

ls /tmp/wheels
$P install --no-deps /tmp/wheels/megatron_core-0.18.2-*.whl
$P install --no-deps /tmp/wheels/flash_attn-2.8.4-*.whl
$P install --no-deps /tmp/wheels/amd_aiter-0.1.0-*.whl

$P install --no-deps -e /workspace/Lumen
$P install --no-deps -e /workspace/Lumen-RL

$P install "ray[default]==2.56.1" \
  "transformers==5.2.0" "datasets==5.0.0" "accelerate==1.14.0" \
  "safetensors==0.8.0" "omegaconf==2.3.1" \
  "math_verify==0.3.3" "wandb==0.28.1"
'
```

rollout 节点只补齐公共依赖并安装 LumenRL；不要重装 torch、Triton 或 vLLM：

```bash
docker exec qwen3-30b-rl bash -lc '
set -e
P=/opt/venv/bin/pip
$P install --no-deps -e /workspace/Lumen
$P install --no-deps -e /workspace/Lumen-RL
$P install "ray[default]==2.56.1" \
  "transformers==5.2.0" "datasets==5.0.0" "accelerate==1.14.0" \
  "safetensors==0.8.0" "omegaconf==2.3.1" \
  "math_verify==0.3.3" "wandb==0.28.1"
'
```

### 8.1 flash-attn ROCm ABI

当前 cached wheel 的 Python wrapper 曾比 native extension 多传末尾 `num_splits`。`run_grpo.sh`
启动时会幂等删除这个不支持的参数。

验证 kernel：

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 qwen3-30b-rl /opt/venv/bin/python - <<'PY'
import torch
from flash_attn import flash_attn_varlen_func

q = torch.randn(12, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, k, v, cu, cu, 7, 7, causal=True)
out.float().sum().backward()
torch.cuda.synchronize()
print("flash_varlen_forward_backward_ok", tuple(out.shape))
PY
```

期望输出：

```text
flash_varlen_forward_backward_ok (12, 2, 128)
```

## 9. 模型和数据

容器内模型路径：

```text
/root/models/Qwen3-30B-A3B
```

已过滤训练/验证数据：

```text
/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
```

启动前验证：

```bash
docker exec qwen3-30b-rl bash -lc '
test -f /root/models/Qwen3-30B-A3B/config.json
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
'
```

## 10. RDMA 和网络预检

两个节点容器内执行：

```bash
docker exec qwen3-30b-rl bash -lc '
ls -l /dev/infiniband
test -e /dev/infiniband/uverbs0
test -d /sys/class/infiniband/mlx5_0
'
```

关键环境变量：

```bash
export NCCL_SOCKET_IFNAME=ens11np0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_DMABUF_ENABLE=0
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO
```

`gdr_mode: auto` 依赖 RCCL 自动探测。正式验证不能只看配置，必须从日志确认：

```text
NCCL INFO Using network IB
NCCL INFO ... via NET/IB/0/GDRDMA ...
```

若只有 socket/TCP 日志，则不算 RDMA 验证通过。

## 11. Ray 集群

先在 rollout 节点启动 head：

```bash
docker exec qwen3-30b-rl bash -lc '
ulimit -n 524288
/opt/venv/bin/ray stop --force || true
/opt/venv/bin/ray start --head \
  --node-ip-address=172.25.12.1 \
  --port=6379 \
  --num-gpus=8 \
  --num-cpus=64 \
  --dashboard-host=0.0.0.0
'
```

再在 trainer 节点加入：

```bash
docker exec qwen3-30b-rl bash -lc '
ulimit -n 524288
/opt/venv/bin/ray stop --force || true
/opt/venv/bin/ray start \
  --address=172.25.12.1:6379 \
  --node-ip-address=172.25.12.3 \
  --num-gpus=8 \
  --num-cpus=64
'
```

验证：

```bash
docker exec qwen3-30b-rl /opt/venv/bin/ray status
```

必须看到：

```text
Active: 2 nodes
Total: 16 GPU
```

## 12. 正式 YAML

唯一正式配置：

```text
examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml
```

关键配置：

```yaml
cluster:
  num_nodes: 2
  gpus_per_node: 8
  ray_address: auto

weight_sync:
  backend: rdma
  shared_folder: /volumes/oss1/lumenrl_weight_sync/qwen3-30b-a3b
  bucket_size_mb: 1024
  timeout_s: 600
  verify_full_load: true
  rdma:
    backend: rccl
    require_rdma: true
    hca: mlx5_0
    interface: ens11np0
    gid_index: 3
    gdr_mode: auto

policy:
  training_backend: megatron
  generation_backend: vllm
  max_total_sequence_length: 9216
  max_response_length: 8192
  train_global_batch_size: 256
  gen_batch_size: 32
  max_token_len_per_gpu: 16384
  learning_rate: 1.0e-6
  training:
    megatron_cfg:
      tensor_parallel_size: 4
      expert_parallel_size: 8
      expert_tensor_parallel_size: 1
      sequence_parallel: true
      use_distributed_optimizer: true
      grad_reduce_in_fp32: true
      attention_softmax_in_fp32: true
      attention_backend: flash
      use_packed_sequences: true
      moe_token_dispatcher_type: alltoall

  generation:
    vllm_cfg:
      tensor_parallel_size: 2
      kv_cache_dtype: auto
      gpu_memory_utilization: 0.70
      max_model_len: 9216
      dtype: bfloat16
      enable_sleep_mode: false

checkpointing:
  save_steps: 5
  save_total_limit: 3

num_training_steps: 200
```

`weight_sync.rdma.backend: rccl` 是配置语义；PyTorch API 仍使用 `backend="nccl"`，ROCm
运行时自动映射至 RCCL。

## 13. 启动正式训练

W&B key 放在 rollout 容器可读取的位置：

```text
/workspace/wandb.key
```

文件格式：

```text
WANDB_API_KEY=...
```

不要把 key 放进 `docker exec -e WANDB_API_KEY=...` 的命令参数，否则会出现在 `ps` 输出。

当前 v2 启动方式：

```bash
docker exec qwen3-30b-rl bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/mnt/raid0
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=longrun
export STEPS=200
export BACKEND=vllm
export CONFIG_OVERRIDE=examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml
export LUMENRL_KEEP_RAY_CLUSTER=1
export RUN_ID=qwen3-30b-a3b-rdma-longrun-200-v2
export LOG=/mnt/raid0/lumenrl_logs/qwen3-30b-a3b-rdma-longrun-200-v2.log
export CKPT_DIR=/mnt/raid0/ckpts/qwen3-30b-a3b-rdma-longrun-200-v2
export WANDB_RUN_NAME=qwen3-30b-a3b-rdma-longrun-200-v2
export RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma

bash /workspace/Lumen-RL/examples/GRPO/run_grpo.sh
'
```

注意：

- `BACKEND=vllm` 与 `CONFIG_OVERRIDE=...vllm...yaml` 必须同时明确设置。
- `run_grpo.sh` 的历史默认值可能仍是 `atom`，不要依赖默认选择。
- 首次从 step 0 启动必须使用 `RESUME_OVERRIDE=false`。
- 从完整 v2 checkpoint 恢复才使用 `RESUME_OVERRIDE=true`。

## 14. 启动验证

日志中依次确认：

```text
Created 1 placement groups for pool 'rollout' ... node_ip=172.25.12.1
Created 1 placement groups for pool 'actor' ... node_ip=172.25.12.3
RDMA preflight: ...
RDMA weight group ready: ... world=9
NCCL INFO Using network IB
NCCL INFO ... NET/IB/0/GDRDMA ...
```

每步确认：

```text
RDMA weight sync committed:
callbacks: step=N
```

重点指标：

```text
rollout_corr/kl
rollout_corr/rollout_is_eff_sample_size
actor/loss
actor/grad_norm
actor/entropy
weight_sync/gbps
timing/weight_sync_rdma_s
```

任一核心指标出现 `nan` 必须立即停止，不得继续写下一个 checkpoint。

## 15. Checkpoint 验证

每 5 步保存一次；actor 节点会在保存前清理旧 shard，避免 controller 与 actor 使用同名但
非共享本地路径时只清理 controller metadata。

检查 step 5：

```bash
P=/mnt/raid0/ckpts/qwen3-30b-a3b-rdma-longrun-200-v2/global_step_5/actor
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

完整 checkpoint 必须满足：

- 8 个 model shard。
- 8 个 optimizer metadata shard。
- 8 个 extra-state shard。
- 8 个大体积 optimizer parameter-state shard。
- controller 侧存在 `checkpoint_N.pt` 与 `latest_checkpointed_iteration.txt`。

检查文件数量：

```bash
docker exec qwen3-30b-rl /opt/venv/bin/python - <<'PY'
from pathlib import Path

p = Path("/mnt/raid0/ckpts/qwen3-30b-a3b-rdma-longrun-200-v2/global_step_5/actor")
for pattern in (
    "model_world_size_8_rank_*.pt",
    "optim_world_size_8_rank_*.pt",
    "optim_parameter_state_world_size_8_rank_*.pt",
    "extra_state_world_size_8_rank_*.pt",
):
    files = list(p.glob(pattern))
    print(pattern, len(files), sum(x.stat().st_size for x in files))
    assert len(files) == 8
PY
```

## 16. 监控

### 16.1 训练状态

```bash
LOG=/mnt/raid0/lumenrl_logs/qwen3-30b-a3b-rdma-longrun-200-v2.log

grep -a "lumenrl.trainer.callbacks: step=" "$LOG" | tail -1
grep -a "RDMA weight sync committed" "$LOG" | tail -1
grep -aiE "Training failed|Traceback|OutOfMemory|NCCL.*timeout|SIGABRT|=nan" "$LOG" | tail
```

### 16.2 进程和 Ray

```bash
docker exec qwen3-30b-rl pgrep -af '[l]umenrl.trainer.main'
docker exec qwen3-30b-rl /opt/venv/bin/ray status
```

### 16.3 磁盘

trainer 节点执行：

```bash
df -h /mnt/raid0
du -sh /mnt/raid0/ckpts/qwen3-30b-a3b-rdma-longrun-200-v2/global_step_*
```

单个完整 checkpoint 当前约 402 GiB。`save_total_limit=3` 需要约 1.2 TiB，加上模型、日志和
Ray 临时文件必须预留安全余量。

### 16.4 W&B

```text
https://wandb.ai/danyzhan-amd/LumenRL/runs/jq122ogj
```

在线 history 的最新 global step 应与本地 callback 日志基本一致。若 run state 为 `crashed` 或
线上 step 长时间不增长，先检查本地 W&B 日志与网络，不要仅根据网页判断训练进程状态。

## 17. 停止和恢复

停止：

```bash
docker exec qwen3-30b-rl pkill -TERM -f '[l]umenrl.trainer.main' || true
```

恢复前必须先确认最新 checkpoint 含大体积
`optim_parameter_state_world_size_8_rank_*.pt`。然后使用相同 `CKPT_DIR`，设置：

```bash
export RESUME_OVERRIDE=true
```

日志必须出现：

```text
Resuming Ray actor checkpoint from ... global_step_N/actor
Ray resume complete. Next training log will be global_step=N+1
```

恢复后至少观察两个完整 step：

- 第一个 step 验证 checkpoint model 可加载。
- 第二个 step 验证 optimizer update 后权重仍为有限值。

只有连续两步 KL、ESS、loss、grad norm、entropy 全部有限，才视为恢复成功。

## 18. 排障

### 18.1 `ModuleNotFoundError: vllm`

原因：Ray 把 rollout placement group 调度到 trainer 节点。

处理：

- 确认 YAML 的 rollout `topology_tags.node_ip=172.25.12.1`。
- 确认 actor `topology_tags.node_ip=172.25.12.3`。
- 日志必须打印两个固定 node IP。

### 18.2 RDMA 退化成 TCP

检查：

```bash
ls -l /dev/infiniband
test -d /sys/class/infiniband/mlx5_0
```

确认 `NCCL_SOCKET_IFNAME=ens11np0`、`NCCL_IB_HCA=mlx5_0`、`NCCL_IB_GID_INDEX=3`。
没有 `NET/IB/.../GDRDMA` 日志就不能宣称 GPU Direct RDMA 已启用。

### 18.3 flash-attn `varlen_fwd()` 参数不兼容

原因：Python wrapper 带 `num_splits`，native ROCm extension 是旧 21 参数 ABI。

处理：使用当前 `run_grpo.sh` 的幂等 ABI patch，并运行 §8.1 kernel 测试。

### 18.4 checkpoint 写满磁盘

症状：

```text
RuntimeError: basic_ios::clear: iostream error
No space left on device
```

处理：

- 删除不完整的当前 step 目录。
- 保留至少一个已验证完整 checkpoint。
- 确认 actor-node `prune_checkpoints` 在保存前执行。
- 不要保留只有 model、没有 optimizer parameter state 的历史 checkpoint。

### 18.5 恢复后第二步 NaN

通常表示 optimizer 没有完整恢复，或 FP32 main parameters 未与 model 同步。

检查：

- `optim_parameter_state_world_size_*.pt` 是否存在且为几十 GiB。
- 是否调用 `load_parameter_state()`。
- 是否调用 `reload_model_params()`。
- 不要用 2 KiB 的 optimizer metadata 文件代替 distributed parameter state。

### 18.6 W&B 没更新

检查：

- `/workspace/wandb.key` 是否存在。
- 本地 callback 是否继续输出 step。
- W&B run ID 是否与当前 v2 一致。
- resume 时不要让 global step 回退，否则 W&B 会拒绝旧 step。

### 18.7 `Too many open files` / Ray socket EOF

容器启动和训练脚本均设置：

```bash
ulimit -n 524288
```

必要时停止两节点 Ray 后重新按 §11 顺序启动。

## 19. 关键源码

```text
lumenrl/core/config.py
lumenrl/utils/independent_process_group.py
lumenrl/engine/inference/rdma_weight_transfer.py
lumenrl/engine/inference/vllm_colocate_worker_ext.py
lumenrl/engine/inference/vllm_ray_server.py
lumenrl/workers/actor_worker.py
lumenrl/controller/ray_worker_group.py
lumenrl/trainer/rl_trainer.py
lumenrl/trainer/callbacks.py
lumenrl/engine/training/megatron_engine.py
examples/GRPO/run_grpo.sh
examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml
```

## 20. 参考

- LumenRL：`https://github.com/ZhangDanyang-AMD/Lumen-RL.git`
- LumenRL branch：`origin/dev/moe-grpo`
- LMSYS MILES RL on AMD：
  `https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/`
- MILES：`https://github.com/radixark/miles`
- Qwen3-30B-A3B：`https://huggingface.co/Qwen/Qwen3-30B-A3B`

## 附录 A：切换到 ATOM rollout

本附录仅说明可选的 ATOM 流程。正文中的正式 200-step 任务仍使用 vLLM + RCCL/RoCE RDMA。

### A.1 支持范围和限制

ATOM 与当前 RDMA backend 不能直接组合：

- `policy.generation_backend: atom` 会创建 `ATOMReplicaManager`。
- 当前 ATOM manager 没有初始化 9-rank RCCL weight group，也没有 RDMA receiver。
- 因此 ATOM 配置不能设置 `weight_sync.backend: rdma`。
- 两节点 ATOM 必须使用 `shared_folder`；同节点可使用 `auto`。

`weight_sync.backend: auto` 的实际选择规则：

- 同节点、ATOM TP=1、8 replicas 对应 8 actor workers：使用 ZMQ CUDA-IPC。
- 同节点、ATOM TP=2、4 replicas 少于 8 actor workers：自动回退到 safetensors。
- ATOM 与 trainer 分布在不同节点：自动回退到 safetensors。

Qwen3-30B-A3B 当前推荐 ATOM TP=2，因此实际稳定路径是：

```text
Megatron BF16 shard
  → actor 聚合并转换为 HF 名称
  → safetensors 导出
  → ATOM TP=2 × 4 replicas reload
```

不要把该路径标记为 RDMA，也不要期待日志出现 `NET/IB/.../GDRDMA`。

### A.2 部署形态

推荐先做单节点 colocated smoke：

```text
单节点 8 × MI308X
├── Megatron actor workers：8
├── ATOM replicas：4
├── 每个 ATOM replica：TP=2
├── ATOM 权重：BF16
├── KV cache：FP8
└── sleep mode：level 2
```

同一批 GPU 在 rollout 和 training 间切换：

1. ATOM wake weights/KV cache。
2. 生成 response，并记录 MoE router logits。
3. ATOM sleep level 2，释放可回收显存。
4. Megatron 使用 R3 distribution replay 计算 log-prob 并更新参数。
5. actor 导出 HF safetensors。
6. ATOM wake weights，reload 新参数，再恢复 KV cache。

单节点同时容纳 ATOM CUDA graph、Megatron 参数和 distributed optimizer，显存压力明显高于正文的
两节点方案。出现 OOM 时优先降低 response length、batch size、ATOM
`gpu_memory_utilization`，不要关闭 checkpoint 的 optimizer parameter-state 保存。

### A.3 ATOM 源码和环境

当前容器挂载：

```text
/home/snx/ATOM-pr-1612 → /workspace/ATOM
```

启动脚本会把 `/workspace/ATOM` 放到 `PYTHONPATH`。验证实际导入位置：

```bash
docker exec qwen3-30b-rl bash -lc '
PYTHONPATH=/workspace/ATOM:/workspace/Lumen-RL \
  /opt/venv/bin/python - <<PY
import atom
from atom.rollout.async_engine import AsyncLLMEngine
print("atom module:", atom.__file__)
print("AsyncLLMEngine:", AsyncLLMEngine)
PY
'
```

期望 `atom.__file__` 位于 `/workspace/ATOM/atom/`，而不是意外导入其他 site-packages 版本。

ATOM TP>1 时保留以下环境：

```bash
export LUMENRL_DISABLE_CUSTOM_AR=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
export ATOM_TORCH_COMPILE_CACHE_ROOT=/tmp/atom_torch_compile_cache
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export USE_ROCM_AITER_ROPE_BACKEND=0
```

`NoCustomARModelRunner` 会关闭 ATOM 的 HIP IPC custom all-reduce，改用 RCCL collective，避免
`ca_comm is None` 或 `hipIpcOpenMemHandle` 崩溃。这里的 RCCL 只用于 ATOM TP collective，
不是正文所述的跨节点 RDMA 权重广播。

### A.4 ATOM YAML

基准文件：

```text
examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml
```

关键配置：

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

weight_sync:
  backend: auto
  shared_folder: /mnt/raid0/lumenrl_weight_sync/atom-qwen3-30b-a3b
  bucket_size_mb: 1024
  timeout_s: 600

controller:
  ray:
    enabled: true
    fuse_actor_ref: false
    actor:
      num_workers: 8

policy:
  training_backend: megatron
  generation_backend: atom
  training:
    megatron_cfg:
      tensor_parallel_size: 4
      expert_parallel_size: 8
      expert_tensor_parallel_size: 1
      use_distributed_optimizer: true
      sequence_parallel: true
      attention_backend: flash
  generation:
    atom_cfg:
      tensor_parallel_size: 2
      expert_parallel_size: 1
      kv_cache_dtype: fp8
      engine_kwargs:
        enforce_eager: false
        compilation_config:
          level: 3
    vllm_cfg:
      gpu_memory_utilization: 0.60
      max_model_len: 9216
      dtype: bfloat16
      quantization: ""
      enable_sleep_mode: true
      sleep_level: 2

moe:
  r3:
    enabled: true
    record_router_logits: true
    replay_mode: distribution
```

注意：

- `atom_cfg` 控制 ATOM engine；当前实现仍从 `vllm_cfg` 读取部分通用 generation/sleep 参数。
- `quantization: ""` 表示 BF16 权重，不启用 `per_block_fp8`。
- FP8 只用于 KV cache。Qwen3-30B-A3B 的 MoE intermediate size 为 768，在线
  `per_block_fp8` 已观测到输出乱码。
- `grpo_qwen3_30b_a3b_atom_ep8_smoke.yaml` 当前内容实际是
  `generation_backend: vllm`，不能作为 ATOM smoke 直接使用；应复制 longrun 文件并缩小参数。

ATOM smoke 建议覆盖：

```yaml
policy:
  max_total_sequence_length: 1088
  max_response_length: 1024
  train_global_batch_size: 64
  gen_batch_size: 8
  max_token_len_per_gpu: 4096

checkpointing:
  save_steps: 9999
  resume: false

logger:
  wandb_enabled: false

num_training_steps: 3
```

### A.5 启动

先准备本地同步目录：

```bash
docker exec qwen3-30b-rl bash -lc '
mkdir -p /mnt/raid0/lumenrl_weight_sync/atom-qwen3-30b-a3b
rm -f /mnt/raid0/lumenrl_weight_sync/atom-qwen3-30b-a3b/*.tmp
'
```

使用单独制作的 ATOM smoke YAML 启动：

```bash
docker exec qwen3-30b-rl bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/mnt/raid0
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=smoke
export STEPS=3
export BACKEND=atom
export CONFIG_OVERRIDE=examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_smoke_local.yaml
export RUN_ID=qwen3-30b-a3b-atom-smoke3
export LOG=/mnt/raid0/lumenrl_logs/qwen3-30b-a3b-atom-smoke3.log
export CKPT_DIR=/mnt/raid0/ckpts/qwen3-30b-a3b-atom-smoke3
export WEIGHT_SYNC_BACKEND=auto
export RESUME_OVERRIDE=false

bash /workspace/Lumen-RL/examples/GRPO/run_grpo.sh
'
```

`BACKEND=atom` 不足以修正一个内容仍为 vLLM 的 `CONFIG_OVERRIDE`；必须确认最终 YAML 中明确写有：

```yaml
policy:
  generation_backend: atom
```

### A.6 ATOM smoke 验证

启动日志应出现：

```text
ATOMReplicaManager: launched 4 colocated rollout replicas (atom_tp=2, workers=8)
Ray ATOM rollout ready: 4 colocated replicas
R3Manager: router recording ENABLED
```

TP=2×4 时权重同步应出现 safetensors export/reload 日志，而不是 RDMA 日志。每步必须确认：

```text
callbacks: step=N
```

并检查：

```bash
LOG=/mnt/raid0/lumenrl_logs/qwen3-30b-a3b-atom-smoke3.log
grep -aE "ATOMReplicaManager|Ray ATOM rollout ready|reloaded weights|callbacks: step=" "$LOG" | tail -30
grep -aiE "Traceback|OutOfMemory|illegal memory access|ca_comm|=nan" "$LOG" | tail
```

通过标准：

- 连续完成至少 3 步。
- reward、loss、entropy、grad norm、KL 和 ESS 全部有限。
- R3 router-logit coverage 完整。
- 每轮训练后 ATOM 成功 reload 权重。
- 无乱码、非法显存访问、OOM 或 stale CUDA/HIP IPC handle。

ATOM smoke 通过后，才可切换到 `grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml`。正式长跑仍应采用
§15 的完整 distributed optimizer checkpoint 规则；不要恢复 v1 的不完整 checkpoint。

---

最后更新：2026-07-28
