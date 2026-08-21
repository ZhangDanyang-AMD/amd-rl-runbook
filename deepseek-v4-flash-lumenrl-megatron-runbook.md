# DeepSeek-V4-Flash 三节点 RL 部署 Runbook

> Megatron 训练（TP4/PP4/EP4）+ vLLM FP8 rollout + RCCL/RoCE GPU Direct RDMA 权重同步
> 硬件：3 节点 — 2 × 8 MI308X 训练 + 1 × 8 MI300X 推理
> LumenRL：`origin/dev/dsv4-grpo`
> 超参对齐 LMSYS MILES DSV4：https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4/

## 0. 2026-08-10 当前已验证状态

- 训练节点为 p19-29 + **p22-05**，p20-29 不属于当前集群；p21-29 继续作为 rollout/Ray head。
- v35 已完成一个完整的 32×8 GRPO step：256 条 rollout、old log-probs、forward/backward、
  streamed Adam、568 GB 级权重同步和 30 条 AIME eval 均执行完毕。
- Megatron 初始化 OOM 已通过 `use_precision_aware_optimizer: true` 解决。该模式避免在 GPU
  上一次性克隆完整 FP32 master weights，再配合 full optimizer CPU offload、BF16 moments 和
  256 MiB streamed-Adam staging，将训练峰值显存控制在约 75.8 GB/GPU。
- rollout GPU memory fault 已消失。当前通过配置为
  `enable_chunked_prefill: true`、`enable_prefix_caching: false`；
  **没有证据表明开启 prefix cache 修复了问题**，正式配置仍应关闭 prefix cache，除非重新完成 A/B。
- 独立权重同步 A/B（不执行 RL、backward 或 optimizer step）确认：
  trainer-side FP8 pre-quantization 会使同步后的 vLLM 输出立即退化为重复的 `Aime`、`0`、`n`
  等模式；inference-side online FP8 同步后输出保持连贯。因此正式训练必须设置
  `fp8_quantization_location: inference`。
- v38 200-step 长跑已启动：rollout log-probs、token IS、batch normalization、R3 全部开启；
  checkpoint 每 5 步保存到共享 NFS，最多保留 2 份。

## 1. 目标架构

### 1.1 节点角色

| 节点 | 主机名 | 角色 | GPU |
|------|--------|------|-----|
| 训练-0 | `banff-ccs-aus-p19-29.cs-aus.dcgpu` (`10.194.132.110`) | Megatron ranks 0–7 | 8 × MI308X |
| 训练-1 | `banff-ccs-aus-p22-05.cs-aus.dcgpu` (`10.194.132.65`) | Megatron ranks 8–15 | 8 × MI308X |
| 推理 | `banff-ccs-aus-p21-29.cs-aus.dcgpu` (`10.194.132.76`) | vLLM rollout + Ray head | 8 × MI300X |

### 1.2 训练配置

- 16 个 Megatron actor worker，分布在 2 个训练节点
- TP=4, PP=4, EP=4, ETP=1
- Pipeline 层分布：11 + 11 + 11 + 10（共 43 层）
- Optimizer：AdamW + full CPU offload + precision-aware optimizer + BF16 moments
- NUMA affinity：自动将每个 Actor 绑定到其 GPU 所在 NUMA node 的 CPU 集合
- 精度：BF16 compute；precision-aware path 避免 GPU 侧完整 FP32 master-weight clone
- Megatron `TransformerConfig.fp8=None`，DSV4 activation QAT 不执行；FP8 仅属于 rollout
  和 inference-side online requantization，不能将两者混为 FP8 actor training
- 初始权重：`/dev/shm/models/DeepSeek-V4-Flash-BF16`
  （`RedHatAI/DeepSeek-V4-Flash-BF16`，原始 BF16 HF safetensors，约 568 GB）
- Engine：`megatron_lumen_dsv4`（LumenRL 的 `MegatronLumenDSV4Engine`）
- Megatron：ROCm/Megatron-LM `rocm_dev` + DSV4 patch
- Kernel：Lumen DSV4 spec + AIter mHC + TileLang indexer + Triton sparse MLA。
  TileLang sparse MLA 已接入为 MI355X/MILES 对照入口，但当前 MI308X 环境 backward
  无法编译（`tirx.exp2`），因此 BF16 actor 不能将其设为默认

### 1.3 推理配置

- 1 个 vLLM 实例，TP=8（全部 8 个 GPU）
- 模型：`DeepSeek-V4-Flash-BF16`（~568 GB）
- 在线 FP8 量化：`fp8_per_block`（BF16 权重加载后在 GPU 上量化为 FP8 e4m3）
- KV cache：`fp8_e4m3`
- `enforce_eager: true`（不使用 cudagraph）
- MoE backend：`triton`
- Chunked prefill：`enable_chunked_prefill: true`
- Prefix cache：`enable_prefix_caching: false`（当前 32×8 rollout 的已验证配置）

### 1.4 权重同步

- RDMA weight sync：独立 `torch.distributed` process group
  - world=9：1 sender（Megatron actor rank 0）+ 8 receivers（vLLM TP=8 workers）
  - Backend: `nccl`（ROCm 下由 RCCL 实现）
  - 每步同步 ~568 GB BF16 参数
  - Actor rank 0 集体执行 TP/EP/PP all-gather 后 broadcast 给 vLLM
- 训练节点间通信也使用 RDMA（PP pipeline P2P + gradient reduction）

### 1.5 RL 超参（MILES 对齐）

| 参数 | 值 |
|------|-----|
| 算法 | GRPO |
| KL coeff | 0.0 |
| Clip ratio | 0.2 / 0.28（asymmetric） |
| LR | 1e-6, constant |
| Adam betas | (0.9, 0.98) |
| Weight decay | 0.1 |
| GBS | 256（32 prompts × 8 generations） |
| Max response | 4096 |
| Temperature | 0.8 |
| R3 | enabled, hard_assignment（LumenRL RouterReplay 实现） |
| Num rollout | 200 |
| 数据集 | `zhuzilin/dapo-math-17k`（JSONL） |
| 验证集 | `zhuzilin/aime-2024`（JSONL） |

## 2. 存储策略

### 2.1 快盘 vs NFS

| 用途 | 路径 | 存储类型 | 原因 |
|------|------|----------|------|
| 模型权重（HF） | `/dev/shm/models/DeepSeek-V4-Flash-BF16` | tmpfs (内存) | 16 worker 并发读，NFS 太慢 |
| 数据集 | `/dev/shm/datasets/` | tmpfs | 小文件，快速访问 |
| Checkpoint | `/nfs/data/danyzhan/lumenrl_ckpts/dsv4-flash-200step-is-r3/` | NFS | 每 5 步保存，最多保留 2 份 |
| 日志 | `/nfs/data/leiwu/logs/` 或宿主机挂载 | NFS/本地 | 需要持久化 |

### 2.2 从 Hugging Face 预加载模型和数据集

三台节点都直接从 Hugging Face 下载相同资产。模型必须使用
`RedHatAI/DeepSeek-V4-Flash-BF16`，不要下载官方 FP4/FP8 mixed 仓库。

```bash
python3 -m venv ~/.cache/dsv4-hf-venv
~/.cache/dsv4-hf-venv/bin/pip install 'huggingface_hub[hf_xet]'

HF_XET_HIGH_PERFORMANCE=1 ~/.cache/dsv4-hf-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "RedHatAI/DeepSeek-V4-Flash-BF16",
    local_dir="/dev/shm/models/DeepSeek-V4-Flash-BF16",
    max_workers=16,
)
snapshot_download(
    "zhuzilin/dapo-math-17k",
    repo_type="dataset",
    local_dir="/dev/shm/datasets/dapo-math-17k",
    max_workers=8,
)
snapshot_download(
    "zhuzilin/aime-2024",
    repo_type="dataset",
    local_dir="/dev/shm/datasets/aime-2024",
    max_workers=8,
)
PY
```

**注意**：`/dev/shm` 是 tmpfs（内存），节点重启后丢失。`--ipc=host` 的 Docker 容器共享宿主机 `/dev/shm`。

## 3. Docker 镜像

### 3.1 最终镜像和节点分配

Docker Hub 上的 260813 tag 是可直接运行的最终环境，已经包含匹配的
`Lumen-RL`、`Lumen`、AIter、Megatron、vLLM 和 ROCm 依赖。部署节点只负责
拉取和启动，不在服务器上修改或重新安装代码。

| 角色 | Docker Hub 最终镜像 | 节点 |
|------|----------------------|------|
| trainer | `zhangdanyangamd/lumen-rl:dsv4-300x-train260813` | f293、f294 |
| rollout | `zhangdanyangamd/lumen-rl:dsv4-300x-rollout260813` | f2b9 |

### 3.2 从 Docker Hub 拉取镜像

```bash
# 需要私有仓库权限时先登录；不要把 token 写进脚本或 runbook
docker login

# f293 / f294
docker pull zhangdanyangamd/lumen-rl:dsv4-300x-train260813

# f2b9（该节点当前需要 sudo 访问 Docker daemon）
sudo docker pull zhangdanyangamd/lumen-rl:dsv4-300x-rollout260813
```

拉取后核对镜像：

```bash
docker image inspect zhangdanyangamd/lumen-rl:dsv4-300x-train260813 \
  --format '{{.Id}} {{.RepoDigests}}'
```

### 3.3 从最终镜像直接启动容器

训练节点 f293、f294：

```bash
docker rm -f dsv4-train 2>/dev/null || true
docker run -d --name dsv4-train \
  --privileged --network=host --ipc=host --shm-size=128g \
  -v /dev/infiniband:/dev/infiniband \
  -v /nfs/data:/nfs/data \
  -e GLOO_SOCKET_IFNAME=ens14np0 \
  -e NCCL_SOCKET_IFNAME=ens14np0 \
  --entrypoint /usr/bin/sleep \
  zhangdanyangamd/lumen-rl:dsv4-300x-train260813 infinity
```

rollout 节点 f2b9：

```bash
sudo docker rm -f dsv4-rollout 2>/dev/null || true
sudo docker run -d --name dsv4-rollout \
  --privileged --network=host --ipc=host --shm-size=128g \
  -v /dev/infiniband:/dev/infiniband \
  -v /nfs/data:/nfs/data \
  -e GLOO_SOCKET_IFNAME=ens14np0 \
  -e NCCL_SOCKET_IFNAME=ens14np0 \
  --entrypoint /usr/bin/sleep \
  zhangdanyangamd/lumen-rl:dsv4-300x-rollout260813 infinity
```

**关键点**：
- `--privileged` 是 InfiniBand/RDMA 正常工作的必要条件
- 显式覆盖基础镜像 ENTRYPOINT，避免容器启动后立即退出
- `--ipc=host` 共享宿主机 `/dev/shm`（可访问预加载的模型权重）
- 不挂载宿主机 Lumen-RL、Lumen 或 AIter；三仓代码已固化在镜像中
- 只挂载需要持久化的 NFS checkpoint/log 路径和 RDMA 设备

启动后检查：

```bash
docker ps --filter name=dsv4-train
docker exec dsv4-train python3 -c \
  'import aiter,lumen,lumenrl; print(aiter.__file__, lumen.__file__, lumenrl.__file__)'
docker exec dsv4-train rocm-smi --showproductname
test -f /dev/shm/models/DeepSeek-V4-Flash-BF16/config.json
```

## 4. Megatron DSV4 Patch

LumenRL 使用 ROCm/Megatron-LM (`rocm_dev`) 而非 Miles fork。ROCm Megatron 没有 DSV4
专属字段，需要用 patch 脚本补齐。

Patch 文件：`examples/GRPO/dsv4/patch_rocm_megatron_dsv4.py`

**改动内容**：
1. `TransformerConfig` 加 `dsv4` 到 `experimental_attention_variant` Literal，加 `dsv4_*` 字段
2. `TransformerBlock` 加 `HCHeadParams`（Hyper-Connection head 参数，最后一个 PP rank）
3. `TransformerLayer` 加 per-layer HC 参数（`hc_attn_fn/base/scale`, `hc_ffn_fn/base/scale`）
4. `experimental_attention_variant_module_specs.py` 加 `dsv4` 分支占位
5. `tensor_parallel/layers.py` 加 `condition_init_method`（Lumen 的 LumenColumnParallelLinear 需要）

**不依赖 Miles**：
- DSV4 的 attention/compressor/indexer/HC 层由 **Lumen** 提供
- R3 路由重放使用 **LumenRL** 的 `RouterReplay` 实现
- MoE kernel 使用 **TileKernels**（jayzlee147 fork）和 **tilelang**（PyPI）
- 零 Miles 依赖

## 5. HF Config 字段映射

DSV4 的 HuggingFace `config.json` 字段名和 Megatron/Lumen 不同：

| HF config.json | Engine 读取 | 说明 |
|---|---|---|
| `head_dim` | kv_lora_rank | MLA head dim (512) |
| `qk_rope_head_dim` | qk_pos_emb_head_dim | RoPE head dim (64) |
| `n_routed_experts` | num_experts | 256 |
| `compress_ratios` | dsv4_compress_ratios | per-layer compress ratio |
| `compress_rope_theta` | dsv4_compress_rope_theta | 160000 |
| `hc_mult` | dsv4_hc_mult | 4 |
| `o_groups` | dsv4_o_groups | 8 |
| `o_lora_rank` | dsv4_o_lora_rank | 1024 |
| `num_hash_layers` | dsv4_n_hash_layers | 3 |
| `sliding_window` | dsv4_window_size | 128 |
| `index_n_heads` | dsa_indexer_n_heads | 64 |
| `index_head_dim` | dsa_indexer_head_dim | 128 |
| `index_topk` | dsa_indexer_topk | 512 |

## 6. Ray 集群

### 6.1 启动

```bash
# Ray head（推理节点 p21-29）
ROLLOUT_IP=10.194.132.76
docker exec dsv4-rl ray start --head \
  --node-ip-address=$ROLLOUT_IP --port=6379 \
  --num-gpus=8 --num-cpus=224 \
  --object-store-memory=200000000000 \
  --min-worker-port=10002 --max-worker-port=19999 \
  --dashboard-host=0.0.0.0

# 两个训练节点分别加入（TRAIN_IP 为 10.194.132.110 / 10.194.132.65）
docker exec dsv4-rl ray start \
  --address=$ROLLOUT_IP:6379 --node-ip-address=$TRAIN_IP \
  --num-gpus=8 --num-cpus=224 \
  --object-store-memory=200000000000 \
  --min-worker-port=10002 --max-worker-port=19999
```

验证：`ray status` 应显示 3 nodes, 24 GPU。

### 6.2 多节点训练 placement

Actor 使用 `process_on_nodes: [8, 8]`（无 topology_tags），Ray STRICT_PACK 自动分配到两个训练节点。
Rollout 使用 `topology_tags: {node_ip: <推理节点IP>}` 固定到推理节点。

### 6.3 失败重试前清理

训练失败后不能只确认 Ray Actor 已消失；ROCm context 可能仍持有 90% 左右显存，下一次
optimizer 初始化会在最后几十 MiB 分配时 OOM。每次失败重试前执行：

```bash
# 三台机器都执行
docker exec dsv4-rl ray stop --force
docker exec dsv4-rl rocm-smi --showmemuse
```

必须确认两个训练节点所有 GPU 的 VRAM 使用率回到 0%，再按 6.1 重建 Ray 集群。不要用降低
batch size 掩盖残留 context 导致的 OOM。

### 6.4 NUMA-aware Actor 初始化

optimizer CPU offload 会为每个 rank 创建约百 GB 的 FP32 master params/Adam state。若 Actor
运行在远离其 GPU 的 NUMA node 上，会出现 `migration_entry_wait_on_locked`，导致 p20
ranks 8–15 初始化明显慢于 p19 ranks 0–7。

Smoke 配置必须启用：

```yaml
policy:
  training:
    megatron_cfg:
      numa_affinity: true
```

实现位于 `lumenrl/workers/numa_affinity.py`。Actor 启动时通过
`rocm-smi --showtoponuma --json` 自动查找 GPU NUMA node，再读取
`/sys/devices/system/node/node<N>/cpulist` 并调用 `os.sched_setaffinity`。
探测失败只告警并继续，不阻断训练。

本批节点的预期绑定：

| 物理 GPU | NUMA node | CPU affinity |
|----------|-----------|--------------|
| 0–3 | 0 | `0-55,112-167` |
| 4–7 | 1 | `56-111,168-223` |

验证运行时绑定：

```bash
# 对每个 LumenActorWorker PID 检查
docker exec dsv4-rl taskset -pc <PID>
```

## 7. 数据集

使用 Miles 同款数据集（JSONL 格式）：

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('zhuzilin/dapo-math-17k', repo_type='dataset', local_dir='/dev/shm/datasets/dapo-math-17k')
snapshot_download('zhuzilin/aime-2024', repo_type='dataset', local_dir='/dev/shm/datasets/aime-2024')
"
```

## 8. 启动训练

### 8.1 生成部署 YAML

```bash
sed "s/REPLACE_ME/${ROLLOUT_NODE_IP}/g" \
  examples/GRPO/configs/grpo_dsv4_flash_vllm_smoke.yaml \
  > /runtime/configs/dsv4-smoke.yaml
```

**关键路径替换**：
- `model_name` → `/dev/shm/models/DeepSeek-V4-Flash-BF16`
- `dataset` → `/dev/shm/datasets/dapo-math-17k/dapo-math-17k.jsonl`
- `checkpoint_dir` → `/nfs/data/leiwu/ckpts/dsv4-flash-lumenrl/smoke`
- `cluster.topology_tags.node_ip` → `10.194.132.76`
- `weight_sync.rdma.interface` → `ens14np0`（不能保留旧节点的 `ens11np0`）
- `policy.training.megatron_cfg.numa_affinity` → `true`
- `policy.training.megatron_cfg.use_precision_aware_optimizer` → `true`
- `policy.generation.vllm_cfg.disable_custom_all_reduce` → `true`
- `policy.generation.vllm_cfg.enable_chunked_prefill` → `true`
- `policy.generation.vllm_cfg.enable_prefix_caching` → `false`
- `weight_sync.fp8_quantization_location` → `inference`

MI300X 上的 vLLM TP=8 rollout 必须禁用 AITER custom all-reduce，改用 RCCL：

```yaml
policy:
  generation:
    vllm_cfg:
      tensor_parallel_size: 8
      disable_custom_all_reduce: true
```

对应配置需要同时存在于 `VLLMConfig` schema，并由
`RLTrainer._setup_ray_vllm_rollout()` 传给 vLLM `AsyncEngineArgs`。

### 8.2 训练前测试清单（必须全部通过）

所有测试必须在 GPU 空闲、三节点 Ray 集群已启动且运行时代码已同步后执行。任一项失败都不能启动正式训练。

#### A. 静态检查与核心单测

在 LumenRL 源码目录执行：

```bash
git diff --check
python -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_fp8_weight_quantizer.py \
  tests/unit/test_rdma_weight_transfer.py \
  tests/unit/test_vllm_colocate_worker_ext.py \
  tests/unit/test_vllm_ray_server.py \
  tests/unit/test_patch_rocm_megatron_dsv4.py \
  tests/unit/engine/test_megatron_lumen_dsv4_engine.py \
  tests/unit/engine/test_megatron_r3_pp.py \
  -q
```

检查范围包括 YAML schema、trainer-side FP8 block quantization、RDMA protocol/capability
handshake、vLLM pre-quantized reload metadata、Megatron DSV4 capability gate 和 R3 PP replay。

#### B. 三节点代码与集群一致性

```bash
# Ray head 容器内
export RAY_ADDRESS=10.194.132.76:6379
ray status
python3 deploy_ray_runtime_files.py
```

验收：
- Ray 显示 3 个 alive nodes、24 GPUs；
- 部署脚本返回的每个文件在三节点 SHA256 完全相同；
- `lumenrl`、Megatron、vLLM 实际 import path 指向预期运行目录；
- 没有遗留的 `lumenrl.trainer.main`、`VLLMRayServer` 或 `EngineCore`；
- 两个训练节点 `rocm-smi --showmemuse` 显示 GPU VRAM 已释放。

#### C. 跨节点 Gloo/RCCL-RDMA

在三节点均空闲时从 Ray head 执行：

```bash
export RAY_ADDRESS=10.194.132.76:6379
python3 diagnose_ray_collectives.py
```

验收：
- Gloo 3-rank `all_reduce` 成功；
- RCCL 3-rank GPU `all_reduce` 成功；
- 所有 rank 使用 `ens14np0`、`mlx5_0`、GID 3，不能出现 Socket fallback。

#### D. ROCm FP8 与 vLLM reload runtime probes

在 rollout 节点容器、GPU 空闲时执行：

```bash
python3 verify_fp8_quantizer_runtime.py
python3 verify_vllm_prequantized_metadata_runtime.py
```

验收：
- trainer quantizer 与 vLLM `per_block_cast_to_fp8` dtype/scale 一致；
- FP8 byte mismatch 不超过 probe 阈值；
- reload 前恢复 `ModelWeightParameter`、`BlockQuantScaleParameter`、loader 和 TP metadata；
- resident weight/scale storage pointer 不变。

#### E. R3 router parity 与 PP replay

```bash
python3 tests/integration/run_dsv4_hash_router_parity.py
torchrun --standalone --nproc-per-node=4 \
  tests/integration/run_dsv4_r3_pp_recompute.py
```

验收：
- 输出 `HASH_TID2EID_WEIGHT_PARITY_OK`；
- PP4 每个 stage 的 forward/recompute route 完全一致；
- 无 missing/duplicate PP layers，无 replay FIFO 错位。

### 8.3 一步分布式 Smoke Test

```bash
export RL_ROOT=/workspace DATA_ROOT=/runtime
export MODE=smoke STEPS=1
export CONFIG_OVERRIDE=/workspace/Lumen-RL/examples/GRPO/configs/grpo_dsv4_flash_vllm_longrun.yaml
export LUMENRL_KEEP_RAY_CLUSTER=1 RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma
export FP8_QUANTIZATION_LOCATION=inference
export GLOO_SOCKET_IFNAME=ens14np0 NCCL_SOCKET_IFNAME=ens14np0
export RAY_ADDRESS=10.194.132.76:6379

bash examples/GRPO/run_grpo_dsv4.sh
```

Smoke 验收条件：
- 进程 `exit=0` 且输出 `step=1`；
- `RDMA weight sync committed`，`fp8_location=inference`，传输约 568 GB；
- 每个 vLLM worker 完成 online FP8 prepare/load/finalize 和 full-load verification；
- `moe/r3_route_coverage=1`；
- `moe/r3_hash_flips=0`、`moe/r3_recompute_flips=0`；
- `moe/r3_pp_missing_layers=0`、`moe/r3_pp_duplicate_layers=0`。

### 8.4 正式 200-step 训练

W&B key 放在 `/workspace/wandb.key`，权限设为 `0600`；文件内容可为裸 key 或
`WANDB_API_KEY=<key>`。禁止提交该文件。

```bash
export RL_ROOT=/workspace DATA_ROOT=/runtime
export MODE=longrun STEPS=200
export TRAIN_FILE=/dev/shm/datasets/dapo-math-17k/dapo-math-17k.jsonl
export VAL_FILE=/dev/shm/datasets/aime-2024/aime-2024.jsonl
export WANDB_RUN_NAME=dsv4-flash-grpo-200step-is-r3-inference-fp8-v38
export LUMENRL_KEEP_RAY_CLUSTER=1 RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma
export FP8_QUANTIZATION_LOCATION=inference
export CHECKPOINT_SAVE_STEPS=5
export CHECKPOINT_SAVE_TOTAL_LIMIT=2
export CKPT_DIR=/nfs/data/danyzhan/lumenrl_ckpts/dsv4-flash-200step-is-r3
export GLOO_SOCKET_IFNAME=ens14np0 NCCL_SOCKET_IFNAME=ens14np0
export RAY_ADDRESS=10.194.132.76:6379

bash examples/GRPO/run_grpo_dsv4.sh
```

W&B project 固定为 `danyzhan-amd/LumenRL`。正式训练启动后先确认 W&B run 已创建，
再确认第一个 step 的 inference-side FP8 RDMA reload 和 R3 diagnostics；第 5 步必须验证
NFS checkpoint 的 Megatron distributed-checkpoint shards、metadata 和恢复 step 完整。

### 8.5 从 rank-local checkpoint 恢复大模型训练（2026-08-21）

本节记录从 `global_step_25` 恢复 DSV4 正式训练的流程。mmap 原位加载和缓存释放已通过
offline recovery gate；原 W&B run 的正式恢复仍在执行中。该 checkpoint 包含 16 个 rank
的 model、optimizer、optimizer parameter state 和 extra state shard。

#### A. 必需修复

恢复前必须部署最新版 `examples/GRPO/dsv4/patch_rocm_megatron_dsv4.py`。补丁对
DP=1 HybridDeviceOptimizer (HDO) 使用 `torch.load(..., mmap=True)`，并将 optimizer
参数原位写回已有 storage，避免为每个 rank 额外创建数十 GiB、集群总计约
600–700 GiB 的临时副本。运行时应看到：

```text
[LumenRL] loading DP=1 HDO checkpoint state in place
```

旧恢复路径还会对 Qwen3 MoE 的原生 FP32 参数查询
`param_to_fp32_param`；这些参数不一定存在于映射中，可能导致部分 rank `KeyError`，
其余 rank 永久阻塞在 distributed barrier。不要通过跳过 optimizer state 来规避该错误。

部署后先运行相关回归测试，并确认三节点运行时代码 SHA256 一致：

```bash
python -m pytest \
  tests/unit/test_patch_rocm_megatron_dsv4.py \
  tests/unit/workers/test_actor_worker_engine_config.py \
  tests/unit/engine/test_megatron_lumen_dsv4_engine.py \
  -q
```

#### B. 恢复前检查与模型暂存

1. 确认 checkpoint 的 16 个 rank shard 全部存在，不能只检查 `global_step_25` 目录。
2. 三节点执行 `ray stop --force`，确认训练 GPU 无残留 context。
3. 训练节点均需暂存完整 BF16 HF 模型；当前模型目录实际约 530 GiB：

```bash
du -sh /dev/shm/models/DeepSeek-V4-Flash-BF16
free -h
```

模型初始化期间两个 trainer 必须保留各自的 `/dev/shm/models`。rollout 节点上的模型副本
必须始终保留，后续 vLLM 仍需使用。

#### C. 以 memory-safe 配置重建 Ray

Ray 自身的 99% memory monitor 会在恢复峰值时提前杀死最后几个 rank，即使 Linux
仍可回收 page cache。三节点启动 Ray 时设置：

```bash
export RAY_memory_monitor_refresh_ms=0
export MALLOC_ARENA_MAX=2
export LUMENRL_CHECKPOINT_FORMAT=rank_local
```

继续设置正确的 `GLOO_SOCKET_IFNAME`、`NCCL_SOCKET_IFNAME`、IB HCA/GID 和
`RAY_TMPDIR`。重建后确认 3 个 alive nodes、24 GPUs。

#### D. 在 engine 初始化后立即释放 trainer 模型缓存

初始化完成且日志首次出现以下 marker 后：

```text
Resuming Ray actor checkpoint from .../global_step_25/actor (step=25).
```

表示 16 个 engine 已不再依赖 HF 模型目录。先用 `lsof +D` 确认没有打开的文件，再只在
两个 trainer 删除模型缓存：

```bash
lsof +D /dev/shm/models/DeepSeek-V4-Flash-BF16
rm -rf /dev/shm/models/DeepSeek-V4-Flash-BF16
free -h
```

此操作可在每个 trainer 立即回收约 530 GiB，为 mmap optimizer restore 留出空间。
删除得过早会破坏 engine 初始化；删除 rollout 节点副本会破坏 vLLM。建议用 watcher
检测上述 marker 后自动执行，避免人工延迟造成 OOM。

#### E. 启动恢复

```bash
export MODE=longrun STEPS=200
export CKPT_DIR=/groupstorage/danyzhan/lumenrl_ckpts/dsv4-lockstep-formal-200-eval8-20260817
export RESUME_OVERRIDE=true
export LUMENRL_CHECKPOINT_FORMAT=rank_local
export RAY_memory_monitor_refresh_ms=0
export MALLOC_ARENA_MAX=2
export LUMENRL_KEEP_RAY_CLUSTER=1

bash examples/GRPO/run_grpo_dsv4.sh
```

`global_step_N` 使用 1-based checkpoint 语义；恢复 `global_step_25` 后内部 resume step
为 25，下一条训练日志应为 step 26。成功标志为：

```text
Ray resume complete. Next training log will be global_step=26.
```

恢复期间持续检查两个 trainer 的 `free -h`、kernel OOM 日志和 16 个
`LumenActorWorker.load_checkpoint` task。任一 rank 失败后都必须停止整个 Ray 集群，
不能只重启失败 rank。

#### F. 恢复原 W&B run

若 W&B 账号未启用 rewind private preview，禁止设置
`WANDB_RESUME_FROM=<run>?_step=25`，否则服务端返回 HTTP 400。继续原 run 时使用：

```bash
export WANDB_RUN_ID=d9ohd8vx
export WANDB_RESUME=must
unset WANDB_RESUME_FROM
```

这种模式不会删除原 run 中已经存在的 step 26–44 历史。训练状态仍从 checkpoint
step 25 恢复，但 W&B 会在已有 history 后继续接受新指标；当前原 run 的新指标从
step 45 起追加。若必须让图表严格回退到 step 25，只能先申请 W&B rewind 权限，或创建
一个新 run。

## 9. Actor→Rollout FP8 权重同步

### 9.1 方案 A：Actor 端 FP8 量化（当前禁用）

Actor 在 BF16 训练后，在 GPU 上对每个 weight 做 per-128×128-block FP8 e4m3 量化，
然后发送 FP8 weight + FP32 scale，vLLM 直接加载跳过 online requant。

```
Actor BF16 → fp8_weight_quantizer.py (per-block FP8)
→ yield (name, fp8_weight) + (name_scale_inv, scale)
→ RDMA broadcast (~290GB, 减半)
→ vLLM model.load_weights → 直接写入 FP8 param + scale
→ 不触发 prepare/finalize online requant
```

YAML 配置：
```yaml
weight_sync:
  backend: rdma
  fp8_quantization_location: trainer
```

2026-08-10 独立验证结果：两次 RDMA 均完整发送 285.9 GB、manifest 检查通过，但第一次同步
后固定 prompts 立即从连贯推理退化为重复 token。当前 reload fingerprint 仅检查 change
correspondence，日志明确标记 `no source/target value-tolerance parity`，因此 manifest 成功不能
证明 FP8 weight/scale 数值正确。修复 trainer quantizer/scale loader 并增加逐值 parity 前，
禁止用于正式训练。

关键文件：
- `lumenrl/engine/inference/fp8_weight_quantizer.py` — per-block FP8 量化
- `lumenrl/workers/actor_worker.py` — `send_weights_rdma(fp8_quantize=True)`
- ROCm 注意：需要 `e4m3fn → e4m3fnuz` 转换（ROCm FP8 格式不同）

### 9.2 方案 B：vLLM 端 online requant（当前正式方案）

发送 BF16 权重，并由 vLLM online FP8 再量化。独立 A/B 中两次同步相同 Megatron 权重后
固定 prompts 仍保持连贯；当前正式训练必须使用此路径。

```
Actor BF16 → RDMA broadcast (~568GB, 全量 BF16)
→ vLLM prepare_online_quantized_weights_for_loading
→ model.load_weights (接收 BF16)
→ finalize_online_quantized_weights_loading (per-block FP8 再量化)
```

YAML 配置：
```yaml
weight_sync:
  backend: rdma
  fp8_quantization_location: inference
```

关键文件：
- `lumenrl/engine/inference/vllm_colocate_worker_ext.py` — `receive_weights_rdma`
  已添加 `prepare/finalize` lifecycle
- `lumenrl/engine/inference/vllm_fp8_utils.py` — online FP8 工具函数

### 9.3 方案对比

| | 方案 A (actor-side FP8) | 方案 B (vLLM online requant) |
|---|---|---|
| 传输量 | ~290 GB (FP8+scale) | ~568 GB (BF16) |
| RDMA 时间 | ~12s @ 200Gb/s | ~23s @ 200Gb/s |
| 额外延迟 | actor 端量化 ~10ms | vLLM requant ~数秒 |
| 兼容性风险 | **已复现同步后输出退化，当前禁止** | 已通过独立同步 A/B |
| ROCm | 需要 e4m3fn→e4m3fnuz 转换 | vLLM 内部已处理 |

## 10. 故障排除

| 症状 | 原因 | 解决 |
|------|------|------|
| `Unknown policy.training_backend: megatron_lumen_dsv4` | `actor_worker.py` 未识别新 backend | 确认代码已同步到所有训练节点 |
| `No module named 'tilelang_cython_wrapper'` | `/opt/tilelang` 旧版覆盖了 pip 安装的 tilelang | `mv /opt/tilelang /opt/tilelang.bak` |
| `No module named 'tile_kernels.modeling.mhc.ops'` | 未安装 TileKernels | `pip install -e /path/to/TileKernels` |
| `config_logger_dir` assert | ROCm Megatron 需要非空 config_logger_dir | Engine 已修复设为 `/tmp/megatron-config-logs` |
| `condition_init_method` ImportError | ROCm Megatron 缺少此函数 | DSV4 patch 已添加 |
| `experimental_attention_variant 'dsv4'` 不接受 | ROCm Megatron 未 patch | 重新执行 `patch_rocm_megatron_dsv4.py` |
| GPU OOM | 其他容器占用 GPU | `docker stop` 其他容器释放 GPU |
| optimizer 初始化只差 64 MiB OOM，Actor 已退出但 `rocm-smi` 仍显示约 90% | 上一轮 Ray worker/ROCm context 未完全清理 | 三节点 `ray stop --force`，确认训练 GPU VRAM 全部回到 0%，重建集群 |
| 初始化 optimizer 时 GPU OOM，即使没有残留进程 | DistributedOptimizer 先在 GPU 创建完整 FP32 master-weight clone，峰值超过 MI308X 容量 | 设置 `use_precision_aware_optimizer: true`，启用 full CPU offload、BF16 moments 和 streamed Adam |
| rank-local HDO checkpoint 恢复时部分 rank `KeyError`，其余 rank 卡在 barrier | 原生 FP32 参数不在 `param_to_fp32_param`，且旧路径产生巨大的临时 optimizer 副本 | 部署 mmap DP=1 HDO 原位恢复 patch，设置 `LUMENRL_CHECKPOINT_FORMAT=rank_local` |
| checkpoint 恢复末期 Ray 杀死 rank，主机尚未发生 kernel OOM | Ray 99% memory monitor 在 page cache 回收前主动终止 worker | 三节点设置 `RAY_memory_monitor_refresh_ms=0`，并在 engine 初始化后释放 trainer 的 HF 模型缓存 |
| W&B 恢复报 `failed to rewind run` / HTTP 400 | 账号未启用 rewind private preview | 使用 `WANDB_RUN_ID=<原ID>`、`WANDB_RESUME=must` 并清除 `WANDB_RESUME_FROM`；接受旧 history 保留 |
| trainer-side FP8 同步成功但 eval/rollout 变成 `Aime`、`0`、`n` 等重复输出 | pre-quantized FP8 weight/scale 数值或 loader 对应错误；manifest 校验不覆盖源/目标逐值 parity | 设置 `fp8_quantization_location: inference`；不要将 `manifest=checked` 当作数值正确 |
| 怀疑 prefix cache 导致或修复 rollout fault | 当前通过配置实际为 `enable_prefix_caching: false`，没有“开启后修复”的 A/B 证据 | 保持 prefix cache 关闭、chunked prefill 开启；修改前必须运行固定 prompts + 32×8 A/B |
| ranks 8–15 optimizer 初始化远慢于 ranks 0–7 | CPU offload 跨 NUMA page migration | 设置 `numa_affinity: true`，用 `taskset -pc <PID>` 验证 GPU 0–3/4–7 分别绑定 NUMA 0/1 |
| NFS 加载慢 | 16 worker 同时读 NFS | 预拷模型到 `/dev/shm` |
| `size mismatch for wq_a.weight` | HF config 字段名不匹配 | Engine 已修复（`head_dim` vs `kv_lora_rank`） |
| RDMA 未生效（Socket fallback） | 容器未用 `--privileged` | 必须用 `--privileged` 运行容器 |
| Gloo transport 失败 | 缺少 `GLOO_SOCKET_IFNAME` | 设置 `GLOO_SOCKET_IFNAME=ens14np0` |
| vLLM 启动时报 `ncclGetUniqueId` / `NCCL error: invalid usage` | YAML 中遗留不存在的 `NCCL_SOCKET_IFNAME=ens11np0`，RCCL 日志为 `Bootstrap: no socket interface found` | 将 `weight_sync.rdma.interface`、容器及启动环境统一设为 `ens14np0` |
| `ConfigKeyError: numa_affinity not in MegatronConfig` | 只更新 YAML/Actor，未同步 config schema | 同步 `lumenrl/core/config.py`，确认 `MegatronConfig.numa_affinity: bool = False` |
| vLLM `EngineCore failed to start`，AITER 报 `hipIpcGetMemHandle ... invalid argument` | vLLM TP=8 默认启用了 AITER custom all-reduce；当前 MI300X/ROCm 组合无法为通信 buffer 创建 HIP IPC handle | 设置 `policy.generation.vllm_cfg.disable_custom_all_reduce: true`，并确认启动日志显示 `disable_custom_all_reduce=True`，使 TP collective 回退到 RCCL |
| vLLM entrypoint 冲突 | vLLM base image 默认入口 | `--entrypoint bash` 覆盖 |
| Python 版本不匹配 | 训练/推理容器 Python 不一致 | 统一使用 `vllm/vllm-openai-rocm:v0.25.1` base |

## 11. 软件版本与代码仓库

### 11.1 Docker 镜像

| 镜像 | Docker Hub Tag | 用途 |
|------|---------------|------|
| `zhangdanyangamd/lumen-rl:dsv4-300x-train260813` | 可直接运行的 trainer 最终环境 | f293、f294 |
| `zhangdanyangamd/lumen-rl:dsv4-300x-rollout260813` | 可直接运行的 rollout 最终环境 | f2b9 |

Base image: `vllm/vllm-openai-rocm:v0.25.1` (Python 3.12, PyTorch 2.11)

### 11.2 软件版本

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.13 | vLLM base image |
| PyTorch | 2.11.0+gitd0c8b1f | ROCm 7.2 |
| Ray | 2.56.1 | |
| vLLM | 0.25.1+rocm723 | rollout 推理；wheel VCS commit `752a3a504` |
| flash-attn | 2.8.0.post2 | ROCm gfx942 编译 |
| tilelang | 0.1.10 | PyPI, sparse MLA kernel |
| transformers | 5.13.1 | |

### 11.3 代码仓库

| 组件 | GitHub 链接 | 分支/Commit | 说明 |
|------|-------------|-------------|------|
| LumenRL | https://github.com/ZhangDanyang-AMD/Lumen-RL.git | `dev/dsv4-grpo` @ `4aac3d4` | DSV4 engine + bridge + configs |
| Lumen | https://github.com/ZhangDanyang-AMD/Lumen.git | PR #8 @ `0223585` | DSV4 spec (MLA, HC, compressor, indexer) |
| Megatron-LM | https://github.com/ROCm/Megatron-LM.git | `rocm_dev` @ `fb4552449f9b33c6f72207a80e80045eadf5267e` | 运行时 working tree 已应用 DSV4/R3/ROCm patch |
| vLLM | https://github.com/vllm-project/vllm.git | `0.25.1` @ `752a3a504` | 运行 wheel 为 `0.25.1+rocm723`；wheel 仅保留短 VCS commit |
| TileKernels | https://github.com/jayzlee147/TileKernels.git | `main` @ `c795a96` | mHC kernel |
| tilelang | https://pypi.org/project/tilelang/ | `0.1.10` (PyPI) | Sparse MLA fwd/bwd + indexer kernel |
| amd-rl-runbook | https://github.com/ZhangDanyang-AMD/amd-rl-runbook.git | `main` | 本 runbook |
