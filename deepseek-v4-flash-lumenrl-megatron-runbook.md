# DeepSeek-V4-Flash 三节点 RL 部署 Runbook

> Megatron 训练（TP4/PP4/EP4）+ vLLM FP8 rollout + RCCL/RoCE GPU Direct RDMA 权重同步
> 硬件：3 节点 — 2 × 8 MI308X 训练 + 1 × 8 MI300X 推理
> LumenRL：`origin/dev/moe-grpo`
> 超参对齐 LMSYS MILES DSV4：https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4/

## 1. 目标架构

### 1.1 节点角色

| 节点 | 主机名 | 角色 | GPU |
|------|--------|------|-----|
| 训练-0 | `banff-ccs-aus-p20-14.cs-aus.dcgpu` | Megatron actor (NODE_RANK=0) | 8 × MI308X |
| 训练-1 | `banff-ccs-aus-p20-38.cs-aus.dcgpu` | Megatron actor (NODE_RANK=1) | 8 × MI308X |
| 推理 | `banff-ccs-aus-p19-29.cs-aus.dcgpu` | vLLM rollout + Ray head | 8 × MI300X |

### 1.2 训练配置

- 16 个 Megatron actor worker，分布在 2 个训练节点
- TP=4, PP=4, EP=4, ETP=1
- Pipeline 层分布：11 + 11 + 11 + 10（共 43 层）
- Optimizer：AdamW + CPU offload（fraction=0.75）
- 精度：BF16 compute + FP32 master params/Adam state
- 初始权重：`/nfs/data/DeepSeek-V4-Flash`（原始 BF16 HF safetensors，568 GB）
- Engine：`megatron_lumen_dsv4`（LumenRL 的 `MegatronLumenDSV4Engine`）
- Megatron：ROCm/Megatron-LM `rocm_dev` + DSV4 patch
- Kernel：Lumen DSV4 spec（MLA attention、HC、compressor、indexer）+ TileKernels（mHC、sparse MLA）+ tilelang（PyPI）

### 1.3 推理配置

- 1 个 vLLM 实例，TP=8（全部 8 个 GPU）
- 模型：`DeepSeek-V4-Flash-BF16`（~568 GB）
- 在线 FP8 量化：`fp8_per_block`（BF16 权重加载后在 GPU 上量化为 FP8 e4m3）
- KV cache：`fp8_e4m3`
- `enforce_eager: true`（不使用 cudagraph）
- MoE backend：`triton`

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
| Num rollout | 3000 |
| 数据集 | `zhuzilin/dapo-math-17k`（JSONL） |
| 验证集 | `zhuzilin/aime-2024`（JSONL） |

## 2. 存储策略

### 2.1 快盘 vs NFS

| 用途 | 路径 | 存储类型 | 原因 |
|------|------|----------|------|
| 模型权重（HF） | `/dev/shm/models/DeepSeek-V4-Flash` | tmpfs (内存) | 16 worker 并发读，NFS 太慢 |
| 数据集 | `/dev/shm/datasets/` | tmpfs | 小文件，快速访问 |
| Checkpoint | `/nfs/data/leiwu/ckpts/dsv4-flash-lumenrl/` | NFS | 需要持久化，save_steps 设大（50+） |
| 日志 | `/nfs/data/leiwu/logs/` 或宿主机挂载 | NFS/本地 | 需要持久化 |

### 2.2 模型预加载

训练前需将 568 GB BF16 模型从 NFS 复制到两个训练节点的 `/dev/shm`：

```bash
# 在每个训练节点执行
mkdir -p /dev/shm/models
cp -a /nfs/data/DeepSeek-V4-Flash /dev/shm/models/
```

**注意**：`/dev/shm` 是 tmpfs（内存），节点重启后丢失。`--ipc=host` 的 Docker 容器共享宿主机 `/dev/shm`。

## 3. Docker 镜像

### 3.1 统一 base image

训练和推理使用同一 base image `vllm/vllm-openai-rocm:v0.25.1`（Python 3.12, PyTorch 2.11），
保证 Ray 集群跨节点 Python 版本一致。

使用 `Dockerfile.dsv4` 构建：

```bash
cd ~/Lumen-RL
docker build -f examples/GRPO/Dockerfile.dsv4 --target trainer -t dsv4-flash:trainer .
docker build -f examples/GRPO/Dockerfile.dsv4 --target rollout -t dsv4-flash:rollout .
```

### 3.2 容器内额外安装

Dockerfile build 后，容器内还需：

```bash
# 1. ROCm Megatron + DSV4 patch
git clone --depth 1 https://github.com/ROCm/Megatron-LM.git -b rocm_dev /tmp/rocm-megatron
cp -a /tmp/rocm-megatron /workspace/Megatron-LM
python3 /workspace/Lumen-RL/examples/GRPO/dsv4/patch_rocm_megatron_dsv4.py /workspace/Megatron-LM

# 2. TileKernels (mHC kernel)
pip uninstall tile_kernels -y
pip install --no-deps -e /path/to/TileKernels  # from jayzlee147/TileKernels.git

# 3. tilelang (PyPI, sparse MLA kernel)
pip install tilelang
```

安装完成后 `docker commit` 保存为镜像。

### 3.3 启动容器

```bash
docker run -d --name dsv4-rl \
  --privileged --network=host --ipc=host --shm-size=128g \
  -v /dev/infiniband:/dev/infiniband \
  -v ~/Lumen-RL:/workspace/Lumen-RL \
  -v /nfs/data:/nfs/data \
  -v ~/dsv4-runtime:/runtime \
  -e GLOO_SOCKET_IFNAME=ens14np0 \
  -e NCCL_SOCKET_IFNAME=ens14np0 \
  --entrypoint bash \
  dsv4-flash:trainer-ready -c "sleep infinity"
```

**关键点**：
- `--privileged` 是 InfiniBand/RDMA 正常工作的必要条件
- `--entrypoint bash` 覆盖 vLLM 默认入口
- `--ipc=host` 共享宿主机 `/dev/shm`（可访问预加载的模型权重）
- 不挂载宿主机 Lumen（用 Docker 镜像内构建的版本，保证和 Megatron 版本匹配）

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
# Ray head（推理节点 p19-29）
docker exec dsv4-rl ray start --head --node-ip-address=$P19_IP --port=6379 --num-gpus=8 --num-cpus=64

# 训练节点加入
docker exec dsv4-rl ray start --address=$P19_IP:6379 --node-ip-address=$TRAIN_IP --num-gpus=8 --num-cpus=64
```

验证：`ray status` 应显示 3 nodes, 24 GPU。

### 6.2 多节点训练 placement

Actor 使用 `process_on_nodes: [8, 8]`（无 topology_tags），Ray STRICT_PACK 自动分配到两个训练节点。
Rollout 使用 `topology_tags: {node_ip: <推理节点IP>}` 固定到推理节点。

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
- `model_name` → `/dev/shm/models/DeepSeek-V4-Flash`
- `dataset` → `/dev/shm/datasets/dapo-math-17k/dapo-math-17k.jsonl`
- `checkpoint_dir` → `/nfs/data/leiwu/ckpts/dsv4-flash-lumenrl/smoke`

### 8.2 Smoke Test

```bash
export RL_ROOT=/workspace DATA_ROOT=/runtime
export MODE=smoke STEPS=3
export CONFIG_OVERRIDE=/runtime/configs/dsv4-smoke.yaml
export LUMENRL_KEEP_RAY_CLUSTER=1 RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma
export GLOO_SOCKET_IFNAME=ens14np0 NCCL_SOCKET_IFNAME=ens14np0
export RAY_ADDRESS=$P19_IP:6379

bash examples/GRPO/run_grpo_dsv4.sh
```

## 9. 故障排除

| 症状 | 原因 | 解决 |
|------|------|------|
| `Unknown policy.training_backend: megatron_lumen_dsv4` | `actor_worker.py` 未识别新 backend | 确认代码已同步到所有训练节点 |
| `No module named 'tilelang_cython_wrapper'` | `/opt/tilelang` 旧版覆盖了 pip 安装的 tilelang | `mv /opt/tilelang /opt/tilelang.bak` |
| `No module named 'tile_kernels.modeling.mhc.ops'` | 未安装 TileKernels | `pip install -e /path/to/TileKernels` |
| `config_logger_dir` assert | ROCm Megatron 需要非空 config_logger_dir | Engine 已修复设为 `/tmp/megatron-config-logs` |
| `condition_init_method` ImportError | ROCm Megatron 缺少此函数 | DSV4 patch 已添加 |
| `experimental_attention_variant 'dsv4'` 不接受 | ROCm Megatron 未 patch | 重新执行 `patch_rocm_megatron_dsv4.py` |
| GPU OOM | 其他容器占用 GPU | `docker stop` 其他容器释放 GPU |
| NFS 加载慢 | 16 worker 同时读 NFS | 预拷模型到 `/dev/shm` |
| `size mismatch for wq_a.weight` | HF config 字段名不匹配 | Engine 已修复（`head_dim` vs `kv_lora_rank`） |
| RDMA 未生效（Socket fallback） | 容器未用 `--privileged` | 必须用 `--privileged` 运行容器 |
| Gloo transport 失败 | 缺少 `GLOO_SOCKET_IFNAME` | 设置 `GLOO_SOCKET_IFNAME=ens14np0` |
| vLLM entrypoint 冲突 | vLLM base image 默认入口 | `--entrypoint bash` 覆盖 |
| Python 版本不匹配 | 训练/推理容器 Python 不一致 | 统一使用 `vllm/vllm-openai-rocm:v0.25.1` base |

## 10. 代码仓库

| 组件 | 仓库 | 分支 |
|------|------|------|
| LumenRL | `github.com/ZhangDanyang-AMD/Lumen-RL.git` | `dev/moe-grpo` |
| Lumen | `github.com/ZhangDanyang-AMD/Lumen.git` | PR #8 |
| Megatron-LM | `github.com/ROCm/Megatron-LM.git` | `rocm_dev` + DSV4 patch |
| TileKernels | `github.com/jayzlee147/TileKernels.git` | main |
