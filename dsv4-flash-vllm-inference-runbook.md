# DeepSeek-V4-Flash vLLM 推理 Runbook（BF16 + 在线 FP8 量化，8×MI300X 单机）

> 目标：在一台 **8 卡 AMD MI300X (gfx942)** 机器上，用 **vLLM** 跑通 **DeepSeek-V4-Flash** 推理服务。
> **加载 BF16 全精度权重，使用 vLLM 在线 `fp8_per_block` 量化推理。**
> 模型与数据放在 `/dev/shm`（tmpfs，内存文件系统，读写极快）。
>
> - **模型**：`RedHatAI/DeepSeek-V4-Flash-BF16`（284B total / 13B active MoE，BF16 全精度，~568 GB）
> - **量化方式**：vLLM 在线 `fp8_per_block`（128×128 block scaling，加载时量化权重，推理时动态量化 activation，FP32 scale）
> - **推理框架**：vLLM v0.25.1（ROCm，Docker `vllm/vllm-openai-rocm:v0.25.1`）
> - **配置来源**：[LMSYS ROCm Miles DSV4 Blog](https://www.lmsys.org/blog/2026-07-10-rocm-miles-dsv4/) rollout 配置
>   + [vLLM Recipes DSV4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) MI300X 验证配置
>   + [vLLM Online Quantization](https://docs.vllm.ai/en/stable/features/quantization/online/)
> - **硬件**：8×MI300X (gfx942, 192 GB HBM3 each, 共 1536 GB), ~2 TB RAM, Ubuntu 24.04

---

## 0. 为什么用 BF16 + 在线 FP8 量化

### ~~预量化 FP8 checkpoint 不可用~~ (MI300X 兼容性问题)

Miles rollout 使用的 `sgl-project/DeepSeek-V4-Flash-FP8` 和官方 `deepseek-ai/DeepSeek-V4-Flash` checkpoint
的 `quantization_config` 均包含 `"scale_fmt": "ue8m0"`。**UE8M0 是 NVIDIA DeepGEMM 的 scale 格式**（unsigned
e8m0 exponent-only），MI300X (gfx942/CDNA3) 不支持。vLLM v0.25.1 在 MI300X 上加载这些 checkpoint 时会报错：

```
RuntimeError: The size of tensor a (2048) must match the size of tensor b (4096) at non-singleton dimension 1
```

错误发生在 `routed_experts.py` 的 `_load_w13` 中——UE8M0 scale tensor 的 shape 与 AMD 代码路径期望的 FP32 scale 不匹配。

此外，**MXFP8 是 MI350X (CDNA4/gfx950) 的硬件特性**，MI300X 不具备。

### 可行方案：BF16 + 在线 FP8 量化

| 方案 | checkpoint | 在 MI300X 上 | 与 Miles 量化数值 |
|---|---|---|---|
| ~~预量化 FP8 (ue8m0)~~ | `sgl-project/...-FP8` (294 GB) | **不兼容** | 一致 |
| ~~官方 FP4+FP8 (ue8m0)~~ | `deepseek-ai/...` (160 GB) | **不兼容** | 不一致（FP4） |
| **BF16 + 在线 FP8 量化（本方案）** | `RedHatAI/...-BF16` (568 GB) | **兼容** | 量化路径相似（128×128 block scaling），scale 格式不同（FP32 vs UE8M0） |

vLLM `--quantization fp8_per_block` 在加载时将 BF16 权重在线量化为 FP8 e4m3（128×128 block scaling），
生成 **FP32 scale tensor**，完全兼容 MI300X。无需校准数据。

### 显存估算

- BF16 权重加载到 CPU：~568 GB
- 在线量化为 FP8 后 GPU 显存：~284 GB 权重 + FP32 scale tensors
- KV cache（FP8, 16K context）：~10–20 GB
- 8×MI300X 总显存：1536 GB，**充裕**
- `/dev/shm` 需求：~568 GB（当前可用 ~1 TB）

### 已知限制与风险

| 项目 | 说明 |
|---|---|
| MI300X FP8 方言 | MI300X 使用 `fnuz` 方言（exponent bias 与 OCP 差 1），v0.25.1 已包含主要修复 |
| AITER kernel 覆盖 | gfx942 上部分 AITER kernel 可能缺失或不稳定；需用 `--enforce-eager` + `--moe-backend triton` 兜底 |
| 在线量化加载时间 | 568 GB BF16→FP8 在线量化在模型加载阶段完成，首次加载较慢（预估 5–15 分钟） |
| 与 Miles rollout 的对齐 | 在线量化使用 FP32 scale，Miles 使用 UE8M0 scale；量化数值非 bit-exact 但 block scaling 方式相同 |

---

## 1. 路径变量

```bash
export MODEL_DIR=/dev/shm/models/DeepSeek-V4-Flash-BF16
export DATA_DIR=/dev/shm/dsv4-data
export CONTAINER=dsv4-vllm-inference
mkdir -p "$DATA_DIR"
```

---

## 2. 下载 BF16 模型到 `/dev/shm`

```bash
pip install -U "huggingface_hub[hf_transfer]" 2>/dev/null

export HF_XET_HIGH_PERFORMANCE=1

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'RedHatAI/DeepSeek-V4-Flash-BF16',
    local_dir='/dev/shm/models/DeepSeek-V4-Flash-BF16',
    allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model', 'tokenizer*', '*.py'],
)
print('download done')
"

# 验证
du -sh /dev/shm/models/DeepSeek-V4-Flash-BF16           # 期望 ~568 GB
ls /dev/shm/models/DeepSeek-V4-Flash-BF16/*.safetensors | wc -l
df -h /dev/shm
```

> **注意**：模型 ~568 GB，下载耗时取决于网络带宽。`/dev/shm` 是内存文件系统，机器重启后数据丢失。

---

## 3. 拉取 Docker 镜像并启动容器

```bash
sudo docker pull vllm/vllm-openai-rocm:v0.25.1

sudo docker rm -f "$CONTAINER" 2>/dev/null

sudo docker run -d \
  --name "$CONTAINER" \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64G \
  -v /dev/shm/models:/dev/shm/models:ro \
  -v "$DATA_DIR":"$DATA_DIR" \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e TORCHDYNAMO_DISABLE=1 \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.25.1 -lc 'sleep infinity'

sudo docker ps --filter name="$CONTAINER"
```

验证容器内 GPU 与 vLLM：
```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 -c "import torch; print(\"GPUs:\", torch.cuda.device_count(), \"arch:\", torch.cuda.get_device_properties(0).gcnArchName)"
python3 -c "import vllm; print(\"vLLM:\", vllm.__version__)"
'
```
> 期望：`GPUs: 8 arch: gfx942:sramecc+:xnack-`，`vLLM: 0.25.1`。

---

## 4. 启动 vLLM 推理服务（BF16 + 在线 FP8 量化）

### 4.1 配置说明

| 参数 | 值 | 说明 |
|---|---|---|
| `dtype` | `bfloat16` | 以 BF16 加载权重 |
| `quantization` | `fp8_per_block` | 在线量化：128×128 block scaling，FP32 scale，兼容 MI300X |
| `tensor-parallel-size` | 8 | 8 卡全部做 tensor parallel |
| `max-model-len` | 16384 | LMSYS rollout (context-length) |
| `kv-cache-dtype` | `fp8_e4m3` | vLLM Recipes MI300X |
| `tokenizer-mode` | `deepseek_v4` | vLLM Recipes |
| `moe-backend` | `triton` | `triton_unfused` 不支持 FP8 MoE，必须用 `triton`（已验证） |
| `enforce-eager` | — | vLLM Recipes MI300X (避免 cudagraph 问题) |
| `trust-remote-code` | — | DSV4 hybrid attention 需要 |
| `enable-chunked-prefill` | — | 提升长序列效率 |
| `max-num-batched-tokens` | 8192 | LMSYS / vLLM Recipes |
| `gpu-memory-utilization` | 0.92 | vLLM Recipes MI300X throughput |
| `distributed-executor-backend` | `mp` | vLLM Recipes MI300X |

> **关于并行策略**：LMSYS 原始配置为 TP1+PP4。前一轮测试中 FP8 checkpoint 使用 PP4 时因 UE8M0 scale
> 在所有 PP worker 上均报 shape 不匹配错误。BF16 + 在线量化不存在该问题，但 PP 在 DSV4 的 AMD 代码
> 路径上可能仍有其他兼容性风险，因此先用 TP8（更成熟的路径）。如果需要可以回退到 TP1+PP4。

### 4.2 启动命令

```bash
sudo docker exec -d "$CONTAINER" bash -lc '
export VLLM_ROCM_USE_AITER=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export TORCHDYNAMO_DISABLE=1

vllm serve /dev/shm/models/DeepSeek-V4-Flash-BF16 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --quantization fp8_per_block \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --moe-backend triton \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --distributed-executor-backend mp \
  > /dev/shm/dsv4-data/vllm-serve.log 2>&1
'
```

查看启动日志（BF16 加载 + 在线量化较慢，耐心等待 5–15 分钟）：
```bash
tail -f $DATA_DIR/vllm-serve.log
```

> 期望日志：
> - `Loading model weights`（568 GB BF16 加载 + FP8 在线量化）
> - `Quantization: fp8_per_block` 或类似提示
> - `INFO: Started server process` / `Uvicorn running on http://0.0.0.0:8000`

---

## 5. 验证推理

### 5.1 检查服务状态

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/health
```

### 5.2 Non-think 模式（快速响应）

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4-Flash-BF16",
    "messages": [{"role": "user", "content": "What is 2+3?"}],
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 256
  }' | python3 -m json.tool
```

### 5.3 Think High 模式（CoT 推理）

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4-Flash-BF16",
    "messages": [
      {"role": "system", "content": "Please reason step by step."},
      {"role": "user", "content": "Find all prime numbers p such that p^2 + 2 is also prime."}
    ],
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 4096
  }' | python3 -m json.tool
```

### 5.4 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="DeepSeek-V4-Flash-BF16",
    messages=[{"role": "user", "content": "Explain the attention mechanism in transformers in 3 sentences."}],
    temperature=1.0,
    top_p=1.0,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

> **注意**：`model` 字段的值取决于 vLLM 注册的模型名。如果上面的名字不工作，
> 先用 `curl -s http://localhost:8000/v1/models` 查看实际注册名，或启动时加 `--served-model-name DeepSeek-V4-Flash`。

---

## 6. 替代配置

### 6.1 TP1 + PP4（LMSYS rollout 原始并行策略）

如果 TP8 遇到问题，可以尝试 LMSYS 原始配置：

```bash
vllm serve /dev/shm/models/DeepSeek-V4-Flash-BF16 \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --quantization fp8_per_block \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4 \
  --max-model-len 16384 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --moe-backend triton \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --distributed-executor-backend mp
```

### 6.2 Per-tensor 在线量化（精度稍低，但兼容性更好）

如果 `fp8_per_block` 在 gfx942 上遇到 kernel 兼容问题，回退到 per-tensor 量化：

```bash
vllm serve /dev/shm/models/DeepSeek-V4-Flash-BF16 \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --quantization fp8_per_tensor \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --moe-backend triton \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --distributed-executor-backend mp
```

### 6.3 纯 BF16 推理（无量化，精度最高，吞吐量最低）

```bash
vllm serve /dev/shm/models/DeepSeek-V4-Flash-BF16 \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --moe-backend triton \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 256 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --distributed-executor-backend mp
```

---

## 7. 监控与运维

### 7.1 状态检查

```bash
# vllm 进程存活
sudo docker exec "$CONTAINER" bash -lc "ps aux | grep 'vllm serve' | grep -v grep | wc -l"

# GPU 显存
rocm-smi --showmeminfo vram | grep 'Used Memory'

# 最新日志
tail -20 $DATA_DIR/vllm-serve.log

# 请求指标（vLLM 内置 Prometheus）
curl -s http://localhost:8000/metrics | grep -E 'vllm_(num_requests|avg_prompt|avg_generation)' | head -10
```

### 7.2 停止服务

```bash
sudo docker exec "$CONTAINER" bash -lc "pkill -f 'vllm serve'"
# 或直接停止容器
sudo docker stop "$CONTAINER"
```

### 7.3 重启服务

```bash
sudo docker restart "$CONTAINER"
sleep 3
# 重新执行第 4.2 节启动命令
```

---

## 8. 配置一览

| 维度 | 值 |
|---|---|
| 模型 | `RedHatAI/DeepSeek-V4-Flash-BF16` (284B/13B active, BF16 全精度, ~568 GB) |
| 在线量化 | `fp8_per_block`（128×128 block scaling, FP32 scale, 动态 activation scaling） |
| 推理框架 | vLLM v0.25.1 ROCm |
| Docker 镜像 | `vllm/vllm-openai-rocm:v0.25.1` |
| 硬件 | 8×MI300X (gfx942), 192 GB HBM3/卡, 共 1536 GB |
| 并行策略 | TP=8 |
| Context length | 16384 |
| KV cache | FP8 e4m3 |
| MoE backend | `triton`（`triton_unfused` 不支持 FP8 MoE） |
| 模式 | eager（无 cudagraph） |
| GPU 显存利用率 | 0.92 |
| 模型路径 | `/dev/shm/models/DeepSeek-V4-Flash-BF16` |
| 服务端口 | 8000 |
| API | OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`) |

### 关键环境变量

| 变量 | 作用 |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | 启用 AITER kernel 加速 |
| `HIP_FORCE_DEV_KERNARG=1` | ROCm 性能优化 |
| `HSA_NO_SCRATCH_RECLAIM=1` | 避免 scratch memory 回收开销 |
| `TORCHDYNAMO_DISABLE=1` | 禁用 TorchDynamo（与 enforce-eager 配合） |

---

## 9. MI300X 上预量化 FP8 Checkpoint 失败记录

### 已测试并失败的配置

| checkpoint | scale_fmt | 错误 |
|---|---|---|
| `sgl-project/DeepSeek-V4-Flash-FP8` | `ue8m0` | `RuntimeError: The size of tensor a (2048) must match the size of tensor b (4096)` in `routed_experts.py:_load_w13` |
| `deepseek-ai/DeepSeek-V4-Flash` | `ue8m0` (FP4+FP8 mixed) | 同样不兼容（ue8m0 scale format） |

### 根因分析

- UE8M0（unsigned e8m0 exponent-only）是 NVIDIA DeepGEMM 的 scale 格式
- MI300X (gfx942/CDNA3) 上 vLLM 的 AMD 代码路径（`vllm/models/deepseek_v4/amd/model.py`）期望 FP32 scale tensor
- UE8M0 scale tensor 的 shape 与 FP32 scale 不同，导致 `expert_data.copy_(loaded_weight)` 时 shape 不匹配
- MXFP8 是 MI350X (CDNA4/gfx950) 硬件特性，MI300X 不可用

### 解决路径

1. **当前方案**：BF16 checkpoint + vLLM 在线 `fp8_per_block` 量化（FP32 scale，兼容 MI300X）
2. **未来可能**：vLLM AMD 路径增加 UE8M0 → FP32 scale 转换支持；或社区提供 FP32 scale 的 FP8 checkpoint

---

## 10. Troubleshooting

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| `tensor a (2048) must match tensor b (4096)` | FP8 checkpoint 使用 UE8M0 scale，MI300X 不支持 | 改用 BF16 + 在线量化（本 runbook） |
| `PDL is not supported` / TileLang 错误 | vLLM 版本过旧（< v0.22），MHC kernel 缺失 | 升级到 v0.25.1+ |
| `mul_cuda not implemented for Float8_e8m0fnu` | gfx942 fnuz 方言未处理 | 确认使用 v0.25.1+；若仍报错，尝试 nightly |
| `triton_unfused` 不支持 FP8 MoE | 在线量化后 MoE 权重是 FP8，`triton_unfused` 不兼容 | 改用 `--moe-backend triton`（已验证） |
| 在线量化报错 / `fp8_per_block` 不支持 | gfx942 缺少对应 FP8 kernel | 回退到 `--quantization fp8_per_tensor`（6.2 节） |
| OOM | 量化后权重 + KV cache 超出显存 | 降低 `gpu-memory-utilization` 至 0.85，或缩短 `max-model-len` |
| 模型加载极慢（>20 分钟） | 568 GB BF16 加载 + 在线量化 | 正常现象；从 `/dev/shm` 读取应较快 |
| 启动后长时间无响应 | warmup regression (v0.21–v0.25) | 等待 5-10 分钟；或升级到 v0.26+ |
| `/dev/shm` 空间不足 | 模型 ~568 GB 占满 tmpfs | `df -h /dev/shm` 检查；确保至少 600 GB 可用 |
| 容器内找不到 GPU | 缺少 `--device=/dev/kfd --device=/dev/dri` | 重新创建容器，确保设备映射正确 |
| `model not found` (curl 报错) | 模型名不匹配 | `curl localhost:8000/v1/models` 查看实际名；或启动时加 `--served-model-name` |
