# DeepSeek-V4-Flash 在 LumenRL + ROCm 上能跑到哪一步（定界 handoff）

> 目标是拿 `deepseek-ai/DeepSeek-V4-Flash-0731` 跑 DAPO RL smoke。**结论：rollout 侧可用，
> 训练侧不可用。** 本文把每一道墙都量化，并记下已经排除的假设，免得下一个人重走。
>
> 环境：4×8 MI355X gfx950，vime-rocm 镜像（vllm `0.22.1rc1.dev392+g43914dd74.rocm702`、
> transformers 5.13.1、megatron-core 0.16.0rc0、TE 2.12.0.dev0）。2026-08-05。

## 0. 一页结论

| 环节 | 状态 |
|---|---|
| vLLM 加载 + 生成 | ✅ **可用**，但必须 TP≥2（实测 TP=8）+ 4 个必需配置项，见 §2 |
| LumenRL rollout 托管 | ✅ 需要 rollout TP>1（已实现，见 `lumenrl-rollout-tp-gt-1-handoff.md`） |
| 训练侧加载权重（FSDP2） | ❌ checkpoint 是 DeepSeek 原生命名，与 transformers **交集 0**，见 §3 |
| 训练侧加载权重（Megatron） | ❌ 更远：LumenRL 的引擎当场 KeyError，且 bridge 是 Qwen3 专用，见 §4 |
| 32 卡显存 | ❌ 284.3B 参数，4 节点装不下，见 §5 |

## 1. 这个模型是什么

`DeepseekV4ForCausalLM` / `model_type: deepseek_v4`，**284.3 B 参数**（meta device 实例化实测），
top-6 激活约 8B——"Flash" 就是这个意思。43 层，每层都是 MoE，256 路由专家 + 1 共享专家。

| 项 | 值 |
|---|---|
| 层数 / hidden | 43 / 4096 |
| 专家 | 256 routed + 1 shared，top-6，`moe_intermediate_size=2048` |
| 专家参数量 | 43 × 256 × 25.2M ≈ **277 B**（占总量的 97%） |
| 注意力 | MLA 变体：`q_lora_rank=1024`、`o_lora_rank=1024`、`head_dim=512`、KV head 1、`o_groups=8` |
| 稀疏注意力 | `index_n_heads=64`、`index_head_dim=128`、`index_topk=512`（Lightning Indexer / DSA） |
| 其它新机制 | `num_hash_layers=3`（hash-MoE bootstrap）、`hc_*`（mHC 超连接 + Sinkhorn）、`compress_ratios`（46 项逐层 KV 压缩表）、`dspark_target_layer_ids=[40,41,42]`、`num_nextn_predict_layers=1`（MTP） |
| 路由打分 | `scoring_func=sqrtsoftplus`、`topk_method=noaux_tc` |
| 量化 | `quantization_config`: fp8 e4m3、`weight_block_size [128,128]`、`scale_fmt=ue8m0`；**`expert_dtype: fp4`** |
| 上下文 | `max_position_embeddings=1048576`，yarn |

同族仓库（**全部是原生命名格式**）：

| 仓库 | 体积 | 分片 |
|---|---|---|
| `DeepSeek-V4-Flash-0731`（本文用的） | 166.9 GB | 48 |
| `DeepSeek-V4-Flash` | 159.6 GB | 46 |
| `DeepSeek-V4-Flash-Base` ← **RL 该用这个** | **294.7 GB** | 46 |
| `DeepSeek-V4-Pro` / `-Pro-Base` / `-Flash-DSpark` / `-Pro-DSpark` | — | — |

⚠️ Base 版比 instruct 版大 1.8 倍（专家应为 FP8 而非 FP4）。而 RL 必须用 Base（vime runbook §16.1：
instruct/thinking 版会让 `filter_groups` 连续 10 轮 kept 0 直接抛错）。**TP=1 时光引擎就要 295GB，
单卡 288GB 直接装不下。**

## 2. ✅ rollout：能跑，但只在 TP≥2

实测能加载并生成通顺文本：

```python
LLM(model=".../DeepSeek-V4-Flash-0731",
    tensor_parallel_size=8,          # 决定性因素，见 §2.2
    kv_cache_dtype="fp8_e4m3",       # 必需
    moe_backend="triton_unfused",    # AMD 验证配方里的
    disable_custom_all_reduce=True,  # 必需
    max_model_len=4096, enforce_eager=True, gpu_memory_utilization=0.80,
    max_num_batched_tokens=2048, max_num_seqs=8)
# 环境：VLLM_ROCM_USE_AITER=1   必需
```

```
Model loading took 20.03 GiB memory and 15.05 seconds    （每个 TP rank）
GPU KV cache size: 1,124,711 tokens
ENGINE LOADED in 70s
'The capital of France is' -> ' Paris. The capital of Spain is Madrid. The capital of Italy is Rome...'
```

vLLM 也正确识别了量化：`Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM`、
`Using DeepSeek's fp8_ds_mla KV cache format`、`Enabled custom fusions: norm_quant, act_quant, mla_dual_rms_norm`。

> **值得注意**：vLLM 能读这份**原生命名**的 checkpoint（它自己有映射），而 transformers 不能（§3）。
> 这两件事必须分开看。

### 2.1 四个必需项各自的原因（都是踩出来的）

| 配置 | 不设的后果 |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | `RuntimeError: Sparse attention indexer ROCm path is only supported on AITER. Please enable aiter with VLLM_ROCM_USE_AITER=1` —— ROCm 上 DSv4 的稀疏注意力索引器**只有 AITER 实现**，硬 `raise`，没有回退。⚠️ 这与 vime runbook 的 BF16 配方（`VLLM_ROCM_USE_AITER=0`）**相反** |
| `kv_cache_dtype=fp8_e4m3` | `AssertionError: DeepseekV4 FlashMLA fp8 layout only supports fp8 kv-cache, got auto` |
| `disable_custom_all_reduce=True` | TP≥2 时 `init_device` 阶段死在 `custom_all_reduce.py:297 create_shared_buffer` → `HIP error: invalid argument` |
| `max_model_len` 必须显式压小 | config 声明 1048576 个位置，让 vLLM 按那个尺寸算 KV cache 会 OOM（与本文要回答的问题无关的 OOM） |

### 2.2 TP=1 是不可用的：五种组合复现同一个 fault

```
Memory access fault by GPU node-2 (Agent handle: 0x...) on address 0x... Reason: Unknown.
```

发生在权重加载完成（155 GiB）之后的**第一次前向**（profiling dummy run）：

| TP | MoE backend | AITER | 结果 |
|---|---|---|---|
| 1 | 默认（fused） | 开 | Memory access fault（`module_gemm_a8w8_blockscale.so`，镜像 aiter） |
| 1 | vLLM `fp8_utils` W8A8 Block FP8 | 关 | Memory access fault |
| 1 | 同上 + `VLLM_USE_DEEP_GEMM=0` | 关 | Memory access fault |
| 1 | 仓库 aiter `blockscale_cktile` | 开 | Memory access fault |
| 1 | `triton_unfused` | 开 | **Memory access fault** ← 隔离实验，证明决定因素是 TP 而非 MoE backend |
| 1 | `triton_unfused` | 关 | raise（索引器要 AITER） |
| **8** | **`triton_unfused`** | **开** | ✅ 加载 + 生成正常 |

profiling batch 压到 2048 token 也一样。上游 PR #40889 的验证矩阵是 `TP2/h_q=64、TP4/h_q=32、TP8/h_q=16`
——**TP=1 从来没被验证过**。这个模型 64 个注意力头、`index_n_heads=64`、`o_groups=8`，TP=1 时某个
kernel 拿到了它处理不了的形状。

相关上游工作（都比我们这个构建新）：vLLM PR #49714（gfx950 上 AITER paged-MQA 导致 memory access
fault 的 sanitize 修复，描述与我们的现象高度相似但场景是并发混合负载）、#41601（ROCm 优化 + AMD
在 MI355X 上验证过的完整启动参数）、#40889（AITER MLA decode）、#41820（ROCm 使能跟踪）。

## 3. ❌ 训练侧：checkpoint 与 transformers 交集为 0

用真实 config 在 meta device 上实例化 `DeepseekV4ForCausalLM`，与权重索引做 key diff：

```
transformers 期望 1285 个张量, checkpoint 提供 72317 个
交集: 0
模型参数量: 284.3 B  (BF16 = 569 GB)
```

| 一侧 | 命名样例 |
|---|---|
| transformers 期望 | `model.embed_tokens.weight`、`model.layers.0.self_attn.q_a_proj.weight`、`model.layers.0.mlp.experts.gate_up_proj`（每层把 256 个专家融合成一个 3D 张量，所以只有 1285 个） |
| checkpoint 提供 | `embed.weight`、`head.weight`、`layers.0.attn.q_norm.weight`、`layers.0.ffn.experts.0.w1.weight`、`layers.0.hc_attn_base`、`mtp.0.*`（逐专家拆开，所以有 72317 个） |

`_checkpoint_conversion_mapping` 是 `None`，transformers 不做映射。而 LumenRL 的 FSDP2 路径正是
`AutoModelForCausalLM.from_pretrained`（`fsdp_backend.py:120`），所以**读不了**。

仓库里的 `inference/convert.py` 是 **HF → 本项目格式**的方向（README 原文 "First convert huggingface
model weight files to the format of this project"），没有反向转换器。同族 7 个仓库的 HF 风格张量数
**全部为 0**，也就是说**整个 V4 家族都没有 HF 格式的权重**。

**要补的转换器**：每专家的 `w1`/`w3` 拼成融合的 `gate_up_proj` 3D 张量、`w2` 转 `down_proj`、注意力和
`hc_*` 一批改名、丢掉 mtp 与 dspark 部分，还要把 FP4(e2m1) + FP8 块量化按 `ue8m0` scale 反量化成 BF16。

> 好消息：**架构本身不用写**。transformers 5.13.1 已经完整实现了 V4（`self_attn.compressor`、
> `hc_head`、`attn_hc` 等模块都在，纯 PyTorch，不依赖自定义内核）。FSDP2 缺的只有权重转换 + 显存。

## 4. ❌ Megatron 这条路更远

### 4.1 LumenRL 的 megatron 引擎当场 KeyError

`megatron_native_engine.py` 构造 `TransformerConfig` 时读 `hf["intermediate_size"]`，而 V4 的 config
**没有这个键**（只有 `moe_intermediate_size`）。而且它建的是标准 GQA（`num_query_groups`、`kv_channels`），
从头到尾没请求过 MLA。权重 bridge 也是 Qwen3 专用（`qwen3_megatron_bridge.py` / `qwen3moe_megatron_bridge.py`）。

### 4.2 镜像里的 megatron-core 0.16.0rc0 能表达一部分

| V4 机制 | megatron-core 0.16 |
|---|---|
| DeepSeek-V2/V3 式 MLA | ✅ `MLATransformerConfig`（`q_lora_rank` / `kv_lora_rank` / `qk_head_dim` / `qk_pos_emb_head_dim` / `v_head_dim`） |
| MTP | ✅ `mtp_num_layers` |
| 滑动窗口 | ✅ `window_size` |
| `noaux_tc` 路由 | ✅ `moe_router_enable_expert_bias` |
| `o_lora_rank` / `o_groups`（输出侧 LoRA） | ❌ |
| `index_n_heads` / `index_topk`（DSA 索引器） | ❌ |
| `num_hash_layers` / `hc_*`（mHC 哈希聚类） | ❌ |
| `compress_ratios`（逐层 KV 压缩） | ❌ |
| `dspark_*` | ❌ |
| `scoring_func=sqrtsoftplus` | ❌ 只有 `Literal['softmax','sigmoid']` |

⚠️ 那些 `hc_*` 不是可选项——checkpoint 里每层都有 `hc_attn_base` / `hc_attn_fn` / `hc_attn_scale` /
`hc_ffn_base` 实打实的权重。

### 4.3 上游 Megatron-LM 已经实现了，但在 dev 分支且是 NVIDIA 侧

NVIDIA 的跟踪 issue [#4468](https://github.com/NVIDIA/Megatron-LM/issues/4468) 状态 "Ready in Dev"：

- 混合 CSA/HCA 注意力（`DSv4HybridSelfAttention`，`--experimental-attention-variant dsv4_hybrid`、
  `--csa-window-size`、`--csa-compress-ratios`、`--dsa-indexer-n-heads/head-dim/topk`）：已合入 dev（PR #4458、#4894）
- hash-routed MoE + ClampedSwiGLU（`--moe-n-hash-layers`）：已合入 dev（PR #4481）
- mHC 与 MTP：已完成
- Megatron-Bridge：有 DSv4 配置和 HF↔Megatron 权重映射
- **未完成**：FP4 QAT "partially supported / needs DeepSeek-V4 specific validation"、FP8 indexer "In Progress"
  ——而 V4 的专家权重恰好是 FP4

**但 ROCm 上没有人做过 Megatron 的 DSv4 训练。** 搜到的 ROCm + DSv4 工作全是**推理**（vLLM #41820/#40889/#43718、
SGLang 的 AMD fork、ROCm/ATOM #707）。ATOM 的跟踪里连推理都还有不少通用回退：
`mHC kernel gfx950-tuned path（today: generic gfx942 fallback）`、`MLA sliding_window=128 passthrough
（HCA layers go through generic path today）`、`sqrtsoftplus CK kernel ... PR is currently Triton-only`。
训练侧的 CSA/indexer 反向内核是另一套代码，issue 里还引用了 cudnn-frontend。

而且升级 megatron-core 碰不到 LumenRL 那一半：引擎自己建 `GPTModel`/`TransformerConfig`、用自己的
bridge，既不调 Megatron 的训练脚本也不用 Megatron-Bridge（vime runbook §6 明确写了不要装
megatron-bridge）。再加上跳到 dev 分支会放大 megatron ↔ TE 的 API 漂移面——0.16 上已经踩到一个
（见 4 节点 runbook §9.2 的 `general_gemm(workspace=)` 补丁）。

## 5. ❌ 32 卡装不下 284B

| 项 | 数值 |
|---|---|
| BF16 权重 | 569 GB |
| FSDP2 训练态（权重 + 梯度 + fp32 master + Adam 两动量） | ≈ 4.55 TB → **142 GB/卡**（32 卡） |
| rollout 引擎，TP=1，保持发布时的 FP8/FP4 | **155 GiB/卡**（实测） |
| 合计 | **297 GB > 288 GB**，还没算激活和 KV cache |

两条能装下的路：

- **16 节点 / 128 卡**：训练态 4.55TB/128 ≈ 36 GB/卡；引擎那份拷贝不随节点数缩小。配 TP=8 后引擎
  约 20 GB/卡 → 合计约 56 GB，余量充足。`amd-burst-qos` 上限 128 节点，16 节点在配额内。
- **4 节点 + rollout TP=8**：引擎 20 GB + 训练 142 GB = 162 GB，账面能过。**前提是训练侧的权重
  转换器先写出来**（§3）。

## 6. 下一步的最短路径

> 要接着做的人：**先读 `deepseek-v4-base-rl-train-handoff.md`**，那份把下面这些拆成了带判据的步骤，
> 并且指出了一个本文没有强调的风险——RL 必须用的 **Base 版（FP8 专家）在 vLLM ROCm 上尚未使能**
> （上游 checklist 里那一项没打勾），所以第一步是花 1 小时验证它能不能被托管，而不是先写转换器。

1. **写 native→HF 权重转换器 + FP4/FP8 反量化**（§3）。这是唯一让 V4 进入训练的门。架构不用写，
   transformers 已有。
2. 用 **4 节点 + rollout TP=8** 先跑 smoke（§5），显存账面可行。
3. 若要真实吞吐再考虑 16 节点。

Megatron 路线在上面之外还要加"把 NVIDIA dev 分支的实现搬到 ROCm"和"改 LumenRL 的引擎与 bridge"，
少两层风险的是 FSDP2。

## 7. 产物位置

| 内容 | 路径 |
|---|---|
| 模型（166.9 GB，48 分片） | `/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-0731`（**node-local，只在 crsuse2-m2m-068 上**） |
| config + 权重索引（可离线分析） | `~/4node/dsv4/` |
| vLLM 加载探针 | `~/4node/probe_dsv4_vllm.py` |
| checkpoint 与 transformers 的 key diff | `~/4node/probe_dsv4_keys.py` |
| 六次尝试的完整日志 | `~/logs/4node/dsv4_vllm_tp1{,b,c,d,e,f}.log`、`dsv4_vllm_tp8{,b,c}.log` |
