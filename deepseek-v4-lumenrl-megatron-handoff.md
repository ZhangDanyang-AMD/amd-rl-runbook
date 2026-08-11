# 交接：在 LumenRL 的 megatron_native 上跑 DeepSeek-V4 RL

> 给下一个 agent。目标：`deepseek-ai/DeepSeek-V4-Flash-Base` 在 **LumenRL、megatron_native 训练、
> vLLM rollout** 上跑通 DAPO RL。
>
> **先读这一句：`LumenRL 的 Megatron` ≠ `miles 的 Megatron`。**
> LumenRL 的 `megatron_native` 用的是镜像里的 **stock megatron-core 0.16.0rc0**；
> miles 用的是 **`radixark/Megatron-LM@miles-main`**，比 NVIDIA 上游领先 339 个 commit。
> DeepSeek-V4 的 mHC（超连接）残差是 `[s, b, hc, d]` 四维流，stock Megatron 的
> `TransformerBlock` 契约是 `[s, b, h]`，**表达不了**——miles 之所以要 fork，首先就是为了这个。
>
> 所以"LumenRL 能跑 Qwen3-MoE + miles 能跑 DSv4 ⇒ LumenRL 能跑 DSv4"这个推理不成立。
> 中间隔着一次移植，规模见 §3 / §4。本文把要搬的、要改的、以及**必须先做的三个 go/no-go 探针**
> 列清楚。
>
> 姊妹文档：
> - `deepseek-v4-miles-megatron-4node-runbook.md` —— **另一条已跑通的路**。若目标变成"尽快拿到
>   DSv4 RL 结果"而不是"在 LumenRL 里跑"，直接看那份，重起 Ray 就能跑
> - `deepseek-v4-lumenrl-fsdp-handoff.md` —— FSDP2 那条路，**已否决**，理由见 §1
> - `rocm-gfx950-strided-bmm-oob-issue.md` —— gfx950 的 `torch.bmm` 越界，与训练后端无关
> - `dapo-lumenrl-4node-32gpu-runbook.md` —— 4 节点环境怎么搭

## 0. 一页现状

| 环节 | 状态 |
|---|---|
| vLLM 托管 Base（FP8 专家，TP=8） | ✅ 已验证，`moe_backend=triton`，新机器上复现过 |
| DSv4 权重能被读懂 | ✅ transformers 5.13.1 自带原生→HF 映射，全量 43 层验过，数值与 vLLM 对得上 |
| **stock Megatron 能否表达 mHC** | ❌ **不能。第一号 go/no-go**，见 §5.1 |
| mHC 的反向 | ❌ 只存在于 `tile_kernels`，**没有 PyTorch 回退** |
| LumenRL megatron 的权重加载 | ❌ 每 rank 读全量 + 峰值双倍 + **完全没有 FP8 反量化** |
| LumenRL megatron 的 optimizer offload | ❌ **不支持**，基类写死 `return False` |
| LumenRL megatron 的逐层异构 | ❌ 不支持，V4 有 3 种注意力 + 2 种 MLP |
| Megatron → vLLM 的名字映射 | ❌ **不存在**，miles 那份是发给 SGLang 的，命名不同，见 §4.6 |

**规模估计**：要搬的模型代码约 2,900 行 + Megatron fork 的 DSv4 增量约 500 行 + 一个
`miles_plugins` 形状的包 + `tile_kernels`/`tilelang` 依赖，再加 LumenRL 侧六项基础设施改造，
以及一份没人写过的 Megatron→vLLM 名字映射。**这是一个项目，不是一次接入。**

## 1. 为什么不是 FSDP2

FSDP2 本轮推到了"32 卡起得来、rollout 跑得通、训练步进得去"，然后停在两件结构性的事上：

1. **没有 EP，而跨节点没有 RDMA。** FSDP2 每层 all-gather 整层 256 个专家（13.2 GB），而每
   token 只用 6 个。43 层 ≈ 每次前向 426 GB 走跨节点链路，而这台集群只能走 ens3 上的 TCP
   （约 25 GB/s，8 张 400Gb/s 的 ionic RDMA 用不了）。**光前向权重 all-gather 就约 17 秒。**
   Qwen3-30B 能这么跑是因为它小 20 倍。
2. **加载装不下。** `_load_hf_model` 每 rank 在 CPU 上完整加载，`model_dtype` 写死 fp32，
   284B 要 9.1 TB/节点，机器只有 2.7 TB。

⚠️ 第 2 条在 megatron_native 上**不但没解决，还更重**（§4.1）。选 Megatron 的理由是 **EP**，
不是加载。

## 2. 两个 Megatron 的差异

`radixark/Megatron-LM@miles-main` 相对上游：**领先 339 commit、落后 1380 commit**，约 200 个文件。
这不是 DSv4 fork，是 miles 的通用 fork，DSv4 只是其中一个 PR。

### 2.1 DSv4 增量本身很小、可 cherry-pick

[radixark/Megatron-LM#28](https://github.com/radixark/Megatron-LM/pull/28)，**19 个文件，+511 / −146**：

| 文件 | +/− | 内容 |
|---|---|---|
| `transformer/transformer_layer.py` | 88/29 | 逐层 mHC 参数、HC pre/post 取代 bias-dropout-add、`input_ids` 透传给 hash 路由 |
| `transformer/experimental_attention_variant/dsa.py` | 99/38 | `DSAIndexer` 的 `dsv4_mode` 分支 |
| `transformer/moe/router.py` | 88/24 | `sqrtsoftplus`、hash 路由（`tid2eid`）、`moe_router_freeze_gate` |
| `transformer/transformer_config.py` | 59/7 | 10 个 `dsv4_*` 字段 + `dsv4_mode` |
| `tensor_parallel/mappings.py` | 59/9 | `all_reduce_grad_fp32`、`split_along_nth_dim` |
| `transformer/transformer_block.py` | 31/0 | 块边界的 HC expand/head、`hc_head_params` |
| `pipeline_parallel/schedules.py` | 8/2 | **4 维 PP 形状** |
| `pipeline_parallel/p2p_communication.py` | 5/4 | **4 元素形状交换** |
| 其余 11 个文件 | 约 60 | 单行改动 |

**4 维 PP 只有约 20 行，且全部由一个布尔量把守**：

```python
# schedules.py:1152 和 :2104
if config.dsv4_mode:
    tensor_shape = [seq_length, micro_batch_size, config.dsv4_hc_mult, config.hidden_size]
else:
    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]

# p2p_communication.py:184
num_dims = 4 if config.dsv4_mode else 3
```

展开/收拢在 block 边界完成（`transformer_block.py:829` `block_expand`、`:917` `block_head`），
**只有 PP 的 send/recv 看得见 4 维张量**。`dsv4_mode` 在 fork HEAD 上只出现在 8 个文件的 16 处。

### 2.2 但 fork 的核心 import 了 miles

fork 的 Megatron 核心文件直接 import miles 的包（都是函数内 import，所以 `import megatron`
本身不挂，但代码路径会）：

| fork 文件:行 | import |
|---|---|
| `transformer/transformer_block.py:452` | `miles_plugins.models.deepseek_v4.ops.hyper_connection` |
| `transformer/transformer_layer.py:641, 838, 972` | 同上 |
| `.../experimental_attention_variant/dsa.py:783-785, 863-864` | `miles_plugins.models.deepseek_v4.ops.{compressor,utils,qat,cp_utils,ref_model}` |
| `.../dsa.py:802, 947` | `miles.utils.replay_base.indexer_replay_manager` |
| `transformer/moe/moe_utils.py:702` | `miles.utils.replay_base.routing_replay_manager` |
| `transformer/moe/router.py:216` | 同上 |
| `training/training.py:1293` | `miles.backends.megatron_utils.fp32_param_utils` |

⚠️ **`moe_utils.py:702` 在 `topk_routing_with_score_function` 里，那是每个 MoE 模型每次前向都走的
路径**——整体拿这个 fork 的话，连 Qwen3-30B-A3B 都需要 `miles.utils.replay_base` 存在。

另有约 20 处模块级 import `miles_megatron_plugins.true_on_policy.*`，那个包**内嵌在 fork 里**，
跟着 clone 走，不用额外处理，但说明这个 fork 结构上是 miles 的产物。

⚠️ **版本偏斜陷阱**：fork HEAD import 了 `ops.ref_model.apply_rotary_emb` 和
`ops.utils.wrapped_precompute_freqs_cis`，但 8 月 3 日的 miles 快照里没有 `ref_model.py`，
且 `wrapped_precompute_freqs_cis` 在 `rope.py` 不在 `utils.py`。**fork 和 miles 必须成对锁版本。**

**实务判断**：cherry-pick 那 500 行增量到 LumenRL 已有的 Megatron 上，比整体接管 fork 便宜得多。

## 3. 要从 miles 搬的东西

### 3.1 模型插件（约 2,396 行）

全部在 `miles_plugins/models/deepseek_v4/`，通过 `--spec` 挂进 Megatron：

```
--spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec
```

| 文件 | 行数 | 内容 |
|---|---|---|
| `deepseek_v4.py` | 358 | `DeepSeekV4Attention` 全套 + `get_dsv4_spec`（猴补 Megatron 的 variant 分发） |
| `ops/compressor.py` | 183 | CSA（ratio 4，含 overlap 变换）和 HCA（ratio 128）压缩器 |
| `ops/hyper_connection.py` | 217 | mHC，`tile_kernels.modeling.mhc.ops` 的薄封装 |
| `ops/v4_indexer.py` | 136 | ratio-4 层的 DSA indexer |
| `ops/cp_utils.py` | 130 | CP 的 topk 索引 / rope 切片（只支持连续 CP） |
| `ops/rope.py` | 97 | YaRN 频率预计算 + 原地复数 RoPE |
| `ops/qat.py` / `ops/utils.py` | 36 / 15 | FP8 QAT 直通反向；Hadamard 变换 |
| `ops/kernel/tilelang_sparse_mla_{fwd,bwd,}.py` | 196+288+31 | 稀疏 MQA 前向/反向/autograd 绑定 |
| `ops/kernel/tilelang_indexer_{fwd,bwd,}.py` | 194+237+93 | indexer 前向/反向/autograd 绑定 |
| `ops/kernel/act_quant.py` | 137 | FP8 E4M3 逐块激活量化（ue8m0 scale） |
| `ops/kernel/precision_aligned_ops.py` | 46 | `linear_bf16_fp32`，**与 SGLang 逐位对齐的契约** |

目录外：`miles_plugins/mbridge/deepseekv4.py`（123 行，离线 HF↔mcore 名字映射）、
`miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py`（156 行，运行时 mcore→HF）、
`tools/param_name_remap.py`（19 行）、`scripts/models/deepseek-v4-flash.sh`（85 行 MODEL_ARGS）。

### 3.2 tilelang / tile_kernels：mHC 是硬门

两个不同依赖：`tilelang`（miles 自己 kernel 用的 DSL，pin 0.1.8，实跑 0.1.13）和
`tile_kernels==1.0.0`（DeepSeek 预编译 kernel 包）。

| op | 提供方 | 前/反向 | 回退 |
|---|---|---|---|
| 稀疏 MQA 注意力 | miles tilelang | 都要 | **无**，`deepseek_v4.py:308` 无条件调用 |
| DSA indexer 打分 | miles tilelang | 都要 | **有** —— `V4_INDEXER_IMPL != "tilelang"` 走 Megatron 的稠密纯 PyTorch `DSAIndexer`，代价 O(S²) 显存 |
| **mHC pre/post/head** | **tile_kernels** | **都要** | **无** |
| FP8 QAT cast-back | tile_kernels | 仅前向 | 只能关掉 FP8 |
| Hadamard rotate | `fast_hadamard_transform` | 前向 | 无 |

**mHC 的反向只存在于 tile_kernels**，树内那份 legacy 实现已删，只剩 `hyper_connection.py`
开头的注释为证（"the legacy in-tree implementation only had a no-grad forward path"）。

好消息是分解写得很清楚：RMS-norm 后投影成 mixes → 拆成 `pre_mix` / `post`（sigmoid ×2）/
`comb` → 对 `comb` 做 20 轮 Sinkhorn 归一化 → apply。**这套用 PyTorch autograd 完全写得出来，
代价是吞吐不是正确性。** 而且训练路径本来就已经走非融合版本（`hyper_connection.py:81-111`，
因为 Megatron 的调用点破坏了 tile_kernels `fuse_grad_acc=True` 的存储假设），移植难度比想象低。

gfx950 上还需要三个补丁（都在 miles 的 `docker/amd_patch/latest/`）：`tile_kernels.patch`
（36 行，去掉 Hopper 专属的 `T.pdl_sync()`）、`tilelang_hip_fp8.patch`（32 行）、以及
`~/dsv4/01_fix_tile_kernels.sh:33-42` 的 `determine_target` 兼容垫片。
**tilelang 0.1.10 不可用**——HIP codegen 降不了 `exp2`，而稀疏 MLA 反向需要。

### 3.3 权重怎么进 Megatron：miles 的两阶段路径

⚠️ **反直觉的一点：原生→HF 的改名发生在第 1 阶段，不是第 2 阶段。**

1. `tools/fp8_cast_bf16.py` —— 反量化 FP8 块量化**同时**给每个张量改名（改名函数就是 SGLang
   那个 `remap_weight_name_to_dpsk_hf_format`，靠嗅探 `embed.weight` 判断是不是原生格式）。
   产物已经是 "dpsk HF format"。
2. `tools/convert_hf_to_torch_dist.py` —— 建 Megatron 模型、经 mbridge 加载，产出 mcore 命名的
   分片张量（`decoder.layers.N.self_attention.wq_a.weight` 这类）。映射表在
   `miles_plugins/mbridge/deepseekv4.py:15-71`；`_weight_to_mcore_format` 会抑制基类的 bf16 降精度，
   这是 `attn_sink` / `compressor.ape` / `hc_*` 能保持 fp32 的原因。

产物尺寸（都在节点本地盘，每台一份）：`DeepSeek-V4-Flash-FP8` 274G（SGLang 用）、
`-bf16` 542G（只是中间产物，不分发）、`_torch_dist` 530G（Megatron 用，每个 rank 都读）。

两个有用的性质：

- **`torch_dist` 加载时会重新分片**——写出来是 `[t 1/1, p 1/8]`，加载成 `[t 1/8, p 1/4]`。
  **转换时的并行度不约束训练时的并行度**，转换用 TP=PP=EP=1 就行。
- ⚠️ **转换陷阱**：`convert_hf_to_torch_dist.py:58-64` 在 `pipeline_model_parallel_size == 1 且
  world_size > 1` 时会**静默把 PP 改写成 world_size**，叠加 `_prepare_spmd` 硬编码的 EP=8，
  Megatron 会拿 `ETP × EP × PP = 64` 去校验 `world_size=8` 然后挂掉。miles 的解法是转换时用 EP=1。

**对我们的启示**：既然 transformers 5.13.1 能直读原生 checkpoint 并做反量化（§8），第 1 阶段
可以不依赖 SGLang——但要注意 transformers 的目标命名和 SGLang 的"dpsk HF format"**不一致**，
mbridge 的映射表是照着后者写的，直接换会全部对不上。要么沿用 SGLang 那份改名（vendor 过来，
它是纯字符串函数），要么把 mbridge 的映射表改成对齐 transformers 的命名。

## 4. LumenRL 侧要改的东西

来自对 `lumenrl/engine/training/megatron_native_engine.py` 的通读，按阻塞程度排序。

### 4.1 加载：每 rank 读全量，峰值双倍，且无 FP8 反量化

```python
# qwen3_megatron_bridge.py:125-138
def load_hf_safetensors(model_dir: str) -> dict[str, torch.Tensor]:
    for f in files:
        state.update(load_file(f))
    return state
```

没有 rank 判断、没有 `safe_open` 流式、没有按 key 过滤，全仓库**没有 `mbridge`**。
更糟的是 MoE 分支在 `hf_state` 还活着时又构造了完整的非专家 Megatron dict
（`megatron_native_engine.py:449`），**峰值双倍**。284B bf16 ≈ 568 GB/rank × 8 = 4.5 TB/节点。

**而且完全没有 FP8 反量化。** V4 是原生 FP8 块量化，裸 `load_file` 拿到 `float8_e4m3fn` 后
`_shard_hf_for_moe` 直接 `.to(torch.bfloat16)`（476、489 行），**无视 `_scale_inv`，得到静默
错误的权重**；那些 scale key 没有映射会被丢弃，`load_state_dict(strict=False)` 也不报警。
FSDP2 那边已解决（`FineGrainedFP8Config(dequantize=True)`），可照搬思路。

### 4.2 没有 optimizer offload

```python
# megatron_base_engine.py:106-119
@property
def is_optimizer_offload_enabled(self) -> bool:
    return False
```

`OptimizerConfig` 构造里没有任何 offload 参数。`param_offload` / `optimizer_offload` /
`grad_offload` 三个字段 `actor_worker.py:204-206` 确实转发了，但引擎从不读——**纯死配置**，
而且 `MegatronConfig` dataclass 里根本没声明它们。

284B 的 FP32 master + Adam 双动量约 3.4 TB，只靠 distributed optimizer 按 DP 切分不够。
**miles 正是靠 `--optimizer-cpu-offload --optimizer-offload-fraction 0.75 --no-offload-train`
才在 4 节点上装下的**（§6）。这一项必须补。

### 4.3 不支持逐层异构

`TransformerConfig` 用单一 `num_layers`，layer spec 由 `get_gpt_decoder_block_spec` 生成**同质**层。
`moe_layer_freq` 和 `first_k_dense_replace` **全仓库零命中**。

bridge 侧更硬：`_non_expert_hf_to_megatron` 对每一层无条件索引同一组 key
（`qwen3moe_megatron_bridge.py:95-109`）。V4 的 43 层有 3 种注意力 + 2 种 MLP，`hash_moe` 层
没有 `mlp.gate.weight` → 立刻 KeyError。反方向 `megatron_to_hf_moe` 的 if/elif 链不认识的 key
**静默丢弃**（172 行 `continue`）。需要引入 Megatron 的 `TransformerBlockSubmodules`
（逐层 spec 列表），引擎目前完全没有这个概念。

### 4.4 注意力形状全是 GQA 假设

`TransformerConfig`（230-252 行）硬编码 `num_query_groups=hf["num_key_value_heads"]`、
`qk_layernorm=True`、`kv_channels=head_dim`，**没有任何按 model_type 分支**。
`_hf_qkv_to_megatron` 要求 `q/k/v_proj` 三件套并按 KV group 交织，对 V4 的低秩分解完全不适用。

另外每次前向强制 `qkv_format="thd"` 打包 TE 注意力（`megatron_native_engine.py:823-835`），
**没有 eager 回退**。`TEDotProductAttention` 能否吃 head_dim 512 + per-head sink 是前置项。

### 4.5 没有按 model_type 的分发

`EngineRegistry` 的 `model_type` 是**角色**（`language_model` / `value_model`）不是架构。
bridge 是文件顶部**静态 import**，分发逻辑就是 `self._is_moe` 一个布尔量（三处）。
新模型族**没有 registry 可注册**。

### 4.6 ⚠️ Megatron → vLLM 的名字映射不存在

**最容易被低估的一项。** miles 的 `megatron_to_hf/deepseekv4.py` emit 的是 "dpsk HF format"
（`model.layers.N.self_attn.wq_a.weight` 这类），那是发给 **SGLang** 的。而 **vLLM 直读原生
checkpoint、保留原生模块名**。本轮实测的 vLLM 侧参数名（完整清单
`~/logs/4node/vllm_dsv4_param_names.txt`，134 个参数 / 39 个 loader 模块）：

| transformers / SGLang 侧 | vLLM 侧 |
|---|---|
| `self_attn.sinks` | `attn.attn_sink` |
| `self_attn.q_a_proj` + `self_attn.kv_proj` | **`attn.fused_wqa_wkv`**（两个融成一个） |
| `self_attn.q_b_proj` / `o_a_proj` / `o_b_proj` | `attn.wq_b` / `wo_a` / `wo_b` |
| `input_layernorm` | `attn_norm` |
| `mlp.experts.gate_up_proj`（3D） | `ffn.experts.routed_experts.w13_weight` |
| `mlp.shared_experts.gate_proj` + `up_proj` | **`ffn.shared_experts.gate_up_proj`**（两个融成一个） |
| （BF16，无 scale） | 每个权重还有 `weight_scale_inv`，vLLM 保持 FP8 |

FSDP2 那条路就是死在这里：`Worker failed with error ''layers.0.self_attn.sinks''`。
Megatron 会撞同一堵墙，**而且 miles 那份映射帮不上忙，方向不对**。

还要处理三个**原子更新组**（miles `megatron_to_hf/deepseekv4.py:6-23`），因为接收侧会融合：
`wqkv_a`、`compressor_wkv_gate`、`indexer_compressor_wkv_gate`。vLLM 的融合点和 SGLang 不同
（vLLM 是 `fused_wqa_wkv`），要重新推。

### 4.7 router 语义不匹配

1. `moe_router_pre_softmax=False` 的等价性论证（`megatron_native_engine.py:195-205` 注释）
   **只对 Qwen3 的 full-softmax → top-k → renorm 成立**。DeepSeek 用 sigmoid scoring +
   group-limited top-k，论证不成立。误用会导致 gate 权重不归一化 → rollout/train log-prob 错位。
2. **`e_score_correction_bias` 没有加载路径**，`moe_router_enable_expert_bias` 从未设置。
3. **group-limited routing（`n_group` / `topk_group`）完全没有配置字段。**

V4 的 `scoring_func` 是 `sqrtsoftplus`，fork 的 `router.py` 专门加了这个分支，stock 没有。

### 4.8 其它

- **shared expert 命名**：bridge 读**单数** `mlp.shared_expert.*`（Qwen2-MoE 风格），DeepSeek 用
  **复数** `mlp.shared_experts.*`。而且分支是 `if ... in hf` 保护的，**不报错，静默跳过**
  ——shared expert 完全不加载，输出全错但无提示。
- **RoPE 无 scaling**：`rope_scaling` 在 megatron 路径零命中。V4 用 YaRN。
- **hyper-connection 参数是 fp32-pinned 的**，而引擎把一切强制 bf16，会静默降精度。
- **`moe_grouped_gemm: true`（默认）走 `TEGroupedLinear` 的 batched GEMM**，gfx950 上
  `torch.bmm` 对交错批视图有越界 bug（见 `rocm-gfx950-strided-bmm-oob-issue.md`），同类风险存在。
  降级手段是 `moe_grouped_gemm: false`（bridge 的 `_EXP_SEQ` 正则已支持这种命名）。
- **权重同步要把完整 284B gather 到每个 rank 的 GPU 上**（`_full_megatron_named_params_moe`，
  `stage` 和 `out` 都在 cuda），几乎肯定 OOM，需要改成流式/分块 gather + 立即发送。

## 5. ⚠️ 先做这三个 go/no-go 探针，再写任何 bridge

**如果 5.1 无解，整个 megatron_native 方案要重新评估，后面全是白做。**

### 5.1 mHC 能不能在 LumenRL 的 Megatron 上表达

- **A（推荐）**：把 §2.1 那 500 行增量 cherry-pick 到镜像里的 megatron-core 0.16.0rc0。
  两边都是 0.16 系，冲突可控。验证点：`dsv4_mode=True` 时 PP 的 4 维 send/recv 能否跑通。
  若 LumenRL 自己算 PP 张量形状（而不是委托 Megatron 的 `get_tensor_shapes`），这段要自己复刻。
- **B**：不碰 PP（`pipeline_model_parallel_size=1`），让 4 维流只存在于单 stage 内部。
  §2.1 里 PP 那 13 行就不需要，但 43 层 284B 在 PP=1 下的显存要重算——miles 用的是 PP=4。

**判据**：4 层裁剪模型（`~/dsv4/make_native_subset.py`，`DSV4_SUBSET_LAYERS=4`）前向 + 反向
不报错、梯度有限。

### 5.2 mHC 的反向要不要自己写

把 `hyper_connection.py` 那套分解用 PyTorch autograd 重写，和 `tile_kernels.modeling.mhc` 对拍。
**判据**：前向相对误差 < 1e-3，反向梯度一致。通了就摆脱 tile_kernels 依赖，代价是吞吐；
不通就必须把 tile_kernels + 两个 gfx950 补丁一起装进 LumenRL 的镜像。

### 5.3 稀疏注意力和 TE 的兼容性

`TEDotProductAttention` 能否吃 head_dim 512 + per-head learnable sink。稀疏 MQA 没有回退
（必须 tilelang），但 DSA indexer 有稠密 PyTorch 回退（`V4_INDEXER_IMPL != "tilelang"` +
`--miles-dsa-topk-backend torch`），可以先用回退把管路跑通再谈性能。
**判据**：4 层模型上 attention 前向数值与 transformers 的 eager 实现对得上。

## 6. miles 那份已跑通的配方（参照与目标）

`scripts/amd/run_deepseek_v4.py:279-290`。⚠️ **这是唯一存在的非单节点配置**，
`_get_parallel_config` 对其它任何组合抛 `NotImplementedError`：

```
--tensor-model-parallel-size 8 --sequence-parallel
--pipeline-model-parallel-size 4
--decoder-first-pipeline-num-layers 11 --decoder-last-pipeline-num-layers 10
--context-parallel-size 1
--expert-model-parallel-size 8 --expert-tensor-parallel-size 1
```

43 层 = 11+11+11+10。CP 硬编码为 1（zigzag CP 对 DSv4 不支持，CP>1 必须 `--allgather-cp`）。

**优化器与 offload**（`0.75` 和 `--no-offload-train` 只在 4 节点时施加）：

```
--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--optimizer-cpu-offload --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d
--optimizer-offload-fraction 0.75 --no-offload-train
--accumulate-allreduce-grads-in-fp32
```

**批量与序列**：`--rollout-batch-size 32 --n-samples-per-prompt 8 --micro-batch-size 1
--max-tokens-per-gpu 2048 --rollout-max-response-len 4096 --recompute-granularity full
--recompute-method uniform --recompute-num-layers 1`

**gfx950 必需的精度与开关**：

```
--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise
--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}'   # gfx950 要求 fp32 scale
--no-gradient-accumulation-fusion                                # ROCm TE MoE FP8 缺融合 wgrad
--qkv-format bshd --moe-router-freeze-gate --freeze-e-score-correction-bias
--attention-softmax-in-fp32 --deterministic-mode --use-rollout-routing-replay
```

**实测 step 0**：`grad_norm 0.1061`、`train_rollout_kl 0.0707`、`logprob_abs_diff 0.1529`、
`step_time 1306.6 s`。⚠️ **step 0 的 rollout 只用了 85 s，但 step 1/2 分别是 1473 s 和 1397 s**
（引擎冷了），`wait_time_ratio` 到 0.84。**按 40 分钟/步估，不是 21 分钟。**

⚠️ `train_rollout_kl = 0.07` 比 Qwen3-8B BF16 基线的 1.2e-3 高两个数量级。首要嫌疑是
**AMD 分支丢掉了 TE 精度覆盖 YAML**——NVIDIA 版把 DSA indexer 的 `linear_weights_proj` 在 FP8
训练下钉成 BF16（`scripts/run_deepseek_v4.py:59-70`），AMD 版既没有这个常量也没有
`--te-precision-config-file`。**移植时不要照抄 AMD 分支的这个遗漏。**

另外：稀疏 MLA 和 indexer 反向用 `atomic_add` 累加到 fp32，**本质不确定**，
`--deterministic-mode` 也管不住。

## 7. 建议的推进顺序

1. **§5 的三个探针。** 先做 5.1，它决定整条路生死。
2. **§4.1 + §4.2**：分片流式加载 + FP8 反量化 + optimizer offload。与模型结构无关的纯基础设施，
   可并行推进，FSDP2 路径同样受益。
3. **§4.3**：逐层异构 spec，这是写 bridge 的前提。
4. **§4.5**：引入按 `model_type` 分发的 bridge registry。
5. **搬模型插件（§3.1）+ cherry-pick Megatron 增量（§2.1）。**
6. **写 `deepseek_v4_megatron_bridge.py`，以及 §4.6 那份 Megatron → vLLM 名字映射**
   （没有现成实现可抄，vLLM 侧名单在 `~/logs/4node/vllm_dsv4_param_names.txt`）。
7. **§4.7 router 语义**，用 `rollout_corr/kl` 验证 train/rollout 对齐。
8. **§4.8 最后一条**：`_full_megatron_named_params_moe` 改成流式 gather。

## 8. 不要重做的事（都有实测依据）

- **不要再试 FSDP2 跑 DSv4。** 理由见 §1，两条都是结构性的。
- **不要再试 rollout TP=1。** TP=1 在 gfx950 上必然 `Memory access fault`，五种 MoE backend /
  AITER 组合全部复现过。
- **不要写 native→HF 权重转换器。** transformers 5.13.1 自带，全量 43 层验过，1537 个张量
  missing/unexpected 全为 0，数值与 vLLM 对得上。详见 `deepseek-v4-lumenrl-fsdp-handoff.md` §2。
- **不要照抄 SGLang 的 `remap_weight_name_to_dpsk_hf_format` 当作 HF 映射。** 实测它的目标命名和
  transformers **不一致**——保留 `wq_a`/`wkv` 这类论文记号，只做前缀改写。
- **不要把 vLLM 的 `prompt_logprobs` 当金标准。** 实测它在这个模型上有位置会给出完全不合理的分布。
- **不要在 instruct/thinking 版上训练。** 必须用 Base。
- **不要再排查跨节点 RDMA。** 走 ens3 上的 TCP（`NCCL_IB_DISABLE=1`）能跑通，坑记在 4 节点
  runbook §6。
- **不要 fork `run_dapo.sh`。** 用 `~/4node/bin/ray` 那个 PATH shim 吞掉开头会拆集群的 `ray stop`。

## 9. 本轮已完成、可直接用的东西

### 9.1 环境（spur JobID 43970，`crsuse2-m2m-[029,172,206,272]`，head = 029 / 10.245.146.21）

- 四台 `rl-vime-4node` 容器都起着（`bash ~/4node/01_container.sh`）
- **每台本地盘各有一份完整 275G 权重**：`/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-Base`
  （46 分片。作业没了要重下，约 12 分钟，命令见 `deepseek-v4-base-rl-train-handoff.md` §2，
  记得带 `HF_HUB_DISABLE_XET=1` 和 `--max-workers 16`）
- **每台还有一份 4 层切片**（27G）：`/mnt/m2m_nobackup/xysheng/models/dsv4-native-L4`，覆盖
  sliding / CSA+indexer / HCA 和 hash_moe / moe 全部分支，做探针用它
- Ray 用 `~/4node/02_ray_start_dsv4.sh head|worker` 起（与 Qwen3 那份唯一区别是
  `VLLM_ROCM_USE_AITER=1`，V4 的稀疏注意力索引器在 ROCm 上只有 AITER 实现）

⚠️ **`/mnt/m2m_nobackup` 是节点本地盘，作业一没就够不着。** 共享卷 `/home` 上没有备份。
另外 **GPU memory fault 会往进程 cwd 写 core dump，单个 50–80 GB**——本轮排查 bmm bug 时
一口气写了 215 GB，把共享卷从 997G 打到 611G。跑可能崩 GPU 的东西前先 `cd` 到 `/mnt/m2m_nobackup`。

### 9.2 已提交的代码（`ZhangDanyang-AMD/Lumen-RL`，`dev/vllm-fsdp-dapo`）

- `99cdccf` —— rollout TP>1（六个文件）+ `megatron_te_gemm_compat.py` + `moe_backend` 透传。
  ⚠️ `megatron_te_gemm_compat` **megatron 路径必须**：megatron-core 0.16.0rc0 给 TE 2.12 的
  `general_gemm` 传了已被移除的 `workspace=`，只在 MoE router 触发，dense 模型看不到。
- `0138104` —— DSv4 的 FSDP2 加载改动 + **TP>1 的 rank→节点固定** + `VLLMConfig.moe_backend`。
  其中两项对 Megatron 路线同样有用：`ray_worker_group.py` 的 STRICT_SPREAD placement group
  （rank 0-7 同机，TP=8 rollout 的前提，与训练后端无关）和 `VLLMConfig.moe_backend`
  （DSv4 必须传 `triton`）。`fsdp_backend.py` 那 122 行是 FSDP 专属、对 Megatron 惰性，
  但里面记录的 gfx950 bmm 两面性（前向要 contiguous、反向不能 contiguous）与后端无关。

### 9.3 探针与产物

| 内容 | 位置 |
|---|---|
| **vLLM DSv4 参数名清单**（§4.6 的依据） | `~/logs/4node/vllm_dsv4_param_names.txt`、探针 `~/dsv4/probe_vllm_dsv4_param_names.py`（需 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`） |
| 切 N 层原生副本 | `~/dsv4/make_native_subset.py` |
| 权重加载验收（双向 key diff + 前向） | `~/dsv4/probe_dsv4_load_native.py` |
| 与 vLLM 的数值核对 | `~/dsv4/refcheck_dsv4_{vllm,hf}.py`、`refcheck_compare.py`、`refcheck_decode.py` |
| 单卡训练步（前向+反向）探针 | `~/dsv4/probe_dsv4_train_step.py`（支持 6 种 grouped-linear 变体） |
| gfx950 bmm 最小复现 | `~/dsv4/probe_rocm_grouped_bmm.py` |
| rollout 验证 | `~/4node/probe_dsv4_vllm.py` |
| Megatron fork 缓存 | `/tmp/megfork/Megatron-LM-miles-main`（临时盘，建议重新 clone） |

### 9.4 没做完的

- 4 层 smoke 在 FSDP2 上停在 §4.6 那个命名错位（按决定不再推进）
- `~/dsv4/probe_dsv4_train_step.py` 的 `contig_x` 变体在 6144 长度下的结果没跑完——它能把
  `rocm-gfx950-strided-bmm-oob-issue.md` 里"没区分是哪个操作数触发"那句补完整
