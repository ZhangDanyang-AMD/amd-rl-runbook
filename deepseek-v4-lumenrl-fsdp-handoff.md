# 交接：用 LumenRL FSDP2 + vLLM 跑 DeepSeek-V4 RL

> 给下一个 agent。目标：`deepseek-ai/DeepSeek-V4-Flash-Base` 在 **LumenRL、FSDP2 训练、
> vLLM rollout** 上跑通 DAPO RL。
>
> **rollout 那半已经验证过了,不用再碰**（vLLM TP=8 可加载可生成,见
> `deepseek-v4-base-rl-train-handoff.md` §3.1）。
>
> **本文原本认为"唯一的门"是训练侧的权重转换——那个门已经开了,而且根本不用写转换器：**
> `transformers 5.13.1` 自带 DeepSeek-V4 原生 → HF 的完整映射,`from_pretrained` 直读原生
> checkpoint。全量 43 层实测 1537 个张量、missing/unexpected 全为 0、4096 前向干净,
> 与 vLLM 同 prompt 的 logits 也对得上（top-1 一致 91.4%,贪心续写 2/4 逐 token 相同）。详见 §2。
>
> **换上来的新门是一个 gfx950 的 `torch.bmm` 越界 bug**,序列长度进 2040 以上就会安静地产出
> NaN,4k 回复长度正中靶心。有一行补丁,但打的时机有讲究。见 §2.6 与
> `rocm-gfx950-strided-bmm-oob-issue.md`。
>
> 姊妹文档,按需读,不用全读：
> - `rocm-gfx950-strided-bmm-oob-issue.md` —— **上面那个 bmm bug 的完整复现与修法**,先读它
> - `deepseek-v4-base-rl-train-handoff.md` —— rollout 侧的实测结论（§3.1）与显存账（§5）
> - `deepseek-v4-miles-megatron-4node-runbook.md` —— **另一条已跑通的路**（miles + Megatron +
>   SGLang）。如果你要的是"尽快拿到 DSv4 RL 训练结果"而不是"在 LumenRL 里跑",直接看那份
> - `dapo-lumenrl-4node-32gpu-runbook.md` —— 4 节点环境怎么搭（本文引用它的脚本）

## 0. 一页现状

| 环节 | 状态 |
|---|---|
| vLLM 托管 Base（FP8 专家,TP=8） | ✅ **已验证**,`moe_backend=triton`,35.15 GiB/rank |
| transformers 侧的 V4 架构实现 | ✅ **已验证完整且纯 PyTorch**,零自定义内核,见 §2.1 |
| 原生 checkpoint → HF 布局 | ✅ **transformers 自带,不用写转换器**,全量 43 层已验收,数值也和 vLLM 对上了,见 §2.2 / §2.5 |
| LumenRL FSDP2 读 HF 权重 | 通路现成,但**必须显式传 `dequantize=True`**,见 §2.4 |
| gfx950 `torch.bmm` 越界 | ❌ **新的门**,≥2040 token 出 NaN,有一行补丁,见 §2.6 |
| MoE 权重同步到 vLLM | ⚠️ 布局天然对得上,还剩一处小活,见 §4 |
| 32 卡显存 | ⚠️ 账面可行,激活是未知量,见 §3 |

**为什么是 FSDP2 而不是 Megatron。** 结论来自实测,不是偏好：

| | FSDP2 + vLLM（本文） | Megatron + vLLM |
|---|---|---|
| Megatron fork（`radixark/Megatron-LM@miles-main`） | 不需要 | 必需,其 mHC 改动把 PP buffer 全局改成 4 维 `[s,b,hc,d]` |
| tilelang + tile_kernels | 不需要 | 必需（mHC backward 只在 tile_kernels 里,无回退） |
| 移植 miles 的 DSv4 插件 | 不需要 | 约 1500 行,且精度选择是承重的 |
| 权重映射 | 一份 | 两份（离线 mbridge + 运行时 megatron_to_hf） |
| LumenRL 侧改造 | 小 | 大：**没有架构注册表**;`load_hf_safetensors` 把所有分片读进**每个 rank 的 CPU 内存**（284B 下必爆） |
| 与 vLLM 的关系 | 天然（专家已是融合 3D） | Megatron 发逐专家 2D 名,绕过 `FusedMoEWeightRouter` |

代价只有一个：transformers 的注意力走 `ALL_ATTENTION_FUNCTIONS` / `eager_attention_forward`,
是**稠密注意力**,DSv4 稀疏注意力（`index_topk=512`）的收益拿不到。4k 回复长度下可接受,
**这条路不适合长上下文**。

## 1. 环境准备

**沿用 vime 那条已验证的线,不要新建环境。** 需要的东西镜像里全都有,不用装 tilelang、
不用 Megatron fork。

```bash
# 环境变量与节点清单（换机只改这一份）
source ~/4node/env.sh          # HEAD_NODE / HEAD_IP / NODES / CONTAINER=rl-vime-4node
bash   ~/4node/01_container.sh  # 四台各起一次；含 --ulimit memlock=-1 与 ionic provider
bash   ~/4node/02_ray_start.sh head|worker
```

镜像 `vllm/vime-rocm` 里的实测版本：

```
torch 2.10.0+rocm7.0 · vllm 0.22.1rc1.dev392+g43914dd74.rocm702
transformers 5.13.1  ← 关键：它自带完整的 deepseek_v4 实现
megatron-core 0.16.0rc0（本路线不用）· mbridge 0.15.1（不用）
```

⚠️ **`~/4node/ray_env.sh` 要为 DSv4 单开一份副本再改。** `VLLM_ROCM_USE_AITER` 在 Qwen3 BF16
配方里必须是 `0`,而 DSv4 的稀疏注意力索引器在 ROCm 上**只有 AITER 实现**,关掉会硬 `raise`。
这两个要求冲突,别把已验证的 Qwen3 那条线改坏。

模型放在 head 的本地盘上（275G,46 分片,`expert_dtype: fp8`）：

```
/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-Base
```

⚠️ **`/mnt/m2m_nobackup` 是节点本地盘,作业一没就够不着了。** 共享卷 `/home` 上没有备份,
也不要往那儿放（10T 全组共享,余量在 0.6–1.3T 之间浮动）。作业换了就得重下,约 12 分钟
（400–680 MB/s）,命令见 `deepseek-v4-base-rl-train-handoff.md` §2 —— 记得带
`HF_HUB_DISABLE_XET=1` 和 `--max-workers 16`。

四节点训练时每台都要有一份。节点间**没有 ssh**（`Permission denied (publickey)`）,
分发方式见 miles runbook §4 的 rsyncd 办法。

⚠️ 另外注意：**GPU memory fault 会往进程 cwd 写 core dump,单个 50–80 GB。** 容器里 cwd 通常是
`/home/$USER`,也就是 NFS。排查 §2.6 那个 bmm bug 时我一口气写了 10 个、共 215 GB,把共享卷从
997G 打到 611G。跑任何可能崩 GPU 的东西之前,先 `cd` 到 `/mnt/m2m_nobackup` 下。

**先复现一次 rollout 验证**（约 4 分钟,确认环境没退化）：

```bash
~/nx.sh <head> 'source /home/$USER/4node/env.sh
  docker exec -d "$CONTAINER" bash -lc "
    source /home/$USER/4node/ray_env.sh
    unset HIP_VISIBLE_DEVICES
    export VLLM_LOGGING_LEVEL=INFO VLLM_ROCM_USE_AITER=1
    export HF_HOME=/mnt/m2m_nobackup/$USER/hf_home
    DSV4_PATH=/mnt/m2m_nobackup/$USER/models/DeepSeek-V4-Flash-Base \
    DSV4_TP=8 DSV4_UTIL=0.30 DSV4_MAXLEN=4096 DSV4_MAXBT=2048 \
    DSV4_KV_DTYPE=fp8_e4m3 DSV4_MOE_BACKEND=triton \
      timeout 1200 python3 /home/$USER/4node/probe_dsv4_vllm.py \
      > /home/$USER/logs/4node/dsv4base_recheck.log 2>&1"'
```

判据：`Model loading took 35.15 GiB` + `ENGINE LOADED` + 通顺文本 + `DSV4 VLLM OK`。
⚠️ `moe_backend` 必须显式传 `triton`——`triton_unfused` 是 FP4 专用会 `ValueError`,
而默认值会自动挑中 AITER,在第一次前向 `moe_sorting_opus_fwd` 处硬失败。

## 2. 权重：不用写转换器,transformers 直读原生 checkpoint

### 2.1 架构不用写,transformers 已经全实现了

实测 `transformers 5.13.1`（探针 `~/dsv4/probe_transformers_dsv4.py`）：

```
modeling_deepseek_v4.py  1526 行  22 个类
  DeepseekV4Indexer / IndexerScorer / CSACompressor / HCACompressor
  HyperConnection / HyperHead / HashRouter / TopKRouter
  CSACache / HCACache / GroupedLinear / SparseMoeBlock ...
自定义内核依赖:  tilelang 0   tile_kernels 0   deep_gemm 0   triton 0
```

V4 的机制一个都没缺,而且几处细节都对得上：

- **attention sink 有**,叫 `sinks`（`nn.Parameter(torch.empty(num_heads))`,在
  `eager_attention_forward` 里拼进 logits 再丢掉最后一列）
- **`sqrtsoftplus` 在 `ACT2FN` 里**（`self.score_fn = ACT2FN[config.scoring_func]`,实测 `True`）
- **`deepseek_v4` 在 `AutoConfig` 里原生注册**——不用像 miles 那样合成 config alias
- **config 的向后兼容折叠是正确的**：原始 `compress_ratios`（44 元素）被转成 43 项
  `layer_types`（`0→sliding_attention`、`4→compressed_sparse_attention`、
  `128→heavily_compressed_attention`）,`num_hash_layers=3` 转成前 3 层 `mlp_layer_types=hash_moe`。
  ⚠️ 折叠之后 `cfg.compress_ratios` 就取不到了,别拿它判层型,要读 `cfg.layer_types`

### 2.2 结论：映射已经在 transformers 里,`from_pretrained` 直接指原生目录就行

`transformers/conversion_mapping.py` 里有一条 `"deepseek_v4"` 条目（5.13.1 中在第 357 行起）,
约 35 条 `WeightRenaming` 加 2 条 `WeightConverter`。**它是照着我们手上这个 checkpoint 写的**——
FP8 的 `.scale` → `.weight_scale_inv` 改名在 `FineGrainedFP8HfQuantizer.update_weight_conversions`
里,注释直接写着 "e.g. DeepSeek-V4-Flash"；`_process_model_after_weight_loading` 的注释点名
"dsv4-flash-base stores its (power-of-two) ue8m0 scales in a float32 container under `.scale`"。

它覆盖了下面 §2.3 里所有容易搞错的地方,包括 `attn.indexer.compressor.*` → `self_attn.compressor.indexer.*`
的嵌套翻转、`ape` → `position_bias`、`gate.bias` → `e_score_correction_bias`、
`w1`/`w3` 融合成 3D `gate_up_proj`（`MergeModulelist` + `Concatenate`）。
`mtp.*` 由 `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` 吃掉。

**验收结果（全量 43 层,8 卡 `device_map="auto"`,BF16 dequantize,加载 346 s）：**

| 检查项 | 结果 |
|---|---|
| 张量总数 | **1537**,与 §2.3 的期望清单完全一致 |
| missing / unexpected / mismatched / error | 0 / 0 / 0 / 0 |
| `mtp.*`（可忽略的那类 unexpected） | 0 |
| 与 meta-device 规格书的双向差集 | 空 / 空 |
| 残留 meta 张量 | 0 |
| 前向 8 / 128 / 512 / 1024 / 1536 / 2048 / 3072 / 4096 | 全部 finite,logits absmax 稳定在 31–32 |

⚠️ 前向那一行的前提是先打了 §2.6 的 bmm 补丁,不打的话 4096 全是 NaN。

最小可用调用：

```python
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
M.DeepseekV4GroupedLinear.forward = fixed_grouped      # §2.6,必须在加载之前

cfg = AutoConfig.from_pretrained(MODEL)                 # MODEL = 原生 checkpoint 目录
qc = {k: v for k, v in cfg.quantization_config.items() if k not in ("quant_method", "fmt")}
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    quantization_config=FineGrainedFP8Config(dequantize=True, **qc),   # §2.4
)
```

想快速自查用 `~/dsv4/probe_dsv4_load_native.py`（会打印 missing/unexpected、双向 key diff、
残留 meta 张量、逐张量 absmax,再跑一次前向）。只想验层级映射不想加载 275GB 的话,
`~/dsv4/make_native_subset.py` 能切出前 N 层的原生副本（`DSV4_SUBSET_LAYERS=4` 约 27GB,
覆盖 sliding / CSA+indexer / HCA 和 hash_moe / moe 全部分支;config 只改 `num_hidden_layers`,
`DeepseekV4Config.__post_init__` 会自动截断 `compress_ratios` 和 `num_hash_layers`）。

### 2.3 期望清单（当验收对照用,不再是"要造的东西"）

meta device 实例化实测（探针 `~/dsv4/probe_dsv4_hf_target.py`）：
**1537 个张量,284.3 B 参数,52 种命名模式**。下面这张表现在的用途是核对,不是照着写转换器。

顶层：

```
model.embed_tokens.weight        (129280, 4096)     ← 原生 embed.weight
model.norm.weight                (4096,)
model.hc_head.hc_fn              (4, 16384)         ← 原生 hc_head_fn
model.hc_head.hc_base            (4,)
model.hc_head.hc_scale           (1,)
lm_head.weight                   (129280, 4096)     ← 原生 head.weight
model.rotary_emb.*_inv_freq      (32,)              ← 非持久 buffer,不用转
```

每层（以 layer 0 = `sliding_attention` 为例,是所有层的公共部分）：

```
self_attn.sinks                        (64,)          ← 原生 attn.attn_sink
self_attn.q_a_proj.weight              (1024, 4096)   ← attn.wq_a
self_attn.q_a_norm.weight              (1024,)        ← attn.q_norm
self_attn.q_b_proj.weight              (32768, 1024)  ← attn.wq_b
self_attn.kv_proj.weight               (512, 4096)    ← attn.wkv
self_attn.kv_norm.weight               (512,)         ← attn.kv_norm
self_attn.o_a_proj.weight              (8192, 4096)   ← attn.wo_a
self_attn.o_b_proj.weight              (4096, 8192)   ← attn.wo_b
mlp.gate.weight                        (256, 4096)
mlp.gate.tid2eid                       (129280, 6)    ← hash 路由表
mlp.experts.gate_up_proj               (256, 4096, 4096)  ← 3D 融合,见下
mlp.experts.down_proj                  (256, 4096, 2048)  ← 3D 融合
mlp.shared_experts.{gate,up,down}_proj.weight           ← 2D,不融合
input_layernorm.weight / post_attention_layernorm.weight
attn_hc.{fn,base,scale}                (24,16384)/(24,)/(3,)  ← 原生 hc_attn_*
ffn_hc.{fn,base,scale}                 (24,16384)/(24,)/(3,)  ← 原生 hc_ffn_*
```

**逐层差异（按 `layer_types` 分支）**：

| 层型 | 额外张量 |
|---|---|
| `sliding_attention`（ratio 0） | 无 |
| `compressed_sparse_attention`（ratio 4） | `self_attn.compressor.{kv_proj,gate_proj,kv_norm,position_bias}` **+ 嵌套的 `compressor.indexer.{q_b_proj,kv_proj,kv_norm,gate_proj,position_bias,scorer.weights_proj}`** |
| `heavily_compressed_attention`（ratio 128） | 只有 `self_attn.compressor.*`,**没有 indexer** |
| `mlp_layer_types = moe`（第 4 层起） | `mlp.gate.e_score_correction_bias` (256,)（`hash_moe` 层没有） |

注意两点容易搞错的：**indexer 嵌在 compressor 下面**（`compressor.indexer.*`,不是
`self_attn.indexer.*`）；`compressor.position_bias` 就是原生的 `ape`,形状随层型变
（CSA `(4, 1024)`、HCA `(128, 512)`）。

### 2.4 FP8 vs BF16：训练必须走 dequantize

Base 的 `quantization_config` 是 FP8 块量化（`quant_method: fp8` + `weight_block_size [128,128]`
+ ue8m0 scale 装在 float32 容器里）。两条路都实测过：

| | 保留 FP8 | `dequantize=True` |
|---|---|---|
| 加载 | ✅ 干净,`gate_up_proj_scale_inv` 是 `(256, 32, 32) float8_e8m0fnu`,形状 dtype 都对 | ✅ 干净 |
| 前向 | ❌ 需要 `pip install kernels==0.15.2`,镜像里没装（容器有外网,能装） | ✅ 纯 PyTorch |
| 能否训练 | ❌ `FineGrainedFP8HfQuantizer.is_trainable` 返回 `False` | ✅ |
| 显存 | 约 295GB | 569GB |

**所以 FSDP2 训练只能走 `FineGrainedFP8Config(dequantize=True, ...)`**,显存里是 BF16。
§3 的 4.55TB 训练态账算的就是 BF16,不用改。

⚠️ 注意这跟本文旧版的建议相反。旧版说"落 295GB 的 FP8 产物、别落 569GB BF16",那是在
"要自己写转换器、要落盘"的前提下讲的。现在既然直读原生 checkpoint,**磁盘上只有原始的
275GB,没有任何中间产物**,BF16 只存在于显存里,那条取舍不复存在。

### 2.5 验收现状

| 步骤 | 状态 |
|---|---|
| 1) 单层 `from_pretrained` | ✅ 通过（L1 33 个张量,双向差集为空） |
| 2) key 覆盖双向差集 | ✅ 空（L1 / L4 / 全量 43 层都验过） |
| 3) 与 §2.3 的 1537 个张量对照 | ✅ 完全一致 |
| 4) 数值核对（抽张量 vs 官方解码逻辑） | ⬜ 未做,但在第 5 步通过之后边际价值不大 |
| 5) 端到端：全模型前向 + 与 vLLM 同 prompt 比 logits | ✅ **通过**,见下 |

absmax 抽查能排除"加载成默认初始化"这一类错误（`compressor.position_bias` 初始化是全零,
实测 1.602 / 3.331 / 0.5974；`e_score_correction_bias` 初始化全零,实测 9.029；
`hc_scale` 初始化全一,实测 0.7908）,但排除不了张量放错位置——那要靠第 5 步。

**第 5 步：与 vLLM 的数值核对（`~/dsv4/refcheck_dsv4_{vllm,hf}.py` + `refcheck_compare.py`）**

同一批 prompt、**同一份 token id**（直接把 token id 喂给 vLLM,避开 tokenizer 差异）。vLLM 用
已验证的那套参数（TP=8 / `moe_backend=triton` / `kv_cache_dtype=fp8_e4m3` / `VLLM_ROCM_USE_AITER=1`）,
transformers 用训练要走的那条路（`dequantize=True` + §2.6 的补丁）。

| | 结果 |
|---|---|
| prompt 位置 top-1 一致 | **32/35（91.4%）** |
| \|Δ logprob\|（实际 token）均值 / 最大 | 0.217 / 4.94；**剔掉下面那个异常点后 0.078 / 0.94** |
| 贪心续写完全一致的 prompt | 2/4,且是 24/24 token 逐个相同 |

⚠️ **别指望逐位相等,那不是判据。** vLLM 的专家保持 FP8 并跑 FP8 GEMM、KV cache 是 fp8_e4m3、
注意力是稀疏的（`index_topk=512`）；transformers 反量化成 BF16、注意力是稠密 eager。
判据是**排序一致**：argmax 基本一致、logprob 差很小、贪心文本一样。映射错了会是乱码,不是小数点。

prompt 要短（这里 4–16 token）。长到几百 token 以后稀疏与稠密注意力看到的 KV 集合就不同了,
那时的分歧是注意力造成的,说明不了权重。

两条最长的 prompt 逐 token 完全一致；另两条在真正的近似平局处分叉（三处 top-1 翻转的领先幅度
是 0.125 / 0.375 / 0.5 nats）,两边都通顺——`The capital of France is Paris.` 之后一个接着数德国、
另一个接着数西班牙。

**⚠️ 唯一一处实质分歧,错的是 vLLM。** 上下文 `def fibonacci(n):\n    if n < `,下一个 token 是 `2`：

```
transformers top-5:  '0'@-0.771, '2'@-0.771, '1'@-2.771, '3'@-4.771, '4'@-6.771
vLLM         top-5:  ' inertia'@-2.711, ' rent'@-3.086, ' perm'@-3.148, '碎'@-3.273, '惯'@-3.586
```

`if n < ` 后面该是数字。对齐没错（同一次比对里其它位置两边的候选都互相对应）,就是 vLLM 在
这个位置崩了。结论：**vLLM 的 `prompt_logprobs` 在这个模型上不是每个位置都可信**,拿它当参考
时要看分布本身合不合理,别只看数字。

留给下一个 agent 的一个问号：DAPO 用的是**采样出来的 token** 的 logprob（decode 路径）,和
`prompt_logprobs`（prefill 路径）不是同一段代码,所以未必影响 `rollout_corr/kl`。要确认就再比
一次 decode 路径的 logprob——那正好就是 RL 里的那个量。

### 2.6 ⚠️ gfx950 的 `torch.bmm` 越界：不打补丁,4k 长度必出 NaN

`DeepseekV4GroupedLinear.forward`（也就是 `self_attn.o_a_proj`）喂给 `torch.bmm` 的两个操作数
都是非连续视图,A 的 batch stride 比矩阵跨度小、八个 batch 在内存里交错。在
MI355X / ROCm 7.0 / torch 2.10 上,**M ≳ 2040 时会越界读**：单独跑直接 `Memory access fault`,
跑在模型里则安静地产出 inf / NaN 并污染整个前向。

全量 43 层实测：未打补丁 `len=4096` 有 529,530,880 个非有限值（= 整个 logits）；打了之后
8 → 4096 全干净。两批不同节点上都复现过,不是偶发。

补丁：

```python
def fixed_grouped(self, x):
    input_shape, hidden_dim = x.shape[:-2], x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2).contiguous()
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1).contiguous()
    return torch.bmm(x, w).transpose(0, 1).reshape(*input_shape, self.n_groups, -1)
```

⚠️ **用 `device_map` 加载时,补丁必须打在 `from_pretrained` 之前。** accelerate 的
`add_hook_to_module` 会把 `functools.partial(new_forward, module)` 写成**实例属性**,遮蔽掉类上的
`forward`;加载之后再改类,补丁完全不生效,而且不报错不告警。我第一次全量跑就是栽在这里,
白白多花了三次加载去排查。FSDP2 那条路不装 accelerate hook,没有这个限制,但统一按
"先打补丁再加载"写不会错。

⚠️ **"我在长度 X 上试过没事"不是结论。** 越界与否取决于 kernel tile 选择,实测 2040 / 2043 /
2044 / 2048 / 4096 都崩,1792 和 2052 恰好没事。

完整的复现、已排除的七个假设、上游报告要点见 `rocm-gfx950-strided-bmm-oob-issue.md`。
那份文档里还有两个复现时的坑（core dump 会往 cwd 写 50–80 GB;一次崩溃会污染同进程后续所有
调用,所以扫描长度必须一个尺寸一个进程）。

## 3. 显存账与并行度

参数 284.3B,BF16 权重 569 GB。FSDP2 训练态 = 权重 569 + 梯度 569 + fp32 master 1137 +
Adam 两动量 2274 ≈ **4.55 TB**。

| 方案 | 训练态/卡 | 引擎/卡（TP=8 实测） | 合计 | 判断 |
|---|---|---|---|---|
| 4 节点 32 卡 + rollout TP=8 | 4.55TB/32 = **142 GB** | **40.5 GiB** | 约 **182 GiB / 288 GiB** | 账面可行,余量约 105 GiB |
| 16 节点 128 卡 + TP=8 | 4.55TB/128 = **36 GB** | 40.5 GiB | 约 77 GiB | 宽裕,`amd-burst-qos` 配额够 |

**推理侧的第一份实测**（全量 43 层,BF16,8 卡 `device_map="auto"`,4096 token 前向）：

```
每卡 allocated  62.3 – 73.6 GiB      （权重 569 GB / 8 ≈ 71 GB,对得上）
每卡 reserved   45.5 – 117.6 GiB     （峰值比 allocated 高 55 GiB）
```

这还只是**推理**,没有梯度、优化器状态和反向的激活。那 55 GiB 的差额就是稠密 eager 注意力的
工作区——4096 长度下 `[1, 64, 4096, 4096]` 的 bf16 logits 单个就 2.1 GB,而
`eager_attention_forward` 里 `qk` / `+mask` / `centered` / `probs` 是四份独立的临时量。

**激活是唯一的未知量,也是最可能让 4 节点方案失败的地方。** 参照 Qwen3-30B 那次：
`actor_allocated` 只有 10.8 GB,而 `max_reserved` 到了 113.9 GB——差额全是激活与工作区。
284B 上这个量级可能吃掉全部余量。所以：

- **第一次跑务必采样 `rocm-smi`,不要只看框架指标**
- 撞 OOM 的退路是 16 节点,不是调小 batch（`max_tokens_per_gpu` 已经很小了）
- 若 LumenRL 的 FSDP2 支持 optimizer CPU offload,那是比加节点更便宜的一步（miles 在
  4 节点上就是靠 `--optimizer-offload-fraction 0.75` 才装下的）

引擎侧：`gpu_memory_utilization` 是"占**整卡**的比例",不看训练 actor 已占多少。
**从 `0.30` 起步,别照抄 Qwen3 那条线的 0.45。** 而且实测它**不是硬上限**——
`util=0.30` 名义 86.4 GiB、实占 91.5 GiB,超出约 5 GiB,算账要预留。

## 4. 权重同步：布局天然对得上,但有两处小活

**好消息**：transformers 的路由专家是**融合 3D** 的——

```python
self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden))
self.down_proj    = nn.Parameter(torch.empty(num_experts, hidden, intermediate))
```

这正是 LumenRL `FusedMoEWeightRouter._load_fused` 要求的
（`must be 3D (experts, out, in)`）。所以 `~/lumen_rl/Lumen-RL` 里现有的 MoE 权重同步机制
**直接对得上**,不用像 Megatron 那条路一样绕开它。

要改的地方只剩一处：

1. **shared expert 命名。** DSv4 是复数 `mlp.shared_experts.{gate,up,down}_proj.weight`,
   而现有 bridge 里的 Qwen3 写法是单数 `mlp.shared_expert.gate_proj.weight`。
   漏掉会让 `assert_weight_sync_coverage` 报
   `weight sync (colocate-ipc) left N/M rollout parameters untouched`。

~~2. FP8 scale 被当成"不该发"而忽略。~~ **不再是问题。** 本文旧版担心
`vllm_moe_weight_sync.py` 的 `_COVERAGE_IGNORE_SUFFIXES` 会把真权重当成 scale 忽略掉——那是
"保留 FP8 训练"方案下的顾虑。既然 §2.4 定死了训练走 `dequantize=True`,trainer 侧压根不会有
`_scale_inv` 参数,发出去的全是 BF16,和现有的 Qwen3 写法同构。

诊断信号（都在 `lumenrl/engine/inference/vllm_moe_weight_sync.py`）：
`untouched`（覆盖率不足,`LUMENRL_WEIGHT_SYNC_CHECK=warn` 可降级）、
`must be 3D`（训练侧没发融合布局）、`verify failed`（需 `LUMENRL_WEIGHT_SYNC_VERIFY=1`）。

⚠️ **`~/lumen_rl/Lumen-RL` 有未提交的改动,别碰、别 `git checkout`**：rollout TP>1 的 6 个
文件 + 新增的 `megatron_te_gemm_compat.py`。热点在 `_setup_ray_vllm_rollout` 的 kwargs 块、
`VLLMReplicaManager.__init__/create`、`update_weights_ipc_send` 的 ZMQ handle、
`FusedMoEWeightRouter._dispatch/_verify_written`——改这些地方前先 `git diff` 看一眼。

## 5. 不要重做的事（都有实测依据）

- **不要再试 rollout TP=1。** TP=1 在 gfx950 上必然 `Memory access fault`,五种 MoE backend /
  AITER 组合全部复现过。上游验证矩阵里 TP=1 从来没被覆盖。
- **不要写 native→HF 转换器,也不要落中间产物。** transformers 5.13.1 自带映射,全量 43 层已验收
  （见 §2.2）。~~本文旧版说"缺的是权重格式,不是版本问题"——那句是错的,5.13.1 连权重格式映射都有。~~
  顺带,本文旧版建议去抄的那两份映射（vLLM 的 `vllm/models/deepseek_v4/`、SGLang 的
  `DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format`）现在都用不上了。补一句实测：
  SGLang 那份的目标命名和 transformers **并不一致**——它保留 `wq_a` / `wkv` 这类论文记号,
  只做 `attn`→`self_attn`、`ffn`→`mlp` 的前缀改写,照抄会得到一套错的名字。
- **不要试图靠调 batch / 长度绕开 4k 处的 NaN。** 那是 gfx950 的 bmm 越界,不是数值问题,
  换长度只是换运气。用 §2.6 的补丁。
- **不要把 vLLM 的 `prompt_logprobs` 当成金标准。** 实测它在这个模型上有位置会给出完全不合理的
  分布（§2.5 那个 `if n < ` 的例子）。拿它做参考时要看分布本身讲不讲得通,不要只比数字——
  否则会把 transformers 正确的那一侧判成错的。
- **不要走 Megatron。** 理由见 §0 的对照表。若确实要走,看
  `deepseek-v4-miles-megatron-4node-runbook.md`——那条路已经跑通了,但绑 SGLang。
- **不要用 `sgl-project/DeepSeek-V4-Flash-FP8` 当"HF 版"。** 名字里有 FP8 但**不是** HF 布局,
  它也是原生命名（69187 张量）。我们的 Base 是它的严格超集（69189,多两个可丢的 MTP 张量）。
- **不要在 instruct/thinking 版上训练。** 它在 `max_response_length` 内永远不闭合,reward 恒
  −1,`filter_groups` 连续 10 轮 kept 0 直接抛 `RuntimeError`。必须用 Base。
- **不要再排查跨节点 RDMA。** ionic RoCE 那条线的坑都记在 4 节点 runbook §6,含已排除的六个
  假设。走 ens3 上的 TCP（`NCCL_IB_DISABLE=1`）能跑通。
- **不要 fork `run_dapo.sh`。** 用 `~/4node/bin/ray` 那个 PATH shim 吞掉开头会拆集群的
  `ray stop`。

## 6. 建议的推进顺序

1. ~~单层转换 + `from_pretrained`~~ ✅ **已完成**,而且发现根本不用写转换器。见 §2.2。
2. ~~全量转换,输出 295GB~~ **这一步整个消失了**——直读原生 checkpoint,磁盘上没有中间产物。
   §2.5 的第 5 项（与 vLLM 比 logits）也 ✅ **已完成**。**权重这条线到此收口,可以往工程量走了。**
3. 分发**原生的 275GB**到 4 台（无 ssh,用 miles runbook §4 的 rsyncd 办法）。**下一个该做的就是这步。**
4. 写 DAPO 配置：以 `dapo_qwen3moe_a3b_ray_*` 那几份 MoE 配置为模板,
   `policy.training_backend=fsdp2`、`policy.generation.vllm_cfg.tensor_parallel_size=8`、
   `kv_cache_dtype=fp8_e4m3`、`gpu_memory_utilization=0.30`。
   ⚠️ `vllm_cfg` 目前**没有 `moe_backend` 字段**,而 DSv4 必须传 `triton`——要加透传,
   两处：`lumenrl/core/config.py` 的 `VLLMConfig` dataclass（OmegaConf structured schema,
   不加字段会直接拒绝 YAML 里的未知键）+ `_setup_ray_vllm_rollout` 的 kwargs 块。
5. 显存采样 + smoke。判据同 Qwen3 那条线：`exit=0`、`untouched` 0 次、`verify failed` 0 次、
   `rollout_corr/kl` 不随步数爬升。
6. tokenizer 与数据：V4 的 vocab 是 129280,和 Qwen3 完全不同,所以
   `data_cached/qwen3-8b-maxprompt1024/` 的**过滤阈值不再准确**（原始 parquet 是文本可复用,
   但 prompt token 数会变）。要么按 V4 tokenizer 重跑 `filter_prompts.py`,要么把
   `max_total_sequence_length` 留足余量。

## 7. 交接清单

| 内容 | 位置 |
|---|---|
| **权重加载验收探针** | `~/dsv4/probe_dsv4_load_native.py`（missing/unexpected + 双向 key diff + 残留 meta + 逐张量 absmax + 一次前向;`DSV4_DEQUANT=1 DSV4_PATCH_GROUPED=1 DSV4_DEVICE=auto`） |
| 切 N 层原生副本 | `~/dsv4/make_native_subset.py`（`DSV4_SUBSET_LAYERS=4` 约 27GB,覆盖全部层型分支） |
| **与 vLLM 的数值核对** | `~/dsv4/refcheck_dsv4_vllm.py`（参考侧,顺带也是 rollout 没退化的复现）、`refcheck_dsv4_hf.py`、`refcheck_compare.py`、`refcheck_decode.py`（把分歧位置解成可读 token） |
| 期望清单 | `~/dsv4/probe_dsv4_hf_target.py`（打印 §2.3 那 1537 个张量名） |
| 原生 checkpoint 张量清单 | `~/dsv4/dump_native_layer0.py`（只读 index + 分片头,几 MB IO） |
| transformers 实现体检 | `~/dsv4/probe_transformers_dsv4.py` |
| **gfx950 bmm bug 相关** | `~/dsv4/probe_rocm_grouped_bmm.py`（最小复现）、`probe_dsv4_full_sweep.py`（全量长度扫描）、`probe_dsv4_patch_timing.py`（accelerate hook 那个坑）、`probe_dsv4_grouped_fix.py`；排除项：`probe_rocm_gemm_nan.py`、`probe_rocm_hc_matmul.py`；定位过程：`probe_dsv4_layer_trace.py`、`probe_dsv4_attn_trace.py` |
| transformers 源码离线副本 | `~/dsv4/vendor/`（`conversion_mapping.py`、`core_model_loading.py`、`finegrained_fp8.py`、`quantizer_finegrained_fp8.py`、`tf_deepseek_v4/`） |
| rollout 探针（已验证） | `~/4node/probe_dsv4_vllm.py` |
| 4 节点环境脚本 | `~/4node/`（NFS,换节点只改 `env.sh`） |
| 模型 | `/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-Base`（275G,46 分片,**node-local,作业结束即失联**;重下约 12 分钟,命令见 `deepseek-v4-base-rl-train-handoff.md` §2） |
| Base 的 config.json 副本 | `~/dsv4/base_config.json`（可离线分析） |
| 验收日志 | `~/logs/4node/dsv4_full43_final.log`（全绿那次）、`dsv4_full43_sweep.log`、`dsv4_patch_timing.log`、`dsv4_ref_vllm.log`、`dsv4_ref_hf.log` |
| rollout 验证日志 | `~/logs/4node/dsv4base_vllm_tp8b.log`（成功那次）、`dsv4base_vllm_tp8.log`（`triton_unfused` 失败）、`dsv4base_vllm_tp8_aiter.log` / `_auto.log`（AITER 失败） |
| LumenRL 未提交改动 | `~/lumen_rl/Lumen-RL`：rollout TP>1（6 文件）+ `megatron_te_gemm_compat.py`,**别动** |
| gfx950 bmm issue | `rocm-gfx950-strided-bmm-oob-issue.md` |
| 另一条已跑通的路 | `deepseek-v4-miles-megatron-4node-runbook.md` |

