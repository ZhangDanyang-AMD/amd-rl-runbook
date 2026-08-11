# 交接：在这台集群上把 DeepSeek-V4 Base 跑成 DAPO RL 训练

> 给下一个 agent。目标是用 **`deepseek-ai/DeepSeek-V4-Flash-Base`** 在 LumenRL 上跑通 DAPO RL
> （先 smoke，再谈长跑）。上一轮（2026-08-05）把能力边界摸清了，本文告诉你**从哪一步接着做、
> 什么已经验证过不用重做、以及哪一步有可能直接判死**。
>
> 背景与证据在两份姊妹文档里，**先读它们再动手**：
> - `deepseek-v4-flash-enablement-handoff.md` —— 三道墙的量化定界（本文不重复证据）
> - `dapo-lumenrl-4node-32gpu-runbook.md` —— 4 节点环境怎么搭（本文直接引用其中的脚本）

## 0. 一页现状

| 环节 | 状态 |
|---|---|
| 4 节点 32 卡编排（Ray / NCCL / 容器） | ✅ 已跑通，脚本在 `~/4node/`，见 §2 |
| rollout 支持 TP>1 | ✅ 已实现并验证（Qwen3-30B-A3B TP=8 smoke `exit=0`） |
| vLLM 托管 **instruct 版**（Flash-0731，FP4 专家） | ✅ TP=8 可加载可生成 |
| vLLM 托管 **Base 版**（FP8 专家） | ✅ **已验证**（2026-08-05）TP=8 可加载可生成，但 `moe_backend` 必须是 `triton`，见 §3.1 |
| 训练侧读权重（FSDP2 / transformers） | ❌ 原生命名与 transformers 交集 0，要写转换器，见 §4 |
| 训练侧读权重（**Megatron via miles**） | ✅ **已有现成实现**，且 Base 是它消费格式的严格超集，见 **§9** |
| 32 卡显存 | ⚠️ 靠 rollout TP=8 账面可行，见 §5 |

**为什么必须用 Base 版**：instruct/thinking 版在 `max_response_length` 内永远不闭合，每条样本被截断、
reward 恒为 −1、`filter_groups` 连续 10 轮 kept 0，直接抛 `RuntimeError: filter_groups collected no
valid groups`（vime runbook §16.1 记过）。所以上一轮验证用的 Flash-0731 只能证明"引擎能跑"，
**不能**用来训练。

**Base 与 instruct 的 config 差异只有两处**（我逐键 diff 过）：`expert_dtype` 是 `fp8` 而不是 `fp4`；
没有 `dspark_*` 那四个键。架构、层数、专家数、284.3B 参数量完全一致。所以：

| | Flash-0731（已验证） | Flash-Base（你要用的） |
|---|---|---|
| 发布体积 | 166.9 GB / 48 分片 | **294.7 GB / 46 分片** |
| 专家精度 | FP4 (e2m1) | **FP8 (e4m3)** |
| BF16 展开后 | 569 GB | 569 GB（同参数量） |
| TP=8 每 rank 引擎占用 | 20.03 GiB（实测） | **35.15 GiB**（实测，见 §3.1） |
| FP8/FP4 MoE backend | `triton_unfused` | **`triton`**（`triton_unfused` 是 FP4 专用，见 §3.1） |

## 1. 环境：先拿到 4 个节点

上一轮的作业（38564）到你接手时**大概率已经结束**。重新申请：

```bash
# 1) 配额：默认 QOS 是 account 共享的 19 节点，通常不够；用 burst
spur accounts show qos | grep -E "aifw-dev|burst"

# 2) 真人在真终端跑（agent 不要自己申请 GPU，见 SPUR_NODE_ACCESS_GUIDE.md §0 铁律）
spur run -q amd-burst-qos -N 4 --gpus-per-node=8 \
  -t 1-00:00:00 -A amd-aifw-dev -p amd-spur --pty bash -l
```

⚠️ 三个坑：`spur alloc` 不支持 `-q`（所以撞配额时只能用 `spur run`）；`--pty` 不能省；
**不要在已有分配的 shell 里跑**（`SLURM_JOB_ID` 会让它退化成 job step）。详见
`SPUR_NODE_ACCESS_GUIDE.md` §2.1 / §4c。

拿到 JobID 后，**必做**：把 `~/4node/env.sh` 里的 `HEAD_NODE` / `HEAD_IP` / `NODES` 改成新节点，
`HEAD_IP` 取自 head 节点的 `ip -o -4 addr show ens3`。然后按 4 节点 runbook §4–§5 起容器和 Ray。

在非 head 节点上执行命令一律用 `JOBID=<id> ~/nx.sh <node> '...'`（`spur exec` 只到 head）。

## 2. 已经建好、可以直接复用的东西

`~/4node/` 下（都在 NFS，换节点不用重建）：

| 文件 | 用途 |
|---|---|
| `env.sh` | 路径与节点清单，**换机只改这一份** |
| `01_container.sh` | 起容器（含 `--ulimit memlock=-1`、ionic provider、nobackup 挂载） |
| `ray_env.sh` | **raylet 环境**——Ray 不传 driver 环境，这份必须在每台 `ray start` 前 source |
| `02_ray_start.sh head\|worker` | 起 Ray |
| `bin/ray` | PATH shim，吞掉 `run_dapo.sh` 开头那句会拆集群的 `ray stop` |
| `03_smoke.sh` / `04_smoke_megatron.sh` / `05_smoke_moe_tp8.sh` | Qwen3-30B-A3B 的三个 smoke，可作为改 DeepSeek 版本的模板 |
| `probe_ray.py` / `probe_nccl.py` / `probe.sh` | 上训练前的两个预检（env 传播、32 rank all-reduce） |
| `probe_dsv4_vllm.py` | **vLLM 单独加载 DeepSeek 的探针，§3 直接用** |
| `probe_dsv4_keys.py` | checkpoint 与 transformers 的 key diff，写转换器时用来验收 |
| `dsv4/` | Flash-0731 的 config.json 与权重索引，可离线分析 |
| `nx.sh`（在 `~/`） | 在任意节点执行命令 |

`~/nx.sh` 的输出会夹一个 `^@`，用 `sed 's/\^@//'` 滤掉。

**Lumen-RL 的代码改动还在工作树里未提交**（`git -C ~/lumen_rl/Lumen-RL status`）：rollout TP>1 那 5 个
文件 + Megatron/TE 的 compat 补丁。别不小心 `git checkout` 掉。

## 3. 第一步（最便宜，也可能直接判死）：Base 版能不能被 vLLM 托管

**为什么先做这个**：vLLM 的 ROCm 使能跟踪 issue（vllm-project/vllm#41820）里，
`[x] Base PR ... for DeepSeek-V4-Pro and DeepSeek-V4-Flash` 是打勾的，但

```
[ ] [Functionality] DeepSeek-V4-Flash Base FP8 enablement PR:
```

**是未打勾的**。也就是说 Base 版的 FP8 专家路径在 ROCm 上很可能还没使能。如果它跑不起来，
写转换器就毫无意义——RL 需要 rollout 和训练同时可用。**先花 1 小时验证这件事，再谈其它。**

```bash
# 下到 node-local 大盘（294.7 GB；别放 NFS，见 §4 的空间说明）
~/nx.sh <head> 'source /home/$USER/4node/env.sh
  docker exec -d "$CONTAINER" bash -lc "
    export HF_HUB_DISABLE_XET=1 HF_HOME=/mnt/m2m_nobackup/$USER/hf_home
    hf download deepseek-ai/DeepSeek-V4-Flash-Base \
      --local-dir /mnt/m2m_nobackup/$USER/models/DeepSeek-V4-Flash-Base --max-workers 16 \
      > /home/$USER/logs/4node/dsv4base_download.log 2>&1"'
# 本地 NVMe 实测约 400 MB/s，294.7 GB 约 12–15 分钟
```

然后用现成探针（**这四个参数一个都不能少**，原因见 enablement handoff §2.1）：

```bash
~/nx.sh <head> 'source /home/$USER/4node/env.sh
  docker exec -d "$CONTAINER" bash -lc "
    source /home/$USER/4node/ray_env.sh
    unset HIP_VISIBLE_DEVICES
    export VLLM_LOGGING_LEVEL=INFO VLLM_ROCM_USE_AITER=1
    export HF_HOME=/mnt/m2m_nobackup/$USER/hf_home
    DSV4_PATH=/mnt/m2m_nobackup/$USER/models/DeepSeek-V4-Flash-Base \
    DSV4_TP=8 DSV4_UTIL=0.80 DSV4_MAXLEN=4096 DSV4_MAXBT=2048 \
    DSV4_KV_DTYPE=fp8_e4m3 DSV4_MOE_BACKEND=triton \
      timeout 3000 python3 /home/$USER/4node/probe_dsv4_vllm.py \
      > /home/$USER/logs/4node/dsv4base_vllm_tp8.log 2>&1"'
```

判据：`Model loading took … GiB`（记下这个数，§5 要用）+ `ENGINE LOADED` + 生成出通顺文本 + `DSV4 VLLM OK`。

**如果失败**：把报错原文对照 enablement handoff §2 的表，先确认不是那四个已知项；再判断是不是
FP8 专家路径缺失。若确属上游未使能，**这条线到此为止**，需要更新的 vLLM（`rocm/vllm-dev:deepseek-v4-mi35x`
镜像是 AMD 自己的 DSv4 镜像，可用于纯 rollout 验证；但 RL 训练栈绑在 vime 镜像上，混用不了）。
这时应当回来和用户确认方向，而不是继续往下做。

### 3.1 结论（2026-08-05 实测）：Base 版能被托管，但 `moe_backend` 要换

**上游 checklist 那一项没打勾，不代表 FP8 路径不存在。** 实测这个构建里 FP8 专家路径是完整的：
vLLM 自己就把量化方式识别成 `quantization=deepseek_v4_fp8`（instruct 版走的是 fp4），并启用了
`Using FP8 indexer cache for Lightning Indexer` 和 `Using DeepSeek's fp8_ds_mla KV cache format`。

唯一要改的是 **`moe_backend`**：§3 命令里照抄 instruct 配方的 `triton_unfused` 会在 8 个 worker 上一起 `raise`：

```
ValueError: moe_backend='triton_unfused' is not supported for FP8 MoE.
Expected one of ['triton', 'deep_gemm', 'cutlass', 'flashinfer_trtllm', 'flashinfer_cutlass', 'marlin', 'aiter'].
```

`triton_unfused` 是 **FP4 专用**的 backend。换成 `triton` 后一次通过（`DSV4_MOE_BACKEND=triton`）：

```
Using TRITON Fp8 MoE backend out of potential backends: ['AITER', ..., 'TRITON', ...]
Model loading took 35.15 GiB memory and 14.0 seconds      （每个 TP rank）
Available KV cache memory: 191.04 GiB                      （util=0.80）
GPU KV cache size: 1,041,802 tokens
ENGINE LOADED in 74s
'The capital of France is' -> ' Paris. The capital of Spain is Madrid. The capital of Italy is Rome...'
DSV4 VLLM OK
```

⚠️ **`moe_backend=aiter` 和 `auto` 都是坏的**，这直接决定 §6 第 2 条不是可选项：

| `moe_backend` | 结果 |
|---|---|
| `triton_unfused`（instruct 配方） | ❌ `ValueError`，FP4 专用 |
| **`triton`** | ✅ **加载 + 生成正常** |
| `aiter` | ❌ 权重已加载（35.16 GiB）后死在第一次前向：`moe_sorting_opus_fwd: workspace needs to be Optional[aiter_tensor_t] but got tensor(...)` |
| `auto`（不传） | ❌ **自动挑中 AITER**（它在候选列表里排第一），同上报错 |

`aiter` 那条是 `~/lumen_rl/aiter/aiter/fused_moe.py:76` 与镜像里编译好的 `module_moe_sorting_opus.so`
之间的 API 漂移（仓库 aiter 被 `PYTHONPATH` 前置，遮住了镜像自带的那份），不是模型问题。
**所以 `moe_backend=triton` 必须显式传进 vLLM**——靠默认值会掉进 AITER 那条坏路。

#### 每 rank 显存实测（`rocm-smi` 采样，8 卡，288 GiB/卡）

| 阶段 | GPU0 | GPU1–7 |
|---|---|---|
| 权重加载完、KV 未分配 | 39.9 GiB | 40.5–40.7 GiB |
| + KV cache（`util=0.30`） | 90.4 GiB | 91.1–91.7 GiB |

vLLM 自报的权重是 **35.15 GiB/rank**；`rocm-smi` 多出的约 5 GiB 是 HIP context、aiter JIT 模块与工作区。
`util=0.30` 时 KV 得到 47.05 GiB（254,830 token），引擎合计约 **91.5 GiB/卡**。

⚠️ 注意 `util=0.30` 的名义预算是 288×0.30 = 86.4 GiB，实测 **91.5 GiB，超出约 5 GiB**——
`gpu_memory_utilization` 不是硬上限，§5 算账要留这份余量。

日志：`~/logs/4node/dsv4base_vllm_tp8{,b,_util030,_aiter,_auto}.log`、
`~/logs/4node/dsv4base_vram_sample.txt`。模型在 `crsuse2-m2m-068:/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-Base`
（275 GiB，46 分片，`expert_dtype: fp8`，无 `dspark_*` 键，69189 个张量）。

## 4. 第二步：写 native→HF 权重转换器（真正的门）

> 📌 **这一节已经有了专门的接续文档：`deepseek-v4-lumenrl-fsdp-handoff.md`。**
> 那份把转换器的**规格书**写全了（transformers 侧期望的 1537 个张量名与形状、逐层差异、
> 两份可抄的现成名字映射、单层验收方式），并且实测确认 transformers 5.13.1 的 V4 实现
> 是**完整且纯 PyTorch** 的（零自定义内核）。本节以下内容保留为当时的分析,
> 实操以那份为准。

`AutoModelForCausalLM.from_pretrained` 读不了这份 checkpoint：transformers 期望 1285 个张量
（`model.layers.N.mlp.experts.gate_up_proj` 这种融合 3D 布局），checkpoint 提供 72317 个
（`layers.N.ffn.experts.E.w1.weight` 这种逐专家布局），**交集 0**，且 `_checkpoint_conversion_mapping`
是 `None`。仓库自带的 `inference/convert.py` 是 **HF → 原生**的反方向。

**先做设计选择，两条路**：

**(a) 只改名字、保留 FP8 块量化**（推荐先试）。transformers 有 FP8 块量化的加载支持
（`quant_method: fp8` + `weight_block_size` + scale 张量），如果输出的 HF 布局能被它认出来，
加载时会自动反量化，`torch_dtype=bfloat16` 就得到 BF16 参数。**输出仍约 295 GB，代码量最小。**
验证成本很低：只转 1 层，`from_pretrained` 那一层看能不能加载。

**(b) 直接反量化成 BF16 落盘**。输出 **569 GB**，稳妥但大。

⚠️ **盘位**：NFS（`/home`）现在只剩 **2.5T**（10T 的共享卷，已用 76%），而且是全组共用。
569 GB 扔上去不合适。建议**转换输出放 node-local `/mnt/m2m_nobackup`**（每台 24T 空闲），
在一台上转好、验证通过后再分发到另外 3 台。代价是 4 份拷贝，收益是加载不再受 NFS 带宽制约
（上一轮 32 个 actor 从 NFS 读 57GB 花了 13 分钟，见 4 节点 runbook §11）。

**转换要做的映射**（用 `probe_dsv4_keys.py` 的两个 key 集合对照着写）：

| 源（原生） | 目标（transformers） |
|---|---|
| `embed.weight` / `head.weight` | `model.embed_tokens.weight` / `lm_head.weight` |
| `layers.N.ffn.experts.E.w1.weight` + `w3.weight` | 拼成 `model.layers.N.mlp.experts.gate_up_proj`（3D，w1 在前半、w3 在后半） |
| `layers.N.ffn.experts.E.w2.weight` | `model.layers.N.mlp.experts.down_proj`（3D） |
| `layers.N.attn.*`（`q_norm`/`kv_norm`/`attn_sink`…） | `model.layers.N.self_attn.*`（`q_a_norm`/`kv_norm`/`sinks`/`compressor.*`…） |
| `layers.N.hc_attn_base/fn/scale`、`hc_ffn_*`、`hc_head_*` | `model.layers.N.attn_hc.base/fn/scale`、`model.hc_head.*` |
| `mtp.0/1/2.*`、dspark 相关 | Base 版没有 dspark；MTP 训练时可丢（`num_nextn_predict_layers` 与 RL 无关） |
| 各 `*.scale` | 按 (a) 保留为 HF 的 scale 命名，或按 (b) 用掉后丢弃 |

**验收标准**（别跳过）：

```bash
# 1) key 覆盖：转换后 want-have 双向差集都应为空（改 probe_dsv4_keys.py 指向新目录）
python3 ~/4node/probe_dsv4_keys.py

# 2) 数值：抽若干张量，和 inference/convert.py 里的 FP4/FP8 解码逻辑对照
#    （那个文件有 FP4_TABLE 和 cast_e2m1fn_to_e4m3fn，是官方的解码参考）

# 3) 端到端：先用 1–2 层的裁剪 config 走一遍 from_pretrained，再上全模型
```

## 5. 第三步：显存账与并行度

参数 284.3B，BF16 权重 569 GB。FSDP2 训练态 = 权重 569 + 梯度 569 + fp32 master 1137 +
Adam 两动量 2274 ≈ **4.55 TB**。

| 方案 | 训练态/卡 | 引擎/卡（Base，TP=8） | 合计 | 判断 |
|---|---|---|---|---|
| 4 节点 32 卡 + rollout TP=8 | 4.55TB/32 = **142 GB** | **40.5 GiB**（实测，含 context/工作区，KV 另算） | **约 182 GiB / 288 GiB** | 账面可行，余量约 105 GiB 给激活与 KV |
| 4 节点 + rollout TP=1 | 142 GB | 约 295 GB | 超 | ❌ 不可能 |
| 16 节点 128 卡 + TP=8 | 4.55TB/128 = **36 GB** | 40.5 GiB | 约 77 GiB | 宽裕，`amd-burst-qos` 128 节点配额够 |

**结论：4 节点 + `policy.generation.vllm_cfg.tensor_parallel_size=8` 是首选**，16 节点作为撞 OOM 后的退路。
激活那部分无法预估（Qwen3-30B 那次 `actor_allocated` 10.8 GB 而 `max_reserved` 113.9 GB，
差额都是激活与工作区），所以**第一次跑务必采样 `rocm-smi` 而不是只看指标**。

⚠️ `gpu_memory_utilization` 是"占**整卡**的比例"，不看训练 actor 已占多少（vime runbook §16.2 有
一次 MoE 因此在引擎初始化阶段就 OOM）。TP=8 时引擎权重实测 35.15 GiB（+ 约 5 GiB context/工作区），
KV 另算——从 `GPU_UTIL=0.30` 起步，别照抄 Qwen3 那条线的 0.45。

⚠️ 而且**它连整卡比例都不是硬上限**：§3.1 实测 `util=0.30` 名义 86.4 GiB、实占 **91.5 GiB**，
超出约 5 GiB。单跑引擎时无所谓，和 142 GiB 训练态挤在一张卡上时这 5 GiB 要预留出来。

## 6. 第四步：跑 smoke

以 `~/4node/05_smoke_moe_tp8.sh` 为模板，改这几处：

```bash
MODEL_PATH=/mnt/m2m_nobackup/$USER/models/DeepSeek-V4-Flash-Base-hf   # 转换后的目录
EXTRA_OVERRIDE='cluster.num_nodes=4 cluster.gpus_per_node=8
  cluster.ray_address=<HEAD_IP>:6379
  controller.ray.actor.num_workers=32
  policy.generation.vllm_cfg.tensor_parallel_size=8
  policy.generation.vllm_cfg.kv_cache_dtype=fp8_e4m3          # DSv4 必需
  policy.generation.vllm_cfg.gpu_memory_utilization=0.30'
```

还要处理的三件事：

1. **`VLLM_ROCM_USE_AITER=1`**。`ray_env.sh` 里现在是 `0`（Qwen3 BF16 配方要求），但 DSv4 的稀疏注意力
   索引器在 ROCm 上**只有 AITER 实现**，关掉会硬 `raise`。这两个要求是冲突的——DeepSeek 跑起来之前
   需要在 `ray_env.sh` 里为这条线单独开一份，别直接改坏 Qwen3 那条已验证的线。
2. **`moe_backend=triton`**（不是 instruct 那条线的 `triton_unfused`，见 §3.1）。LumenRL 的 `vllm_cfg`
   目前没有这个字段，需要加一个透传（`_setup_ray_vllm_rollout` 里组 `engine_kwargs` 的地方）。
   ⚠️ **这条是必需项，不是优化项**：已实测不传的话 vLLM 自动挑中 AITER，权重加载完之后在第一次
   前向 `moe_sorting_opus_fwd` 处硬失败（§3.1 的表）。
3. **tokenizer 与数据**。V4 的 vocab 是 129280，和 Qwen3 完全不同，所以
   `data_cached/qwen3-8b-maxprompt1024/` 那份**过滤阈值不再准确**（原始 parquet 是文本，可以复用，
   但 prompt 的 token 数会变）。要么按 V4 的 tokenizer 重跑一遍 `filter_prompts.py`，要么
   把 `max_total_sequence_length` 留足余量。仓库里 `encoding/encoding_dsv4.py` 是它自带的分词实现。

判据同 Qwen3 那条线：`exit=0`、`untouched` 0 次、`verify failed` 0 次、`rollout_corr/kl` 不随步数爬升。
⚠️ 但 **MoE 权重同步的路由器是按 Qwen3 的 FusedMoE 命名写的**（`routed_experts.w13_weight`），
DSv4 在 vLLM 里的专家布局不同（256 专家 + 共享专家 + FP8/FP4），`FusedMoEWeightRouter` 大概率需要
适配。这一步会在第一次权重同步时以 `untouched` 或 `must be 3D` 的形式暴露出来，不会静默。

## 7. 不要重做的事（已排除，都有实测依据）

- **不要再试 rollout TP=1**。TP=1 在 gfx950 上必然 `Memory access fault`，五种 MoE backend / AITER
  组合全部复现过（enablement handoff §2.2）。上游验证矩阵里 TP=1 从来没被覆盖。
- **不要指望升级 transformers**。整个 V4 家族 7 个仓库的 HF 风格张量数全为 0，不存在 HF 格式权重；
  这不是版本问题。
- ~~**不要走 Megatron**~~ ⚠️ **这一条已于 2026-08-05 被推翻，见 §9。** 原文如下，其中"megatron-core 0.16
  缺 DSA 索引器"是**错的**：本镜像的 `/root/Megatron-LM` 是 NVIDIA dev（`core_v0.15.0rc7-1047-g1dcf0dafa`），
  DSA 索引器与 `experimental_attention_variant` 都在。
  > LumenRL 的 megatron 引擎在 `hf["intermediate_size"]` 就 KeyError（Base 版同样缺
  > 这个键，我确认过），bridge 是 Qwen3 专用；megatron-core 0.16 缺 `o_lora_rank` / DSA 索引器 /
  > mHC / `sqrtsoftplus`；上游实现只在 dev 分支且 ROCm 未验证（enablement handoff §4）。
  （引擎的 KeyError 和 bridge 是 Qwen3 专用这两条仍然成立，那是 LumenRL 侧要改的工作量，不是不可行。）
- **不要再排查跨节点 RDMA**。ionic RoCE 那条线的坑（memlock / provider / GID index / `ibv_reg_mr`
  EINVAL / ANP 插件的 glibc）都记在 4 节点 runbook §6，包括已经排除的六个假设。现在走 TCP 能跑通。
- **不要 fork `run_dapo.sh`**。用 PATH shim。

## 8. 交接清单

| 内容 | 位置 |
|---|---|
| 4 节点脚本与探针 | `~/4node/`（NFS，换节点只需改 `env.sh`） |
| 上一轮三个 smoke 的日志 | `~/rl_data/logs/moe-{fsdp2,megatron}-smoke-4node*.log` |
| DeepSeek 六次 vLLM 尝试的日志 | `~/logs/4node/dsv4_vllm_tp1{,b..f}.log`、`dsv4_vllm_tp8{,b,c}.log` |
| Flash-0731 模型（166.9 GB） | `/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-0731`，**只在 crsuse2-m2m-068，作业结束即失联** |
| Lumen-RL 代码改动（未提交） | `~/lumen_rl/Lumen-RL`：rollout TP>1（5 文件）+ `megatron_te_gemm_compat.py` |
| 文档 | 本文 + `deepseek-v4-flash-enablement-handoff.md` + `dapo-lumenrl-4node-32gpu-runbook.md` + `lumenrl-rollout-tp-gt-1-handoff.md` + `SPUR_NODE_ACCESS_GUIDE.md` |

**建议的推进顺序**：§3（1 小时，可能判死）→ §4(a) 单层验证（半天）→ §4 全量转换 + 验收 →
§5 显存采样 → §6 smoke。每一步都有明确的判据，任何一步失败先回来对齐方向，不要硬推。

⚠️ **2026-08-05 追加**：§4–§6 这条 FSDP2 + native→HF 转换器的路线**可能整条都不必要了**，
先读 §9。

## 9. 更好的一条路：miles 已经在同样的硬件上跑通了 DSv4 Megatron 训练

> ✅ **这条路已经走通了（2026-08-05 当天）。完整步骤见
> `deepseek-v4-miles-megatron-4node-runbook.md`** —— 4 节点 32 卡 TP8/PP4/EP8、FP8 blockwise
> 训练、rollout + optimizer step 全部跑通，`grad_norm=0.106`。本节保留为当时的可行性论证，
> 实操以那份 runbook 为准（它还记了三个 miles 自身的坑：镜像 triton 版本、
> `prepare-spmd` 的 EP=8 冲突、`--sglang-mem-fraction-static` 前后矛盾）。

`/home/xysheng/working/miles-rl/miles`（radixark/miles）里有一套完整的 DeepSeek-V4 Megatron RL 训练实现，
而且 `scripts/amd/run_deepseek_v4.py` 的注释写着：

> Verified full-model profile: **4 nodes x 8 GPUs on MI355X (gfx950)**

**跟我们这套硬件一模一样**（TP8 / PP4 / EP8，43 层切 11+11+11+10）。这条线把 §4 那个"要自己写
native→HF 转换器"的前提整个换掉了。

### 9.1 为什么它绕开了 §4 的门

miles **不走 transformers**，它是 native → Megatron `torch_dist`：
`tools/fp8_cast_bf16.py`（FP8→BF16，保持原生命名）→ `tools/convert_hf_to_torch_dist.py`
+ `miles_plugins/mbridge/deepseekv4.py`（原生命名 → Megatron）。

关键是它消费的 checkpoint 和我们的 Base **就是同一种布局**。我逐键 diff 过
`sgl-project/DeepSeek-V4-Flash-FP8`（miles 用的那份，注意它**不是** HF 布局，也是
`embed.weight` / `layers.N.ffn.experts.E.w1.weight` 那套原生命名）：

| | 张量数 |
|---|---|
| `sgl-project/DeepSeek-V4-Flash-FP8`（miles 已验证） | 69187 |
| `DeepSeek-V4-Flash-Base`（我们要用的） | 69189 |
| 交集 | **69187** |

**Base 是它的严格超集**，只多两个 MTP 张量（`mtp.0.emb.tok_emb.weight`、`mtp.0.head.weight`），
而 `--enable-mtp` 默认关、§4 也早说了 MTP 与 RL 无关可丢。两边都是 FP8 e4m3 + `weight_block_size
[128,128]` + `scale_fmt=ue8m0`。也就是说 **miles 那条已验证的流水线本来就能吃我们的 Base**。

### 9.2 本镜像（vime-rocm）到底缺什么——逐项实测

| 项 | 结论 |
|---|---|
| miles DSv4 插件需要的 11 个 megatron API | ✅ **全都有**。`/root/Megatron-LM` 是 NVIDIA dev `core_v0.15.0rc7-1047-g1dcf0dafa`，`experimental_attention_variant/{dsa,absorbed_mla}.py` 都在（探针：`~/4node/probe_megatron_dsv4_api.py`） |
| `mbridge` | ✅ 镜像里有 0.15.1（miles 钉的是 `ISEEKYAN/mbridge@89eb1088`，需比对） |
| `moe_n_hash_layers`（TransformerConfig 字段） | 不需要——miles 自己塞 `config.dsv4_n_hash_layers`，由它的插件消费 |
| `tilelang` | ⚠️ 镜像里**没装**，但 vLLM 自己就钉 `tilelang==0.1.10`，说明这一套在 ROCm 上是通的；`pip install` 即可 |
| `tile_kernels`（mHC + FP8 QAT cast-back） | ✅ `pip install tile_kernels==1.0.0`，`modeling.mhc.ops` / `quant.per_token_cast_back` 都能 import |
| TE | ⚠️ 镜像 2.12.0.dev0；miles ROCm 用 `ROCm/TransformerEngine@v2.8_rocm`。FP8 blockwise 训练（`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`）未验证——**smoke 先用 BF16**（`--fp8-training false`）绕开 |

### 9.3 DSv4 训练内核在 gfx950 上的实测结果（这是最关键的一节）

miles 的 DSv4 注意力全靠 tilelang 自研内核，**训练要 fwd + bwd 两个方向**，且
`V4Indexer` 硬 import，没有 torch 回退。我把 miles 自带的两个 manual test 直接在
vime 镜像 + gfx950 上跑了（`--noconftest` 绕开需要 sglang 的 conftest）：

| 内核 | tilelang 0.1.10 | tilelang 0.1.13 |
|---|---|---|
| sparse MLA **forward**（含真实 config H=64/D=512/topk=512） | ✅ 16/16 | ✅ |
| sparse MLA **backward** | ❌ `tvm.error.InternalError: Unresolved call tirx.exp2`（HIP codegen 不认 `exp2`） | ✅ **5/5 全过** |
| DSA indexer **fwd + bwd** | — | ⚠️ 先 45/45 全挂，见下 |

**indexer 那 45 个全挂的原因是显存预算，不是算法**：
`hipModuleLaunchKernel ... failed with error: invalid argument`。算一下
`tilelang_indexer_fwd.py` 的默认 tile：`index_k_shared` = `block_N(256) × index_dim(128) × 2B`
= 64 KB，乘 `num_stages=3` = 192 KB，加上 q 的 32 KB ≈ **224 KB**——Hopper/Blackwell 有 228 KB
放得下，**gfx950 每 block 只有 163840 B（160 KB）**，所以直接 launch 失败。

把这两个 kwarg 改成 `block_N=128, num_stages=2`（≈128 KB）之后：

```
45 passed, 18 warnings in 39.43s
```

**fwd 和 bwd 全过。** 所以 indexer 不是要重写内核，是要按 gfx950 的 LDS 预算重新 tune 两个数。

⚠️ 还有一个已知的 Hopper-only 残留：`tile_kernels/mhc/post_kernel.py` 里的 `T.pdl_sync()`，
miles 的 `docker/amd_patch/latest/tile_kernels.patch` 就是删掉它。import 阶段不会炸，
**JIT 编译 `mhc_post` 时才会**，到时候照那个 patch 删一行即可。

### 9.4 tilelang 0.1.13 会不会搞坏已验证的 vLLM rollout？——不会

vLLM 声明 `tilelang==0.1.10`，但那只是 requirement，不是运行期约束。在装了 0.1.13 的
同镜像容器里重跑 §3.1 那个 TP=8 探针：

```
Using TRITON Fp8 MoE backend ...
Model loading took 35.15 GiB memory      （与 0.1.10 下完全一致）
ENGINE LOADED in 101s
'The capital of France is' -> ' Paris. The capital of Germany is Berlin. ...'
DSV4 VLLM OK
```

日志 `~/logs/4node/dsv4base_vllm_tl0113.log`。所以 rollout 与训练可以共存在一个环境里。

### 9.5 还没验证的部分（真正的剩余风险）

1. **LumenRL 侧的集成**。miles 的 DSv4 是长在 miles 自己的 `miles_plugins` + mbridge + arguments 上的；
   LumenRL 的 megatron 引擎自己建 `TransformerConfig`/`GPTModel`、用 Qwen3 专用 bridge、**不用 mbridge**。
   这部分是实打实的移植工作量，不是装个包能解决的。
2. **Megatron→vLLM 的权重同步**。miles 同步的对象是 SGLang（`megatron_to_hf/deepseekv4.py`
   + `update_weight/common.py`），我们要的是 vLLM，且 §6 已经说了 `FusedMoEWeightRouter`
   是按 Qwen3 FusedMoE 命名写的。
3. **TE FP8 blockwise 训练**（见 9.2）。
4. **mHC 的 4-D PP buffer**：miles 文档说 HC 让 PP buffer 变成 `[s, b, hc_mult, d]` 四维，
   PP>1 时 LumenRL 的流水线代码要能吃这个形状。

### 9.6 复现用的命令

```bash
# 装依赖（建议先在一次性容器里做，别动已验证的 rl-vime-4node）
pip install tilelang==0.1.13 tile_kernels==1.0.0

# 跑 DSv4 训练内核（miles 的 conftest 依赖 sglang，用 --noconftest 绕开）
cd /home/xysheng/working/miles-rl/miles
PYTHONPATH=$PWD python3 -m pytest --noconftest -q \
  tests/manual/models/deepseek_v4/test_v4_tilelang_sparse_mla.py \
  tests/manual/models/deepseek_v4/test_v4_tilelang_indexer.py

# megatron API 探针
python3 ~/4node/probe_megatron_dsv4_api.py
# tilelang/tile_kernels 在 gfx950 上的可用性探针
python3 ~/4node/probe_tilelang_rocm.py
```

gfx950 的 indexer retune 已在 `/mnt/m2m_nobackup/xysheng/miles_scratch` 里改好（**没有动
`~/working/miles-rl` 原始树**）：`tilelang_indexer_fwd.py` 的 `block_N=128, num_stages=2`。
