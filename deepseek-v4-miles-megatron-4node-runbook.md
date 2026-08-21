# DeepSeek-V4 Runbook：用 miles 在 4 节点 32 卡 MI355X 上跑 Megatron RL 训练

> 在 **4 节点 × 8 卡 MI355X（gfx950）** 上，用 **miles**（SGLang rollout + Megatron 训练）把
> **DeepSeek-V4-Flash（284B MoE）** 的 GRPO RL 训练跑起来。
>
> 与 `dapo-miles-fsdp-sglang-runbook.md` 的关系：那份是 miles 在这个集群上的**环境配方**
> （镜像 tag、容器参数、`--ulimit nofile`），用 Qwen3-8B + FSDP。本文复用它的环境部分，
> 换成 **Megatron 后端 + DSv4**，并补上它没有的四节点编排与权重分发。
>
> 与 `deepseek-v4-base-rl-train-handoff.md` 的关系：那份的结论是"要自己写 native→HF 转换器"。
> **本文使这条路线作废**——miles 已经有完整的 native→Megatron 通路，见 §2。
>
> **已验证**（2026-08-05，spur JobID 38564）：4 节点 32 卡，TP8/PP4/EP8，FP8 blockwise 训练，
> rollout 与 optimizer step 全部跑通，`grad_norm=0.106` 非零，无 OOM。判据见 §8。

## 0. 一页现状

| 环节 | 状态 |
|---|---|
| 镜像选择 | ✅ 必须 `rocm720`，不是 `rocm700`；判据是 triton 版本，见 §1 |
| native checkpoint → Megatron `torch_dist` | ✅ miles 自带，无需写转换器，见 §2 |
| 全模型单节点转换 | ⚠️ miles 的 `prepare-spmd` 这条路是坏的，要绕开，见 §3.2 |
| 4 节点权重分发 | ⚠️ 节点间无 ssh，miles 的 `rsync_simple` 用不了，见 §4 |
| 跨节点 Ray | ⚠️ 与 vime 集群共享 host 端口空间，必须错开，见 §5 |
| 4 节点训练 | ✅ 跑通；但 `--sglang-mem-fraction-static` 要覆盖成 0.6，见 §6.2 |
| 4 层裁剪版单节点 smoke | ✅ 跑通，但产不出学习信号（设计如此），见 §7 |
| 训练/推理一致性 | ⚠️ `train_rollout_kl` 0.07，比 Qwen3 基线高两个数量级，见 §9.1 |

**这条线只支持一种形状。** `scripts/amd/run_deepseek_v4.py` 里 `_get_parallel_config` 对
32 卡以外的规模（除单节点）直接 `NotImplementedError`，`--context-parallel-size 1` 写死，
Pro 模型与 MXFP8 都从 AMD 分支里删掉了。换节点数意味着自己写 recipe 并重新验证。

## 1. 选镜像：判据是 triton 版本，不是宿主机 ROCm 版本

**这一节是本文最省时间的一节。** 我第一次按宿主机 `/opt/rocm/.info/version = 7.0.1` 选了
`rocm700`，浪费了两小时。容器自带 ROCm 用户态，与宿主机版本不必一致；**决定因素是 triton**。

DSv4 的 DSA indexer 在 ROCm 上只有一条能用的路：aiter 的 preshuffle paged-MQA 内核，而它要
`triton >= 3.5`（`sglang/srt/layers/attention/dsa/utils.py:aiter_can_use_preshuffle_paged_mqa`）。
拿不到就会被 `sglang/srt/arg_groups/overrides.py` 强制 `page_size = 1` 走 legacy 路径。

实测两个同期 tag：

| | `miles-rocm700-mi35x-20260804` | `miles-rocm720-mi35x-20260803` |
|---|---|---|
| torch | 2.9.0a0 | **2.9.1**+rocm7.2 |
| triton | **3.4.0** | **3.6.0** |
| sglang | 0.5.15.dev1138 | 0.5.17.dev32 |
| transformer_engine | 2.8.0 | 2.14.0.dev0 |
| DSA preshuffle 可用 | ❌ | ✅ |
| `transformer_engine.pytorch` 可 import | ❌（见下） | ✅ |
| 解压体积 | 95GB | 115GB |

**rocm700 的两个独立故障：**

1. `flash_attn_2_cuda.so` 是针对另一个 torch 编的，`undefined symbol:
   _ZNK3c106SymInt22maybe_as_int_slow_pathEv`。TE 的 `backends.py` 只用
   `get_pkg_version("flash-attn")` 判断"是否安装"而不判断"是否可导入"，于是
   `transformer_engine.pytorch` 整个加载失败，Megatron 起不来。
   （权宜之计是 `pip uninstall -y flash-attn` 让 TE 走 `PackageNotFoundError` 分支跳过它；
   DSv4 的注意力走自己的 tilelang 内核，不需要 flash-attn。但既然要换镜像，不必这么做。）
2. legacy DSA 路径本身有上游 bug：`page_size=1` 时 plan 张量非连续，
   `sglang/srt/layers/attention/dsv4/compressor_v2.py:329` 的裸
   `plan[1].view(torch.int32)` 在 **decode CUDA graph 捕获阶段**抛
   `RuntimeError: self.stride(0) must be divisible by 4 to view Byte as Int ... got 1`。
   同一文件第 47 行的正确写法是带 `.contiguous()` 的——说明这条路径平时没人测。

**为什么不能就地升级 triton**：aiter 自己的 `.github/scripts/install_triton.sh` 在
`torch < 2.9.1` 时会主动跳过并打印 "please upgrade torch to 2.9.1 or later"。
rocm700 正好是 2.9.0a0，死路。AOT gluon 预编译产物（`paged_mqa_logits_aot_kernel.zip`）
也没打进镜像，所以 `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` 这条备选也不通。

```bash
# 列 tag，挑一个日期 >= 你的 miles 源码快照的 rocm720
curl -s "https://hub.docker.com/v2/repositories/rocm/sgl-dev/tags?page_size=100&name=miles" \
  -o /tmp/tags.json && python3 -c "
import json
d=json.load(open('/tmp/tags.json'))
for t in sorted(d['results'], key=lambda x:x['last_updated'], reverse=True)[:20]:
    print('%-46s %s' % (t['name'], t['last_updated'][:10]))"

docker pull rocm/sgl-dev:miles-rocm720-mi35x-20260803   # 约 30GB 压缩 / 115GB 解压
```

⚠️ **四台都要拉**。而且 `nohup docker pull ... &` 在 srun step 里活不下来（step 结束就被杀），
必须前台执行；三台并行约 3.5 分钟：

```bash
for N in <三个非 head 节点>; do
  ( JOBID=$JOBID ~/nx.sh $N 'docker pull rocm/sgl-dev:miles-rocm720-mi35x-20260803 2>&1 | tail -1' ) &
done; wait
```

## 2. 为什么不需要写 native→HF 转换器

`deepseek-v4-base-rl-train-handoff.md` §4 把"写 native→HF 转换器"当作唯一的门。**不必了。**
miles 不走 transformers，它是 native → Megatron `torch_dist`：

```
FP8 native ckpt --tools/fp8_cast_bf16.py--> BF16 native
                --tools/convert_hf_to_torch_dist.py + miles_plugins/mbridge/deepseekv4.py--> torch_dist
```

关键是**它消费的 checkpoint 和 DeepSeek 官方发布的是同一种原生布局**。注意
`sgl-project/DeepSeek-V4-Flash-FP8` 名字里有 "FP8" 但**不是** HF 布局，它也是
`embed.weight` / `layers.N.ffn.experts.E.w1.weight` 那套原生命名。逐键 diff：

| | 张量数 |
|---|---|
| `sgl-project/DeepSeek-V4-Flash-FP8`（miles 已验证） | 69187 |
| `deepseek-ai/DeepSeek-V4-Flash-Base`（RL 该用的） | 69189 |
| 交集 | **69187** |

**Base 是它的严格超集**，只多两个 MTP 张量（`mtp.0.emb.tok_emb.weight`、`mtp.0.head.weight`），
而 `--enable-mtp` 默认关。两边都是 FP8 e4m3 + `weight_block_size [128,128]` + `scale_fmt=ue8m0`。
所以 **miles 这条已验证流水线本来就能吃 Base**，本文先用 instruct 版拿健康基线。

> 代价：这条路把你绑在 **SGLang** 上。miles 整个仓库没有 vLLM 后端，而且 DSv4 与 SGLang 的耦合
> 不止生成层——`precision_aligned_ops.py` 的存在就是为了和
> `sglang.jit_kernel.deepseek_v4.linear_bf16_fp32` 逐位对齐，还有 R3 routing replay 和
> 权重同步的原子融合组。换 vLLM 不是加个 shim，是重做这些数值契约。

## 3. 环境与权重准备（在 head 上做）

三个脚本都在 `~/dsv4/`，NFS 上，四台可见。`~/dsv4/miles_env.sh` 是唯一要改的路径清单。

### 3.1 起容器

照 `dapo-miles-fsdp-sglang-runbook.md` §5，**`--ulimit nofile=1048576:1048576` 不能省**
（默认软限制 1024，8 个引擎 + 训练 actor 会在第一次 rollout 结束时耗尽，raylet 崩成
`Too many open files`，看着像 Ray 的 bug）。

⚠️ **不要加 `--pid=host`。** miles 的启动器会 `pkill -9 sglang / miles / redis`；独立的 PID
namespace 才能保证它碰不到同节点上 vime 容器里那个已验证的 Qwen3+vLLM 集群。

```bash
bash ~/dsv4/10_miles_container.sh          # 四台都要跑
docker exec miles-dsv4 bash -lc 'ulimit -n'   # 确认不是 1024
```

用镜像内置的 `/root/miles`，**不要**挂自己 clone 的源码树：镜像里的 miles、Megatron fork、
tilelang 是同一天构建的配套集合。我试过混用自己 8-03 的快照，撞上版本错位——
那份快照期望 `--dsv4-*` CLI 参数，而 fork HEAD 已经不再定义它们。

### 3.2 FP8 → BF16 → torch_dist

`prepare-single`（FP8→BF16）可以直接用启动器。**但 `prepare-spmd` 对全模型 + 单节点是坏的**：

它给全模型写死 `--expert-model-parallel-size 8`，而 `convert_hf_to_torch_dist.py:58-64` 在
`PP == 1 且 world_size > 1` 时会**把 PP 悄悄改写成 world_size**（这里 8）。Megatron 随后校验
`ETP × EP × PP = 1 × 8 × 8 = 64` 对不上 world_size 8：

```
RuntimeError: world_size (8) is not divisible by
              expert_tensor_model_pipeline_parallel size (64)
```

启动器只声明 4 节点那一条 profile 验证过，这条单节点全模型转换路径显然没人跑过。
最小修法是 **EP=1**，让乘积回到 `1×1×8 = 8`，并让工具自己算 43 层怎么切：

```bash
bash ~/dsv4/17_convert_torch_dist.sh      # 内含 EP=1 与理由注释
```

两个实现细节：

- 脚本里 `source scripts/models/deepseek-v4-flash.sh` 前要 `set +u`——那个文件在定义
  `COMPRESS_RATIOS` 之前就用 `${#COMPRESS_RATIOS[@]}` 探测它，在 `set -u` 下会
  `unbound variable`（启动器自己是在没有 `-u` 的环境里 source 的）。
- **转换布局不约束训练。** torch_dist 加载时会 reshard：实测 `TP1/PP8/EP1` 写出的
  checkpoint 被训练侧以 `[ t 1/8, p 1/4 ]` 正确加载。上游 NVIDIA 文档也是 PP8/EP4 转换、
  TP8/PP8/EP8 训练。

产物尺寸（每节点都要有）：

| 目录 | 大小 | 谁需要 |
|---|---|---|
| `DeepSeek-V4-Flash-FP8` | 274G | SGLang 引擎 |
| `DeepSeek-V4-Flash-FP8-bf16` | 542G | 只是中间产物，**不用分发** |
| `DeepSeek-V4-Flash-FP8_torch_dist` | 530G | Megatron 每个 rank |

## 4. 分发到 4 节点：节点间没有 ssh

三条显而易见的路都不通：

- miles 的 `U.rsync_simple` 在**每个** Ray 节点上跑同一条 `rsync <local_src> <local_dst>`，
  也就是它假设 `--model-dir` 在共享存储上。我们的 `/mnt/m2m_nobackup` 是节点本地 NVMe，
  源端根本不存在。
- `rsync -e ssh` 不行：`xysheng@<node>: Permission denied (publickey)`。
- 放 NFF（`/home`）不行：805GB，而那是 10T 的全组共享卷、只剩 2.4T，runbook 明令禁止。

可行做法是在 head 的容器里起一个 rsync daemon（host 网络），三台并行拉。**6 分钟传完 805GB。**

```bash
# head：起 daemon（docker exec -d 能在 srun step 之间存活）
docker exec -d miles-dsv4 bash -lc 'bash ~/dsv4/18_serve_models_rsyncd.sh > ~/logs/dsv4/rsyncd.log 2>&1'

# 三台并行拉（宿主机侧就有 rsync，不必进容器）
for N in <三个非 head 节点>; do
  ( JOBID=$JOBID ~/nx.sh $N 'H=<HEAD_IP>:8730; D=/mnt/m2m_nobackup/xysheng/models
    for M in DeepSeek-V4-Flash-FP8 DeepSeek-V4-Flash-FP8_torch_dist; do
      rsync -a --partial rsync://$H/models/$M/ $D/$M/; echo "$(hostname) $M rc=$?"
    done' ) &
done; wait
```

四台校验应完全一致：`FP8=274G  td=530G  tracker=release`。

> 各节点是各自独立的一份拷贝而非共享存储。这没问题：转换是权重的纯重排，确定性的，
> 四份内容相同，等价于共享盘。

## 5. 跨节点 Ray：必须错开全套端口

同节点上 vime 容器的 Ray 集群占着 6379 / 8265 / **52365**，而 `--network=host` 让两个集群
共享 host 端口空间。

**只漏掉 dashboard agent 端口就会以一种完全看不出因果的方式失败**：agent 抢不到 52365，
把自己的端口注册成 `None`，`ray job submit` 于是 POST 到
`http://<ip>:None/api/job_agent/jobs/` → `ValueError: Invalid URL: port can't be converted
to integer`。

```bash
bash ~/dsv4/16_ray_node.sh head        # 在 head 上
bash ~/dsv4/16_ray_node.sh worker      # 在另外三台上
```

脚本里错开的端口：`--port 6380`、`--dashboard-port 8266`、
`--dashboard-agent-listen-port 52366`、`--dashboard-agent-grpc-port 52466`、
`--runtime-env-agent-port 52566`、`--node-manager-port 52666`、
`--object-manager-port 52766`、`--min/max-worker-port 20002/29999`。

另外脚本里固化了跨节点集合通信的三个变量。**这个集群的容器里 ionic RoCE 走不通**
（插件需要 glibc 2.38，且 buffer 注册返回 EINVAL；六个已排除的假设见
`dapo-lumenrl-4node-32gpu-runbook.md` §6），所以走 ens3 上的 TCP：

```bash
export NCCL_SOCKET_IFNAME=ens3 GLOO_SOCKET_IFNAME=ens3 NCCL_IB_DISABLE=1
```

⚠️ 起 head 之后要等 `ray stop` 收尾再让 worker 加入。我第一次把两步串在一条命令里，
worker 一直在连一个还没起来的 head，整个调用挂住。

期望：`ray status` 显示 4 个 node、`0.0/32.0 GPU`。

## 6. 启动 4 节点训练

```bash
docker exec -d miles-dsv4 bash -lc '
  export MILES_DIR=/root/miles MODEL_DIR=... DATA_DIR=... SAVE_DIR=...
  export MODEL_NAME=DeepSeek-V4-Flash-FP8 HEAD_IP=<HEAD_IP> RAY_DASH=8266
  export EXTRA="--extra-args --sglang-mem-fraction-static=0.6"
  bash ~/dsv4/22_run_dsv4_full_4node.sh > ~/logs/dsv4/dsv4_full_4node.log 2>&1'
```

### 6.1 用外部 Ray，别让启动器自己起

脚本里设了 `MILES_SCRIPT_EXTERNAL_RAY=1` + `RAY_ADDRESS=http://<HEAD_IP>:8266`。
看 `command_utils.py:131-153`：`external_ray` 为真时它会跳过 `ray stop --force` 和
`pkill -9 ray`，只保留无害的 `pkill sglang/miles/redis`。这是保护 vime 集群的关键开关。

`MASTER_ADDR` 必须是 head 的可路由地址（`execute_train` 直接从环境读，默认 `127.0.0.1`，
那对跨节点 torch.distributed 是错的）。`NCCL_IB_DISABLE` **不在** miles 的透传白名单里
（白名单只有 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `NCCL_DEBUG` / `NCCL_DEBUG_FILE`），
所以要用 `--extra-env-vars` 送进 runtime_env。

### 6.2 `--sglang-mem-fraction-static` 必须覆盖成 0.6

**miles 脚本内部自相矛盾。** 4 节点分支的注释明确写着要配 0.6：

```python
if args.actor_num_nodes == 4:
    # 4-node PP4 memory balance: partial optimizer offload (keep ~25% on GPU) + keep train
    # weights on GPU; pair with --sglang-mem-fraction-static 0.6.
    optimizer_args += "--optimizer-offload-fraction 0.75 " "--no-offload-train "
```

但 `misc_args` 里硬编码的是 `--sglang-mem-fraction-static 0.7`。这一分支同时带
`--no-offload-train`（训练权重常驻显存），0.7 直接把引擎挤爆：

```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 76.00 MiB.
GPU 1 has a total capacity of 287.98 GiB of which 0 bytes is free.
Of the allocated memory 200.12 GiB is allocated by PyTorch
```

`train_args` 里 `extra_args` 拼在最后，argparse 后者生效，所以
`--extra-args --sglang-mem-fraction-static=0.6` 就能覆盖。确认方式是看日志里
`sglang_mem_fraction_static ... 0.6`（前面还会出现一行 0.7，那是被覆盖掉的）。

### 6.3 启动器自动给出的并行度

`_get_parallel_config` 对 4×8 给出启动器声明验证过的 profile，不用自己填：

```
TP=8 + --sequence-parallel, PP=4 (43 层 = 11+11+11+10), CP=1, EP=8, ETP=1
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
--micro-batch-size 1 --max-tokens-per-gpu 2048
--optimizer-cpu-offload --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d
--optimizer-offload-fraction 0.75 --no-offload-train
--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise
--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}'   # gfx950 用 fp32 scales
--no-gradient-accumulation-fusion                               # ROCm TE MoE FP8 缺 fused wgrad
--qkv-format bshd --moe-router-freeze-gate --freeze-e-score-correction-bias
```

## 7. 可选：先跑 4 层裁剪版单节点 smoke

`Pinaster/DeepSeek-V4-Flash-FP8-4layer`（27G）是 `sgl-project/DeepSeek-V4-Flash-FP8` 的
4 层裁剪，`COMPRESS_RATIOS=(0 0 4 128)` 恰好覆盖三种层型（2 个无压缩、1 个 CSA ratio=4
带 indexer、1 个 HCA ratio=128）。单节点 8 卡走 `TP=8 + SP / PP=1 / EP=8`。

```bash
bash ~/dsv4/20_run_dsv4_4layer_smoke.sh   # full-train：download → cast → spmd → train
```

**它验证管线，不验证收敛。** 实测 `grad_norm` 恰好 0.0、`zero_std/count_0 = 32`、
`truncated_ratio = 1.0`、回复长度全是 4096 —— 裁剪模型产不出有意义文本，GRPO 组内归一化后
优势恒为 0。启动器 docstring 自己写了 "Cannot generate meaningful output - pipeline-only
sanity check"。**别把 grad_norm=0 当成故障。**

它对 4 层的 `prepare-spmd` 是好的（那条分支用 EP=1，不触发 §3.2 的冲突）。

## 8. 判据（4 节点全模型实测）

### 8.1 阶段性里程碑

| 阶段 | 期望证据 |
|---|---|
| torch_dist 转换 | `successfully saved checkpoint ... [ t 1/1, p 1/8 ]`，tracker 内容 `release` |
| 训练侧加载 + reshard | `successfully loaded checkpoint from ..._torch_dist [ t 1/8, p 1/4 ] at iteration 0` |
| 每 rank 参数量 | 8.5–9.2 B（284B / 32 ≈ 8.9B） |
| SGLang 引擎就绪 | `The server is fired up and ready to roll!` × 8 |
| 权重同步 | `update_weights phase=end ok=true`，引擎侧 `POST /update_weights_from_tensor 200 OK` |
| decode CUDA graph | 无异常通过（rocm700 正是死在这里） |

### 8.2 step 0 指标

```
train/grad_norm                       0.1061      <- 非零是关键
train/loss / pg_loss                  1.98e-09
train/ppo_kl                          5.97e-13
train/ess_ratio                       1.0
train/train_rollout_logprob_abs_diff  0.1529
train/train_rollout_kl                0.0707
train/lr                              1e-06
```

`loss ≈ 0` 是**正确的**：`--num-steps-per-rollout 1` 下 old==new，ratio 恒为 1，而 GRPO
组内归一化让优势均值为 0；但优势本身非零，所以梯度不为零。要看的是 `grad_norm`。

### 8.3 rollout 0 指标

```
rollout/response_len/{mean,median,max,min}  3113 / 4096 / 4096 / 253
rollout/truncated_ratio                     0.5625
rollout/zero_std/{count_0,count_1}          13 / 10        <- count_1>0 说明真在解题
rollout/repetition_frac                     0.0
```

对照 4 层裁剪版（全 4096、截断率 1.0、count_0=32、grad_norm=0），差别一目了然。

### 8.4 性能与显存

```
perf/step_time         1306.6 s   (约 21 分钟/步)
perf/train_time         942.4 s
perf/actor_train_time   590.0 s
perf/log_probs_time     352.0 s
perf/update_weights_time 276.2 s
perf/rollout_time        85.3 s
perf/wait_time_ratio     0.279
perf/actor_train_tok_per_s  1413.8
```

显存：训练期 119–123 GiB/卡；`update_weights` 前 `used 179 GB / free 109 GB`
（`allocated 60.3 GB`）。健康状态下 GPU 利用率 86–89%、实时功耗约 480–540W。

> ⚠️ **两条 step 之间会有长时间"安静期"**，日志尾部只有 SGLang 的 `GET /health` 在刷。
> 那不是卡住：训练阶段引擎被 TorchMemorySaver 睡掉，只有健康轮询打日志，Megatron 要到
> 阶段边界才吐指标。判断是否 hang 看三样：日志 mtime、`ray job list` 状态、
> `rocm-smi --showuse`（应 85%+）。

## 9. 已知问题与剩余风险

### 9.1 `train_rollout_kl` 比 Qwen3 基线高两个数量级

实测 0.0707 / `logprob_abs_diff` 0.153，而 Qwen3-8B BF16 基线是 1.2e-3
（`dapo-miles-fsdp-sglang-runbook.md` §10.1）。**不能直接比**（那边是 BF16 稠密 + vLLM，
这边是 FP8 blockwise 训练 + FP8 SGLang rollout 的 284B MoE），但它是训练/推理一致性的核心
指标，值得盯。

**一个具体嫌疑**：NVIDIA 版启动器会生成一个 TE 精度覆盖 YAML，用
`--te-precision-config-file` 把 DSA indexer 的 `linear_weights_proj` 在 FP8 训练下钉在 BF16：

```yaml
matchers:
  dsa_indexer_weights_proj_bf16:
    pattern: "*.self_attention.indexer.linear_weights_proj"
    config: "bf16"
```

**AMD 分支把这一项整个删掉了**（`scripts/amd/run_deepseek_v4.py` 里没有
`_DSV4_TE_PRECISION_CONFIG`，也没有 `--te-precision-config-file`）。同时 AMD 的 compressor
GEMM 走 upcast-to-FP32 的另一条路（`precision_aligned_ops.py:20-24`，因为 HIP 没有
bf16-in/fp32-out 的 `torch.mm`）。两者都是刻意的 parity 近似，但确实是**近似**。

两个可验证方向，都还没做：

1. `--fp8-training false` 跑 BF16 对照。若 `train_rollout_kl` 掉回 1e-3 量级，
   就说明偏差来自 FP8 路径而非权重同步。**这是更干净的判别实验。**
2. 给 AMD 这条线补回那个 YAML。

判据：**看它是常数偏移还是随步数爬升**。爬升说明权重同步或 kernel 真有对不齐的地方；
常数偏移则更像 FP8 量化噪声。

前两步实测：

| step | `grad_norm` | `train_rollout_kl` | `logprob_abs_diff` | `ppo_kl` | `ess_ratio` |
|---|---|---|---|---|---|
| 0 | 0.1061 | 0.07068 | 0.1529 | 5.97e-13 | 1.000 |
| 1 | 0.1367 | 0.07921 | 0.1658 | 0.00 | 1.000 |

量级稳定在 0.07–0.08，没有失控；但 **两个点分辨不出"带噪声的常数偏移"和"缓慢爬升"**
（kl +12%、logprob_diff +8%），需要 5–10 步才能判。`ess_ratio` 恒为 1.0、`ppo_kl` ≈ 0
说明单步优化本身是自洽的，偏差全部来自训练与 rollout 两套实现之间。

### 9.2 上游没有 ROCm CI 覆盖 DSv4

ROCm CI 套件（`stage-c-{2,4,8}-gpu-mi350`）只覆盖 Qwen/MiMo/LoRA；DSv4 的门是
`stage-c-4-gpu-h200`，**只在 NVIDIA 上跑**。gfx950 的正确性依据是
`scripts/amd/run_deepseek_v4.py:7` 的一句 docstring
（"Verified full-model profile: 4 nodes x 8 GPUs on MI355X (gfx950)"）加三个 commit。
另外那个 4 层 CI 的五个 metric gate 全是 `enforce=False`，只报告不拦。

### 9.3 其它

- **32 卡以外没有余量。** `_get_parallel_config` / `_prepare_spmd` 对其它规模抛
  `NotImplementedError`；CP 写死 1（且 DSv4 只支持 contiguous all-gather CP，
  `arguments.py:3172` 断言 `--allgather-cp`）；Pro 与 MXFP8 在 AMD 分支里被删。
- **AMD 分支是 launcher 的硬 fork**，与 mainline 无共享代码，上游 DSv4 改动不会自动传播。
- **被绕过而非修好的 ROCm 不稳定项**：MTP 下 aiter all-gather 死锁（故 `SGLANG_USE_AITER_AG=false`）、
  aiter GEMM autotune 慢到触发 SGLang watchdog（故 `--sglang-watchdog-timeout 1800`）、
  ROCm 7.2 colocate 下 ROCr VMM 泄漏（镜像层 `HSA_ENABLE_IPC_MODE_LEGACY=1`）。
- **sparse-MLA / indexer 的 backward 用 atomic_add 累加到 fp32 buffer**，
  因此**天生非确定性**——尽管启动器设了 `--deterministic-mode` 和
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`。
- 本轮 `--no-enable-eval --skip-saving`，所以 **AIME eval 与 checkpoint 落盘都没验证过**。

## 10. 排障速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `AttributeError: module 'transformer_engine' has no attribute 'pytorch'` | rocm700 的 flash-attn wheel 与 torch ABI 不匹配，把 TE 带崩 | 换 rocm720（§1）；或 `pip uninstall -y flash-attn` |
| `RuntimeError: self.stride(0) must be divisible by 4 to view Byte as Int` | triton<3.5 → DSA 降级到 `page_size=1` legacy 路径的上游 bug | 换 rocm720（§1） |
| `world_size (8) is not divisible by ... size (64)` | `prepare-spmd` 的 EP=8 与工具自动改写 PP=world_size 冲突 | 绕开，用 EP=1（§3.2） |
| `aiohttp ... InvalidUrlClientError: http://<ip>:None/api/job_agent/jobs/` | dashboard agent 抢不到 52365（vime 集群占着），端口注册成 None | 错开全套 Ray 端口（§5） |
| `torch.OutOfMemoryError: HIP out of memory ... 0 bytes is free`，引擎侧 | `--sglang-mem-fraction-static 0.7` 与 `--no-offload-train` 冲突 | 覆盖成 0.6（§6.2） |
| `COMPRESS_RATIOS: unbound variable` | 自己的脚本开了 `set -u` 去 source 模型脚本 | source 前后 `set +u` / `set -u`（§3.2） |
| worker `ray start` 一直挂住 | head 的 `ray stop` 还没收尾就让 worker 去连 | 分两步、确认 head 起来再加入（§5） |
| `docker pull` / `nohup ... &` 悄悄没执行 | 后台进程随 srun step 结束被杀 | 前台执行，多台用并行 subshell（§1） |
| `grad_norm` 恰好 0.0 | 用的是 4 层裁剪版 | 正常，见 §7 |
| 日志长时间只刷 `GET /health` | 训练阶段引擎已 sleep | 不是 hang，看 mtime / job 状态 / `rocm-smi --showuse`（§8.4） |
| 节点间 `rsync -e ssh` 被拒 | 该用户没有 inter-node ssh | 用 rsyncd（§4） |

## 11. 产物位置

| 内容 | 位置 |
|---|---|
| 本条线的脚本 | `~/dsv4/`（NFS）：`miles_env.sh`（路径清单）、`10_miles_container.sh`、`15_prepare_full_on_head.sh`、`16_ray_node.sh`、`17_convert_torch_dist.sh`、`18_serve_models_rsyncd.sh`、`20_run_dsv4_4layer_smoke.sh`、`22_run_dsv4_full_4node.sh` |
| 环境探针 | `~/dsv4/probe_miles_dsv4_env.py`（镜像里 megatron fork / tilelang / tile_kernels / indexer tile 是否齐备） |
| 日志 | `~/logs/dsv4/`：`dsv4_full_4node_mf06.log`（成功那次）、`dsv4_full_4node.log`（0.7 OOM 那次）、`dsv4_4layer_720.log`、`dsv4_4layer_smoke2.log`（rocm700 崩那次）、`convert_td.log`、`prepare_full.log` |
| 模型（每节点本地） | `/mnt/m2m_nobackup/xysheng/models/DeepSeek-V4-Flash-FP8{,-bf16,_torch_dist}`、`...-4layer{,-bf16,_torch_dist}` |
| 镜像 | `rocm/sgl-dev:miles-rocm720-mi35x-20260803`（用这个）、`...-rocm700-mi35x-20260804`（不要用，见 §1） |
| miles 源码快照 | `~/working/miles-rl/miles`（8-03；**运行时用镜像内置的 `/root/miles`**，这份只作离线阅读） |

## 12. 一句话流程

```bash
# 1. 四台拉 rocm720 镜像（前台，并行）                        §1
# 2. 四台起容器（--ulimit nofile 必需，别加 --pid=host）      §3.1
# 3. head：下模型 → FP8→BF16 → torch_dist（EP=1 绕坑）        §3.2
# 4. rsyncd 分发 274G + 530G 到另外三台（约 6 分钟）          §4
# 5. 起跨节点 Ray，端口全部错开，确认 0.0/32.0 GPU            §5
# 6. 启动训练，务必覆盖 mem-fraction=0.6                     §6
docker exec -d miles-dsv4 bash -lc '... EXTRA="--extra-args --sglang-mem-fraction-static=0.6" \
  bash ~/dsv4/22_run_dsv4_full_4node.sh > ~/logs/dsv4/dsv4_full_4node.log 2>&1'
# 7. 判据：grad_norm 非零、count_1>0、reshard 成 [t 1/8, p 1/4]  §8
# 8. 盯 train_rollout_kl 是常数偏移还是爬升                    §9.1
```
