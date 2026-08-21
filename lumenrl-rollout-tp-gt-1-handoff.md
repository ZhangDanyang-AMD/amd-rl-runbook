# LumenRL colocated rollout 支持 TP>1（改动与验证）

> 原来的 colocated 设计是**每卡一个 TP=1 的 vLLM 引擎**：`_setup_ray_vllm_rollout` 里
> `tensor_parallel_size=1` 是硬编码的，配置项 `vllm_cfg.tensor_parallel_size` 存在但被忽略。
> 本文记录把它做成"一个引擎跨 N 张卡"的改动、其中一个不容易发现的陷阱、以及验证方式。
>
> 改动落在 **Lumen-RL 仓库**（`dev/vllm-fsdp-dapo` 分支工作树），2026-08-05。

## 0. 为什么需要

两个用途，第二个是硬需求：

1. **省显存。** 引擎的模型副本按 TP 分片。实测 DeepSeek-V4-Flash 在 TP=1 下单卡占 **155 GiB**，
   TP=8 下每 rank 只占 **20.03 GiB**（7.7 倍）。一个 280B 级别的 policy 在 TP=1 下根本没法和训练态
   共存于同一张卡（142GB 训练 + 155GB 引擎 > 288GB），TP=8 之后账面就能过。
2. **有些模型没有 TP=1 的可用路径。** DeepSeek-V4 在 gfx950 上 TP=1 必然
   `Memory access fault`，TP=8 正常——上游 PR #40889 的验证矩阵是 `TP2/h_q=64、TP4/h_q=32、TP8/h_q=16`，
   **TP=1 从来没被验证过**。详见 `deepseek-v4-flash-enablement-handoff.md`。

## 1. 编号契约（整个改动的核心）

> **actor rank `r` → replica `r // tp`，TP rank `r % tp`。**

manager 按 rank 顺序把该组 `tp` 张卡的物理编号拼成引擎的 `CUDA_VISIBLE_DEVICES`，于是引擎的第 `j` 个
可见设备就是 actor `r*tp+j` 的那张卡，也就是 vLLM worker 的 `local_rank=j`。发送端的 socket key
`replica-{r//tp}-rank-{r%tp}` 因此必然落到**坐在自己这张卡上的那个 worker**。

这个前提**不是假设而是强制的**：如果 Ray 把一组 actor 摆到了不同节点，manager 直接抛错，而不是
静默产生错误的权重。

接收端本来就已经按 `local_rank` 区分 socket（`ipc:///tmp/lumen-colocate-zmq-{job}-replica-{r}-rank-{local}.sock`），
所以多 worker 的骨架早就在，缺的只是发送端的编号和 manager 的分组。

编排也天然对称：`server.update_weights_from_ipc` 是 `collective_rpc`，会让一个 replica 的全部 `tp` 个
worker 各自开接收器；而 `tp` 个 actor 并发发送。每个 worker 收到的是**完整张量**，由 vLLM 的
`weight_loader` 自己取分片。

## 2. ⚠️ 陷阱：vLLM 的 `full_load` 分支会跳过 TP 切片

这是整个改动里唯一不容易看出来的地方。`vllm/model_executor/layers/fused_moe/routed_experts.py`：

```python
# _load_w13 / _load_w2
if not load_full and loaded_weight.ndim > 0:
    tp_size = self.moe_config.moe_parallel_config.tp_size
    loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
    start_offset = loaded_per_rank * tp_rank
    ...
    loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)
```

`tp_rank` 的 narrow 被 `if not load_full` 守着。而 `full_load = len(loaded_weight.shape) == 3` ——
也就是说**传整个 3D 融合专家张量时，loader 假定你传进来的已经是本 rank 的分片，不做任何切片**。

`FusedMoEWeightRouter._dispatch` 原本在 EP=1 时正是走这条 3D 快路。TP>1 下它会把未分片的数据
copy 进只有 1/tp 大小的参数里 → 形状不匹配报错（好在是响亮的失败，不是静默错误）。

**修法**：EP>1 那条逐专家（2D）路径本来就会触发 loader 自己的 `tp_rank` narrow，所以把触发条件从
"EP>1" 扩成 "**EP>1 或 TP>1**"，复用已经正确的路径。代价是每个 (专家, shard) 一次 `weight_loader`
调用而不是一次，但 router gemm 那点开销相对整步可忽略。

## 3. 改动清单（5 个文件）

| 文件 | 改动 |
|---|---|
| `lumenrl/engine/inference/vllm_ray_server.py` | `VLLMReplicaManager` 接受 `tensor_parallel_size`；校验能整除 actor 数；按组切分并**强制同节点**；按 rank 顺序拼 `CUDA_VISIBLE_DEVICES`；`num_cpus=tp`（mp executor 每 TP rank 一个进程）；日志带上 TP |
| `lumenrl/trainer/rl_trainer.py` | 读 `vllm_cfg.tensor_parallel_size` 透传给 engine_kwargs 和 manager；**TP>1 时自动设 `disable_custom_all_reduce=True`**（见 §4） |
| `lumenrl/workers/actor_worker.py` | `update_weights_ipc_send` 的 socket key 改为 `replica-{rank//tp}-rank-{rank%tp}` |
| `lumenrl/engine/inference/vllm_moe_weight_sync.py` | `_dispatch` 在 TP>1 时走逐专家路径（§2）；`_verify_written` 改成 TP 感知（§5） |
| `lumenrl/tests/test_moe_weight_sync.py` | fake FusedMoE 复刻 `if not load_full` 守卫；新增 2 个测试 |

**TP=1 的行为一字未变**：`disable_custom_all_reduce` 只在 `tp > 1` 时设置，`_dispatch` 的快路条件在
tp=1 时与原来等价，socket key 在 tp=1 时化简为原式。

## 4. ⚠️ TP>1 必须关掉 custom all-reduce

colocated 路径需要 `NCCL_CUMEM_ENABLE=0`（CUDA-IPC 权重同步的要求），而 vLLM 的 custom all-reduce
通过 cuMem 分配共享缓冲。两者同开，每个 TP worker 都会死在 `init_device`：

```
custom_all_reduce.py:297 create_shared_buffer
  -> _custom_ops.py:3040 allocate_shared_buffer_and_handle
RuntimeError: HIP error: invalid argument
（下游表现为 'CustomAllreduce' object has no attribute '_ptr'）
```

改成 pynccl 走节点内 all-reduce。AMD 自己的 DeepSeek-V4 测试也是这么做的（vLLM PR #48728 的测试
说明里写着 "custom all-reduce disabled (PYNCCL)"）。因为这是设计的固有约束而不是用户该记住的
细节，所以由代码自动设置。

## 5. TP 感知的逐位校验

`LUMENRL_WEIGHT_SYNC_VERIFY=1` 是权重同步最强的正确性证据，不能因为 TP>1 就退化成"跳过"。
`_verify_written` 现在按 loader 相同的规则取切片：`per_rank * tp_rank`，w1/w3 沿维 1、w2 沿维 2
（都比 per-expert 的 shard_dim 多一个前导专家维）。

EP>1 仍然跳过——那里参数只持有一部分专家，全局到局部的专家映射在 vLLM 内部。

**任何形状不一致只降级为跳过并告警，绝不制造假报警**：这条路径新，宁可少一次校验，也不要一个
让人怀疑数据正确性的假失败。

## 6. 验证

### 6.1 单测：13 项（原 11 + 新 2）

```bash
docker exec "$CONTAINER" bash -lc 'source /home/$USER/4node/ray_env.sh
  export HIP_VISIBLE_DEVICES=0
  cd $RL_ROOT/Lumen-RL && python3 -m lumenrl.tests.test_moe_weight_sync'
```

新增两项：

- `test_tensor_parallel_takes_the_per_expert_path_and_the_right_slice` —— tp_size=2 / tp_rank=1 / I=6，
  断言所有 `weight_loader` 调用都是 2D（**没有**走 3D 快路），且 `w13[:, :3] == gate_up[:, 3:6]`、
  `w13[:, 3:] == gate_up[:, 9:12]`、`w2 == down[:, :, 3:6]`。
- `test_tensor_parallel_verify_catches_the_wrong_slice` —— 一个"声称是 rank 1 却写了 rank 0 切片"的
  loader 必须被 verify 抓住。

⚠️ **测试有牙齿是验证过的**，不是假设。把 `_tp_size` monkeypatch 成恒返回 1（模拟改动前的行为），
两个测试都以 `size of tensor a (3) must match the size of tensor b (6)` 失败——正是从 vLLM 源码
推断出的失败形态：

```bash
docker exec "$CONTAINER" bash -lc 'source /home/$USER/4node/ray_env.sh
  cd $RL_ROOT/Lumen-RL && python3 /home/$USER/4node/check_tp_test_teeth.py'
# 期望：两行 "FAILS as it should" + "both TP tests have teeth"
```

### 6.2 端到端：4 节点 32 卡 MoE smoke，TP=8

```bash
EXTRA_OVERRIDE='... policy.generation.vllm_cfg.tensor_parallel_size=8'
```

日志判据：

```
VLLMReplicaManager: launched 4 colocated rollout replicas (TP=8).
Ray vLLM rollout ready: 4 colocated replicas (TP=8, ZMQ IPC weight sync).
```

`exit=0`，3 步，指标与 TP=1 对照见 `dapo-lumenrl-4node-32gpu-runbook.md` §10.1。最关键的一行：
**`verify failed` 0 次、`verify skipped` 0 次、`untouched` 0 次** —— 96 个融合专家张量在
4 replica × 8 TP rank 上都落进了正确的切片。

## 7. 尚未验证

- **只测过 TP=8**（以及单测里的 TP=2 逻辑）。TP=2/4 的端到端没跑过。
- **只测过 Qwen3-30B-A3B MoE + FSDP2 训练后端**。Megatron 后端配 TP>1 rollout 没测；dense 模型
  （无 FusedMoE，router 不激活）没测，但那条路径上 §2 的陷阱不存在。
- **EP>1 与 TP>1 同时开**没测。`_dispatch` 会走逐专家路径（正确），但 `_verify_written` 会跳过，
  等于失去逐位校验。
- **只在 smoke 规模验证**。长跑下 TP all-reduce 的稳定性、以及 `rollout_corr/kl` 那个常量偏移
  （0.0019 vs 0.0016）是否随步数漂移，都没有数据。
- **这些改动尚未提交、未提 PR**，还在工作树里。
