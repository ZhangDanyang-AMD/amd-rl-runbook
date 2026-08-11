# LumenRL Megatron 后端「方案 B」重构交接文档（原生 Megatron-Core 路径）

> 目的：把 LumenRL 自写的轻量 `MegatronEngine`（为 "8B + DP8" 定制）重构为**复用 Megatron-Core 原生路径**
> （TransformerEngine spec + pipeline schedule + dist-checkpoint），以支持 **TP/PP/CP**、可重分片 checkpoint、
> HF 互转，并解决自写路径的 save/load 冗余、不能重分片等问题。
>
> 本文档供**新 agent 无上下文接手**。已完成阶段 1（TE spec DP8）+ 阶段 3（dist-checkpoint）+ **阶段 2a（TP）** + **阶段 2b（PP）** + **阶段 2c（CP）**；DP8、TP、PP、CP、TP×PP 3D、**TP×PP×CP 4D** 均已全链路验证，CP2 dist-checkpoint save/resume 也已通过。
> 剩余仅为性能优化项（主要是 vocab-parallel logprob、PP 权重同步分桶等），无功能性 topology 缺口。参考 runbook：
> - `lumenrl-megatron-native-env-build-runbook.md`（**新机从零构建环境：固定镜像 + Apex + ROCm TE 2.15 + 完整 smoke**）
> - `dapo-lumenrl-native-vllm-megatron-runbook.md`（自写 engine 路线，§7.1 megatron_cfg / FA+chunk+packing）
> - `dapo-vime-vllm-megatron-runbook.md`（**vime 原生 Megatron**，§5 HF→torch_dist 转换、TP/PP 参数——阶段 2 的直接参考）

---

## 0. 环境现状（先读，避免踩坑）

| 项 | 值 |
|---|---|
| 容器 | `rl-vllm-megatron`（基于 `vllm/vllm-openai-rocm:v0.23.0` 起，rocm7.2.3 / gfx950 / MI355X ×8） |
| RL_ROOT（代码） | `/home/xysheng/working/11/lumen_rl` |
| DATA_ROOT（模型/数据/ckpt/日志） | `/mnt/m2m_nobackup/xysheng/data`（28T nvme，充足） |
| Lumen-RL 分支 | `dev/vllm-fsdp-dapo`（`$RL_ROOT/Lumen-RL`） |
| 容器内挂载 | `-v $RL_ROOT:$RL_ROOT -v $DATA_ROOT:$DATA_ROOT`；`_te_apex_transfer`、`te_src`、`apex_src` 已拷到 `$DATA_ROOT/`（容器可见） |

**⚠️ 关键约束：**
1. **容器未 `docker commit`**（用户明确要求不 commit）。TE 2.15 + apex 是**源码编译**装在容器 site-packages（overlay 层）——`docker rm` 会丢失（TE 编译约 22min）；`docker stop/start` **不丢**。若要保存：`docker commit rl-vllm-megatron <tag>`（需用户同意）。
2. **代码改动全部在工作区、未 commit**（用户要求 only edit，不 commit/push）。`git -C $RL_ROOT/Lumen-RL status` 可见。
3. `/home` 是共享 NFS、**99% 满**（别的用户占的，与我们无关）。训练日志/ckpt/wandb 都写 `$DATA_ROOT`（nobackup），不受影响；但 `/home` 上 git/编辑偶发 `EDQUOT (-122)`，重试即可。
4. **cumem sleep 在 rocm7.2.3 上坏**（`hipMemUnmap/Release` no-op，`wake_up` 会 OOM）→ 所有 config 已设 `enable_sleep_mode: false`。别改回 true。
5. 装 apex 后 **local spec 默认用 apex FusedLayerNorm，不支持 RMSNorm 会报错**；自写 `MegatronEngine` 因强制 `WrappedTorchNorm` 不受影响，原生 TE spec 用 TE norm 也不受影响。别在 local spec 下不覆盖 norm。
6. ckpt 目录由容器内 **root** 建，host（xysheng）删不掉；清理用 `docker exec rl-vllm-megatron rm -rf <path>`。

### 依赖（已装并验证）
- `megatron.core` 0.18.2 + **`megatron.training`（完整 Megatron-LM）** + `dist_checkpointing` + `pipeline_parallel.get_forward_backward_func` + TP vocab-parallel CE —— 都在。
- **TransformerEngine 2.15.0.dev0**（源码 gfx950 编译，含 `USE_FUSED_ATTN_CK`+`USE_FUSED_ATTN_AOTRITON`）：`import transformer_engine.pytorch`、TE Linear/RMSNorm/DotProductAttention fwd+bwd 均验证 OK；megatron TE spec = `TEDotProductAttention` + `TELayerNormColumnParallelLinear`。
- **apex 1.14.0a0**（源码 gfx950）：FusedLayerNorm/FusedAdam 验证 OK。
- 缺 `megatron.bridge`（HF↔megatron 自动转换；阶段 2 若走 vime 的 `convert_hf_to_torch_dist` 需要，或用手写 bridge）。
- 重新编译命令（若换容器）：
  ```bash
  # apex
  cd $DATA_ROOT/apex_src && rm -rf build && PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=64 python setup.py install --cpp_ext --cuda_ext
  # TE（约 22min；CK 走 JIT、aotriton 用预下载 image）
  cd $DATA_ROOT/te_src && NVTE_FRAMEWORK=pytorch PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=64 pip install -v . --no-build-isolation
  # 注意先 git config --global --add safe.directory "*"（源码 owner 非 root）
  ```

---

## 1. 已完成（阶段 1/2/3，均已验证）

### 阶段 1：原生 TE spec engine（DP8 跑通）
新增 `lumenrl/engine/training/megatron_native_engine.py`：`MegatronNativeEngine(MegatronEngine)`
- override `initialize()`：用 `get_gpt_layer_with_transformer_engine_spec`（TE fused attn + TELayerNormColumnParallelLinear + TE RMSNorm），pipeline-stage-aware（`pre_process=mpu.is_pipeline_first_stage()` / `post_process=...last_stage()`），HF 权重经 **TE 命名 bridge** 加载，DDP + 分布式优化器 + scheduler。
- override `get_per_tensor_param()`：`megatron_to_hf(..., te=True)` 导出给 vLLM。
- **继承复用**父类的 forward / packing（`_forward_logits`/`_forward_logits_packed`，TE 原生支持 packed thd）/ fused-chunked logprob / optimizer-step。
- 注册 `backend="megatron_native"`。
- 阶段 1 最初只支持 TP=PP=CP=1；该限制已由后续阶段 2a/2b/2c 全部移除。

验证（DP8 smoke，`training_backend=megatron_native`）：entropy≈0.6、grad_norm≈0.8、ppo_kl≈0、稳态 **train_s≈1.4s**（比 local spec 的 1.9s 更快）。

### 阶段 3：dist-checkpoint（sharded、可重分片）
`MegatronNativeEngine` 加 `save_dist_checkpoint` / `load_dist_checkpoint`：
- `module.sharded_state_dict()` + `optimizer.sharded_state_dict(model_ssd, metadata={"distrib_optim_sharding_type": "dp_zero_gather_scatter"})` → `megatron.core.dist_checkpointing.save/load`。
- **关键坑**：默认 `fully_sharded_model_space` 会产生 `flattened_range` ShardedTensor，torch_dist save 报 `ShardedTensor.flattened_range is not supported` → 必须用 `dp_zero_gather_scatter`（或研究 `fully_reshardable`）。
- `actor_worker.save_checkpoint/load_checkpoint` 已改：`hasattr(engine, "save_dist_checkpoint")` 时走 dist-ckpt，否则旧的 per-rank torch.save。

验证（DP8）：save→`resume_step=2`→续训健康（optimizer fp32 master+Adam、scheduler lr、step 都正确恢复）。**这是 LumenRL Megatron 路径首次验证 load/resume 通过。**

### 阶段 2a：张量并行 TP（DP4×TP2 全链路跑通 + 验证）
`MegatronNativeEngine` 现支持 **TP>1（并可与 PP/CP 组合）**，全链路（rollout→logprob→DAPO 更新→dist-ckpt save/resume→权重同步回 vLLM）在 DP4×TP2 验证通过：
- `GPTModel(parallel_output=False)` 让 output_layer 把 TP 分片的 logits all-gather 成全 vocab（每个 TP rank 得到相同全 vocab logits），从而**复用父类全 vocab logprob/entropy** 不改。`sequence_parallel` 默认 **False**（RL 变长序列不满足 seq%tp==0；纯 TP 无此约束）。
- **TP 权重加载** `_shard_hf_for_mp()`：full HF→`hf_to_megatron(te=True)` 得全量 Megatron 权重，再用 `model.sharded_state_dict()` 每个 ShardedTensor 的 `global_slice()[prepend_axis_num:]` 切本 rank 分片（qkv 交织/vocab/row/col 都自动对；融合 SwiGLU `linear_fc1` 是 `ShardedTensorFactory`，按 `[gate(ffn);up(ffn)]` 各列并行拼接**显式**处理）。
- **TP 权重同步回 vLLM** `_full_megatron_named_params()`：每个 colocated rollout replica 是 TP=1、需要完整模型，故 `get_per_tensor_param` 先在 TP 组 `all_gather` 每个 param 还原全量张量（`_shard_hf_for_tp` 的逆），再 `megatron_to_hf`。**HF→TP-shard→load→TP-gather→HF roundtrip 在 8B 上逐张量 bit-exact 验证通过。**
- **qk_layernorm TP 梯度同步**由 Megatron `finalize_model_grads`（父类 `_optimizer_step` 已调用）自动处理（`config.qk_layernorm=True` → q/k layernorm grad 跨 TP SUM all-reduce）；SP=False 下其余 replicated norm 各 rank 输入相同→梯度天然一致，无需额外同步。
- **控制器 DP×TP 集成**（全部 gated 在 TP>1，DP8 路径零改动）：
  - `rl_trainer` 通过各 actor 的真实 Megatron DP rank 构造 `mesh_mapping`，同一模型并行组收到相同数据分片；
  - `engine.is_mp_src_rank_with_outputs()` 在纯 TP 下返回 `tp_rank==0`（组合 topology 还要求 cp0 + PP last）；非 src rank 返回空 DataProto，避免 controller 重复 merge；
  - 损失归一使用纯 DP 宽度（组合 topology 为 `num_workers // (tp*pp*cp)`）。

验证：`training_backend=megatron_native` + `megatron_cfg.tensor_model_parallel_size=2`（8 卡=DP4×TP2）2-step smoke，指标健康（entropy≈0.67、grad_norm≈0.83、ppo_kl≈0、rollout_corr/kl≈0.002）；TP2 dist-ckpt save→`resume_step=2`→step3/4 续训健康（lr/step/optimizer 正确恢复）。DP8 native 回归同样健康（train_s≈1.3s 不变）。

### 阶段 2b：流水并行 PP（DP4×PP2 全链路跑通 + 验证）
`MegatronNativeEngine` 现支持 **PP>1（并可与 TP/CP 组合）**，全链路（rollout→logprob→DAPO 更新→dist-ckpt save/resume→权重同步回 vLLM）在 DP4×PP2 验证通过：
- **pipeline schedule 前反向**：`engine_update_policy`/`engine_compute_log_probs` 在 PP>1 时改走 `megatron.core.pipeline_parallel.get_forward_backward_func()`（1F1B）。每个 **microbatch = 一个 packed bin**（复用 `_build_bins`/`_forward_logits_packed` 的 thd 打包）；`forward_step_func` 拉一个 bin→`model(input_ids,position_ids,packed_seq_params)`（中间 stage 输出 hidden 经 P2P 传下一 stage），`loss_func`/collect 只在 last stage 跑（复用父类 `_row_policy_loss` / `_logprob_entropy_nograd`）。变长序列靠 `TransformerConfig(variable_seq_lengths=True)`（P2P 动态交换 shape；同时设 `moe_token_dispatcher_type="alltoall"` 过 dense 模型的配置校验）。
- **loss 归一**：CP=1 时 schedule 内部把 loss 除以 `num_microbatches`，故 `loss_func` 返回 `bin_loss * num_mb` 抵消；CP>1 还需除 schedule 的隐式 CP 因子（见阶段 2c）。梯度 finalize（DP reduce + TP layernorm + PP embedding 同步）由 `config.finalize_model_grads_func=finalize_model_grads` 在 schedule 末尾统一做，之后 `optimizer.step()`。
- **PP 权重加载** `_shard_hf_for_mp`（`_shard_hf_for_tp` 泛化）：state_dict key 是**本 stage 局部层号**，但 ShardedTensor `global_offset[0]` 是**全局层号** → 用它把局部 key 映射到 `meg_full` 的全局层 key，只取本 stage 的层（TP 切分逻辑不变）。
- **PP 权重同步** `_full_megatron_named_params`：先 TP-all-gather 本 stage 参数（全局层号命名），再**跨 PP 组 broadcast** 各 stage 的层（先 `all_gather_object` 交换 (key,shape,dtype) 元数据，再逐张量 `dist.broadcast`），使每个 rank 都拿到完整模型给 TP=1 的 vLLM replica。
- **控制器 mesh**：source 条件现为 `tp_rank==0 && cp_rank==0 && pipeline_last_stage`；trainer 通过 `actor_worker.get_mp_info()` 查每个 actor 的真实纯-DP rank 建 `mesh_mapping`（对 Megatron 排序鲁棒）；`dp_size` 取不含 CP 的 DP world size；非 last-stage 的 `engine_update_policy` 只回 `grad_norm/lr`（避免 metric 平均被稀释）。

验证：`megatron_cfg.pipeline_model_parallel_size=2`（8 卡=DP4×PP2）2-step smoke 健康（entropy≈0.6–0.75、grad_norm≈0.75–0.85、ppo_kl≈0、rollout_corr/kl≈0.002）；PP2 dist-ckpt save→`resume_step=2`→step3/4 健康；独立引擎测试（HF→PP-shard→load→PP-gather→HF **逐张量 bit-exact** + pipeline update grad 有限非零 + logprob 形状正确）通过。

### 阶段 2c：上下文并行 CP（zigzag thd，已完成 + 全链路验证）
`MegatronNativeEngine` 现支持 **CP>1**，并可与现有 TP/PP 同时开启：
- **CP-aware packed 输入**：`_pp_forward_model` 对每条真实序列长度 `L` 取 `chunk=ceil(L/(2*cp))`，右 pad 到 `2*cp*chunk`；CP rank `r` 取 `chunk[r]` 与 `chunk[2*cp-r-1]`。bin 内拼接各序列的本地 token，`PackedSeqParams.cu_seqlens = local_cumsum * cp_size`、`qkv_format="thd"`，TE `TEDotProductAttention` 负责跨 CP rank 的 causal ring attention。
- **shift-by-one 与重构**：`_cp_row_logprob_entropy` 只在本 rank 持有的两个 chunk 上算 logits/logprob，但保留 chunk 尾部 logit 对下一 chunk 首 token 的预测（target 从原始完整 `ids` 取）；再按全局 logit 位置散回 `[L-1]`。训练用 `torch.distributed.nn.functional.all_reduce`（可微 SUM），推理的 logprob/entropy 用普通 CP SUM，最终恢复完整响应顺序。
- **DAPO/global-token 归一**：controller 传入的 `dp_size` 始终是**不含 CP**的纯 DP 宽度。Megatron two-value schedule 会隐式做 `loss *= cp/num_microbatches`，所以 loss closure 必须返回 `bin_loss * num_mb / cp`。可微 CP gather 的 backward SUM 提供一个 `cp` 因子，随后 DDP 在 `DP×CP` 组上的 AVG 恰好消掉它；若不除 schedule 的 CP 因子，CP2 梯度会精确放大 2 倍。
- **与 PP 组合**：CP>1（即使 PP=1）统一走 `get_forward_backward_func()`；PP=1 用 no-pipeline schedule，PP>1 用原生 1F1B，同一个 packed microbatch/重构逻辑，不另写旁路。
- **controller/source rank**：`_compute_actor_mp = tp*pp*cp`；`get_mp_info()` 显式用 `with_context_parallel=False` 返回纯 DP rank/size，并回传 cp rank/size。一个 DP shard 内所有 TP/PP/CP rank 收到相同数据；仅 `tp_rank==0 && cp_rank==0 && pipeline_last_stage` 向 controller 返回重构后的输出。

验证（2026-07-23，所有日志在 `$DATA_ROOT/logs/`）：
- **重构/shift 单测**（临时脚本已删）：长度 2/3/4/5/7/8/9/13 的 synthetic logits，CP2 重构 logprob、entropy、可微 gather 梯度对直接完整序列结果均 `max_error=0`。
- **tiny TE GPTModel CP1/CP2 数值对齐**（2 layers / hidden 256 / BF16；临时脚本与输出已删）：logprob max/mean abs diff=`5.687e-3/1.166e-3`，entropy=`8.011e-5/1.816e-5`；CP2 backward finite 且 nonzero。差异来自 BF16 下 CP/非 CP attention reduction 顺序，不是 token 错位。
- **DP4×CP2 Ray 2-step smoke**：mesh `[0,0,1,1,2,2,3,3]`；step1/2 entropy=`0.652/0.716`、grad_norm=`0.874/0.960`、ppo_kl≈0、rollout_corr/kl=`0.00222/0.00233`，稳态 train_s=`1.54s`；日志 `native-cp2-smoke.log`。
- **CP2 dist-checkpoint**：step2 保存约 107GB，随后 `resume_step=2`，step3/4 续训；step4 entropy=`0.662`、grad_norm=`0.976`、ppo_kl≈0；日志 `native-cp2-ckpt-save.log` / `native-cp2-ckpt-resume.log`。大 checkpoint 已清理。
- **DP8 回归**：step2 entropy=`0.625`、grad_norm=`0.855`、ppo_kl≈0、稳态 train_s=`1.32s`；日志 `native-dp8-cp-regression.log`。
- **4D 组合 DP1×TP2×PP2×CP2**：mesh 全 0，2 steps 健康；step2 entropy=`0.638`、grad_norm=`0.906`、ppo_kl≈0、rollout_corr/kl=`0.00248`，峰值显存 `45.2GB/GPU`；日志 `native-tp2-pp2-cp2-smoke.log`。

### 代码改动清单（均未 commit，工作区）
| 文件 | 改动 |
|---|---|
| `engine/training/megatron_native_engine.py` | **新增**（MegatronNativeEngine + dist-ckpt）；TP/PP 权重 shard/gather、pipeline schedule；**阶段 2c（CP）**：zigzag per-sequence pad/slice、`cu_seqlens*cp`、`_cp_row_logprob_entropy` 可微完整顺序重构、schedule CP loss 因子抵消、source=`tp0/cp0/last-stage` |
| `engine/training/qwen3_megatron_bridge.py` | `hf_to_megatron`/`megatron_to_hf` 加 `te=True`（融合 layernorm 命名：`linear_qkv.layer_norm_weight` / `linear_fc1.layer_norm_weight`） |
| `engine/training/__init__.py` | 导入注册 native engine |
| `workers/actor_worker.py` | 识别 `megatron_native` backend；save/load 走 dist-ckpt；非 source rank 返回空 proto；`get_mp_info()` 回 dp/tp/pp/**cp** rank/size，DP 查询显式排除 CP |
| `core/config.py` | **阶段 2a（必修 bug）**：`MegatronConfig` 字段重命名为 `tensor_model_parallel_size`/`pipeline_model_parallel_size`/`context_parallel_size`/`expert_model_parallel_size`（+ `sequence_parallel`），与 `actor_worker._build_engine_config` 读的名字统一 |
| `core/types.py` | `TrainingBackend.MEGATRON_NATIVE`；`MegatronConfigDict` 字段同步重命名 |
| `trainer/rl_trainer.py` | `_compute_actor_mp()`=`tp*pp*cp`；mp>1 时用 `get_mp_info` 查真实纯-DP rank 建 `actor.mesh_mapping`；DAPO `dp_size` 使用不含 CP 的 DP world size |
| `configs/*.yaml`（6 个 MoE recipe） | megatron_cfg 下 `tensor_parallel_size`→`tensor_model_parallel_size` 等（配合字段重命名；atom_cfg 的同名字段不动） |
| `examples/DAPO/run_dapo.sh` | 末尾加 `${EXTRA_OVERRIDE:-}` 透传（可传 checkpointing/backend 等 CLI override） |
| （另有之前几处已 commit 的：FA/chunk/packing、sleep=false、config `${oc.env:DATA_ROOT}` 插值——见 git log） |

---

## 2. 剩余工作（仅优化）

> 8B + DP8 场景本身用不到 TP/PP/CP（单卡放得下），不阻塞当前训练；价值在更大模型。
> TP（2a）+ PP（2b）+ CP（2c）及其组合均已完成并验证（见 §1）。

### ✅ 前置必修 bug：MegatronConfig 字段名不匹配 —— 已修
统一为 `*_model_parallel_size` + `context_parallel_size`/`sequence_parallel`（`core/config.py`、`core/types.py`、6 个 MoE recipe yaml）。

### ✅ 阶段 2c CP —— 已完成
实现、归一数学、数值对齐与全链路指标见 §1「阶段 2c」。后续若改 CP loss 或 schedule 返回契约，必须同时重验 CP1/CP2 logprob 与梯度尺度。

### 待验证/优化项
1. ✅ **TP×PP×CP 同开**：除 DP2×TP2×PP2 外，DP1×TP2×PP2×CP2 也已验证；权重 shard/gather、1F1B、CP ring attention 与 controller mesh/source 均可组合。
2. **vocab-parallel logprob**（TP 高效版，未做）：`parallel_output=True` + 移植 vime `utils/ppo_utils.py::_VocabParallelLogProbEntropy`（跨 TP all-reduce，单 [L,V/tp] 缓冲），省 all-gather 通信 + 全 vocab 内存。当前 `parallel_output=False`（all-gather 全 vocab）已正确、简单。
3. **PP 权重同步优化**：`_full_megatron_named_params` 逐张量 `dist.broadcast` 跨 PP 组（正确但集合通信次数多）；大模型可改为分桶广播。
4. **PP 负载均衡**：`num_microbatches`=bin 数，取决于该 DP 组数据；stage 间气泡未特别优化（1F1B 默认）。
5. **train_s 严格对比**（DAPO dynamic sampling 使每步 batch 不同 → 当前 train_s 非严格可比）：关掉 dynamic sampling、固定同批数据才能精确比较。已知趋势：8B 单卡放得下时 DP8 最快（~1.5s），TP2 最慢（~5s，纯通信开销），PP2≈DP8（~1.3s），3D 居中（~1.9s）；TP/PP 用速度换显存，价值在更大模型。

---

## 3. 验证方法（smoke + resume，复制即用）

```bash
export RL_ROOT=/home/xysheng/working/11/lumen_rl
export DATA_ROOT=/mnt/m2m_nobackup/xysheng/data
export CONTAINER=rl-vllm-megatron
S=$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT';"

# DP8 native smoke（阶段1 回归）
docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  STEPS=2 MODE=bf16 EXTRA_OVERRIDE='policy.training_backend=megatron_native' \
  LOG=\$DATA_ROOT/logs/native-smoke.log bash '$S'"

# DP4×TP2 native smoke（阶段2a）：8 卡自动 DP4×TP2
docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  STEPS=2 MODE=bf16 EXTRA_OVERRIDE='policy.training_backend=megatron_native \
    policy.training.megatron_cfg.tensor_model_parallel_size=2' \
  LOG=\$DATA_ROOT/logs/native-tp2-smoke.log bash '$S'"

# DP4×PP2 native smoke（阶段2b）：8 卡自动 DP4×PP2（改 pipeline_model_parallel_size）
docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  STEPS=2 MODE=bf16 EXTRA_OVERRIDE='policy.training_backend=megatron_native \
    policy.training.megatron_cfg.pipeline_model_parallel_size=2' \
  LOG=\$DATA_ROOT/logs/native-pp2-smoke.log bash '$S'"

# DP4×CP2 native smoke（阶段2c）
docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  STEPS=2 MODE=bf16 EXTRA_OVERRIDE='policy.training_backend=megatron_native \
    policy.training.megatron_cfg.context_parallel_size=2 checkpointing.resume=false' \
  LOG=\$DATA_ROOT/logs/native-cp2-smoke.log bash '$S'"

# DP1×TP2×PP2×CP2（8 卡全模型并行组合）
docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_megatron_smoke.yaml \
  STEPS=2 MODE=bf16 EXTRA_OVERRIDE='policy.training_backend=megatron_native \
    policy.training.megatron_cfg.tensor_model_parallel_size=2 \
    policy.training.megatron_cfg.pipeline_model_parallel_size=2 \
    policy.training.megatron_cfg.context_parallel_size=2 checkpointing.resume=false' \
  LOG=\$DATA_ROOT/logs/native-tp2-pp2-cp2-smoke.log bash '$S'"

# CP2 dist-ckpt save→resume：Run A 存，Run B resume（改 checkpointing.resume=true）
#   ⚠️ checkpoint_dir 用**绝对路径**（EXTRA_OVERRIDE 经 run_dapo.sh 二次展开不认 $DATA_ROOT）
#   Run A: EXTRA_OVERRIDE='...context_parallel_size=2 checkpointing.save_steps=2 \
#     checkpointing.resume=false checkpointing.checkpoint_dir=/mnt/m2m_nobackup/xysheng/data/ckpts/lumenrl-dapo/native-cp2-ckpt'
#   Run B: 同上但 STEPS=4 checkpointing.resume=true（save_steps=100 避免再存）
# 期望：Run B 日志 "resume_step=2" + step3/4 指标健康。（ckpt≈107GB，测完 docker exec rm -rf 清理）
```
- `run_dapo.sh` 用 `MODE=bf16`（`VLLM_ROCM_USE_AITER=0`）、`STEPS`=步数、`CONFIG_OVERRIDE`=config、`EXTRA_OVERRIDE`=额外 CLI override（新加的透传）。
- 健康判据：entropy≈0.6–0.75、grad_norm≈0.8、ppo_kl≈0、rollout_corr/kl≈0.002、无 Traceback/OOM/NaN。
- 本轮验证日志均写在 `$DATA_ROOT/logs/`；不要把 `LOG` 指到 `/home`。

---

## 4. 已知坑速查
- ~~**MegatronConfig 字段名不匹配**~~ → **已修**（统一为 `*_model_parallel_size`）。
- **TP + sequence_parallel**：SP 需 seq%tp==0，RL 变长序列不满足 → native engine `sequence_parallel` 默认 False（纯 TP，无此约束）。别为 TP 强开 SP。
- **PP + variable_seq_lengths**：PP>1 必须 `variable_seq_lengths=True`（变长 microbatch 的 P2P 动态交换 shape），且要设 `moe_token_dispatcher_type="alltoall"` 过 dense 模型配置校验（allgather dispatcher 在 variable_seq 下报错）。
- **CP thd 格式**：必须逐序列 pad 到 `2*cp*chunk`、取 zigzag 两块，并让 `cu_seqlens=local_cumsum*cp`；不能先把整个 bin 连起来再任意切，否则序列边界/RoPE 都会错。
- **CP shift-by-one**：chunk 最后一个有效 logit 预测的 target 可能在另一个 CP rank；target 必须从原始完整 ids 的 `global_logit_pos+1` 取，不能简单对本地 token 流做 `[:-1]`/`[1:]`。
- **CP loss 三个因子**：可微 gather backward 是 SUM，DDP 在 DP×CP 上 AVG，Megatron two-value schedule 还会额外 `*cp/num_mb`。当前 closure 返回 `bin_loss*num_mb/cp`；不要再把 controller `dp_size` 改成 DP×CP，否则梯度尺度会错。
- **mesh/source**：CP rank 持有不同 token，但 gather 后每个 CP rank 都有完整输出；controller mesh 必须按不含 CP 的 DP rank 复制数据，且只让 `tp0 && cp0 && PP-last` 返回输出。
- **PP metric 稀释**：非 last-stage 的 `engine_update_policy` 只回 grad_norm/lr（不回 loss/kl），否则 controller 平均会稀释指标。
- **TP 权重同步**：vLLM replica 是 TP=1，`get_per_tensor_param` 必须先 TP all-gather 还原全量（已实现 `_full_megatron_named_params`）；且**所有 actor 必须并发调用**（TP 集合通信），trainer 的 `execute_all_async`/`execute_all_sync` 已满足。
- **run_dapo.sh EXTRA_OVERRIDE 不二次展开 `$DATA_ROOT`** → checkpoint_dir 传绝对路径，否则会在 cwd 下建字面量 `$DATA_ROOT` 目录（曾误写 107GB 到 /home）。
- **cumem sleep broken (rocm7.2.3)** → `enable_sleep_mode: false`（别改回）。
- **dist-ckpt optimizer** 必须 `distrib_optim_sharding_type=dp_zero_gather_scatter`（默认 flattened_range 不被 torch_dist 支持）。
- **apex + local spec** → RMSNorm 报错（TE spec / 强制 WTN 不受影响）。
- **/home 99% 满**（共享 NFS）→ git/编辑偶发 EDQUOT，重试；训练数据/ckpt 全写 nobackup（别用相对/字面量路径把大文件落到 /home）。
- **ckpt 目录 root 权限** → 清理用 `docker exec ... rm -rf`。
- **容器未 commit** → TE/apex 在 overlay，`docker rm` 丢失（22min 重编）。
- HF→dist-ckpt bootstrap：仅特殊模型才可能需要；当前 TP/PP 已用 `_shard_hf_for_mp` 直接从 HF 切分，无需 bootstrap。参考 vime `tools/convert_hf_to_torch_dist.py`。

---

## 5. 参考
- 新机环境构建（含 TE/Apex）：`lumenrl-megatron-native-env-build-runbook.md`
- 本仓库 runbook：`dapo-lumenrl-native-vllm-megatron-runbook.md`、`dapo-vime-vllm-megatron-runbook.md`
- vime 原生 Megatron 源码：`/home/xysheng/working/vime-rl/vime/vime/backends/megatron_utils/` + `utils/ppo_utils.py`
- LumenRL 现有 Megatron：`$RL_ROOT/Lumen-RL/lumenrl/engine/training/megatron_engine.py`（自写，FA+chunk+packing）、`megatron_native_engine.py`（原生，本次新增）、`qwen3_megatron_bridge.py`
