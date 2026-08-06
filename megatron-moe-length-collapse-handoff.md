# 交接：Qwen3-30B-A3B MoE 上 Megatron 响应长度崩塌的排查

> 读者：接手这台机器继续排查的编码 agent。全程中文回复；代码/命令/路径/标识符保持原文。
> 最后更新：2026-08-03（第二轮）。机器：spur 集群 job `16358`，节点 `crsuse2-m2m-267`（8×MI355X gfx950, 288GB/卡）。

---

## 0. 一句话现状

同一份 DAPO 配方、同一份数据、同一个 rollout 引擎，**FSDP2 训练正常（响应长度持续增长、准确率爬到 0.56），
Megatron-Native 在 step ~140 之后响应长度单调崩塌、准确率腰斩**。

**第二轮找到了根因候选：Megatron 侧 `response_mask` 对齐差一位，导致每条序列的 EOS 位置从策略梯度里被剔除，
同时把最后一个 prompt 位置错误地算进了 loss。这是 Megatron 独有的，FSDP2 那条路径是对的。** 详见 §5.0。
第一轮记的两个 bug（§5.1 / §5.2）里，5.1 其实是这个错位的**症状**而不是独立 bug。

三处都已修好，4K smoke 三步通过，正在跑 4K 从零长跑验证长度轨迹（§11）。

---

## 1. 怎么连上这台机器

```bash
# 1) 计算节点只能通过 spur exec 进，不能 ssh（见 SPUR_NODE_ACCESS_GUIDE.md）
spur exec 16358 bash -lc 'hostname; rocm-smi --showmeminfo vram | grep -i used | head -2'

# 2) 训练全部在容器里跑。注意：本机没有 sudo，但 xysheng 在 docker 组，命令要去掉 sudo
spur exec 16358 bash -lc 'docker ps --format "{{.Names}}\t{{.Status}}"'
```

| 项 | 值 |
|---|---|
| 容器 | `rl-vllm-fsdp`，镜像 `vllm/vllm-openai-rocm:v0.23.0`（digest `sha256:3813e31c...`）|
| `RL_ROOT` | `/home/xysheng/lumen_rl`（共享 NFS，代码）|
| `DATA_ROOT` / `SCRATCH_ROOT` | `/mnt/m2m_nobackup/xysheng/rl_data`（**节点本地盘**，28T/剩 24T）|
| 模型 | `$DATA_ROOT/models/Qwen3-30B-A3B-Base`（57G）|
| 数据 | `$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/*.parquet`（8B 与 MoE 共用，tokenizer 相同）|
| 日志 / ckpt | `$DATA_ROOT/logs/`、`$DATA_ROOT/ckpts/lumenrl-dapo/verlref-moe-a3b-bf16-megatron-ep8/` |

⚠️ `/mnt/m2m_nobackup` 是**节点本地盘**，换节点后模型和数据要重新准备（模型下载只要 20 多秒，成本很低）。

环境已装好且**两条后端都可用**：megatron-core 0.18.2、ROCm Apex 1.14.0a0、ROCm TransformerEngine
`2.15.0.dev0+6e541a10`（源码编译，约 9 分钟）、flydsl 0.1.8。重建方法见
`dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md`，本机已完整执行过并通过其 §7 的双 engine 注册验证。

> flydsl 有个坑：`run_dapo.sh` 把本地 `$AITER_DIR` 放在 PYTHONPATH 最前，运行时用的是**仓库里的 aiter 源码**，
> 它要求 flydsl ≥ `0.1.5.dev515`，所以必须 0.1.8。而镜像自带的 wheel `amd-aiter 0.1.13.post1` 反过来 pin
> `flydsl<0.1.5`，升级后它会报 `cannot import name 'fly_values'`。两者互斥，只有走 `run_dapo.sh` 的
> PYTHONPATH 才是对的。别按 wheel 的 pin 回退。

---

## 2. 怎么跑起来

```bash
# smoke（3 步，约 11 分钟）
spur exec 16358 bash -lc "docker exec -d rl-vllm-fsdp bash -lc \
  'bash /home/xysheng/lumen_rl/run_moe_a3b_4k_smoke_megatron.sh'"     # Megatron
spur exec 16358 bash -lc "docker exec -d rl-vllm-fsdp bash -lc \
  'bash /home/xysheng/lumen_rl/run_moe_a3b_4k_smoke.sh'"              # FSDP2 对照

# 长跑（1000 步，约 9.3 分钟/步）。配置里 resume=true，会自动从 global_step_105 接上
spur exec 16358 bash -lc "docker exec -d rl-vllm-fsdp bash -lc \
  'bash /home/xysheng/lumen_rl/run_moe_a3b_verlref_longrun_megatron.sh \
     > /mnt/m2m_nobackup/xysheng/rl_data/logs/longrun-megatron-driver.log 2>&1'"

# 只跑 N 步做诊断：STEPS=107 LOG=<路径> 前置即可

# 停止
spur exec 16358 bash -lc "docker exec rl-vllm-fsdp bash -lc \
  'ray stop --force; pkill -9 -f \"[l]umenrl.trainer.main\"; pkill -9 -f \"[V]LLMRayServer\"; \
   pkill -9 -f \"[E]ngineCore\"; sleep 10'"
```

启动后先确认：`MoE+EP spec: ... EP=8 ... router_dtype=fp32`、`Resuming Ray actor checkpoint ... (step=105)`、
无 `Traceback` / `HSA_STATUS`。首步约 14 分钟（含 vLLM 加载），之后约 9.3 分钟/步。

---

## 3. 现象（这是要解释的东西）

同一 step 区间、同一配方，两个后端的对照：

| | FSDP2（run `z8aeo3cf`）| Megatron（run `lepges5o`）|
|---|---|---|
| steps 101–120 `resp_len` | 3204 →（121–140）4253，**在涨** | 1884，**在掉** |
| steps 101–120 `seq/max_len` | 20811 / 20912（打满 20480 预算）| **11074** |
| `reward/accuracy` | 0.539 → 0.560 | 0.503（step 94 见顶 0.53 后回落）|
| `entropy` | 0.084 | 0.100 |

Megatron 的完整轨迹（20 步分块均值，run `lepges5o`）：

| block | acc | resp_len | max_len | entropy |
|---|---|---|---|---|
| 1–20 | 0.295 | 844 | 20672 | 0.448 |
| 41–60 | 0.477 | 1144 | 20129 | 0.093 |
| 81–100 | 0.516 | 2150 | 18138 | 0.090 |
| 101–120 | 0.503 | 1884 | **11074** | 0.100 |

更早还有一个 Megatron 长跑 `p819dsox`（fp32 router、TP2·PP2·CP2·EP2·ETP2、batch 512）跑到 228 步崩溃，
同样的形状：step 1–140 健康（acc 0.15→0.54、长度增长），step 142 起 `max_len` 单调收缩
18099→10014→5629→3723→1896，acc 掉到 0.28。所以这个现象**可复现、跨配置、跨拓扑**。

---

## 4. 已经排除的假设（都有数据，别重复走）

| 假设 | 结论与证据 |
|---|---|
| **`rollout_corr/kl` 爆炸导致崩塌** | **证伪。kl 本身是度量假象，见 §5。** |
| 权重同步漏参数 / 写错 | **证伪。**用 `LUMENRL_WEIGHT_SYNC_CHECK=error` + `LUMENRL_WEIGHT_SYNC_VERIFY=1` 跑完整一步，无覆盖率缺失、无逐位比对失败。Megatron→vLLM 同步是正确的 |
| 续跑丢 optimizer state | **证伪。**wandb `_runtime` 在转折点前后连续（每步 300–1200s 无断口），`lr` 恒定，从未重启 |
| MoE 路由配置不一致 | **证伪。**`norm_topk_prob=True`（HF）与 Megatron `pre_softmax=False` 在选择集合和权重上严格等价；`score_function` / `topk_scaling_factor` 两边都未设；`num_experts=128`/`topk=8` 一致 |
| bf16 router 翻转专家 | **证伪。**改成 `moe_router_dtype: fp32` + `LUMENRL_FP32_MOE_ROUTER=1` 后，干净步的 kl 仍是 0.0355 |
| `log_probs_chunk_size` 分块有误 | **证伪。**`log_softmax` 沿 vocab 维逐 token 计算，沿 token 维分块数学上精确 |
| checkpoint 落盘 / validation 周期性扰动 | **证伪。**kl 增长平滑（每步约 +16%），不是阶梯 |
| kernel 数值差异（attention/norm/GEMM）| **证伪。**剔除假象 token 后，中位 |Δlogprob| 是 3e-6，训练与 vLLM 前向基本位级一致 |
| CP/TP/PP 改拓扑省显存 | **无效。**任何缩小 DP 的改动都会让 distributed optimizer 的 state 每卡翻倍（DP 8→4 多 8.5GB），抵消掉激活的节省。CP=2 实测当场 OOM |

---

## 5. 找到的三个真实 bug（**已全部修好**）

### 5.0 Megatron `_row_policy_loss` 里 `response_mask` 对齐差一位 ★根因候选

`megatron_base_engine.py:_row_policy_loss` 用 `_col(name, shift)` 从整批张量里取一行的有效段，
`shift=True` 就多跳一格。原来这里写的是：

```python
resp_mask = _col("response_mask", shift=True)
```

但 `rl_trainer._build_response_mask` 返回的已经是 `mask[:, 1:]`——宽度 `S-1`、第 i 项标记 token i+1，
**和 `old_log_probs` / `token_lp` 是同一个坐标系**。再 shift 一次，整个 loss 窗口就往前滑了一格：

| | 应该覆盖的位置 | 实际覆盖的位置 |
|---|---|---|
| token_lp 下标 | `plen-1 .. L-2` | `plen-2 .. L-3` |
| 含义 | 首个 response token 的预测 … EOS 的预测 | 最后一个 **prompt** token 的预测 … EOS 前一个 token |

后果两条：

1. **每条序列的 EOS 位置拿不到任何策略梯度。** EOS 恰好是决定响应长度的那个 token。
2. 最后一个 prompt 位置被算进 loss（GRPO 组内 advantage 和为 0、同组 prompt 完全相同，
   单步单 epoch 下这项梯度精确抵消，所以危害小）。

FSDP2 那条路径（`actor_worker._policy_loss_fn`）用的是 `if mask.shape[-1] == L + 1: mask = mask[:, 1:]`，
宽度判断，不会重复 shift，**所以只有 Megatron 错**。这正好是两个后端唯一一处 loss 层面的行为差异。

**它还解释了 §5.1 的现象**：窗口前移后被纳入 loss 的那个 prompt 位置，rollout 引擎从来没有为它上报过
logprob，于是 `rollout_lp` 在那里精确为 `0.0`——**每条序列恰好一个**，2048 条序列就是 2048 个，
和当时测到的数字一模一样，`old_lp` 在那里是 −52 也说得通（prompt 模板的固定 token，模型自己不会生成）。

验证脚本 `$RL_ROOT/test_response_mask_alignment.py`（容器内直接 `python3` 跑，不用 GPU）：
重建 trainer 的张量坐标系后回放 `_col` 切片。修复后 PASS（选中的正好是 response token 集合、
每个位置都有真实 rollout logprob）；把 `shift` 强制改回 `True` 就能看到窗口整体前移一格、
且 `rollout_lp_zero_in_mask=1`（每行一个）。

**修复**：按宽度判断要不要 shift（`response_mask` 宽度 ≥ `input_ids` 宽度才是 token-indexed、才需要 +1）。
顺带把 2D `advantages` 分支的 `start + 1` 也改成 `start`（同一坐标系，当前 DAPO/GRPO 走 1D 分支，未触发）。

### 5.1 `rollout_log_probs` 的未上报位置被当成 `log p = 0`（是 5.0 的症状）

`rl_trainer.py` 里 `rollout_lp = torch.zeros((b, max_len-1))` 零初始化后逐生成 token 填充，
而引擎返回的 logprob 个数比 response token 数**少一个**，所以每条序列的最后一个 response 位置
永远保持 `0.0`。`log p = 0` 意味着概率 1.0，是可能的最大值。

实测（step 107，3,456,397 个 token）：

```
恰好 2048 个 token（= train_global_batch_size，即每序列一个）|Δ| > 10
它们贡献了 rollout_corr/kl 的 97.4%
在这些 token 上：rollout_lp 均值 = 0.0（精确），old_lp 均值 = -52.3
```

剔除它们之后：

| | mean(rollout − train) |
|---|---|
| 全部 token | +0.031807 |
| **剔除 2048 个** | **+0.000816** |
| FSDP2 同期 | +0.00066 |

**所以 Megatron 的真实训练/rollout 一致性和 FSDP 是同一量级，`rollout_corr/kl` 那条曲线不可信。**
它随策略变尖锐而放大（这些位置的 `old_lp` 从约 −10 掉到 −52），所以看起来像指数增长。

指标本身没有 bug：恒等式 `rc − ppo = mean(rollout_lp − old_lp)` 精确成立（+0.031807 vs 报告 +0.031779）。

**已修**：`rl_trainer._apply_rollout_is_weights()` 在算 TIS 权重前，把 response mask 内
`rollout_lp == 0.0` 的位置回填成 `old_log_probs`（比值 1、对 kl 零贡献）。

⚠️ 修 5.0 之后再看，这个判据没有原来想的那么干净。实测 4K：**206171 个 response 位置里有 522 个
`rollout_lp` 精确为 0.0**，不是每序列一个（256 条序列），而且这些位置上 `old_log_probs` 的均值是
**−2.7e-07**——它们是**真的**极度确信的 token（float32 下 log p 下溢到 0，需要 p > 1−6e-8），
不是上报缺失。对照第一轮测到的 −52.3，两者一眼可分。所以回填代码保留但降为 INFO，
并且会打出这些位置上 `old_log_probs` 的均值——**接近 0 说明 token 真的确信，很负说明是上报缺口**。
回填对前者无害（old_lp 在那里也≈0，比值仍是 1）。

第一轮的补丁打在了 `rl_trainer.py` 的 `train()` 里，而 Ray 路径走 `_train_with_ray_controller()`，
是死代码。现在两条路径共用 `_apply_rollout_is_weights()`。

### 5.2 Ray controller 路径上 TIS 完全没生效（已修）

`compute_rollout_is_weights` 全仓库**只在 `rl_trainer.py:2888`（`train()` 内）被调用一次**。
Ray 路径不经过那里，所以 `rollout_is_weights` 从未写入 batch，引擎里 `ris` 恒为 `None`，
`asymmetric_clip_loss(..., rollout_is_weights=None)` 不做任何修正。

配置里 verl 对齐的 `rollout_is: token` / `rollout_is_threshold: 2.0` **是空转的**，
且 FSDP 和 Megatron 都一样。

副作用（也是好消息）：正因为 TIS 空转，5.1 的坏 token 没有通过重要性权重污染训练，只污染了指标。

**已修**：`_train_with_ray_controller()` 在 `compute_advantages` / `apply_rollout_correction` 之后、
`_update_actor_with_ray()` 之前调用 `_apply_rollout_is_weights(batch)`，写入 `rollout_is_weights`
（`[B, S-1]`，和 `rollout_log_probs` 同坐标系），引擎/worker 侧的 `_col("rollout_is_weights", shift=False)`
和 `ris[..., :L]` 都能直接对上。同时把 `rollout_correction/*` 四个诊断量并进 step metrics。

⚠️ 这**同时改变了 FSDP2 基线的行为**（两条后端都会开始真的做 TIS）。4K smoke 实测
`is_weight_mean=0.9997`、`is_weight_max=2`（有 token 打到 clip 上限），影响很小，但要做严格 A/B 时
把 config 里 `rollout_is` 置空即可关掉。

---

## 6. 排查这类问题的两个陷阱（我各踩了两次）

1. **MoE 永远走 `_pp_update_policy`。**
   `megatron_native_engine.py:949`：
   ```python
   if self._pp == 1 and self._cp == 1 and not getattr(self, "_is_moe", False):
       return super().engine_update_policy(batch)
   return self._pp_update_policy(batch)
   ```
   即使 PP=1、CP=1，MoE 也走流水线调度路径。基类的 `engine_update_policy` 对这条线是死代码。
   `_row_policy_loss` 是共用的。往引擎里加探针要加在 `_pp_update_policy`。

2. **`train()` vs `_train_with_ray_controller()`。**
   Ray 路径不经过 `train()` 的主体。往 trainer 里加逻辑先确认在哪个函数里。

3. **env 变量传不到 Ray actor。** actor 创建时没带 `runtime_env`，driver 侧 export 的变量到不了 actor。
   我用 `/tmp/lumenrl_gap_dump_dir` sentinel 文件绕开（`megatron_base_engine.py:_gap_dump_dir()`）。

4. **vLLM worker 里的 `logger.info` 不进 driver 日志。** 所以 `weight sync coverage` 那行看不到，
   **不能**据此认为断言没跑。要判断断言是否触发，看它有没有抛异常。

---

## 7. 下一步建议（原问题：为什么长度会崩）

kl 已被排除，回到长度本身。以下都不占 GPU，wandb 和日志里有数据：

1. **奖励构成对比**。在匹配的 step 上，比较两个后端：被截断在 20480 上限的响应占比、
   长响应 vs 短响应的平均 reward、`overlong_buffer` 惩罚命中率。
   配置里 `overlong_buffer.len` 只有 512（DAPO 论文对 L_max=20480 用的是 **4096**，
   FSDP runbook §15.6 里 verl 长跑也是 4096），惩罚带很窄、越界即断崖，这是个具体的可疑点。

2. **advantage 的长/短分布**。GRPO 在一个 prompt 的 16 条生成里做组归一化，而 `filter_groups`
   只保留准确率混合的组。如果 Megatron 侧存活组里"短的 +1、长的 −1"的比例系统性高于 FSDP，
   那就是自我强化的缩短压力。日志里有 `filter_groups round N: kept X/384`。

3. **确认它是不是真的后端相关**。FSDP 那条线（run `z8aeo3cf`，另一台机器 gpu03）需要跑过 step 200，
   才能排除"只是时间早晚不同"。`p819dsox` 的转折点在 142，FSDP 现在才 135。

先做 1 和 2；如果都看不出偏向，再考虑 3。

---

## 8. 工作区改动清单（全部未提交）

`Lumen-RL` HEAD = `021025a`，分支 `dev/vllm-fsdp-dapo`。

```
 M lumenrl/engine/training/megatron_base_engine.py     §5.0 的修复（response_mask 按宽度决定是否 shift、
                                                       2D advantages 同步）；诊断探针 _GAP_ROWS / _gap_dump_dir()
 M lumenrl/engine/training/megatron_native_engine.py   诊断探针：_pp_update_policy 末尾落盘
 M lumenrl/trainer/rl_trainer.py                       新增 _apply_rollout_is_weights()（§5.1 回填 + §5.2 TIS 接线），
                                                       train() 与 _train_with_ray_controller() 共用；driver 侧 dump
?? examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml
?? examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml
?? examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml   ← 新，见 §11
```

`$RL_ROOT/` 下新增：`test_response_mask_alignment.py`（§5.0 的对齐单测）、
`run_moe_a3b_verlref_4k_longrun_megatron.sh`（§11 的启动脚本）。

两个新 config 都是 FSDP2 `verlref` 对应文件的逐字段拷贝，**只改 `training_backend` 和 `megatron_cfg`**，
拓扑 EP=8/TP=PP=CP=1 → DP=8，和 FSDP2 的 DP8 对齐（同一全局 batch、同样的切分）。长跑配置相对 FSDP2 的偏离只有三处，
都只影响显存/吞吐不影响优化问题，且都写在文件头注释里：
`gpu_memory_utilization` 0.40→0.25、`max_tokens_per_gpu` 22528→8192、`recompute_granularity: full`。

> 显存那一段的结论值得记住：崩溃不是 allocated 峰值的问题（改前后都是 ~130GB），
> 而是**碎片**——ROCm 没有 `expandable_segments`，原来每步约 7 个打满 22.5k 的 bin 反复申请释放巨块，
> reserved 比 allocated 多出 42GB。把 bin 压到 8192 后碎片间隙塌到 4–11GB，peak reserved 177→134GB。

`moe_router_dtype` 现在是 `fp32`，启动脚本配套 `LUMENRL_FP32_MOE_ROUTER=1`（这个环境变量只作用于 vLLM
worker，**必须和 megatron_cfg 那个字段一起翻**，否则变成反向不匹配）。这是基于后来被证伪的假设改的，
但 fp32 本来就是 LumenRL 对 MoE 的默认值，留着无害；若要严格对齐 FSDP 基线做 A/B，两处一起改回 `null` / `0`。

**诊断脚本**（都在 `$RL_ROOT/`，可复用）：

| 文件 | 用途 |
|---|---|
| `wandb_fetch_run.py` / `wandb_analyze.py` | 拉 wandb run 历史、按 20 步分块出趋势与相关系数 |
| `analyze_logprob_gap.py` | driver 侧 dump 的 \|Δlogprob\| 分布与尾部集中度 |
| `analyze_engine_gap.py` | 引擎侧三张量（train/old/rollout）对账，验证指标恒等式 |
| `megatron_verify.py` | TE / Apex / megatron-core / 双 engine 注册的一站式验证 |
| `run_moe_a3b_4k_smoke{,_megatron}.sh`、`run_moe_a3b_verlref_longrun_megatron.sh` | 启动脚本 |

打开诊断 dump：`echo <目录> > /tmp/lumenrl_gap_dump_dir`（容器内），跑完记得删掉该文件。

---

## 9. 红线

- 两条后端**不能同时占卡**，起之前先确认无残留进程、显存回到 ~298MB/卡。
- 不改基础镜像的 torch/triton/vLLM/flash-attn；**不要 `pip install transformer_engine`**（会装成 NVIDIA 版）。
- vLLM 保持 `enable_sleep_mode: false`（ROCm 7.2.3 上 cumem sleep 会在 wake 时 OOM）。
- 大文件只放 `DATA_ROOT`，别放 NFS `/home`。
- 两个后端**不要共用 checkpoint 目录**。
- 不改 git config；提交用 `--author="xysheng-AMD <xysheng@amd.com>"`，推 `dev/vllm-fsdp-dapo`。

---

## 11. 第二轮的验证跑（4K，从零开始）

修完 §5.0/5.1/5.2 之后，用一个**压缩到 4K 响应预算**的配方复现长度轨迹，而不是再等 20K 那条线跑 140 步
（9.3 min/步 ≈ 22 小时）。依据：`lepges5o`（batch 2048）和 `p819dsox`（batch 512）的转折点都在 step ~140，
**崩塌跟的是 optimizer step 数，不是样本数**，所以可以缩 batch。

| | 4K 验证跑 | 20K 原始跑 |
|---|---|---|
| config | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml` | `..._verlref_longrun.yaml` |
| prompt/response | 2048 / **4096** | 2048 / 20480 |
| global batch | **256**（16 prompts × 16） | 2048（128 × 16） |
| gen_batch | 48 | 384 |
| ckpt 目录 | `…/ckpts/lumenrl-dapo/verlref-moe-a3b-megatron-ep8-4k`，`resume: false` | 另一个目录 |
| wandb | `LUMEN-RL-MOE / verlref-moe-a3b-megatron-ep8-4k-maskfix` | `lepges5o` |
| 步时 | 约 3–4 分钟（4K smoke 实测 128 条 ≈ 100s/步，81% 在 generation） | 9.3 分钟 |

```bash
spur exec 16358 bash -lc "docker exec -d rl-vllm-fsdp bash -lc \
  'bash /home/xysheng/lumen_rl/run_moe_a3b_verlref_4k_longrun_megatron.sh \
     > /mnt/m2m_nobackup/xysheng/rl_data/logs/longrun-4k-driver.log 2>&1'"
# 日志：$DATA_ROOT/logs/longrun-verlref-moe-a3b-megatron-ep8-4k.log
```

**修复后 4K smoke（3 步）的基线数字**，长跑对照用：

```
step=1 resp_len=871 rollout_corr/kl=0.00184 rollout_correction/{kl=0.00173, is_weight_mean=0.9997, is_weight_max=2}
step=2 resp_len=714 rollout_corr/kl=0.00180
step=3 resp_len=971 rollout_corr/kl=0.00163  ppo_kl≈1e-4  step_s≈100
```

`rollout_correction/*` 这一族指标是新的——它们出现本身就说明 §5.2 的 TIS 接上了。

**判读长跑的标准**：看 `seq/mean_response_len` 和 `reward/accuracy` 是否单调改善到 step 200 以后。
崩塌的特征是 `seq/max_len` 单调收缩（20K 那条线上是 18099→10014→5629→3723→1896）。
拉数据用 `$RL_ROOT/wandb_fetch_run.py` + `wandb_analyze.py`（20 步分块）。

如果 4K 这条线**仍然崩**，说明 §5.0 不是根因，回到 §7 的奖励构成 / advantage 长短分布两条线；
这时可以直接用 4K 配方做 A/B，成本只有原来的 1/6。

---

## 10. 相关文档

- `SPUR_NODE_ACCESS_GUIDE.md` — spur 节点访问（本机没有 sudo，命令要去掉）
- `dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md` — 双后端环境从零重建（TE/Apex/megatron-core）
- `dapo-lumenrl-native-vllm-fsdp-runbook.md` §13 — FSDP2 MoE 这条线的完整 runbook 与健康判据
- `dapo-lumenrl-megatron-moe-ep-handoff.md` — MoE+EP 实现说明（含 R3 为什么默认关）
- wandb：`LUMEN-RL-MOE` 项目，Megatron = `lepges5o`，FSDP2 = `z8aeo3cf`；旧 Megatron 长跑 = `LumenRL/p819dsox`
