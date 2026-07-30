# 交接：Qwen3-MoE 在 LumenRL 原生 FSDP2 + vLLM（BF16 DAPO）上的 train/rollout 一致性

## 读者
你是编码 agent，在**一台新机器**上接手。全程中文回复；代码/命令/路径/标识符保持原文。

这条线跑的是 **FSDP2 训练 + colocated vLLM rollout**（`training_backend: fsdp2`、`generation_backend: vllm`），
**不是** Megatron。Megatron+EP 那条线是独立的，见 `dapo-lumenrl-megatron-moe-ep-handoff.md`，别混。

---

## 一句话现状

Qwen3-30B-A3B-Base 能在这条路径上跑起来，但 `rollout_corr/kl` 会从 0.0005 一路爬到 0.0166（55 步），
**根因已定位**：IPC 权重同步把**全部融合 MoE 专家张量静默丢弃**，vLLM 的专家永远停在磁盘加载值。
修复方案已设计好但**尚未实现**——这就是你要做的第一件事。

---

## 0. 先同步代码

仓库 `git@github.com:ZhangDanyang-AMD/Lumen-RL.git`，分支 `dev/vllm-fsdp-dapo`，作者 `xysheng-AMD <xysheng@amd.com>`。
相关提交（**都已 push**，按时间顺序）：

| commit | 内容 |
|---|---|
| `970160b` | `feat(metrics)` 非抵消的 mismatch 指标（5 个）+ 单元测试 |
| `69634e0` | `feat(r3)` `DataProto.ragged` 一等字段 + R3 采集侧 |
| `00ff83f` | `feat(r3)` R3 注入侧（训练前向/反向复放 rollout 路由）+ 单元测试 |
| `f3706fd` | `feat(dapo)` 长跑配置开启 R3，并把同步 bug 写进注释 |

```bash
git -C "$RL_ROOT/Lumen-RL" fetch origin
git -C "$RL_ROOT/Lumen-RL" status --short          # 有本地改动先确认，别 reset --hard
git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
git -C "$RL_ROOT/Lumen-RL" log --oneline -5        # 应看到 f3706fd
```

Lumen-RL 是 editable 安装，改文件下次启动即生效。

先跑 3 个纯 CPU 测试确认代码完整（不需要 GPU、不需要模型）：

```bash
cd "$RL_ROOT/Lumen-RL"
python3 -m lumenrl.tests.test_dataproto_ragged      # 10 项
python3 -m lumenrl.tests.test_mismatch_metrics      #  4 项
python3 -m lumenrl.tests.test_rollout_routing       #  9 项
```

---

## 1. 环境（新机器需自建）

参照 `dapo-lumenrl-native-vllm-fsdp-runbook.md` 的第 2–6 节，但注意这条线的**额外差异**：

- 镜像 `vllm/vllm-openai-rocm:v0.23.0`。**vLLM 0.23.0 已原生支持 `enable_return_routed_experts`**（R3 需要），不用打 patch、不用升级。
- 容器内 `flydsl` 必须升到 **0.1.8**。镜像自带 0.1.4.2 会让 `from aiter import flash_attn_varlen_func` 报版本不兼容，训练前向直接挂：
  ```bash
  pip install "flydsl==0.1.8"
  ```
- **BF16 路线不需要 `aiter setup.py develop`**（`VLLM_ROCM_USE_AITER=0`）。如果 aiter 目录被别的任务共用，重建它会打断对方的 JIT。
- 模型用 **`Qwen/Qwen3-30B-A3B-Base`**，不是 `Qwen/Qwen3-30B-A3B`。
  instruct/thinking 版在 `max_response_length` 内**永远不闭合 `</think>`**（实测给到 3072 token 仍不闭合、无 `\boxed`），
  于是每条样本都被截断、reward 恒为 −1、`filter_groups` 连续 10 轮 kept 0，直接抛
  `RuntimeError: filter_groups collected no valid groups`。Base 版能正常出 `Answer:`。
- transformers 版本 5.12。**这个版本把 Qwen3-MoE 的专家融合了**，是下面那个 bug 的前提。

**上一台机器的模型和 checkpoint 在节点本地盘 `/mnt/m2m_nobackup`，新机器拿不到，需要重新下载模型。**
step-50 checkpoint（342G，FSDP2 分片 + optimizer state）也不可迁移。

---

## 2. 根因：IPC 权重同步丢掉了所有 MoE 专家权重

### 现象
`rollout_corr/kl` 随步数单调增长，与 step 相关 +0.98：

| step | 1 | 10 | 20 | 30 | 40 | 54 |
|---|---|---|---|---|---|---|
| `rollout_corr/kl` | 0.00052 | 0.00055 | 0.00216 | 0.00592 | 0.00916 | 0.0166 |
| `mismatch/abs_diff` | 0.0136 | 0.0144 | 0.0214 | 0.0323 | 0.0423 | 0.0493 |

### 机制
transformers 5.x 的 `Qwen3MoeExperts` 用融合 3D 张量，所以训练侧 `state_dict()`（也就是 IPC 同步发送的内容）发的是：

```
model.layers.N.mlp.experts.gate_up_proj    (128, 1536, 2048)   fp32
model.layers.N.mlp.experts.down_proj       (128, 2048,  768)   fp32
model.layers.N.mlp.gate.weight             (128, 2048)
```

而 vLLM 把同样的 buffer 叫：

```
model.layers.N.mlp.experts.w13_weight      (128, 1536, 2048)
model.layers.N.mlp.experts.w2_weight
model.layers.N.mlp.gate.weight
```

vLLM 的 `expert_params_mapping`（`make_expert_params_mapping`）**只认 per-expert 名**
`experts.{id}.{gate|up|down}_proj.weight`。融合名匹配不上任何 mapping，就走了 `Qwen3MoeForCausalLM.load_weights`
里那个 `if is_expert_weight: continue` 静默分支——**不报错**，而 LumenRL 又把 `load_weights()` 的返回值丢掉了，
所以 96 个融合专家张量（约 57GB、**93% 的参数**）每次同步全部被丢，54 步都没人发现。

### 实测证据
`moe_sync_probe.py`（在上一台机器的 `$RL_ROOT` 下，可按需重写）用**训练侧真实发送的名字**推张量，检查目标 buffer 有没有变：

| 发送名 | 目标 | 结果 |
|---|---|---|
| `...experts.gate_up_proj` | `experts.w13_weight` | **静默丢弃**，返回 0 个、无异常 |
| `...experts.down_proj` | `experts.w2_weight` | **静默丢弃** |
| `...mlp.gate.weight` | `mlp.gate.weight` | 正常写入 |
| `...self_attn.q_proj.weight` | `self_attn.qkv_proj.weight` | 正常写入（对照） |

### 影响
step 54 的 `reward/accuracy` 从 0.14 涨到 0.43 **不可信**——rollout 是用"专家冻结在 step 0、attention 却训练了 54 步"的
混合模型生成的。那一跑的结果作废，日志备份为 `longrun-bf16-vllm-moe-a3b.step54.bak`。

---

## 3. 你的第一件事：修同步

### 第 1 步（约 10 分钟，决定方案）：验证布局是否逐元素对应
`gate_up_proj (128,1536,2048)` 和 `w13_weight (128,1536,2048)` 形状完全一致，但**必须验证内部布局**
（gate/up 在 dim=1 上的拼接顺序、是否转置）。用磁盘上同一份原始权重交叉验证：

- 用 vLLM 正常加载 → 取 `w13_weight` / `w2_weight`
- 用 HF `AutoModelForCausalLM` 加载同一份 → 取 `experts.gate_up_proj` / `down_proj`
- 逐位比对（注意 dtype：HF 侧 fp32/bf16，vLLM 侧 bf16）

### 第 2 步：按结论实现
- **若逐元素对应**（推荐路径）：在 `lumenrl/engine/inference/vllm_colocate_worker_ext.py` 里加名字映射，
  把 `experts.gate_up_proj → experts.w13_weight`、`experts.down_proj → experts.w2_weight` 直接整块 `copy_`。
  高效，96 个张量照旧。
- **若不对应**：在发送端（`lumenrl/workers/actor_worker.py` 的 `update_weights_ipc_send`）把融合张量拆成
  128×3 个 per-expert 张量，用 vLLM 认识的名字发。**这种名字已实测能正确落盘**。
  代价是张量数从 96 变成 18432，分桶开销上升。

### 第 3 步（必做）：加覆盖率断言
这个 bug 能藏 54 步，唯一原因是 `load_weights()` 的返回值被丢弃。
在 `vllm_colocate_worker_ext.py` 的 `update_weights_from_ipc` 里累积各 bucket 的 `loaded_params`，
最后断言它覆盖 `dict(model.named_parameters()).keys()`。这一条把任何静默漏同步变成显式报错。

同时建议顺手补上 verl 的 `patch_vllm_moe_model_weight_loader`（见
`verl/verl/utils/vllm/patch.py:77-142` 和 `verl/verl/workers/rollout/vllm_rollout/utils.py:200-205,243-246`）。
我早前判定它"在这版 vLLM 上无害"——**那个判断是错的**，它正是用来处理融合 MoE 权重加载的。

### 第 4 步：验证
```bash
# 3 步 smoke，R3 关，看 mismatch/abs_diff 是否在多步后保持平稳（修好前它会涨）
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_smoke.yaml \
MODEL_PATH=$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 bash run_dapo.sh
```
真正的判据是**长跑 30+ 步 `rollout_corr/kl` 不再单调爬升**。修好前它必然爬。

---

## 4. 已完成的工作（可直接用）

### 4.1 mismatch 指标（`970160b`）
`rollout_corr/kl` 是**有符号**均值，对称分歧会互相抵消。MoE 上它把一个数量级藏住了：
实测 `mean|delta| = 0.0226` 而有符号均值只有 0.0019。新增（全部 over response tokens，δ = rollout_logp − train_logp）：

| 指标 | 含义 |
|---|---|
| `mismatch/abs_diff` | mean \|δ\|——被抵消掉的量级 |
| `mismatch/k3_kl` | E[e^δ−δ−1] ≥ 0，低方差 KL 估计 |
| `mismatch/chi2_token` | token 级 IS 权重二阶矩，预测梯度方差 |
| `mismatch/chi2_seq` | 序列级 |
| `mismatch/frac_abs_gt_0.1` | 尾部占比 |

`abs_diff` 和 `k3_kl` 也进了 wandb 的 `core/` 面板。
实现约定：每个都上报 SUM + 分母，per-micro 和 per-worker 两次取均值后 `mean(sum)/mean(tok)` 仍严格等于全局比值。

**教训**：不要只看有符号 kl。我因此走了两轮弯路——fp32 router 的 A/B 在 2k token 上"改善 30%"，
样本加到 15k 就归零了，因为有符号均值的标准误和效应量同级。**任何 A/B 至少 15k token，并看标准误。**

`mismatch/chi2_seq` 在长序列下是均值指标、对异常值极敏感（clamp 在 ±10，单条序列打满就贡献约 9.5e5），
长跑里意义有限，建议改成中位数或超阈值占比。

### 4.2 `DataProto.ragged`（`69634e0`）
per-row 变长负载的一等字段。`meta` 不行（浅拷贝进每个 chunk、`select_idxs` 不重排、还会复制给全部 worker）；
padded 张量也不行（`[seq,48,8]` uint8 补齐到统一长度，长跑一个 batch 是 **12.7GB**，ragged 只要约 350MB）。
`select_idxs` / `split` / `concat` / `merge` / `reorder` / `repeat` / `sample_level_repeat` /
`pad_to_divisor` / `unpad` / `mini_batches` / `check_consistency` 全部正确重排，10 项测试逐个覆盖。

### 4.3 R3 = Rollout Routing Replay（`69634e0` + `00ff83f`）
让训练侧复用 rollout 引擎实际选中的 top-k 专家 id，参考 vime 的实现（arXiv 2510.11370）。
开关 `moe.r3.rollout_replay`（长跑配置已开）。

**效果**（3 步 smoke，R3 是唯一变量）：

| 指标 | R3 关 | R3 开 |
|---|---|---|
| `rollout_corr/kl` | 0.00160 | **0.00044** |
| `mismatch/abs_diff` | 0.0207 | 0.0125 |
| `mismatch/chi2_token` | 1.65 | 0.0012 |
| `mismatch/chi2_seq` | 1.33e6 | 1.35 |
| step 时间 | 43.9s | 45.8s（+4.2%） |

`chi2_seq` 从 1.3e6 掉到 1.35 最有分量：之前序列级 IS 权重在 e^7 量级，GSPO 这类序列级方法根本不可用。

**实现要点（改动时别破坏）**：
- **只替换离散选择**，权重从活的 `router_probs` 上 `gather`。直接覆盖 logits 会把 router 冻成常量、切断梯度。
  有测试断言梯度存活。
- **按层号索引，不用调用顺序游标**。HF 的 gradient checkpointing 在 backward 时按**相反层序**重算，
  游标会读到别的层的路由。vime 用双游标是因为 Megatron 的结构让它成立，我们这里不成立。
- **状态必须是模块级全局，不能用 `threading.local()`**。非重入 checkpointing 在 autograd 工作线程上重算前向，
  thread-local 在那里不可见，replay 会静默消失，报
  `CheckpointError: A different number of tensors was saved during the original forward and recomputation`。
  `engine/training/packing.py` 顶部早就写明了同一个坑。
- **`old_log_probs` 也要走同一份 replay**，否则 PPO ratio 比的是两个路由不同的策略。
- **对齐错误直接 raise**：token 数对不上会抛异常，而不是继续。错位不会崩，只会拿从没用过的专家去优化。
- routing 布局：vLLM 返回 uint8 `[prompt + resp − 1, n_layers, top_k]`，第 t 行是位置 t 的前向（产出 token t+1），
  和 `rollout_log_probs` 同一索引空间，**不需要偏移**。每条序列最后一个位置没有路由（引擎没在那里跑前向），
  fall through 到现算。

### 4.4 fp32 router（较早提交，默认开）
`lumenrl/moe/router_precision.py`，`LUMENRL_FP32_MOE_ROUTER=0` 可关。
**实测对 mismatch 无统计显著改善**（15k token：0.001898±0.000618 vs 0.001833±0.000456），
长跑逐步和 baseline 重合到 step 16。保留是因为它消除了一个真实的不确定性来源、成本可忽略。

它顺带承载了 R3 的 top-k 拦截点，所以**即使 fp32 关掉，HF router 的 patch 也仍会安装**。

背景数据（供参考，说明为什么 fp32 不够）：bf16 router 下 top-k 间距被量化到 bf16 步长
（p25=0.0312、p50=0.0625），**11.7%** 的 (token,layer) 决策是精确并列；仅把 router 换成 fp32 重算就有
**6.4%** 的决策改变专家集合，即只有约 4% 的 token 在全部 48 层路由一致。
但翻转的主因是 router 的**输入**（HF 和 vLLM 每层 hidden states 在 bf16 尺度上就有差异），所以提高 router 精度没用。

---

## 5. 已排除的假设（别重复走）

| 假设 | 结论 |
|---|---|
| bf16 router 并列导致翻转 → 提到 fp32 | **无效**，实测 15k token 无统计显著差异 |
| 熵坍缩在 log 空间放大数值分歧 | **不是根因**。entropy 与 `abs_diff` 相关 −0.83 只是相关性，entropy 是"训练了多久"的代理 |
| R3 能解决增长 | **只降水平不压斜率**。R3 修路由，修不了权重本身是错的 |
| 权重同步对融合 MoE 张量是正确的 | **错**，见第 2 节。我最初的探针推的是 per-expert 名，**那不是训练侧发送的格式**，测错了输入 |

最后一条是最重要的教训：**验证同步时，必须用训练侧 `state_dict()` 里真实的 key 名**，
先 `torch.load` 一个 checkpoint 分片把 key 打出来，不要凭 HF 文档假设命名。

---

## 6. 还没做的事

1. **修同步**（第 3 节）——最高优先级，在此之前任何 MoE 长跑结果都不可用。
2. `load_weights` 覆盖率断言 + `patch_vllm_moe_model_weight_loader`。
3. `ppo_kl` 在 R3 开启后从 −3.6e-5 变成 −8.4e-4，涨一个量级，**我没有完整解释**。
   猜测是 old_logprob 前向按整批 `max_tok` 切、训练前向按分发后的分片切，打包组合不同导致 varlen kernel 数值差异。
   绝对值仍远小于告警线，但值得查。
4. `mismatch/chi2_seq` 在长序列下改成中位数或超阈值占比。
5. **熵坍缩**：entropy 从 0.835 掉到 0.185（54 步），配置里没有 `entropy_coeff`（继承自 8B longrun）。
   同步修好后需要重新评估——现在无法区分这是真实的策略坍缩，还是被"用错权重的 rollout"训出来的假象。
6. 残留的非路由分歧：R3 开启、同步正确之后，仍有 attention/MLP kernel 差异（HF SDPA + 自定义 varlen vs vLLM 自己的实现）。
   注意 **vime 并没有对齐 kernel**，它是靠 R3 + TIS 吸收差异，所以"让训练侧贴近 vLLM kernel"不是 vime 的解法。

---

## 7. 参考：vime 对比结论

vime（`xysheng-AMD` 那份 checkout）在同一问题上的做法，值得借鉴的部分：

- **R3 是它的核心机制**，而且它在已经有 `--moe-router-dtype fp32` 的前提下**仍然**实现 R3
  ——独立印证了 fp32 不够。
- 它的 MoE + R3 参考配置用 **GSPO（序列级 IS）+ `eps-clip 4e-4`**，不是 token 级 GRPO。
  序列级对个别 token 的翻转更鲁棒，但前提是 `chi2_seq` 得先降下来（R3 做到了）。
- 指标体系比我们丰富（`k3_kl`、`chi2`、`abs_diff`、`ppl_ratio`），已抄了主要几个。
- 它还有 **rollout top-p nucleus 复放**（vLLM 在核内重新归一化，分母和训练侧全词表 softmax 不同）。
  当前 `top_p=1.0` 所以不影响，一旦调 top-p 就会变成新的 mismatch 源。
- 它有 `--check-weight-update-equal` 开机自检权重同步正确性。**我们正是因为缺这类检查才让 bug 藏了 54 步。**

---

## 8. 可复用的诊断脚本

上一台机器 `$RL_ROOT` 下（不在 git 里，需要时按描述重写）：

- `kl_compare.py`：两阶段（vLLM 生成 → HF 重算）对比同权重下的 logprob 分歧，输出有符号均值 + 标准误、
  `mean|δ|`、分位数、尾部占比、router top-k 间距分布、bf16→fp32 的翻转率。`--fp32-router` 开关。
  **修完同步后用它验证**：同权重下 `abs_diff` 应回到 0.0136 量级并且不随步数涨。
- `moe_sync_probe.py`：用训练侧真实 key 名推张量，检查是否真的写进 vLLM 的 buffer。第 2 节的证据就是它出的。
- `r3_probe.py`：确认 `enable_return_routed_experts` 在 ROCm 上工作、以及返回的形状/dtype/覆盖范围。

写这类探针时注意：vLLM 的 `collective_rpc` 传闭包需要 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（否则序列化失败），
`LLM(...)` 构造要包在 `if __name__ == "__main__":` 里（spawn 模式）。
