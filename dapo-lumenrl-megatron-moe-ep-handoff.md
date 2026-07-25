# 交接（新机器 / SSH 登录）：在 Lumen-RL 上继续 MoE + Expert Parallel 开发

## 读者
你是编码 agent，直接 SSH 登录在这台新 GPU 机器上（不是 spur exec 那套；本机就是干活的机器）。
本机之前用旧代码跑过 Qwen3-8B（dense），**该任务已停止，GPU 现已空闲**（起 MoE 前仍先按第 2 步确认无残留进程占卡）。
你的任务：**同步已完成的 MoE+EP 代码，在本机把 MoE（Qwen3-30B-A3B）跑起来并继续 MoE 相关开发**。
全程中文回复；代码/命令/路径/标识符保持原文。

## 最重要：MoE+EP 已完成，代码在 dev 分支（先同步）
MoE + Expert Parallel 已在另一台机器完整实现、验证并 **push 到 GitHub**：
- 仓库 `https://github.com/ZhangDanyang-AMD/Lumen-RL.git`，分支 `dev/vllm-fsdp-dapo`
- 相关提交（作者 xysheng-AMD <xysheng@amd.com>）：
    · `ddb3b4c` feat(megatron): add MoE + Expert Parallel to native engine
    · `84c7264` feat(megatron): R3 router record/replay + PP/SP-eff/parallel combos

### 第 0 步：先自发现本机环境
```bash
hostname; whoami
# 找到本机的 Lumen-RL 代码位置（跑 qwen8b 用的那份）
ls -d ~/lumen_rl* /mnt/*/lumen_rl* 2>/dev/null; echo "---"
find / -maxdepth 5 -name run_dapo.sh -path '*examples/DAPO*' 2>/dev/null | head
# 是否在 docker 容器里训练？还是裸机？
docker ps 2>/dev/null | head
# 确认卡已空（qwen8b 已停；应无残留训练进程）
pgrep -af "lumenrl.trainer.main|VLLMRayServer|EngineCore" ; rocm-smi 2>/dev/null || nvidia-smi
# 训练/数据盘（大文件盘）
df -h
```
确定三个变量（按本机实际填）：`RL_ROOT`（含 Lumen-RL/Lumen/aiter 的目录）、`DATA_ROOT`（大文件盘：models/data/logs/ckpts）、以及训练是否在容器 `CONTAINER` 内。

### 第 1 步：同步 MoE 代码
- Lumen-RL 是 editable 安装（`pip install -e`），**改文件即生效于下次启动**。qwen8b 已停，拉取代码无冲突。
- 拉取（fast-forward，禁止覆盖本地未处理改动）：
```bash
git -C "$RL_ROOT/Lumen-RL" fetch origin
git -C "$RL_ROOT/Lumen-RL" status --short        # 若有本地改动先确认
git -C "$RL_ROOT/Lumen-RL" pull --ff-only origin dev/vllm-fsdp-dapo
git -C "$RL_ROOT/Lumen-RL" log --oneline -3       # 应看到 84c7264 / ddb3b4c
```
  - 若本机 Lumen-RL 就在共享 NFS `/home/xysheng/lumen_rl`，代码可能已是最新（先看 `git log`）。
  - 若 `--ff-only` 因本地分叉失败，别 `reset --hard`；先看 `git status` 再处理。
- 不需要重装（editable）；如担心，可 `pip install -e "$RL_ROOT/Lumen-RL" --no-deps` 重新登记。

## MoE+EP 改了什么（已验证，直接可用）
- `lumenrl/engine/training/megatron_native_engine.py`：HF config 判 MoE → `get_gpt_decoder_block_spec` 建模；
  EP-aware 权重加载（按 EP rank 选本地专家 + ETP fc1列/fc2行切分）；rollout 权重同步（ETP-gather→EP all-gather→PP broadcast）；
  EP lockstep 微批同步（跨 EP 组补齐，防 all-to-all 挂）；TP>1+MoE **自动强制 sequence_parallel** + thd 补齐到 TP 倍数；R3 接线（默认关）。
- `lumenrl/engine/training/qwen3moe_megatron_bridge.py`：Qwen3-MoE↔Megatron 专家/router/shared_expert 权重转换。
- `lumenrl/moe/moe_utils.py`（+ `__init__.py`）：Megatron 兼容 R3 record/replay（`iter_megatron_routers` + 包装 `TopKRouter.routing`）。
- `lumenrl/core/config.py`、`workers/actor_worker.py`：MoE/EP/R3 配置贯通（`megatron_cfg` 里 num_experts/expert_model_parallel_size/expert_tensor_parallel_size/moe_router_dtype 等；`moe.r3.enabled`）。
- 新增配置：
    · `examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_smoke.yaml`（EP8 smoke）
    · `examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_longrun.yaml`（EP8 longrun，**推荐**）
    · `examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_longrun_e_full.yaml`（TP2·PP2·CP2·EP2·ETP2 全组合，R3 已关）

## 已验证结论（8×MI355X gfx950）
- dense 回归无破坏；MoE 建模/加载/前向/训练/权重 roundtrip 精确（HF 键 100%，误差 0）：EP / ETP>1 / TP>1(+SP) / PP>1 / CP×EP 及全组合。
- MoE+EP DAPO smoke（Qwen3-30B-A3B-Base，EP8）2 步健康：ppo_kl≈0、rollout_corr/kl≈0.002、无 OOM/NaN。
- 30B EP8 longrun step1 健康（entropy 0.92、grad_norm 0.16、rollout_corr/kl 0.00195、seq/max_len 20632、峰值 ~147GB/卡）。

## ⚠️ R3 路由回放：默认关，longrun 勿开
- 构建块正确（record→replay 精确复现，CP×EP 及全组合验证）；
- 但**跨调用回放在真实 DAPO 流程不可用**：old_log_probs 的 compute_log_probs 与 update_policy 动态分箱不同 → 回放错配污染路由 → **ppo_kl 飙到 ~0.9**。
- 故 `moe.r3.enabled` 默认 false，配置里也已关。rollout_corr/kl 不开 R3 本就 ~0.002。别在 longrun 打开 R3。

## 在本机把 MoE 跑起来
> 下面命令按本机是否有容器分两种；把 `<RUN>` 换成本机实际执行方式（裸机直接跑，或 `docker exec <CONTAINER> bash -lc '...'`）。PYTHONPATH 需含 Lumen-RL/aiter/Lumen。

1) **准备 MoE 模型**（若没有）：下载 Qwen3-30B-A3B-Base（约57GB）到 `$DATA_ROOT/models/Qwen3-30B-A3B-Base`：
```bash
python - <<PY
import os
from huggingface_hub import snapshot_download
D=os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-30B-A3B-Base", local_dir=f"{D}/models/Qwen3-30B-A3B-Base",
  allow_patterns=["*.json","*.safetensors","*.model","tokenizer*","*.tiktoken","*.py","*.jinja","*.txt"])
print("done")
PY
```
数据集可复用 qwen8b 已过滤的 `$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/*.parquet`（Qwen3 tokenizer 通用）。

2) **确认卡已空再起**（qwen8b 已停；仍先核对无残留进程/显存）：`pgrep -af "lumenrl.trainer.main|VLLMRayServer|EngineCore"`、`rocm-smi`。若有残留：`ray stop --force; pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"; pkill -9 -f "[E]ngineCore"`。

3) **MoE smoke（2步，验证本机环境）**——直接调 run_dapo.sh：
```bash
cd "$RL_ROOT/Lumen-RL"
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_smoke.yaml \
MODE=bf16 STEPS=2 \
MODEL_PATH="$DATA_ROOT/models/Qwen3-30B-A3B-Base" \
RUN_ID=smoke-moe-a3b-ep8 LOG="$DATA_ROOT/logs/smoke-moe-a3b-ep8.log" \
EXTRA_OVERRIDE="checkpointing.resume=false checkpointing.save_steps=1000 logger.wandb_enabled=false policy.generation.vllm_cfg.enable_sleep_mode=false" \
bash examples/DAPO/run_dapo.sh
```
健康判据：日志出现 `MoE+EP spec: ... EP=8 ...`、`callbacks: step=2`、ppo_kl≈0、rollout_corr/kl~0.002、无 OOM/NaN。

4) **MoE longrun（推荐 EP8，1000步/wandb/每50步ckpt）**：
```bash
cd "$RL_ROOT/Lumen-RL"
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_longrun.yaml \
MODE=bf16 STEPS=1000 \
MODEL_PATH="$DATA_ROOT/models/Qwen3-30B-A3B-Base" \
RUN_ID=longrun-moe-a3b-ep8 LOG="$DATA_ROOT/logs/longrun-moe-a3b-ep8.log" \
WANDB_API_KEY="$(cut -d= -f2- "$RL_ROOT/wandb.key" | tr -d '[:space:]')" \
bash examples/DAPO/run_dapo.sh
```
后台跑请 `nohup ... &` 或 `docker exec -d`。按 runbook §13 盯前 2 步（无 OOM/NaN/hang、ppo_kl≈0、grad_norm 有限、entropy 不塌、wandb `View run` 出现）。单步 ~450s+（长上下文 rollout 主导），1000 步需数天。

## 参考资料
- 环境重建（若本机缺容器/TE/Apex/依赖）：/home/xysheng/working/amd-rl-runbook/dapo-lumenrl-vllm-fsdp-megatron-new-machine-runbook.md（TE=ROCm 2.15 源码编译、不 pip install transformer_engine；Apex=ROCm 固定 commit；megatron-core 0.18.2）。
- spur 集群那套脚本（若本机也是 spur 节点）：/home/xysheng/spur_scripts/（含 10_moe_smoke.sh、12_moe_longrun.sh、12_moe_longrun_e_full.sh）。
- vime 参考实现：/home/xysheng/working/vime-rl/vime。
- 独立验证脚本（可复现并行组合/R3 验证）：/home/xysheng/lumen_rl/moe_parallel_test.py、moe_r3_test.py（torchrun --nproc_per_node=8，env 传 TP/PP/CP/EP/ETP/E/L/R3）。

## 红线 / 约束
- 两后端不能同时占卡（qwen8b 已停，起 MoE 前确认卡空即可）。
- 不改基础镜像 torch/triton/vLLM/flash-attn；不 pip install transformer_engine（会装成 NVIDIA 版）。
- 大文件只放 DATA_ROOT（本地大盘），别放 NFS /home。vLLM 保持 enable_sleep_mode=false。
- 不改 git config；提交用 `--author="xysheng-AMD <xysheng@amd.com>"`，推 `dev/vllm-fsdp-dapo`（SSH remote：`git@github.com:ZhangDanyang-AMD/Lumen-RL.git`）。
- MoE+TP 自动强制 sequence_parallel（Megatron 硬约束）；MoE aux_loss 默认 0；别在 longrun 开 R3。

## 当前决定的优先级（照此做）
- **主线：把 30B MoE EP8 longrun 跑到 1000 步、盯健康、拿收敛曲线**（之前因节点故障未跑完）。这是唯一要做的事。
- **R3 保持关闭，不要 debug**：已确认引擎内 R3（old-logprob→update，两次都是 Megatron、权重相同、路由确定性一致）**无收益**——不开 R3 时 ppo_kl≈0 已证明 old/update 路由本就一致；开了反因跨调用分箱错配污染路由（ppo_kl→~0.9）。且 rollout_corr/kl 已 ~0.002。故 longrun 全程 `moe.r3.enabled=false`。

## 未来可选项（本次不做，除非另有指示）
- 真正有价值的 R3 是 **vLLM→训练**（录 vLLM rollout 每 token 路由专家、训练回放，参考 vime `prepare_routed_experts_for_routing_replay`），但需改 vLLM 导出每 token 路由，工程量大，当前不做。
- MoE aux_loss>0 时 autoscaler 与 `num_mb/cp` 梯度校正一致性；PP+MoE / 更大 MoE / ETP≠TP 且 TP>1 的极端组合完整实测。
