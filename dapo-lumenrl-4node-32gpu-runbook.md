# LumenRL DAPO 4 节点 32 卡 Runbook（spur + vime-rocm 镜像）

> 把 `dapo-lumenrl-vime-fsdp-megatron-runbook.md` 那套单节点 8 卡的环境扩到 **4 节点 32 卡**。
> 训练/rollout 的配方一字不改（同一份 config、同样的超参），变的只有编排：
> 多节点 Ray 集群、跨节点 NCCL、以及"哪台机器上跑什么"。
>
> **前置**：先读那份 vime runbook 的 §2–§8。本文只写多节点独有的部分，不重复镜像、依赖、config 的内容。

✅ **已验证**（2026-08-05，4×8 MI355X gfx950 / 288GB per card / 2751GB RAM，Qwen3-30B-A3B-Base）：

| 组合 | 结果 |
|---|---|
| FSDP2 MoE 4k smoke，32 卡，rollout TP=1（32 replica） | `exit=0`，3 步，24 分 03 秒 |
| Megatron-Native MoE 4k smoke，32 卡，EP=8 | `exit=0`，3 步，9 分 17 秒 |
| FSDP2 MoE 4k smoke，32 卡，**rollout TP=8**（4 replica） | `exit=0`，3 步 |

三次都无 Traceback、无 OOM、无 `HSA_STATUS`，权重同步覆盖率断言全过，收尾后 32 张卡回到约 298MB 空闲基线。

⚠️ **两个已知边界，先说清楚**：

1. **跨节点集合通信目前走 TCP（`ens3`，200Gb/s），不是 RDMA。** 那 8 张 400Gb/s 的 ionic RoCE 网卡在容器里用不起来，原因和已经排除的假设都记在 §7。
2. **同一批卡上不能同时跑两个后端**，也不能和别人共用节点——引擎按"占整卡比例"算预算，见 vime runbook §15.2。

---

## 1. 拿到 4 个节点：卡在配额，不是卡在资源

集群有 200+ 个节点，但 `spur alloc -N 4` 会一直 `PENDING (QOSGrpNodeLimit)`。原因是 QOS 的**组配额**：

```bash
spur accounts show qos            # 看 GrpTRES 那一列
spur accounts show user where name="$USER"
```

实测本账号（`amd-aifw-dev`）：

| QOS | GrpTRES | 说明 |
|---|---|---|
| `amd-aifw-dev-qos` | `node=19` | **整个 account 全体成员共享**，实测已被队友用掉 17 个 |
| `amd-burst-qos` | `node=128` | 低优先级（priority 100 对 10000），但配额大 |

所以 4 节点必须走 `amd-burst-qos`。而**`spur alloc` 不接受 `-q`**（参数表里没有），`spur run` 有：

```bash
spur run -q amd-burst-qos -N 4 --gpus-per-node=8 \
  -t 1-00:00:00 -A amd-aifw-dev -p amd-spur --pty bash -l
```

> ⚠️ `--gpus-per-node=8` 而不是 `-G 32`：`-G` 的语义是"整个作业的 GPU 总数"，调度器可以任意摊到 4 个节点上。
>
> ⚠️ **`--pty` 不能省**。非真 tty 申请 GPU 在本集群不可靠，会掉进 `JobHoldMaxRequeue`（实测队列里常年挂着 30+ 个这种僵尸作业）。
>
> ⚠️ 别在已有分配的交互 shell 里跑这条。那里有 `SLURM_JOB_ID`，`spur run` 会把它当成"我在现有分配里"，于是变成那个作业的一个 job step——`-N 4`、`-q` 全被忽略。先确认 `env | grep SLURM_JOB_ID` 无输出。

拿到后固定几个变量，后面全引用它们：

```bash
squeue -u "$USER" -o '%.10i %.14q %.12j %.8T %.6D %.34R'
scontrol show job "$JOBID" | head -12      # 确认 TresPerNode=gpu:8/node、NodeList 4 个
```

### 1.1 rank 与节点的对应关系（后面所有拓扑推导的基础）

Ray 按节点**连续**发 rank。本次实测：

| 节点 | actor rank |
|---|---|
| crsuse2-m2m-068（head） | 0–7 |
| crsuse2-m2m-100 | 8–15 |
| crsuse2-m2m-008 | 16–23 |
| crsuse2-m2m-204 | 24–31 |

这条性质决定了 Megatron 的 EP 分组和 rollout TP 分组能不能落在节点内，见 §10、§11。**它是观测到的行为，不是保证**——所以代码里对"一组必须同节点"是强制检查而非假设。

---

## 2. 在任意节点上执行命令：`nx.sh`

多节点最先撞到的不是训练，是"怎么在第 2、3、4 台机器上跑一条命令"。三条路里两条不通：

| 方式 | 结果 |
|---|---|
| `ssh <node>` | `Permission denied (publickey)`。计算节点 `AllowUsers ubuntu root`，容器里是普通用户 |
| `spur exec <JobID>` | **只到 head 节点**。没有指定节点的参数，控制端固定代理到 head |
| `srun --jobid X --overlap -w <node>` | 可行，**但 stdin 不是真 tty 时静默丢弃输出**（只打一行 `raw mode unavailable` 警告，命令看起来"什么都没发生"） |

第三条加一层 `script -qec` 造 pty 就正常了。封装成 `~/nx.sh`：

```bash
cat > /home/$USER/nx.sh <<'EOF'
#!/usr/bin/env bash
# Run a bash snippet on one specific node of a spur multi-node job.
#
# spur exec <jobid> only ever reaches the job's head node, and srun job steps
# silently produce nothing unless stdin is a real tty (spur 0.7.0 prints
# "raw mode unavailable" and drops the output), hence the script(1) wrapper.
# The snippet travels through $HOME because that is shared NFS across nodes.
#
#   JOBID=38564 ./nx.sh crsuse2-m2m-008 'hostname; rocm-smi'
#   JOBID=38564 ./nx.sh crsuse2-m2m-008 < some_script.sh
set -uo pipefail
: "${JOBID:?set JOBID to the spur job id}"
NODE="${1:?usage: nx.sh <node> [command...]}"
shift
D="$HOME/.nx"
mkdir -p "$D"
F="$D/cmd_${NODE}_$$_$RANDOM.sh"
if [ $# -gt 0 ]; then
  printf '%s\n' "$*" > "$F"
else
  cat > "$F"
fi
script -qec "srun --jobid $JOBID --overlap -w $NODE -N1 -n1 bash -l $F" /dev/null
rc=$?
rm -f "$F"
exit $rc
EOF
chmod +x /home/$USER/nx.sh
```

自检（4 台都该回自己的 hostname 和 8 卡）：

```bash
export JOBID=<你的作业号>
for n in 008 068 100 204; do
  printf "%s: " "$n"
  ~/nx.sh crsuse2-m2m-$n 'hostname; rocm-smi --showid 2>/dev/null | grep -oE "GPU\[[0-9]+\]" | sort -u | wc -l'
done
```

> 输出里会夹一个 `^@`（script 的产物），用 `sed 's/\^@//'` 滤掉即可。

---

## 3. 盘位：模型和数据必须在共享 NFS

单节点 runbook 把 `DATA_ROOT` 放在 `/mnt/m2m_nobackup`（node-local NVMe，28T）。**4 节点不能这么放**——那块盘只挂在各自的机器上，4 台会各看到一个空目录。

```bash
export RL_ROOT=/home/$USER/lumen_rl          # 代码，NFS
export DATA_ROOT=/home/$USER/rl_data         # 模型 + 数据 + 日志，NFS（必须）
export SCRATCH_ROOT=/home/$USER/rl_data      # checkpoint；想放本地盘就每台各指一个
```

代价是加载变慢（§12 算了这笔账）。折中办法是把模型 `cp` 到每台的 `/mnt/m2m_nobackup`，代价是 4 份拷贝和一次分发——本次没做，因为 smoke 不值得。

⚠️ **`aiter` 的 JIT 产物（`$RL_ROOT/aiter/aiter/jit/*.so`）也在 NFS 上**。好消息是它一次编译四台共用；坏消息是**首次运行时 32 个 actor 会并发往同一个目录编译**。本次没踩到，是因为那 16 个 `.so` 早就编好了。新机器第一次上多节点，建议先用单节点跑一次 smoke 把 JIT 预热完，再上 4 节点。

---

## 4. 容器：两处单节点不需要的改动

```bash
cat > /home/$USER/4node/01_container.sh <<'EOF'
#!/usr/bin/env bash
# Start the vime-rocm container on one node. Runs on the node (docker host side).
#
# Two things the single-node runbook does not need, both required as soon as a
# collective spans nodes:
#
#   --ulimit memlock=-1  the host caps memlock at 8MB, and RDMA memory
#                        registration needs more than that.
#   ionic provider       the image ships libibverbs providers for mlx4/mlx5/efa/
#                        ... but not for the AMD Pensando ionic NICs, so inside
#                        the container RCCL only sees mlx5_0 (200Gb/s frontend)
#                        and misses the eight 400Gb/s ionic HCAs entirely.
#                        Bind-mounting the host provider makes them visible.
set -euo pipefail
source /home/$USER/4node/env.sh

IB_MOUNTS=()
IONIC_LIB="$(ls /usr/lib/x86_64-linux-gnu/libionic.so.* 2>/dev/null | head -1 || true)"
if [ -n "$IONIC_LIB" ] && [ -f /etc/libibverbs.d/ionic.driver ]; then
  IB_MOUNTS+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
  IB_MOUNTS+=(-v "$IONIC_LIB":/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
else
  echo "WARNING: no host ionic provider found; RCCL will fall back to TCP" >&2
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add=video --privileged \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v /home/$USER:/home/$USER \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  "${IB_MOUNTS[@]}" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e SCRATCH_ROOT="$SCRATCH_ROOT" \
  -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" -e ATOM_DIR="$RL_ROOT/ATOM" \
  "$IMAGE" -lc "sleep infinity"

docker exec "$CONTAINER" bash -lc '
python3 -c "import torch; print(\"torch\", torch.__version__, \"gpus\", torch.cuda.device_count())"
python3 -c "import ray; print(\"ray\", ray.__version__)"
'
echo "container up on $(hostname)"
EOF
```

配套的 `env.sh`（换机只改这一份）：

```bash
mkdir -p /home/$USER/4node
cat > /home/$USER/4node/env.sh <<'EOF'
#!/usr/bin/env bash
# Shared settings for the 4-node (32 GPU) LumenRL DAPO run.
#
# DATA_ROOT must be on shared NFS, not /mnt/m2m_nobackup: that disk is
# node-local, so a 4-node job would see four different empty directories.
export RL_ROOT=/home/xysheng/lumen_rl
export DATA_ROOT=/home/xysheng/rl_data
export SCRATCH_ROOT=/home/xysheng/rl_data
export CONTAINER=rl-vime-4node
export IMAGE=vllm/vime-rocm

# spur exec <jobid> always lands on the head node, so keeping the Ray head and
# the trainer driver there means monitoring works without the srun/pty detour.
export HEAD_NODE=crsuse2-m2m-068
export HEAD_IP=10.245.147.228
export RAY_PORT=6379
export NODES="crsuse2-m2m-008 crsuse2-m2m-068 crsuse2-m2m-100 crsuse2-m2m-204"

# ens3 is the only routable inter-node interface; the containers also see
# docker0/br-* (172.x), which NCCL and Ray must not pick.
export NET_IF=ens3
export LOG4N=/home/xysheng/logs/4node
EOF
```

`HEAD_IP` 从 head 节点取：`ip -o -4 addr show ens3 | awk '{print $4}' | cut -d/ -f1`。

四台一起起容器 + 装依赖（**依赖要顺序装**，`pip install -e` 会往 NFS 上的仓库写 egg-info，并发会打架）：

```bash
for n in 008 068 100 204; do
  ~/nx.sh crsuse2-m2m-$n 'bash /home/$USER/4node/01_container.sh'
  ~/nx.sh crsuse2-m2m-$n 'source /home/$USER/4node/env.sh
    docker exec "$CONTAINER" bash -lc "bash \$RL_ROOT/install_deps.sh"'
done
```

判据：四台的 `env_verify.sh` 都要打 `ENV OK`。本次四台的版本号逐项一致：
`torch 2.10.0+rocm7.0 / hip 7.0.51831`、`vllm 0.22.1rc1.dev392+g43914dd74`、`ray 2.56.0`、
`transformers 5.13.1`、`flydsl 0.1.8`、`megatron.core 0.16.0rc0`、`transformer_engine 2.12.0.dev0`。

---

## 5. Ray 集群：两个非显然的点

### 5.1 raylet 的环境变量必须自己铺

**Ray 不会把 driver 的环境传给其它节点上的 actor。** 远程 worker 继承的是**它所在节点 raylet 启动时的环境**。所以 `run_dapo.sh` 为单节点 driver 导出的那一整套（`PYTHONPATH`、`NCCL_CUMEM_ENABLE=0`、`LUMENRL_FP32_MOE_ROUTER=0` …）必须在**每台**机器 `ray start` 之前就位，否则另外 3 台上的 24 个 actor 会带着"没有 PYTHONPATH、fp32 router"的状态起来——而且不会报错，只会让 `rollout_corr/kl` 悄悄变大。

```bash
cat > /home/$USER/4node/ray_env.sh <<'EOF'
#!/usr/bin/env bash
# Environment for the raylets, sourced inside the container before `ray start`.
#
# Why this file exists: Ray does not propagate the driver's environment to
# actors on other nodes. Remote workers inherit whatever the raylet was started
# with, so every variable run_dapo.sh exports for a single-node run has to be
# present here too or the 24 actors on the three non-driver nodes come up
# without PYTHONPATH, without NCCL_CUMEM_ENABLE=0 and with a fp32 MoE router.
#
# Mirrors examples/DAPO/run_dapo.sh (MODE=bf16 branch) plus the two switches the
# runbook's wrapper sets for the MoE FSDP2 path.
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
# ROCm/HIP has no expandable_segments; run_dapo.sh unsets it when handed an
# empty value, and the raylet environment must match.
unset PYTORCH_CUDA_ALLOC_CONF
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export ATOM_DISABLE_VLLM_PLUGIN=1
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LUMEN_DISABLE_HF_ATTN_PATCH=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" LUMENRL_LOG_LEVEL=INFO
export MODEL_NAME="$DATA_ROOT/models/Qwen3-30B-A3B-Base"
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen:$RL_ROOT/ATOM"

# FSDP2 chunk_cat aperture-violation workaround, and the BF16 MoE router that
# has to match vLLM on both sides.
export LUMENRL_FSDP_CHUNK_CAT_FALLBACK=1
export LUMENRL_FP32_MOE_ROUTER=0

# First run on new hardware: read the loaded vLLM buffers back and compare them
# bit-for-bit against what was sent. Costs about 0.1s/step. Read inside the
# replica actor, so it has to be here rather than on the launcher command line.
export LUMENRL_WEIGHT_SYNC_VERIFY=1

# Keep collectives and the torch.distributed rendezvous off docker0/br-*.
export NCCL_SOCKET_IFNAME=ens3
export GLOO_SOCKET_IFNAME=ens3

# Inter-node collectives run over TCP on ens3 (200Gb/s) instead of the eight
# 400Gb/s ionic RoCE HCAs -- see the runbook section on RDMA for why.
export NCCL_IB_DISABLE=1
EOF
```

### 5.2 `run_dapo.sh` 开头会把集群拆掉

`examples/DAPO/run_dapo.sh` 第 145 行是 `ray stop --force`，用来清理单节点上一次跑的残留。在多节点路径上它会**把我们刚建好的 head 拆掉**，而 driver 紧接着就要连它。

不要 fork `run_dapo.sh`（vime runbook §8 记过一次漂移 117 行的教训）。放一个只吞掉 `stop` 的 PATH shim：

```bash
mkdir -p /home/$USER/4node/bin
cat > /home/$USER/4node/bin/ray <<'EOF'
#!/usr/bin/env bash
# PATH shim in front of the real ray CLI, used only on the 4-node path.
#
# run_dapo.sh opens with `ray stop --force` to clear leftovers from a previous
# single-node run. Here the 4-node cluster is started before the launcher, and
# that call would tear down the head this driver is about to connect to. Swallow
# `stop` and forward everything else untouched, so run_dapo.sh stays the single
# source of truth instead of being forked.
if [ "${1:-}" = "stop" ]; then
  echo "[4node] ignoring 'ray stop' -- the multi-node cluster is managed externally" >&2
  exit 0
fi
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEAN_PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$SELF_DIR" | paste -sd: -)"
exec env PATH="$CLEAN_PATH" ray "$@"
EOF
chmod +x /home/$USER/4node/bin/ray
```

### 5.3 起集群

```bash
cat > /home/$USER/4node/02_ray_start.sh <<'EOF'
#!/usr/bin/env bash
# Start this node's raylet inside the container. Runs on the node.
#   02_ray_start.sh head|worker
set -euo pipefail
source /home/$USER/4node/env.sh
ROLE="${1:?usage: 02_ray_start.sh head|worker}"
MY_IP="$(ip -o -4 addr show "$NET_IF" | awk '{print $4}' | cut -d/ -f1)"

if [ "$ROLE" = head ]; then
  START="ray start --head --port=$RAY_PORT --node-ip-address=$MY_IP --num-gpus=8 \
    --dashboard-host=0.0.0.0 --disable-usage-stats"
else
  START="ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$MY_IP --num-gpus=8 \
    --disable-usage-stats"
fi

docker exec "$CONTAINER" bash -lc "
set -e
source /home/$USER/4node/ray_env.sh
ray stop --force >/dev/null 2>&1 || true
$START
"
echo "ray $ROLE started on $(hostname) ($MY_IP)"
EOF
```

```bash
~/nx.sh crsuse2-m2m-068 'bash /home/$USER/4node/02_ray_start.sh head'
for n in 008 100 204; do ~/nx.sh crsuse2-m2m-$n 'bash /home/$USER/4node/02_ray_start.sh worker'; done

# 判据：4 个 node、32.0 GPU、accelerator-type AMD-Instinct-MI355X-OAM
~/nx.sh crsuse2-m2m-068 'source /home/$USER/4node/env.sh
  docker exec "$CONTAINER" bash -lc "ray status | grep -E \"GPU|node_\""'
```

> `--node-ip-address` 必须显式给 `ens3` 的地址，否则 Ray 可能挑到 `docker0`（172.17.x）那种跨节点不可路由的地址。

---

## 6. 跨节点通信：为什么现在走 TCP

这一节记的是**一次失败排查的完整过程**。结论是"暂时用 TCP"，但把已经排除的假设写下来，下一个人才不用重走。

### 6.1 硬件

```bash
~/nx.sh crsuse2-m2m-068 'for d in /sys/class/infiniband/*; do n=$(basename $d); \
  printf "%-10s %-12s %s\n" "$n" "$(cat $d/ports/1/state)" "$(cat $d/ports/1/rate)"; done'
```

每台 9 张 HCA 全部 ACTIVE：**8 张 `ionic_*` @ 400Gb/s**（AMD Pensando AINIC，GPU 互联网）+ **1 张 `mlx5_0` @ 200Gb/s**（前端网）。另有 `ens3` @ 200Gb/s 走 TCP。

⚠️ **ionic 网卡没有 IPv4 地址**，只有 IPv6：每张卡一个 `fc01:*::/64`，而且**每台机器的同一张卡在不同的 /64 网段**（068 的 `enP2p0s9` 是 `fc01:800:b00d:2d37::/64`，008 的是 `fc01:800:c80d:2d6f::/64`）。这是路由型 RoCE fabric，所以后面 GID 选择很关键。

### 6.2 四个坑，逐个排掉

**坑 1：容器里看不见 ionic 设备。** 镜像自带 mlx4/mlx5/efa/irdma 等 15 个 provider，**唯独没有 ionic**。所以 RCCL 只看到 `mlx5_0`：

```
NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB ens3:...
```

宿主有 `/etc/libibverbs.d/ionic.driver` 和 `libionic.so.1.0.54.0`，挂进容器（§4 已包含）后 9 张全可见。**跨 rdma-core 大版本（宿主 v54 provider + 容器 v39 libibverbs）实测能正常加载**，不是问题。

**坑 2：memlock。** 宿主 `ulimit -l` 是 8192KB，RDMA 注册内存必失败。容器要 `--ulimit memlock=-1`。验证方法（ctypes 直接调 `ibv_reg_mr`，不需要跑 NCCL）：

```bash
# 容器内：9 张 HCA 注册 4MB 主机内存应全部 OK
python3 /home/$USER/4node/probe_ibv.py
```

实测：容器内（memlock unlimited）**9 张全 OK**；宿主上只有第一张成功、其余 `ENOMEM` —— 正好反证了 8MB 上限。

**坑 3：GID index。** ionic 的 GID 表里 `gid[0]` 是 link-local（`fe80::`），`gid[1]` 才是可路由的 `fc01::`。用默认的 0 会**直接挂死**（channel 建不起来，90 秒超时）。必须 `NCCL_IB_GID_INDEX=1`。

**坑 4（未解决）：`ibv_reg_mr` EINVAL。** 修好前三个之后，QP 和 FIFO 都建起来了，但代理 buffer 注册失败：

```
NET/IB: NCCL Dev 0 IbDev 0 Port 1 qpn 2048 mtu 5 ... GID 1 (372D0DB0000801FC/...) fifoRkey=0
NCCL WARN Call to ibv_reg_mr failed with error Invalid argument
  net_ib.cc:1783 <- net_tmp.cc:1014 <- net_tmp.cc:485 <- transport.cc:256 (connect)
```

已经排除的解释，都有实测依据：

| 假设 | 实测 |
|---|---|
| GPU Direct RDMA 不可用 | 日志本来就写着 `GPU Direct RDMA Disabled for GPU 0 / HCA 0 (distance 7 > 5)`，走的是主机中转 buffer |
| PCI relaxed ordering 不被支持 | `NCCL_IB_PCI_RELAXED_ORDERING=0` 无效 |
| dmabuf / GDR 相关 | `NCCL_DMABUF_ENABLE=0`、`NCCL_NET_GDR_LEVEL=0` 均无效 |
| 某张网卡的问题 | `mlx5_0` 单独用、`ionic_0` 单独用、`NCCL_IB_MERGE_NICS=0` 全部同样失败 → 问题在被注册的**内存**，不在网卡 |
| NCCL 用的锁页内存注册不了 | ctypes 实测 `malloc` 和 `hipHostMalloc`（torch `pin_memory`）**都能注册成功**；只有显存失败，且错误是 `EFAULT(14)` 而不是 `EINVAL(22)` |
| rdma-core 版本混搭（v39 libibverbs + v54 provider） | 主机内存注册在这个组合下正常，见坑 2 |

**真正的原因大概率是"没用厂商插件"。** 集群自带 `/etc/profile.d/99-rccl-anp.sh`，里面是 AMD 官方推荐配置，第一行就是：

```bash
export NCCL_NET_PLUGIN=anp        # AMD-ANP，插件在 /opt/rocm-7.0.1/lib/librccl-net.so
export NCCL_DMABUF_ENABLE=1
export NCCL_IB_GID_INDEX=1        # 和坑 3 的实测一致
export NCCL_IB_TC=96
export NCCL_IB_FIFO_TC=184
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_USE_INLINE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_GDRCOPY_ENABLE=0
export IONIC_LOCKFREE=all
```

RCCL 日志里那句 `NET/Plugin: Could not find: librccl-net.so. Using internal net plugin.` 就是在说这件事——内置的 `net_ib.cc` 路径不适用于这些 AINIC 网卡。

**但插件在容器里加载不了**：它要 `GLIBC_2.38`（只缺 3 个符号：`__isoc23_fscanf`、`__isoc23_strtol`、`__isoc23_strtoll`），而镜像是 Ubuntu 22.04 / glibc 2.35。宿主是 24.04，`/opt/amd/ainic/deb-repo` 里也只有 ub2404 的包。把宿主的 libibverbs v54 一起挂进去同样撞 `GLIBC_2.38`。

### 6.3 现状与下一步

暂时的做法就是 `ray_env.sh` 里那行 `NCCL_IB_DISABLE=1`，走 `ens3` 的 TCP。实测 **32 rank 跨 4 节点 all-reduce 通过**，两个 smoke 也都跑完了。

想接着修，按这个顺序试：

1. 把宿主整套 rdma-core + ANP 插件的依赖树（`libionic.so.1`、`libmpi.so.40`、`libopen-rte`、`libopen-pal`、`libevent`、`libhwloc` …）连同一个 glibc 2.38+ 的 `ld.so` 一起注入容器——技术上可行但很脆。
2. 换一个 glibc ≥ 2.38 的基础镜像重建 vime 那套栈（torch/vllm/megatron/TE/aiter），代价最大但最干净。
3. 找 AMD 要一个为 Ubuntu 22.04 编的 ANP 插件（三个符号的差距，重编一次即可）。

**修好之后的收益是可量化的**：现在 32 卡 FSDP2 的 `timing/weight_sync_s` 是 32 秒（§9），Megatron 是 19 秒（§10）；这些数字里的大头就是跨节点 all-gather 走 TCP。

---

## 7. 上训练之前的两个预检

一次 smoke 从启动到出第一步指标要十几分钟，先用两个便宜的探针把"环境有没有铺对"和"跨节点能不能通"确认掉。

### 7.1 raylet 环境是否真的传到了每个节点

```bash
# 32 个 1-GPU actor，报告每节点的 lumenrl 可导入性和三个开关
~/nx.sh crsuse2-m2m-068 'source /home/$USER/4node/env.sh
  docker exec "$CONTAINER" bash -lc "source /home/$USER/4node/ray_env.sh
    RAY_ADDR=$HEAD_IP:$RAY_PORT python3 /home/$USER/4node/probe_ray.py"'
```

判据：4 个节点都是 `lumenrl+repo_aiter=True FP32_ROUTER=0 CHUNK_CAT=1 SYNC_VERIFY=1 NCCL_CUMEM=0`。

> 这个探针用的是瞬时 task，GPU 槽位会被复用，所以"每节点 actor 数"看起来不均匀是正常的；真实运行是长驻 actor，Ray 按每节点 8 个 GPU 槽严格摊开。

### 7.2 32 rank 跨节点 all-reduce

`probe_nccl.py` 复刻 `_rendezvous_ray_group` 的做法（取 rank0 的 node ip + 空闲端口，32 个 actor 各自 `local_rank=0` 入组），然后 all-reduce 校验和：

```bash
~/nx.sh crsuse2-m2m-068 'bash /home/$USER/4node/probe.sh 32 ""'
```

判据：`actors placed on 4 nodes`、`per-node actor count: {…: 8, …: 8, …: 8, …: 8}`、`all_reduce ok on 32/32 ranks`。

> 探针支持用 `PROBE_ENV="VAR=val,VAR=val"` 通过 Ray `runtime_env` 给 actor 注入变量——**调 NCCL 参数必须用这个**，在 driver 上 export 是无效的（同 §5.1 的原因）。这一点我自己踩过：在 driver 上设 `NCCL_IB_DISABLE=1` 后测试"通过"，其实 actor 根本没收到，跑的还是原配置。

---

## 8. FSDP2 MoE smoke（32 卡，rollout TP=1）

配方和单节点**完全相同**——同一个 config 文件、`GPU_UTIL=0.45`、sleep level 2、3 步。只加了四个 cluster 覆盖把 actor 世界从 8 拉到 32：

```bash
cat > /home/$USER/4node/03_smoke.sh <<'EOF'
#!/usr/bin/env bash
# MoE 4k smoke across the 4-node Ray cluster. Runs on the head node.
set -euo pipefail
source /home/$USER/4node/env.sh
mkdir -p "$LOG4N"

docker exec "$CONTAINER" bash -lc "
set -uo pipefail
source /home/$USER/4node/ray_env.sh
export PATH=/home/$USER/4node/bin:\$PATH
MODEL=moe BACKEND=fsdp2 SCALE=smoke SLEEP=1 STEPS=3 GPU_UTIL=0.45 \
EXTRA_OVERRIDE='cluster.num_nodes=4 cluster.gpus_per_node=8 cluster.ray_address=$HEAD_IP:$RAY_PORT controller.ray.actor.num_workers=32' \
LOG=$DATA_ROOT/logs/moe-fsdp2-smoke-4node.log \
  bash $RL_ROOT/run_lumenrl_dapo_vime.sh
"
EOF
```

> `controller.ray.actor.num_workers=32` 必须显式给：config 里写死的是 8，而 `cluster.num_nodes×gpus_per_node` 只是 `num_workers=0`（自动推断）时才用的默认值。
>
> `cluster.ray_address` 一设，`RayCluster.init()` 就走 `ray.init(address=...)` 连已有集群，而不是自己建一个本地的。

启动判据（依次出现）：

```
Ray cluster initialized: 4 nodes x 8 GPUs
Created resource pool 'actor' with 32 GPUs (layout=[32], colocate=None)
Started 32 workers of type LumenActorWorker (1.0 GPUs each)
Ray rendezvous complete: 32 distributed actors on 10.245.147.228:54287.   ← 跨节点 NCCL 建组成功
[lumenrl] FSDP2 reduce-scatter copy-in: slicing fallback installed         ← 应有 32 条
LumenActorWorker: initialized fsdp2 engine.                               ← 应有 32 条
VLLMReplicaManager: launched 32 colocated rollout replicas (TP=1).
```

> `layout=[32]` 只是元数据。LumenRL 不用 placement group，actor 是普通 `ray.remote(num_gpus=1)`，靠 Ray 的 GPU 记账保证每节点最多 8 个。实测就是 8/8/8/8。

### 8.1 实测（3 步，`exit=0`，24 分 03 秒，其中约 13 分钟在从 NFS 加载权重）

| 指标 | 4 节点 32 卡 (step 1/2/3) | 单节点 8 卡（vime runbook §16.5） |
|---|---|---|
| `rollout_corr/kl` | 0.00178 / 0.00124 / 0.00167 | 0.00168 / 0.00178 / 0.00181 |
| `mismatch/abs_diff` | 0.0203 / 0.0191 / 0.0224 | 0.0202 / 0.0239 / 0.0222 |
| `reward/accuracy` | 0.117 / 0.172 / 0.102 | 0.125 / 0.117 / 0.086 |
| `entropy` | 0.718 / 0.639 / 0.785 | 0.613 / 0.767 / 0.734 |
| `grad_norm` | 0.462 / 0.754 / 0.454 | 0.450 / 0.354 / 0.391 |
| `ppo_kl` | 3.0e-4 / −6.4e-5 / 6.3e-5 | — |
| `mem/actor_allocated_gb` | **10.81**（恒定） | **42.80** |
| `mem/actor_max_reserved_gb` | 113.9 / 27.3 / 27.4 | 114.0 / 75.1 / 75.1 |
| `timing/step_s` | **187.9 / 175.2 / 177.2** | **105.5 / 98.0 / 101.6** |
| `timing/gen_s` | 85.2 / 81.7 / 82.8 | — |
| `timing/train_s` | 65.3 / 61.5 / 62.8 | — |
| `timing/ref_s` | 37.0 / 32.0 / 31.7 | — |
| `timing/weight_sync_s` | **34.4 / 31.8 / 32.1** | **2.61 / 1.20 / 1.18** |
| `filter_groups` | round 1 kept 10/24、9/24、9/24（各一轮凑够 8） | — |

**正确性完全对齐**：`rollout_corr/kl` 落在单节点区间内，`is_weight_mean` 0.99995~1.0，`ppo_kl` 1e-4 量级。开了 `LUMENRL_WEIGHT_SYNC_VERIFY=1`，96 个融合专家张量 × 32 replica × 3 次同步**逐位一致**（`verify failed` 0、`verify skipped` 0），日志里 `untouched` 0 次——vime runbook §15.3 那个 `routed_experts` 修复在 32 replica 下同样成立。

**显存验证了 FSDP2 的分片收益**：actor 常驻 42.80 → **10.81 GB**，正好 1/4。rollout 侧不变（TP=1 时每个 replica 仍揣着完整的 30.5B）。

### 8.2 ⚠️ 4 倍的卡换来 1.8 倍变慢，这是配置的必然

`step_s` 从约 100s 涨到约 180s。**不是配置错了，是"同 config"本身不适合 32 卡**：

`train_global_batch_size=128` 是 8 prompt × 16 生成，摊到 32 rank 每 rank 只有 **4 条序列**。单卡计算量小到可忽略，剩下全是通信。最刺眼的是 `weight_sync_s` 从 1.2s 涨到 32s（26 倍）——参数现在按 32 路分片，每次要把完整张量重建出来交给同卡的 vLLM，就得跨节点 all-gather，而这些流量走的是 TCP（§6）。

所以**这个 smoke 是链路正确性验证，不是吞吐结论**。要看 32 卡的性能，得把 batch 一起放大（longrun config 是 batch 2048 / gen_batch 384，那样每 rank 64 条序列）。

---

## 9. Megatron-Native MoE smoke（32 卡，EP=8）

### 9.1 并行度：EP=8 让 all-to-all 留在节点内

单节点那份 config 是 TP=1 / PP=1 / CP=1 / **EP=8** → DP=8。搬到 32 卡保持同样的形状，得到 **DP=32、expert-DP=4**。

理由是 §1.1 那条 rank 布局加上 Megatron 的默认 rank 序 `tp-cp-ep-dp`：tp=cp=1 时 `rank = ep_idx + 8*dp_idx`，所以 **EP=8 的组恰好是 ranks 0–7、8–15、16–23、24–31，每组就是一台机器**。MoE 最重的 all-to-all 走节点内 XGMI，跨节点只剩梯度归约（dense 32 路、expert 4 路）。

⚠️ **不要用 TP=4 × EP=8**。它能让每 rank 恢复到 16 条序列，但那时 `rank = tp_idx + 4*ep_idx`，每个 EP 组会横跨全部 4 台机器，all-to-all 全变跨节点流量——在 §6 没修好之前是自找麻烦。

```bash
cat > /home/$USER/4node/04_smoke_megatron.sh <<'EOF'
#!/usr/bin/env bash
# Megatron-Native MoE 4k smoke across the 4-node cluster. Runs on the head node.
#
# TP=1, PP=1, CP=1, EP=8 -> DP = 32, expert-DP = 4.
#
# Ray hands out ranks contiguously per node and Megatron's default order is
# tp-cp-ep-dp, so with tp=cp=1 the rank decomposes as rank = ep_idx + 8*dp_idx:
# EP=8 makes every expert-parallel group exactly one node, keeping the MoE
# all-to-all on XGMI. TP=4 x EP=8 would spread each EP group across all four
# nodes instead -- a bad trade while inter-node traffic is still TCP.
set -euo pipefail
source /home/$USER/4node/env.sh
mkdir -p "$LOG4N"

docker exec "$CONTAINER" bash -lc "
set -uo pipefail
source /home/$USER/4node/ray_env.sh
export PATH=/home/$USER/4node/bin:\$PATH
MODEL=moe BACKEND=megatron SCALE=smoke SLEEP=1 STEPS=3 GPU_UTIL=0.45 \
EXTRA_OVERRIDE='cluster.num_nodes=4 cluster.gpus_per_node=8 cluster.ray_address=$HEAD_IP:$RAY_PORT controller.ray.actor.num_workers=32 policy.training.megatron_cfg.expert_model_parallel_size=8' \
LOG=$DATA_ROOT/logs/moe-megatron-smoke-4node.log \
  bash $RL_ROOT/run_lumenrl_dapo_vime.sh
"
EOF
```

判据（handoff 文档里那条健康判据在这里同样适用）：

```
[MegatronNativeEngine] MoE+EP spec: num_experts=128 topk=8 moe_ffn=768 | tp=1 pp=1 cp=1 EP=8 etp=1
  -> local_experts/rank=16 | grouped_gemm=True router_dtype=None pre_softmax=False aux_loss_coeff=0.0
LumenActorWorker: initialized megatron_native engine.     ← 32 条
```

### 9.2 ⚠️ 必须先打一个 TE 兼容补丁，否则 MoE 第一步就死

**症状**（`STEPS=3` 的第 1 步、log-prob 阶段）：

```
TypeError: general_gemm() got an unexpected keyword argument 'workspace'
  moe/router.py:101 gating
  -> moe_utils.py:1321 router_gating_linear
  -> extensions/transformer_engine.py:2534 te_general_gemm
```

**根因**：vime 镜像把 **megatron-core 0.16.0rc0** 和 **TE 2.12.0.dev0** 配在一起，而这两者对 `general_gemm` 的签名已经不一致——TE 现在在函数内部自己 `get_cublas_workspace(...)`，参数表里没有 `workspace` 了，megatron 0.16 却还在传 `workspace=get_workspace()`。其余 14 个参数都还对得上，只差这一个。

**为什么 vime runbook 没记过**：`router_gating_linear` 是 MoE router 独有的路径，而且它**无条件**走 TE gemm（只有 `router_dtype` 是 float64 时才有 torch 回退）。那份 runbook 的 "megatron-core 0.16.0rc0 实测能跑通 megatron_native" 是用 **Qwen3-8B dense** 验证的，dense 模型根本不进这段代码。**所以这是 vime 镜像上 Megatron + MoE 的首次暴露，和节点数无关。**

**修法**（新增文件 `lumenrl/engine/training/megatron_te_gemm_compat.py`，并在 `MegatronNativeEngine.initialize()` 里调一次 `install()`）：

补丁打在 `megatron.core.extensions.transformer_engine.general_gemm` 上，**不是** `te_general_gemm`——因为 `moe_utils` 在模块顶部就把后者绑定了，而前者是 `te_general_gemm` 函数体在**调用时**从自己模块 globals 里查的，所以一处补丁同时覆盖前向和两个反向 gemm（`moe_utils.py` 的 1254、1287、1290 三处）。自禁用：TE 若仍接受 `workspace` 就什么都不做。

单卡验证过输出与 `inp @ weight.t()` **逐位相同**（maxdiff 0.000e+00），没有数值影响：

```bash
docker exec "$CONTAINER" bash -lc 'source /home/$USER/4node/ray_env.sh
  cd $RL_ROOT/Lumen-RL && HIP_VISIBLE_DEVICES=0 python3 /home/$USER/4node/probe_te_gemm.py'
# 期望：takes workspace? False / shim installed: True / maxdiff_vs_torch=0.000e+00 / TE GEMM COMPAT OK
```

> 这个修复该提回 Lumen-RL 仓库，性质和 §15.3 那个 `routed_experts` 修复一样：任何用这个镜像跑 Megatron + MoE 的机器都会中。

### 9.3 实测（3 步，`exit=0`，9 分 17 秒）

| 指标 | Megatron 32 卡 (1/2/3) | FSDP2 32 卡 (1/2/3) |
|---|---|---|
| `timing/step_s` | **106.2 / 100.2 / 102.8** | 187.9 / 175.2 / 177.2 |
| `timing/train_s` | **18.4 / 16.1 / 16.1** | 65.3 / 61.5 / 62.8 |
| `timing/ref_s` | **4.89 / 0.44 / 0.64** | 37.0 / 32.0 / 31.7 |
| `timing/weight_sync_s` | **19.1 / 16.2 / 19.1** | 34.4 / 31.8 / 32.1 |
| `timing/gen_s` | 82.6 / 83.7 / 86.0 | 85.2 / 81.7 / 82.8 |
| `rollout_corr/kl` | 0.00135 / 0.00186 / 0.00200 | 0.00178 / 0.00124 / 0.00167 |
| `ppo_kl` | 6.2e-7 / −8.4e-5 / 3.8e-4 | 3.0e-4 / −6.4e-5 / 6.3e-5 |
| `entropy` | 0.617 / 0.666 / 0.758 | 0.718 / 0.639 / 0.785 |
| `grad_norm` | 0.793 / 0.568 / 0.417 | 0.462 / 0.754 / 0.454 |
| `reward/accuracy` | 0.148 / 0.141 / 0.109 | 0.117 / 0.172 / 0.102 |
| `mem/actor_allocated_gb` | 40.13 | 10.81 |
| `mem/actor_max_reserved_gb` | 95.8 | 113.9 / 27.3 / 27.4 |

**32 卡上 Megatron 比 FSDP2 快 1.7 倍**，这个反转正是并行度选对的结果：FSDP2 把参数分片到全部 32 卡，每次前向反向都要跨节点 all-gather 参数（走 TCP）；Megatron 的 EP=8/DP=32 让每个 rank 常驻自己那 16 个完整专家，跨节点只有梯度归约。`ref_s` 快约 50 倍是 `log_probs_chunk_size: 1024` 的分块融合实现，vime runbook 在 8B 上记录过同样现象（6.03s vs 0.73s）。

`gen_s` 两边一样（约 83s），因为 rollout 侧完全相同——这反过来说明上面的差异确实只来自训练后端。

显存：actor 常驻 40.1GB 是 FSDP2 的 3.7 倍，和 vime runbook "Megatron 约 5 倍" 的记录一致；`max_reserved` 95.8GB 加上 `GPU_UTIL=0.45` 的 129.6GB 引擎预算约 226GB / 288GB，余量还有。

`mismatch/*` 那一族没有是预期的（只有 FSDP2 才报）。另有一条无害告警 `make_tp_sharded_tensor_for_checkpoint received extra kwargs: ['allow_shape...']`，同属 megatron ↔ TE 的 API 漂移，不影响运行。

---

## 10. rollout TP>1（新增能力）

原来的 colocated 设计是**每卡一个 TP=1 引擎**，`_setup_ray_vllm_rollout` 里 `tensor_parallel_size=1` 写死。现在支持一个引擎跨 N 张卡：

```bash
EXTRA_OVERRIDE='... policy.generation.vllm_cfg.tensor_parallel_size=8'
```

32 actor / TP=8 = **4 个 replica，每个正好一台机器**。改动细节和验证见 `lumenrl-rollout-tp-gt-1-handoff.md`；这里只记怎么用和实测。

**两个用途**，第二个是硬需求：

1. **省显存**：引擎的模型副本按 TP 分片。实测 DeepSeek-V4-Flash 在 TP=1 下单卡占 155 GiB，TP=8 下每 rank 只占 **20.03 GiB**（7.7 倍）。这决定了一个很大的 policy 能不能和训练态共存于同一张卡。
2. **有些模型没有 TP=1 的可用路径**。DeepSeek-V4 在 gfx950 上 TP=1 必然 `Memory access fault`，TP=8 正常——详见 `deepseek-v4-flash-enablement-handoff.md`。

### 10.1 实测：同一个 MoE smoke，TP=1 vs TP=8

| 指标 | TP=1（32 replica） | TP=8（4 replica） |
|---|---|---|
| `rollout_corr/kl` | 0.00178 / 0.00124 / 0.00167 | 0.00185 / 0.00199 / 0.00190 |
| `is_weight_mean` | 0.99995 / 1.000 / 0.99995 | 0.99980 / 0.99995 / 0.99986 |
| `reward/accuracy` | 0.117 / 0.172 / 0.102 | 0.102 / 0.172 / 0.164 |
| `entropy` | 0.718 / 0.639 / 0.785 | 0.647 / 0.683 / 0.723 |
| `timing/weight_sync_s` | 34.4 / 31.8 / 32.1 | **23.6 / 24.6 / 24.4** |
| `timing/train_s` | 65.3 / 61.5 / 62.8 | **45.6 / 45.4 / 45.8** |
| `timing/gen_s` | 85.2 / 81.7 / 82.8 | 106.0 / 107.5 / 115.2 |
| `timing/step_s` | 187.9 / 175.2 / 177.2 | 180.3 / 177.7 / 183.4 |
| 权重逐位校验 | 0 失败 0 跳过 | **0 失败 0 跳过** |

如实说明三点：

- **`gen_s` 慢了约 25%**：384 个请求从摊给 32 个引擎变成摊给 4 个，每 token 还多了 TP all-reduce。这个 smoke 的批量太小，TP=8 对 rollout 吞吐是负收益——它的价值在显存和内核兼容性，不在速度。
- **`train_s` 快了约 28%**，我没有证据链解释成因。猜测是每节点的引擎 actor 从 8 个降到 1 个、CPU 与 Ray 开销变小，但这只是猜测。
- **`rollout_corr/kl` 略高**（均值 0.0019 对 0.0016）。这是预期的：TP 改变了引擎内 matmul 的归约顺序，rollout logprob 会有微小差异，训练侧完全没变。两者都远低于 TIS 阈值 2.0，`is_weight_mean` 也都贴着 1.0。**长跑时应当盯这一项是否随步数爬升**，而不是纠结这个常量偏移。

### 10.2 ⚠️ TP>1 时 custom all-reduce 必须关

代码里已经在 `rollout_tp > 1` 时自动设 `disable_custom_all_reduce=True`，这里记原因：colocated 路径需要 `NCCL_CUMEM_ENABLE=0`（CUDA-IPC 权重同步要求），而 vLLM 的 custom all-reduce 通过 cuMem 分配共享缓冲，两者同开的话每个 TP worker 都会死在 `create_shared_buffer`：

```
HIP error: invalid argument     （表现为 'CustomAllreduce' object has no attribute '_ptr'）
```

AMD 自己的 DeepSeek-V4 测试也是关掉它走 pynccl。

---

## 11. 为什么加载要十几分钟

FSDP2 那次 32 个 actor 加载权重花了约 13 分钟（531 个张量、约 1.3s/个）。**它们是并行的，不是排队**——不同节点的不同 pid 在同一时刻停在同一个百分比。慢的原因是**同一份权重被读了 64 遍**：

- **rollout 侧**（TP=1）：每个 replica 都是独立引擎，必须在自己卡上放完整的 30.5B。32 卡就是 32 份。改成 TP=8 后降到 4 份（每份再分 8 片）。
- **训练侧**：`fsdp_backend.py` 在**每个 rank** 上直接 `AutoModelForCausalLM.from_pretrained` 全量加载，之后才交给 FSDP2 分片。没有 meta device 初始化、没有 rank0 加载再 broadcast、没有 `sync_module_states`。所以每个 rank 都把 531 个张量全读一遍、在 CPU 上拼出完整的 30.5B，最后只留 1/32。

逻辑读是 57GB × 64 ≈ 3.6TB。好在每台有 2751GB 内存，同节点后 7 个读者基本命中 page cache，实际过网大约 57GB × 4 ≈ 228GB —— 但第一波 32 个读者是同时 miss 的，所以开头那段最慢。

两条改善路径（都没做）：把模型 `cp` 到每台的 `/mnt/m2m_nobackup`（各读本地 NVMe，代价是 4 份 57GB 和一次分发）；或者改训练侧加载路径做 rank0 广播（改代码，收益是训练侧 32 份读变 1 份）。

> 参照：Megatron 那次只花了 9 分 17 秒**全程**，因为权重已经在 page cache 里了。所以同一批节点上连续跑第二个 smoke 会明显更快，别把首次的加载时间当成常态。

---

## 12. 排障速查表（只列多节点独有的）

| 现象 | 原因 | 处理 |
|---|---|---|
| `spur alloc -N 4` 一直 `PENDING (QOSGrpNodeLimit)` | QOS 的 account 级组配额（`amd-aifw-dev-qos` = 19 节点，全 account 共享） | 用 `spur run -q amd-burst-qos`（128 节点）。`spur alloc` 不支持 `-q` |
| `spur run` 打出 `dispatched to node <单个节点>`、没有新 JobID | 在已有分配的 shell 里跑，`SLURM_JOB_ID` 让它变成 job step，`-N/-q` 被忽略 | 换干净终端，先确认 `env \| grep SLURM_JOB_ID` 无输出 |
| `srun --overlap -w <node> <cmd>` 静默无输出 | stdin 不是真 tty，spur 0.7.0 丢弃输出 | 用 `nx.sh`（`script -qec` 包一层） |
| 只有 head 节点能执行命令 | `spur exec` 没有节点参数，固定代理到 head | 同上 |
| 4 台看到的 `DATA_ROOT` 是空的 | 放在了 node-local 的 `/mnt/m2m_nobackup` | 模型/数据放 NFS（§3） |
| 非 driver 节点的 actor 报 `ModuleNotFoundError: lumenrl` / 指标异常 | Ray 不传 driver 环境，raylet 环境没铺 | 每台 `ray start` 前 source `ray_env.sh`（§5.1） |
| 集群刚建好就被拆、driver 连不上 | `run_dapo.sh` 开头 `ray stop --force` | 用 PATH shim 吞掉 `stop`（§5.2） |
| NCCL 挂死 90 秒后超时，channel 建不起来 | RoCE GID 选了 link-local 的 index 0 | `NCCL_IB_GID_INDEX=1` |
| `Call to ibv_reg_mr failed with error Invalid argument` | 内置 `net_ib` 路径不适用 AINIC，需要 ANP 插件（容器 glibc 不够） | 暂时 `NCCL_IB_DISABLE=1` 走 TCP（§6） |
| 容器里只看到 `mlx5_0`，8 张 ionic 隐形 | 镜像缺 ionic 的 libibverbs provider | 挂宿主的 `ionic.driver` + `libionic.so.*`（§4） |
| `ibv_reg_mr` 报 `ENOMEM` | memlock 8MB | `--ulimit memlock=-1` |
| 调 NCCL 参数"没效果" | 在 driver 上 export，actor 收不到 | 改 `ray_env.sh` 后重启 raylet，或用探针的 `PROBE_ENV`（Ray `runtime_env`） |
| `TypeError: general_gemm() got an unexpected keyword argument 'workspace'` | megatron-core 0.16 ↔ TE 2.12 API 漂移，MoE router 路径 | 打 §9.2 的 compat 补丁 |
| TP>1 时 `'CustomAllreduce' object has no attribute '_ptr'` | custom AR 的 cuMem 共享缓冲与 `NCCL_CUMEM_ENABLE=0` 冲突 | `disable_custom_all_reduce=True`（代码已自动） |
| `replica N spans 2 nodes` | Ray 没把一组 actor 摆在同一节点 | 检查 rank 分布；TP 必须整除每节点 GPU 数 |

---

## 13. 尚未验证（诚实说明）

- **跨节点走的是 TCP，不是 RDMA**（§6）。所有 `weight_sync_s` / `train_s` 数字都带着这个前提，修好 ANP 之后需要重测。
- **只跑了 smoke 规模（3 步、resp=4096、batch 128）**。32 卡的 longrun（batch 2048 / gen_batch 384 / resp=20480）一步都没跑过，吞吐、显存峰值、以及"32 卡上 Megatron 是否仍快于 FSDP2"都没有数据。
- **没有落过 checkpoint**（`save_steps` 大于步数）。多节点下 checkpoint 的分片布局、`SCRATCH_ROOT` 指向 node-local 盘时的可恢复性，全未验证。
- **rollout TP=8 只在 Qwen3-30B-A3B 上验证过**，且只有 smoke 规模。TP=2/4 没测。
- **`aiter` JIT 并发编译没有被真正触发过**（`.so` 早已存在）。全新机器第一次上 4 节点仍有 32 个 actor 并发写同一 NFS 目录的风险，建议先单节点预热。
- **节点故障没有演练**。vime runbook §16.5 记录过一次 `NODE_FAIL`，4 节点的暴露面是 4 倍，而 LumenRL 没有 elastic 恢复——任一节点掉线整个作业结束。

---

## 14. 与其他 runbook 的关系

| 想做的事 | 去哪份 |
|---|---|
| 单节点 8 卡 BF16 + FSDP2/Megatron + MoE（本文的前置） | `dapo-lumenrl-vime-fsdp-megatron-runbook.md` |
| 4 节点 32 卡编排（本文） | 就是这份 |
| rollout TP>1 的代码改动与验证 | `lumenrl-rollout-tp-gt-1-handoff.md` |
| DeepSeek-V4-Flash 能跑到哪一步 | `deepseek-v4-flash-enablement-handoff.md` |
| 只是要连节点、看作业、进容器 | `SPUR_NODE_ACCESS_GUIDE.md` |
| Megatron MoE + EP 的单节点验证 | `dapo-lumenrl-megatron-moe-ep-handoff.md` |
