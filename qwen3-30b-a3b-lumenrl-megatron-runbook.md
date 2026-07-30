# Qwen3-30B-A3B 两节点 RL 从零部署 Runbook

> Megatron 训练 + vLLM rollout + RCCL/RoCE GPU Direct RDMA 权重同步  
> 硬件：2 × 8 MI308X（gfx942，192 GiB HBM/GPU）  
> LumenRL：`origin/dev/moe-grpo`  
> 从两台干净主机开始：用户输入路径 → 自动发现网络 → 启动容器/Ray → 生成配置 → smoke → longrun

## 1. 部署流程和目标架构

严格按以下顺序执行，不要先把当前机器的 IP、RoCE 设备名或目录写死到 YAML：

1. 在两台主机采集用户提供的代码、模型、数据、运行目录和镜像。
2. 在每台主机自动发现 Ray node IP、RoCE HCA、网卡、GID index 和 RoCE IP。
3. 互相验证 Ray 网络和 RoCE 网络可达。
4. 从 `origin/dev/moe-grpo` 拉取一致的 LumenRL 代码。
5. 如果 `Lumen-RL/examples/GRPO/Dockerfile` 存在，优先用它分别构建 rollout/trainer 镜像；
   只有 Dockerfile 不存在时才在运行容器中手工安装依赖。
6. 启动带 GPU 和 `/dev/infiniband` 的容器并核对软件版本。
7. 启动两节点 Ray 集群。
8. 用发现结果生成本次部署专用 YAML，不修改仓库中的基准 YAML。
9. 先跑 3-step RDMA smoke，通过后再启动 200-step longrun。
10. 持续检查 NaN、RDMA transport、验证指标和完整 optimizer checkpoint。

目标角色由运行时变量决定：

- 训练节点 `${TRAIN_NODE_IP}`：8 个 Megatron actor，TP=4、EP=8，ETP=1。
- rollout/Ray head 节点 `${ROLLOUT_NODE_IP}`：4 个 vLLM replica，每个 TP=2，共 8 个 vLLM worker。
- 训练权重：BF16 compute + Megatron distributed optimizer 的 FP32 master/Adam state。
- rollout 权重：BF16；KV cache 使用 `auto`，即当前稳定 BF16 baseline。
- 权重同步：独立 9-rank `torch.distributed` process group。
  - rank 0：Megatron sender。
  - rank 1–8：vLLM TP receivers。
  - ROCm 的 `backend="nccl"` 实际由 RCCL 执行。
  - 网络使用探测得到的 `${RDMA_HCA}` / `${RDMA_IFACE}` / `${RDMA_GID_INDEX}`。
  - 日志必须出现 `Using network IB` 和 `NET/IB/0/GDRDMA`。
- R3：vLLM 记录 top-k expert IDs，Megatron `RouterReplay` 执行 hard assignment replay。
- 算法：GRPO/DAPO 风格 32 prompts × 8 generations，全局 batch 256。
- 正式目标：200 steps。

本流程不使用 ATOM rollout、ZMQ CUDA-IPC 或跨节点 safetensors 作为主路径。镜像名仍包含
`atom-dev`，但 ATOM Python 包只是基础镜像附带组件，不参与当前训练数据流。

### 1.1 宿主机前置条件

“从零部署”从已安装 AMD GPU/RDMA 内核驱动和 Docker 的两台主机开始，不包含刷固件或安装内核驱动。
两台主机先执行：

```bash
for cmd in docker git ip python3 ping; do
  command -v "$cmd" >/dev/null || { echo "missing: $cmd"; exit 1; }
done
docker info >/dev/null
test -e /dev/kfd
test -d /dev/dri
test -d /dev/infiniband
test -d /sys/class/infiniband
rocminfo | grep -c 'Name:.*gfx942'
```

最后一条应发现 8 个 GPU agent；数量不是 8 时先修复主机驱动/设备权限，不要进入容器部署。

## 2. 收集用户输入

### 2.1 两台主机都填写路径

以下变量是宿主机路径。可以两台机器分别填写，但容器内挂载点保持一致。脚本不会猜测用户磁盘布局：

```bash
read -r -p "代码根目录 WORK_ROOT: " WORK_ROOT
read -r -p "模型根目录 MODEL_HOST_DIR（其下放 Qwen3-30B-A3B）: " MODEL_HOST_DIR
read -r -p "数据根目录 DATASET_HOST_DIR: " DATASET_HOST_DIR
read -r -p "日志/checkpoint 目录 RUNTIME_HOST_DIR: " RUNTIME_HOST_DIR
read -r -p "共享目录 SHARED_HOST_DIR（没有则填本地空目录）: " SHARED_HOST_DIR
read -r -p "rollout 镜像 [rocm/atom-dev:vllm-latest]: " ROLLOUT_IMAGE
read -r -p "trainer 镜像 [rocm/atom-dev:latest]: " TRAIN_IMAGE
read -r -p "W&B project [LumenRL]: " WANDB_PROJECT
read -r -p "W&B entity（留空使用当前账号默认）: " WANDB_ENTITY

export WORK_ROOT MODEL_HOST_DIR DATASET_HOST_DIR RUNTIME_HOST_DIR
export SHARED_HOST_DIR
export LUMENRL_HOST_DIR="${WORK_ROOT}/Lumen-RL"
export LUMEN_HOST_DIR="${WORK_ROOT}/Lumen"
export MEGATRON_HOST_DIR="${WORK_ROOT}/Megatron-LM"
export CONTAINER="${CONTAINER:-qwen3-30b-rl}"
export ROLLOUT_IMAGE="${ROLLOUT_IMAGE:-rocm/atom-dev:vllm-latest}"
export TRAIN_IMAGE="${TRAIN_IMAGE:-rocm/atom-dev:latest}"
export WANDB_PROJECT="${WANDB_PROJECT:-LumenRL}"
export WANDB_ENTITY

for p in "$WORK_ROOT" "$MODEL_HOST_DIR" "$DATASET_HOST_DIR" \
         "$RUNTIME_HOST_DIR" "$SHARED_HOST_DIR"; do
  test -n "$p" || { echo "路径不能为空"; exit 1; }
  mkdir -p "$p"
done
mkdir -p "$RUNTIME_HOST_DIR/logs" "$RUNTIME_HOST_DIR/ckpts"
```

建议把每台主机的值保存到仅当前用户可读的环境文件，后续命令先 `source`：

```bash
ENV_FILE="${HOME}/qwen3-rdma-node.env"
umask 077
cat > "$ENV_FILE" <<EOF
export WORK_ROOT='$WORK_ROOT'
export MODEL_HOST_DIR='$MODEL_HOST_DIR'
export DATASET_HOST_DIR='$DATASET_HOST_DIR'
export RUNTIME_HOST_DIR='$RUNTIME_HOST_DIR'
export SHARED_HOST_DIR='$SHARED_HOST_DIR'
export LUMENRL_HOST_DIR='$LUMENRL_HOST_DIR'
export LUMEN_HOST_DIR='$LUMEN_HOST_DIR'
export MEGATRON_HOST_DIR='$MEGATRON_HOST_DIR'
export CONTAINER='$CONTAINER'
export ROLLOUT_IMAGE='$ROLLOUT_IMAGE'
export TRAIN_IMAGE='$TRAIN_IMAGE'
export WANDB_PROJECT='$WANDB_PROJECT'
export WANDB_ENTITY='$WANDB_ENTITY'
EOF
echo "saved $ENV_FILE"
```

### 2.2 容器内固定挂载点

宿主机路径由用户提供；容器内统一使用：

```text
/workspace/Lumen-RL       LumenRL 源码
/workspace/Lumen          Lumen 依赖源码
/workspace/Megatron-LM    ROCm Megatron-LM 源码（含 megatron.training 和 RouterReplay）
/workspace/aiter          aiter 源码（GPU kernel 编译）
/workspace/flash-attention  ROCm flash-attention 源码
/root/models              模型
/root/data_cached         已过滤数据
/runtime                  日志与 checkpoint
/shared                   可选 shared_folder fallback
/dev/infiniband           RoCE verbs 设备
```

## 3. 自动发现节点和 RoCE 网络

两台主机分别执行本节。不要从示例、旧日志或另一台集群复制 IP。

### 3.1 自动发现 Ray node IP

Ray IP 应是另一台节点可直接访问的管理/业务网 IP。先通过默认路由自动选择，并打印全部候选供核对：

```bash
AUTO_RAY_IP=$(
  ip -4 route get "${RAY_PROBE_TARGET:-1.1.1.1}" |
    awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
)
if [ -z "$AUTO_RAY_IP" ]; then
  AUTO_RAY_IP=$(ip -o -4 addr show scope global | awk 'NR==1 {split($4,a,"/"); print a[1]}')
fi

echo "自动选择: $AUTO_RAY_IP"
echo "全部候选:"
ip -o -4 addr show scope global |
  awk '{split($4,a,"/"); printf "  iface=%-16s ip=%s\n",$2,a[1]}'

read -r -p "确认 Ray node IP [${AUTO_RAY_IP}]: " NODE_RAY_IP
export NODE_RAY_IP="${NODE_RAY_IP:-$AUTO_RAY_IP}"
test -n "$NODE_RAY_IP"
```

在 rollout 节点记录：

```bash
export ROLLOUT_NODE_IP="$NODE_RAY_IP"
```

在 trainer 节点记录：

```bash
export TRAIN_NODE_IP="$NODE_RAY_IP"
```

把两个自动探测结果交换后，在两台节点输入：

```bash
read -r -p "rollout 节点探测到的 Ray IP: " ROLLOUT_NODE_IP
read -r -p "trainer 节点探测到的 Ray IP: " TRAIN_NODE_IP
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export ROLLOUT_NODE_IP='$ROLLOUT_NODE_IP'
export TRAIN_NODE_IP='$TRAIN_NODE_IP'
EOF
```

执行：

```bash
source "$HOME/qwen3-rdma-node.env"
test "$ROLLOUT_NODE_IP" != "$TRAIN_NODE_IP"
ping -c 2 "$ROLLOUT_NODE_IP"
ping -c 2 "$TRAIN_NODE_IP"
```

若默认路由选出的地址不能跨节点访问，应从“全部候选”中选择实际互通的地址，不能继续使用自动值。

### 3.2 自动发现 RoCE HCA、网卡、GID 和 IP

以下脚本从 sysfs 枚举 active RDMA port，优先选择带 IPv4 地址的 RoCE v2 GID。两台主机分别执行：

```bash
eval "$(
python3 - <<'PY'
import json
import pathlib
import subprocess
import sys

root = pathlib.Path("/sys/class/infiniband")
addrs = json.loads(subprocess.check_output(["ip", "-j", "-4", "addr", "show"]))
ipv4 = {
    item["ifname"]: [
        a["local"] for a in item.get("addr_info", [])
        if a.get("family") == "inet" and a.get("scope") == "global"
    ]
    for item in addrs
}

candidates = []
for hca_dir in sorted(root.glob("*")):
    for port_dir in sorted((hca_dir / "ports").glob("*")):
        if (port_dir / "state").read_text().strip().split(":", 1)[0] != "4":
            continue
        gids = port_dir / "gids"
        for gid_file in sorted(gids.glob("*"), key=lambda p: int(p.name)):
            idx = gid_file.name
            ndev_file = port_dir / "gid_attrs" / "ndevs" / idx
            type_file = port_dir / "gid_attrs" / "types" / idx
            if not ndev_file.exists() or not type_file.exists():
                continue
            iface = ndev_file.read_text().strip()
            gid_type = type_file.read_text().strip()
            ips = ipv4.get(iface, [])
            if not iface or not ips or "v2" not in gid_type.lower():
                continue
            gid = gid_file.read_text().strip()
            candidates.append((hca_dir.name, port_dir.name, int(idx), iface, ips[0], gid_type, gid))

if not candidates:
    raise SystemExit("没有找到 active RoCE v2 + IPv4 candidate")

for item in candidates:
    print(
        "candidate:",
        f"hca={item[0]} port={item[1]} gid_index={item[2]}",
        f"iface={item[3]} ip={item[4]} type={item[5]} gid={item[6]}",
        file=sys.stderr,
    )

hca, port, gid_index, iface, ip, gid_type, gid = candidates[0]
print(f"export RDMA_HCA={hca!r}")
print(f"export RDMA_PORT={port!r}")
print(f"export RDMA_GID_INDEX={gid_index!r}")
print(f"export RDMA_IFACE={iface!r}")
print(f"export RDMA_IP={ip!r}")
print(f"export RDMA_GID={gid!r}")
print(f"export RDMA_GID_TYPE={gid_type!r}")
PY
)"

printf 'HCA=%s port=%s iface=%s IP=%s GID_INDEX=%s type=%s GID=%s\n' \
  "$RDMA_HCA" "$RDMA_PORT" "$RDMA_IFACE" "$RDMA_IP" \
  "$RDMA_GID_INDEX" "$RDMA_GID_TYPE" "$RDMA_GID"
```

如果脚本打印多个物理 RoCE 网络中的第一个，但该网络不是两节点互联网络，先设置
`RDMA_IFACE` 为正确候选，再根据该网卡对应的 sysfs GID 重新选择；不要仅凭设备名猜测。

在 rollout 节点记录：

```bash
export ROLLOUT_RDMA_IP="$RDMA_IP"
```

在 trainer 节点记录：

```bash
export TRAIN_RDMA_IP="$RDMA_IP"
```

两台节点当前配置要求 HCA 名、网卡名和 GID index 一致。交换自动探测结果后，在两台节点输入：

```bash
read -r -p "rollout 节点探测到的 RoCE IP: " ROLLOUT_RDMA_IP
read -r -p "trainer 节点探测到的 RoCE IP: " TRAIN_RDMA_IP
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export RDMA_HCA='$RDMA_HCA'
export RDMA_PORT='$RDMA_PORT'
export RDMA_IFACE='$RDMA_IFACE'
export RDMA_GID_INDEX='$RDMA_GID_INDEX'
export ROLLOUT_RDMA_IP='$ROLLOUT_RDMA_IP'
export TRAIN_RDMA_IP='$TRAIN_RDMA_IP'
EOF
```

验证 RoCE 专网：

```bash
source "$HOME/qwen3-rdma-node.env"
ip -4 addr show dev "$RDMA_IFACE"

# rollout 节点
ping -I "$RDMA_IFACE" -c 3 "$TRAIN_RDMA_IP"

# trainer 节点
ping -I "$RDMA_IFACE" -c 3 "$ROLLOUT_RDMA_IP"
```

最终角色表应由本次探测结果填写：

| 角色 | Ray node IP | RoCE IP | GPU | 容器镜像 |
|---|---|---|---|---|
| rollout / Ray head | `${ROLLOUT_NODE_IP}` | `${ROLLOUT_RDMA_IP}` | 8 | `${ROLLOUT_IMAGE}` |
| Megatron trainer | `${TRAIN_NODE_IP}` | `${TRAIN_RDMA_IP}` | 8 | `${TRAIN_IMAGE}` |

## 4. 拉取源码

所有组件从 source 安装。各仓库和分支：

| 组件 | 仓库 | 分支 |
|---|---|---|
| LumenRL | `https://github.com/ZhangDanyang-AMD/Lumen-RL.git` | `dev/moe-grpo` |
| Lumen | `https://github.com/ZhangDanyang-AMD/Lumen.git` | `dev/qwen3-30b-a3b` |
| Megatron-LM | `https://github.com/ROCm/Megatron-LM.git` | `rocm_dev` |
| aiter | `https://github.com/ZhangDanyang-AMD/aiter.git` | `lumen/qwen3-30b-a3b` |
| flash-attention | `https://github.com/ROCm/flash-attention.git` | `main`（或与镜像 ROCm 版本匹配的 tag） |

两台节点都执行：

```bash
source “$HOME/qwen3-rdma-node.env”

# LumenRL
if [ ! -d “$LUMENRL_HOST_DIR/.git” ]; then
  git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git “$LUMENRL_HOST_DIR”
fi
cd “$LUMENRL_HOST_DIR”
git fetch origin dev/moe-grpo
git switch dev/moe-grpo 2>/dev/null \
  || git switch -c dev/moe-grpo --track origin/dev/moe-grpo
git merge --ff-only origin/dev/moe-grpo

# Lumen
if [ ! -d “$LUMEN_HOST_DIR/.git” ]; then
  git clone https://github.com/ZhangDanyang-AMD/Lumen.git “$LUMEN_HOST_DIR”
fi
cd “$LUMEN_HOST_DIR”
git fetch origin dev/qwen3-30b-a3b
git switch dev/qwen3-30b-a3b 2>/dev/null \
  || git switch -c dev/qwen3-30b-a3b --track origin/dev/qwen3-30b-a3b
git merge --ff-only origin/dev/qwen3-30b-a3b

# Megatron-LM（ROCm fork，含 megatron.training 和 RouterReplay）
if [ ! -d “$MEGATRON_HOST_DIR/.git” ]; then
  git clone https://github.com/ROCm/Megatron-LM.git “$MEGATRON_HOST_DIR”
fi
cd “$MEGATRON_HOST_DIR”
git fetch origin rocm_dev
git switch rocm_dev 2>/dev/null \
  || git switch -c rocm_dev --track origin/rocm_dev
git merge --ff-only origin/rocm_dev

# aiter
AITER_HOST_DIR=”${WORK_ROOT}/aiter”
if [ ! -d “$AITER_HOST_DIR/.git” ]; then
  git clone https://github.com/ZhangDanyang-AMD/aiter.git “$AITER_HOST_DIR”
fi
cd “$AITER_HOST_DIR”
git fetch origin lumen/qwen3-30b-a3b
git switch lumen/qwen3-30b-a3b 2>/dev/null \
  || git switch -c lumen/qwen3-30b-a3b --track origin/lumen/qwen3-30b-a3b
git merge --ff-only origin/lumen/qwen3-30b-a3b

# flash-attention
FA_HOST_DIR=”${WORK_ROOT}/flash-attention”
if [ ! -d “$FA_HOST_DIR/.git” ]; then
  git clone https://github.com/ROCm/flash-attention.git “$FA_HOST_DIR”
fi

# 保存路径（MEGATRON_HOST_DIR 已在 Section 2.1 写入 env file）
cat >> “$HOME/qwen3-rdma-node.env” <<EOF
export AITER_HOST_DIR='$AITER_HOST_DIR'
export FA_HOST_DIR='$FA_HOST_DIR'
EOF
```

验证两台节点代码一致：

```bash
source “$HOME/qwen3-rdma-node.env”
echo “LumenRL:    $(cd “$LUMENRL_HOST_DIR” && git rev-parse HEAD)”
echo “Lumen:      $(cd “$LUMEN_HOST_DIR” && git rev-parse HEAD)”
echo “Megatron:   $(cd “$MEGATRON_HOST_DIR” && git rev-parse HEAD)”
echo “aiter:      $(cd “$AITER_HOST_DIR” && git rev-parse HEAD)”
echo “fa:         $(cd “$FA_HOST_DIR” && git rev-parse HEAD)”
```

两台节点的五行输出必须分别相同。

注意：

- 不要把本 runbook 写成”主干最新版本”；必须明确各分支名。
- 从零部署不要依赖未提交工作区；所需修复必须已经进入对应远端分支。
- RDMA 实现参考 MILES 架构思想，但没有复制或嵌入 MILES 源码。

## 5. 已验证容器软件版本

### 5.1 两节点公共版本

以下版本来自已验证镜像构建；部署后仍需在本次 `${CONTAINER}` 中复核：

| 包 | 版本 |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.10.0+rocm7.2.4.git3d3aa833` |
| HIP (`torch.version.hip`) | `7.2.53211` |
| Ray | `2.56.1` |
| Megatron-LM (ROCm fork) | `rocm_dev` branch（含 megatron.core + megatron.training） |
| flash-attn | `2.8.4` |
| amd-aiter | `0.1.0` |
| Transformers | `5.2.0` |
| Datasets | `5.0.0` |
| Accelerate | `1.14.0` |
| Safetensors | `0.8.0` |
| OmegaConf | `2.3.1` |
| Tokenizers | `0.22.2` |
| math-verify | `0.3.3` |
| antlr4-python3-runtime | `4.9.3` |
| W&B | `0.28.1` |

### 5.2 rollout 节点特有版本

| 包 | 版本 |
|---|---|
| vLLM | `0.22.1.dev0+g0b3ba88f1.d20260629.rocm724` |
| NumPy | `2.1.3` |
| Triton | `3.7.0+amd.rocm7.2.0.gitd0d77a509` |
| triton_kernels | `1.0.0+amd.rocm7.2.0.gitd0d77a509` |
| 基础镜像附带 ATOM | `0.1.4.dev208+g96ad40621`（当前 flow 不使用） |

### 5.3 trainer 节点特有版本

| 包 | 版本 |
|---|---|
| vLLM | 未安装，这是预期状态 |
| NumPy | `2.4.6` |
| Triton | `3.7.0+amd.rocm7.2.0.git89002410` |
| triton_kernels | `1.0.0+amd.rocm7.2.0.git89002410` |
| 基础镜像附带 ATOM | `0.1.6rc1.dev117+g3321d0ff0`（当前 flow 不使用） |

使用 GRPO Dockerfile 时，LumenRL、Lumen、aiter、flash-attention 和 Megatron-LM 已经烘焙到镜像；
不再依赖容器启动后的手工安装。fallback 流程仍通过 `pip install -e` 和 `.pth` 使用挂载源码。
不能用缺少 `megatron.training` 或 `RouterReplay` 的通用 PyPI `megatron-core` 替代已验证 ROCm
Megatron-LM。

### 5.4 优先用 GRPO Dockerfile 构建环境

如果以下文件存在：

```text
${LUMENRL_HOST_DIR}/examples/GRPO/Dockerfile
```

必须优先使用它构建环境。该 Dockerfile 使用同一份 LumenRL build context，但提供两个独立 target：

| target | 默认基础镜像 | 输出镜像 | 用途 |
|---|---|---|---|
| `rollout` | `rocm/atom-dev:vllm-latest` | `qwen3-30b-a3b:rollout` | Ray head + vLLM TP2 × 4 |
| `trainer` | `rocm/atom-dev:latest` | `qwen3-30b-a3b:trainer` | Megatron TP4/EP8 |

两个 target 不能合并成同一运行镜像。rollout 的 vLLM、NumPy 和 Triton build 与 trainer 不同；
强行统一会破坏已经验证的 vLLM/ROCm 组合。

Docker build context 必须是 **Lumen-RL 仓库根目录**，不能是
`examples/GRPO/`：

```bash
source "$HOME/qwen3-rdma-node.env"
DOCKERFILE="$LUMENRL_HOST_DIR/examples/GRPO/Dockerfile"
test -f "$DOCKERFILE" || {
  echo "GRPO Dockerfile 不存在，使用 §7 fallback 安装流程"
  export USE_GRPO_DOCKERFILE=0
}
```

#### rollout 节点构建

只在 rollout 节点执行：

```bash
source "$HOME/qwen3-rdma-node.env"
cd "$LUMENRL_HOST_DIR"

docker build --network=host --progress=plain \
  -f examples/GRPO/Dockerfile \
  --target rollout \
  -t qwen3-30b-a3b:rollout \
  .

export ROLLOUT_IMAGE=qwen3-30b-a3b:rollout
export USE_GRPO_DOCKERFILE=1
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export ROLLOUT_IMAGE='$ROLLOUT_IMAGE'
export USE_GRPO_DOCKERFILE=1
EOF
```

#### trainer 节点构建

只在 trainer 节点执行：

```bash
source "$HOME/qwen3-rdma-node.env"
cd "$LUMENRL_HOST_DIR"

docker build --network=host --progress=plain \
  -f examples/GRPO/Dockerfile \
  --target trainer \
  -t qwen3-30b-a3b:trainer \
  .

export TRAIN_IMAGE=qwen3-30b-a3b:trainer
export USE_GRPO_DOCKERFILE=1
cat >> "$HOME/qwen3-rdma-node.env" <<EOF
export TRAIN_IMAGE='$TRAIN_IMAGE'
export USE_GRPO_DOCKERFILE=1
EOF
```

首次构建会编译 gfx942 的 aiter、flash-attention 和 Lumen HIP extension，通常需要较长时间。
后续相同源码和 build args 会复用 Docker layer cache。需要强制重建时才添加 `--no-cache`。

Dockerfile 支持覆盖基础镜像和源码 ref。例如固定 rollout 基础镜像 digest：

```bash
docker build --network=host --progress=plain \
  -f examples/GRPO/Dockerfile \
  --target rollout \
  --build-arg ROLLOUT_BASE_IMAGE='rocm/atom-dev@sha256:<digest>' \
  --build-arg LUMEN_REF='<commit>' \
  --build-arg MEGATRON_REF='<commit>' \
  --build-arg AITER_REF='<commit>' \
  --build-arg FLASH_ATTN_REF='<commit>' \
  -t qwen3-30b-a3b:rollout \
  .
```

正式复现时应把两个基础镜像都换成 digest，并把四个源码 ref 都换成已验证 commit。
Dockerfile 在构建末尾会检查 PyTorch/HIP、Ray、Transformers、flash-attn、AITER、NumPy、
Triton、vLLM 和 Megatron import；版本不匹配会直接令 build 失败。

构建完成后核对目标和镜像 ID：

```bash
docker image inspect qwen3-30b-a3b:rollout \
  --format 'rollout {{.Id}} {{index .Config.Labels "org.opencontainers.image.title"}}' \
  2>/dev/null || true
docker image inspect qwen3-30b-a3b:trainer \
  --format 'trainer {{.Id}} {{index .Config.Labels "org.opencontainers.image.title"}}' \
  2>/dev/null || true
```

如果只在一台 build 主机构建，可把两个镜像推送到内部 registry，或使用
`docker save` / `docker load` 传到对应节点。两节点最终必须使用各自角色的 target，不能把
`trainer` 镜像放到 rollout 节点。

## 6. 启动 Docker

当前容器实测配置：

```text
network=host
ipc=private
shm=64 GiB
privileged=false
group-add=video
ulimit memlock=-1
ulimit stack=64 MiB
devices=/dev/kfd,/dev/dri,/dev/infiniband
```

### 6.1 rollout 节点

```bash
source "$HOME/qwen3-rdma-node.env"
SOURCE_MOUNTS=()
if [ "${USE_GRPO_DOCKERFILE:-0}" != "1" ]; then
  SOURCE_MOUNTS=(
    -v "$LUMENRL_HOST_DIR":/workspace/Lumen-RL
    -v "$LUMEN_HOST_DIR":/workspace/Lumen
    -v "$MEGATRON_HOST_DIR":/workspace/Megatron-LM
    -v "$AITER_HOST_DIR":/workspace/aiter
    -v "$FA_HOST_DIR":/workspace/flash-attention
  )
fi
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add=video \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  "${SOURCE_MOUNTS[@]}" \
  -v "$DATASET_HOST_DIR":/root/data_cached \
  -v "$MODEL_HOST_DIR":/root/models \
  -v "$SHARED_HOST_DIR":/shared \
  -v "$RUNTIME_HOST_DIR":/runtime \
  "$ROLLOUT_IMAGE" -lc 'sleep infinity'
```

### 6.2 trainer 节点

```bash
source "$HOME/qwen3-rdma-node.env"
SOURCE_MOUNTS=()
if [ "${USE_GRPO_DOCKERFILE:-0}" != "1" ]; then
  SOURCE_MOUNTS=(
    -v "$LUMENRL_HOST_DIR":/workspace/Lumen-RL
    -v "$LUMEN_HOST_DIR":/workspace/Lumen
    -v "$MEGATRON_HOST_DIR":/workspace/Megatron-LM
    -v "$AITER_HOST_DIR":/workspace/aiter
    -v "$FA_HOST_DIR":/workspace/flash-attention
  )
fi
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
  --network=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add=video \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  "${SOURCE_MOUNTS[@]}" \
  -v "$DATASET_HOST_DIR":/root/data_cached \
  -v "$MODEL_HOST_DIR":/root/models \
  -v "$SHARED_HOST_DIR":/shared \
  -v "$RUNTIME_HOST_DIR":/runtime \
  "$TRAIN_IMAGE" -lc 'sleep infinity'
```

关键约束：

- 必须映射整个 `/dev/infiniband`，只映射 `/dev/kfd` 与 `/dev/dri` 不足以使用 verbs/GDRDMA。
- rollout 节点必须使用包含当前 ROCm vLLM build 的镜像。
- trainer 节点不需要安装 vLLM。
- 两节点容器必须使用 `--network=host`，否则 Ray IP、RoCE IP 与 rendezvous 地址需要重新配置。
- `USE_GRPO_DOCKERFILE=1` 时不要再挂载宿主机源码覆盖 `/workspace/*`，否则运行的不是构建时已验证代码。

## 7. fallback：在运行容器中从源码安装依赖

本节只在 `Lumen-RL/examples/GRPO/Dockerfile` 不存在，或明确设置
`USE_GRPO_DOCKERFILE=0` 时执行。

如果 §5.4 已成功构建并使用角色镜像，**跳过 §7.1–§7.6**，直接执行 §7.7 和 §7.8
做运行时导入及 kernel 验证。不要在已构建镜像启动后再次 `pip install`，否则会破坏
Dockerfile 构建末尾验证过的版本矩阵。

fallback 模式下，所有核心组件从挂载的源码目录构建安装。不使用预编译 wheel。
不要重装 torch、Triton 或容器镜像自带的 vLLM（rollout 节点）。

### 7.1 aiter（两节点都需要，含 HIP C++ 编译）

aiter 包含 HIP/CK GPU kernel，需要编译。首次编译约 15–30 分钟，宿主机挂载的源码目录会
缓存编译产物，后续重建容器时无需重新编译。

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
cd /workspace/aiter
/opt/venv/bin/pip install -e . 2>&1 | tail -5
/opt/venv/bin/python -c "import aiter; print(\"aiter ok:\", aiter.__file__)"
'
```

如果编译报错缺少 HIP 头文件或 `hipcc` 不在 PATH，确认容器镜像包含完整 ROCm toolkit：

```bash
docker exec "$CONTAINER" bash -lc 'hipcc --version && ls /opt/rocm/include/hip/hip_runtime.h'
```

### 7.2 flash-attention（两节点都需要，含 HIP C++ 编译）

ROCm flash-attention 同样需要编译 HIP kernel。编译约 10–20 分钟。

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
cd /workspace/flash-attention
GPU_ARCHS="gfx942" /opt/venv/bin/pip install -e . 2>&1 | tail -5
/opt/venv/bin/python -c "from flash_attn import flash_attn_varlen_func; print(\"flash_attn ok\")"
'
```

`GPU_ARCHS="gfx942"` 限制只编译 MI308X 架构，大幅缩短编译时间。如果目标 GPU 是其他
架构（如 MI350X 的 gfx950），需要相应修改。

### 7.3 Lumen（两节点，含 HIP C++ extension 编译）

Lumen 的 `lumen/csrc/` 包含 HIP C++ extension（`fused_quant_transpose.cu`、
`fp8_quant_dispatch.cu`），`pip install -e` 会自动调用 `hipcc` 编译。首次编译约 1–2 分钟。

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
/opt/venv/bin/pip install --no-deps -e /workspace/Lumen
/opt/venv/bin/python -c "import lumen; print(\"lumen ok:\", lumen.__file__)"
'
```

### 7.4 LumenRL（两节点，纯 Python editable install）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
/opt/venv/bin/pip install --no-deps -e /workspace/Lumen-RL
/opt/venv/bin/python -c "import lumenrl; print(\"lumenrl ok:\", lumenrl.__file__)"
'
```

### 7.5 Python 依赖（两节点）

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
P=/opt/venv/bin/pip
$P install "ray[default]==2.56.1" \
  "transformers==5.2.0" "datasets==5.0.0" "accelerate==1.14.0" \
  "safetensors==0.8.0" "omegaconf==2.3.1" \
  "math_verify==0.3.3" "wandb==0.28.1" \
  "numpy>=1.26" "pybind11>=3.0"
'
```

### 7.6 Megatron-LM（两节点，通过 .pth 引入）

**不能用 PyPI 的 `megatron-core`**，也不能用 `pip install -e`。原因：

- PyPI 包只含 `megatron.core.*`，不含 `megatron.training`。
- ROCm fork 的 `pyproject.toml` 同样只把 `megatron.core` 列为可安装包，
  `pip install -e .` 不会安装 `megatron.training`。
- Lumen 在 `lumen/models/megatron.py` 顶层 `from megatron.training import get_args`。
- ROCm fork 包含 `RouterReplay`（`megatron/core/transformer/moe/router_replay.py`）和
  `moe_enable_routing_replay` config field，是 R3 MoE replay 的核心依赖。

正确做法：在 venv 的 site-packages 中放一个 `.pth` 文件指向 `/workspace/Megatron-LM`。
这样所有使用该 venv 的进程（包括 Ray worker）都能导入 `megatron.core` 和 `megatron.training`，
不需要每个进程单独设置 `PYTHONPATH`。

先卸载容器镜像中可能自带的旧版 `megatron-core`（PyPI），再写入 `.pth`：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
/opt/venv/bin/pip uninstall -y megatron-core 2>/dev/null || true
SITE=$(/opt/venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "/workspace/Megatron-LM" > "$SITE/megatron-lm-source.pth"
echo "wrote $SITE/megatron-lm-source.pth"
'
```

验证 `.pth` 生效——无需设置 `PYTHONPATH` 即可导入：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
set -e
/opt/venv/bin/python -c "
import megatron.core; print(\"megatron.core ok:\", megatron.core.__file__)
from megatron.training import get_args; print(\"megatron.training ok\")
from megatron.core.transformer.moe.router_replay import RouterReplay; print(\"RouterReplay ok\")
"
'
```

### 7.7 验证完整导入链

两节点都执行：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i -e HIP_VISIBLE_DEVICES=0 \
  "$CONTAINER" /opt/venv/bin/python - <<'PY'
import sys
checks = []

import aiter; checks.append(("aiter", aiter.__file__))
import flash_attn; checks.append(("flash_attn", flash_attn.__file__))
import lumen; checks.append(("lumen", lumen.__file__))
import lumenrl; checks.append(("lumenrl", lumenrl.__file__))
import megatron.core; checks.append(("megatron.core", megatron.core.__file__))

try:
    from megatron.training import get_args
    checks.append(("megatron.training", "ok"))
except ImportError:
    checks.append(("megatron.training", "MISSING — need ROCm Megatron-LM, not PyPI megatron-core"))

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay
    checks.append(("RouterReplay", "ok"))
except ImportError:
    checks.append(("RouterReplay", "MISSING — need ROCm fork with router_replay"))

try:
    import vllm; checks.append(("vllm", vllm.__file__))
except ImportError:
    checks.append(("vllm", "NOT INSTALLED (ok for trainer node)"))

for name, path in checks:
    print(f"  {name:12s} {path}")

# 确认 editable install 指向挂载目录
for name, expected_prefix in [
    ("lumen", "/workspace/Lumen/"),
    ("lumenrl", "/workspace/Lumen-RL/"),
    ("megatron", "/workspace/Megatron-LM/"),
    ("aiter", "/workspace/aiter/"),
    ("flash_attn", "/workspace/flash-attention/"),
]:
    mod = sys.modules.get(name)
    if mod and hasattr(mod, "__file__") and mod.__file__:
        assert mod.__file__.startswith(expected_prefix), \
            f"{name} imported from {mod.__file__}, expected {expected_prefix}"
print("all source installs verified")
PY
```

### 7.8 flash-attn ROCm ABI 验证

从源码编译的 flash-attn 通常不会有 ABI 不匹配问题。但如果容器镜像自带的
flash-attn 被保留而非从源码重建，Python wrapper 可能比 native extension 多传
末尾 `num_splits`。`run_grpo.sh` 启动时会幂等删除这个不支持的参数。

验证 kernel：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" /opt/venv/bin/python - <<'PY'
import torch
from flash_attn import flash_attn_varlen_func

q = torch.randn(12, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, k, v, cu, cu, 7, 7, causal=True)
out.float().sum().backward()
torch.cuda.synchronize()
print("flash_varlen_forward_backward_ok", tuple(out.shape))
PY
```

期望输出：

```text
flash_varlen_forward_backward_ok (12, 2, 128)
```

## 8. 准备模型和数据

容器内模型路径：

```text
/root/models/Qwen3-30B-A3B
```

已过滤训练/验证数据：

```text
/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
```

如果用户提供的目录中还没有模型，在任一节点下载；两台节点的 `$MODEL_HOST_DIR` 都必须最终包含
同一模型，或使用真正共享的模型目录：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i "$CONTAINER" /opt/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-30B-A3B",
    local_dir="/root/models/Qwen3-30B-A3B",
)
PY
```

如果用户没有提供过滤后的 parquet，在任一节点生成，再把结果同步到另一节点相同的容器内路径：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i "$CONTAINER" /opt/venv/bin/python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

out = Path("/root/data_cached/qwen3-30b-a3b-maxprompt1024")
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("/root/models/Qwen3-30B-A3B")

def prompt_len(row):
    prompt = row["prompt"]
    return len(tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))

def normalize_prompt(row):
    prompt = (
        row.get("prompt")
        or row.get("question")
        or row.get("problem")
        or row.get("input")
        or ""
    )
    if not isinstance(prompt, list):
        prompt = [{"role": "user", "content": str(prompt)}]
    return {"prompt": prompt}

jobs = (
    ("BytedTsinghua/DAPO-Math-17k", "train",
     out / "dapo-math-17k.filtered.parquet"),
    ("HuggingFaceH4/aime-2024", "train",
     out / "aime-2024.filtered.parquet"),
)
for repo, split, dst in jobs:
    ds = load_dataset(repo, split=split)
    ds = ds.map(normalize_prompt, num_proc=16)
    ds = ds.filter(lambda row: prompt_len(row) <= 1024, num_proc=16)
    ds.to_parquet(dst)
    print(dst, len(ds))
PY
```

启动前验证：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
test -f /root/models/Qwen3-30B-A3B/config.json
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
test -f /root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
'
```

## 9. RDMA 和网络预检

两个节点容器内执行：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e RDMA_HCA="$RDMA_HCA" \
  -e RDMA_IFACE="$RDMA_IFACE" \
  "$CONTAINER" bash -lc '
ls -l /dev/infiniband
test -e /dev/infiniband/uverbs0
test -d "/sys/class/infiniband/$RDMA_HCA"
ip -4 addr show dev "$RDMA_IFACE"
'
```

关键环境变量：

```bash
source "$HOME/qwen3-rdma-node.env"
export NCCL_SOCKET_IFNAME="$RDMA_IFACE"
export NCCL_IB_HCA="$RDMA_HCA"
export NCCL_IB_GID_INDEX="$RDMA_GID_INDEX"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_DMABUF_ENABLE=0
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO
```

`gdr_mode: auto` 依赖 RCCL 自动探测。正式验证不能只看配置，必须从日志确认：

```text
NCCL INFO Using network IB
NCCL INFO ... via NET/IB/0/GDRDMA ...
```

若只有 socket/TCP 日志，则不算 RDMA 验证通过。

## 10. 启动 Ray 集群

先在 rollout 节点启动 head：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e ROLLOUT_NODE_IP="$ROLLOUT_NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
ulimit -n 524288
/opt/venv/bin/ray stop --force || true
/opt/venv/bin/ray start --head \
  --node-ip-address="$ROLLOUT_NODE_IP" \
  --port=6379 \
  --num-gpus=8 \
  --num-cpus=64 \
  --dashboard-host=0.0.0.0
'
```

再在 trainer 节点加入：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e ROLLOUT_NODE_IP="$ROLLOUT_NODE_IP" \
  -e TRAIN_NODE_IP="$TRAIN_NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
ulimit -n 524288
/opt/venv/bin/ray stop --force || true
/opt/venv/bin/ray start \
  --address="$ROLLOUT_NODE_IP:6379" \
  --node-ip-address="$TRAIN_NODE_IP" \
  --num-gpus=8 \
  --num-cpus=64
'
```

验证：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" /opt/venv/bin/ray status
```

必须看到：

```text
Active: 2 nodes
Total: 16 GPU
```

## 11. 生成本次部署 YAML

不要直接编辑仓库中的基准 YAML。只在 rollout/driver 节点执行下面命令，生成带本次探测值的
`/runtime/configs/` 配置：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i \
  -e ROLLOUT_NODE_IP="$ROLLOUT_NODE_IP" \
  -e TRAIN_NODE_IP="$TRAIN_NODE_IP" \
  -e RDMA_HCA="$RDMA_HCA" \
  -e RDMA_IFACE="$RDMA_IFACE" \
  -e RDMA_GID_INDEX="$RDMA_GID_INDEX" \
  -e WANDB_PROJECT="$WANDB_PROJECT" \
  -e WANDB_ENTITY="$WANDB_ENTITY" \
  "$CONTAINER" /opt/venv/bin/python - <<'PY'
import copy
import os
from pathlib import Path
from omegaconf import OmegaConf

src = Path("/workspace/Lumen-RL/examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml")
out_dir = Path("/runtime/configs")
out_dir.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(src)
cfg.cluster.num_nodes = 2
cfg.cluster.gpus_per_node = 8
cfg.cluster.ray_address = "auto"
cfg.controller.ray.actor.topology_tags = {"node_ip": os.environ["TRAIN_NODE_IP"]}
cfg.controller.ray.rollout.topology_tags = {"node_ip": os.environ["ROLLOUT_NODE_IP"]}
cfg.weight_sync.backend = "rdma"
cfg.weight_sync.shared_folder = "/shared/lumenrl_weight_sync/qwen3-30b-a3b"
cfg.weight_sync.rdma.backend = "rccl"
cfg.weight_sync.rdma.require_rdma = True
cfg.weight_sync.rdma.hca = os.environ["RDMA_HCA"]
cfg.weight_sync.rdma.interface = os.environ["RDMA_IFACE"]
cfg.weight_sync.rdma.gid_index = int(os.environ["RDMA_GID_INDEX"])
cfg.weight_sync.rdma.gdr_mode = "auto"
cfg.policy.model_name = "/root/models/Qwen3-30B-A3B"
cfg.reward.dataset = (
    "/root/data_cached/qwen3-30b-a3b-maxprompt1024/"
    "dapo-math-17k.filtered.parquet"
)
cfg.val_dataset = (
    "/root/data_cached/qwen3-30b-a3b-maxprompt1024/"
    "aime-2024.filtered.parquet"
)
cfg.checkpointing.checkpoint_dir = "/runtime/ckpts/qwen3-30b-a3b-rdma-longrun"
cfg.checkpointing.resume = False
cfg.eval.enabled = True
cfg.logger.wandb.project = os.environ["WANDB_PROJECT"]
cfg.logger.wandb.entity = os.environ.get("WANDB_ENTITY") or None
cfg.num_training_steps = 200

longrun = out_dir / "qwen3-30b-a3b-rdma-longrun.yaml"
OmegaConf.save(cfg, longrun)

smoke_cfg = copy.deepcopy(cfg)
smoke_cfg.policy.max_total_sequence_length = 128
smoke_cfg.policy.max_response_length = 64
smoke_cfg.policy.train_global_batch_size = 8
smoke_cfg.policy.gen_batch_size = 8
smoke_cfg.policy.max_token_len_per_gpu = 512
smoke_cfg.policy.generation.vllm_cfg.max_model_len = 128
smoke_cfg.val_steps = 0
smoke_cfg.eval.enabled = False
smoke_cfg.checkpointing.checkpoint_dir = "/runtime/ckpts/qwen3-30b-a3b-rdma-smoke"
smoke_cfg.checkpointing.save_steps = 9999
smoke_cfg.checkpointing.resume = False
smoke_cfg.logger.wandb_enabled = False
smoke_cfg.num_training_steps = 3
smoke = out_dir / "qwen3-30b-a3b-rdma-smoke.yaml"
OmegaConf.save(smoke_cfg, smoke)

print(longrun)
print(smoke)
PY
```

核对生成值，输出中不得出现旧机器 IP 或示例 RoCE 名称：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -i "$CONTAINER" /opt/venv/bin/python - <<'PY'
from omegaconf import OmegaConf

for path in (
    "/runtime/configs/qwen3-30b-a3b-rdma-smoke.yaml",
    "/runtime/configs/qwen3-30b-a3b-rdma-longrun.yaml",
):
    cfg = OmegaConf.load(path)
    print(path)
    print("  actor:", cfg.controller.ray.actor.topology_tags.node_ip)
    print("  rollout:", cfg.controller.ray.rollout.topology_tags.node_ip)
    print("  RDMA:", cfg.weight_sync.rdma.hca, cfg.weight_sync.rdma.interface,
          cfg.weight_sync.rdma.gid_index)
PY
```

`weight_sync.rdma.backend: rccl` 是配置语义；PyTorch API 仍使用 `backend="nccl"`，ROCm
运行时自动映射至 RCCL。

## 12. 先 smoke，再启动正式训练

可选 W&B key 由用户放到 rollout 节点的：

```text
${RUNTIME_HOST_DIR}/wandb.key
```

文件格式：

```text
WANDB_API_KEY=...
```

不要把 key 放进 `docker exec -e WANDB_API_KEY=...` 的命令参数，否则会出现在 `ps` 输出。

先前台运行 3-step smoke：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/runtime
export AITER_DIR=/workspace/aiter
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=smoke
export STEPS=3
export BACKEND=vllm
export CONFIG_OVERRIDE=/runtime/configs/qwen3-30b-a3b-rdma-smoke.yaml
export LUMENRL_KEEP_RAY_CLUSTER=1
export RUN_ID=qwen3-30b-a3b-rdma-smoke3
export LOG=/runtime/logs/qwen3-30b-a3b-rdma-smoke3.log
export CKPT_DIR=/runtime/ckpts/qwen3-30b-a3b-rdma-smoke3
export RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma

bash /workspace/Lumen-RL/examples/GRPO/run_grpo.sh
'
```

确认 3 步均通过 §13 的验证条件后，再分离启动 200-step longrun：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec -d \
  -e NCCL_SOCKET_IFNAME="$RDMA_IFACE" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_IB_GID_INDEX="$RDMA_GID_INDEX" \
  "$CONTAINER" bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/runtime
export AITER_DIR=/workspace/aiter
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=longrun
export STEPS=200
export BACKEND=vllm
export CONFIG_OVERRIDE=/runtime/configs/qwen3-30b-a3b-rdma-longrun.yaml
export LUMENRL_KEEP_RAY_CLUSTER=1
export RUN_ID="qwen3-30b-a3b-rdma-longrun-$(date +%Y%m%d-%H%M%S)"
export LOG="/runtime/logs/${RUN_ID}.log"
export CKPT_DIR="/runtime/ckpts/${RUN_ID}"
export WANDB_RUN_NAME="$RUN_ID"
export RESUME_OVERRIDE=false
export WEIGHT_SYNC_BACKEND=rdma

if [ -f /runtime/wandb.key ]; then
  export WANDB_API_KEY="$(cut -d= -f2- /runtime/wandb.key | tr -d "[:space:]")"
fi

echo "$RUN_ID" > /runtime/current_run_id.txt
echo "$LOG" > /runtime/current_run_log.txt
echo "$CKPT_DIR" > /runtime/current_ckpt_dir.txt
bash /workspace/Lumen-RL/examples/GRPO/run_grpo.sh
'
```

注意：

- `BACKEND=vllm` 与本次生成的 `CONFIG_OVERRIDE` 必须同时明确设置。
- `run_grpo.sh` 的历史默认值可能仍是 `atom`，不要依赖默认选择。
- 首次从 step 0 启动必须使用 `RESUME_OVERRIDE=false`。
- 只有从 §14 验证完整的 checkpoint 恢复时才使用 `RESUME_OVERRIDE=true`。

## 13. 启动验证

日志中依次确认：

```text
Created 1 placement groups for pool 'rollout' ... node_ip=${ROLLOUT_NODE_IP}
Created 1 placement groups for pool 'actor' ... node_ip=${TRAIN_NODE_IP}
RDMA preflight: ...
RDMA weight group ready: ... world=9
NCCL INFO Using network IB
NCCL INFO ... NET/IB/0/GDRDMA ...
```

每步确认：

```text
RDMA weight sync committed:
callbacks: step=N
```

重点指标：

```text
rollout_corr/kl
rollout_corr/rollout_is_eff_sample_size
actor/loss
actor/grad_norm
actor/entropy
weight_sync/gbps
timing/weight_sync_rdma_s
```

任一核心指标出现 `nan` 必须立即停止，不得继续写下一个 checkpoint。

## 14. Checkpoint 验证

每 5 步保存一次；actor 节点会在保存前清理旧 shard，避免 controller 与 actor 使用同名但
非共享本地路径时只清理 controller metadata。

在 trainer 节点检查 step 5；先填写启动时记录的 `RUN_ID`：

```bash
source "$HOME/qwen3-rdma-node.env"
read -r -p "RUN_ID: " RUN_ID
P="$RUNTIME_HOST_DIR/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

完整 checkpoint 必须满足：

- 8 个 model shard。
- 8 个 optimizer metadata shard。
- 8 个 extra-state shard。
- 8 个大体积 optimizer parameter-state shard。
- controller 侧存在 `checkpoint_N.pt` 与 `latest_checkpointed_iteration.txt`。

检查文件数量：

```bash
source "$HOME/qwen3-rdma-node.env"
read -r -p "RUN_ID: " RUN_ID
docker exec -i -e RUN_ID="$RUN_ID" "$CONTAINER" /opt/venv/bin/python - <<'PY'
import os
from pathlib import Path

p = Path("/runtime/ckpts") / os.environ["RUN_ID"] / "global_step_5" / "actor"
for pattern in (
    "model_world_size_8_rank_*.pt",
    "optim_world_size_8_rank_*.pt",
    "optim_parameter_state_world_size_8_rank_*.pt",
    "extra_state_world_size_8_rank_*.pt",
):
    files = list(p.glob(pattern))
    print(pattern, len(files), sum(x.stat().st_size for x in files))
    assert len(files) == 8
PY
```

## 15. 监控

### 15.1 训练状态

```bash
source "$HOME/qwen3-rdma-node.env"
LOG="$RUNTIME_HOST_DIR/logs/$(basename "$(cat "$RUNTIME_HOST_DIR/current_run_log.txt")")"

grep -a "lumenrl.trainer.callbacks: step=" "$LOG" | tail -1
grep -a "RDMA weight sync committed" "$LOG" | tail -1
grep -aiE "Training failed|Traceback|OutOfMemory|NCCL.*timeout|SIGABRT|=nan" "$LOG" | tail
```

### 15.2 进程和 Ray

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" pgrep -af '[l]umenrl.trainer.main'
docker exec "$CONTAINER" /opt/venv/bin/ray status
```

### 15.3 磁盘

trainer 节点执行：

```bash
source "$HOME/qwen3-rdma-node.env"
read -r -p "RUN_ID: " RUN_ID
df -h "$RUNTIME_HOST_DIR"
du -sh "$RUNTIME_HOST_DIR/ckpts/$RUN_ID"/global_step_*
```

单个完整 checkpoint 当前约 402 GiB。`save_total_limit=3` 需要约 1.2 TiB，加上模型、日志和
Ray 临时文件必须预留安全余量。

### 15.4 W&B

W&B 页面使用 §2 输入的 `${WANDB_ENTITY}` / `${WANDB_PROJECT}` 和启动时生成的 `RUN_ID`。
不要复制历史任务的 run URL。

在线 history 的最新 global step 应与本地 callback 日志基本一致。若 run state 为 `crashed` 或
线上 step 长时间不增长，先检查本地 W&B 日志与网络，不要仅根据网页判断训练进程状态。

## 16. 停止和恢复

停止：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" pkill -TERM -f '[l]umenrl.trainer.main' || true
```

恢复前必须先确认最新 checkpoint 含大体积
`optim_parameter_state_world_size_8_rank_*.pt`。然后使用相同 `CKPT_DIR`，设置：

```bash
export RESUME_OVERRIDE=true
```

日志必须出现：

```text
Resuming Ray actor checkpoint from ... global_step_N/actor
Ray resume complete. Next training log will be global_step=N+1
```

恢复后至少观察两个完整 step：

- 第一个 step 验证 checkpoint model 可加载。
- 第二个 step 验证 optimizer update 后权重仍为有限值。

只有连续两步 KL、ESS、loss、grad norm、entropy 全部有限，才视为恢复成功。

## 17. 排障

### 17.1 `ModuleNotFoundError: vllm`

原因：Ray 把 rollout placement group 调度到 trainer 节点。

处理：

- 确认 YAML 的 rollout `topology_tags.node_ip=${ROLLOUT_NODE_IP}`。
- 确认 actor `topology_tags.node_ip=${TRAIN_NODE_IP}`。
- 日志必须打印本次探测并写入 YAML 的两个 node IP。

### 17.2 RDMA 退化成 TCP

检查：

```bash
source "$HOME/qwen3-rdma-node.env"
ls -l /dev/infiniband
test -d "/sys/class/infiniband/$RDMA_HCA"
```

确认 `NCCL_SOCKET_IFNAME=$RDMA_IFACE`、`NCCL_IB_HCA=$RDMA_HCA`、
`NCCL_IB_GID_INDEX=$RDMA_GID_INDEX`。
没有 `NET/IB/.../GDRDMA` 日志就不能宣称 GPU Direct RDMA 已启用。

### 17.3 flash-attn `varlen_fwd()` 参数不兼容

原因：Python wrapper 带 `num_splits`，native ROCm extension 是旧 21 参数 ABI。

处理：使用当前 `run_grpo.sh` 的幂等 ABI patch，并运行 §7.1 kernel 测试。

### 17.4 checkpoint 写满磁盘

症状：

```text
RuntimeError: basic_ios::clear: iostream error
No space left on device
```

处理：

- 删除不完整的当前 step 目录。
- 保留至少一个已验证完整 checkpoint。
- 确认 actor-node `prune_checkpoints` 在保存前执行。
- 不要保留只有 model、没有 optimizer parameter state 的历史 checkpoint。

### 17.5 恢复后第二步 NaN

通常表示 optimizer 没有完整恢复，或 FP32 main parameters 未与 model 同步。

检查：

- `optim_parameter_state_world_size_*.pt` 是否存在且为几十 GiB。
- 是否调用 `load_parameter_state()`。
- 是否调用 `reload_model_params()`。
- 不要用 2 KiB 的 optimizer metadata 文件代替 distributed parameter state。

### 17.6 W&B 没更新

检查：

- `/runtime/wandb.key` 是否存在且容器内可读。
- 本地 callback 是否继续输出 step。
- W&B run name 是否与本次 `RUN_ID` 一致。
- resume 时不要让 global step 回退，否则 W&B 会拒绝旧 step。

### 17.7 `Too many open files` / Ray socket EOF

容器启动和训练脚本均设置：

```bash
ulimit -n 524288
```

必要时停止两节点 Ray 后重新按 §10 顺序启动。

## 18. 关键源码

```text
lumenrl/core/config.py
lumenrl/utils/independent_process_group.py
lumenrl/engine/inference/rdma_weight_transfer.py
lumenrl/engine/inference/vllm_colocate_worker_ext.py
lumenrl/engine/inference/vllm_ray_server.py
lumenrl/workers/actor_worker.py
lumenrl/controller/ray_worker_group.py
lumenrl/trainer/rl_trainer.py
lumenrl/trainer/callbacks.py
lumenrl/engine/training/megatron_engine.py
examples/GRPO/run_grpo.sh
examples/GRPO/configs/grpo_qwen3_30b_a3b_vllm_ep8_longrun.yaml
```

## 19. 参考

- LumenRL：`https://github.com/ZhangDanyang-AMD/Lumen-RL/tree/dev/moe-grpo`
- Lumen：`https://github.com/ZhangDanyang-AMD/Lumen/tree/dev/qwen3-30b-a3b`
- Megatron-LM（ROCm fork）：`https://github.com/ROCm/Megatron-LM/tree/rocm_dev`
- aiter：`https://github.com/ZhangDanyang-AMD/aiter/tree/lumen/qwen3-30b-a3b`
- flash-attention：`https://github.com/ROCm/flash-attention`
- LMSYS MILES RL on AMD：
  `https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/`
- MILES：`https://github.com/radixark/miles`
- Qwen3-30B-A3B：`https://huggingface.co/Qwen/Qwen3-30B-A3B`

## 附录 A：切换到 ATOM rollout

本附录仅说明可选的 ATOM 流程。正文中的正式 200-step 任务仍使用 vLLM + RCCL/RoCE RDMA。

### A.1 支持范围和限制

ATOM 与当前 RDMA backend 不能直接组合：

- `policy.generation_backend: atom` 会创建 `ATOMReplicaManager`。
- 当前 ATOM manager 没有初始化 9-rank RCCL weight group，也没有 RDMA receiver。
- 因此 ATOM 配置不能设置 `weight_sync.backend: rdma`。
- 两节点 ATOM 必须使用 `shared_folder`；同节点可使用 `auto`。

`weight_sync.backend: auto` 的实际选择规则：

- 同节点、ATOM TP=1、8 replicas 对应 8 actor workers：使用 ZMQ CUDA-IPC。
- 同节点、ATOM TP=2、4 replicas 少于 8 actor workers：自动回退到 safetensors。
- ATOM 与 trainer 分布在不同节点：自动回退到 safetensors。

Qwen3-30B-A3B 当前推荐 ATOM TP=2，因此实际稳定路径是：

```text
Megatron BF16 shard
  → actor 聚合并转换为 HF 名称
  → safetensors 导出
  → ATOM TP=2 × 4 replicas reload
```

不要把该路径标记为 RDMA，也不要期待日志出现 `NET/IB/.../GDRDMA`。

### A.2 部署形态

推荐先做单节点 colocated smoke：

```text
单节点 8 × MI308X
├── Megatron actor workers：8
├── ATOM replicas：4
├── 每个 ATOM replica：TP=2
├── ATOM 权重：BF16
├── KV cache：FP8
└── sleep mode：level 2
```

同一批 GPU 在 rollout 和 training 间切换：

1. ATOM wake weights/KV cache。
2. 生成 response，并记录 MoE router logits。
3. ATOM sleep level 2，释放可回收显存。
4. Megatron 使用 R3 distribution replay 计算 log-prob 并更新参数。
5. actor 导出 HF safetensors。
6. ATOM wake weights，reload 新参数，再恢复 KV cache。

单节点同时容纳 ATOM CUDA graph、Megatron 参数和 distributed optimizer，显存压力明显高于正文的
两节点方案。出现 OOM 时优先降低 response length、batch size、ATOM
`gpu_memory_utilization`，不要关闭 checkpoint 的 optimizer parameter-state 保存。

### A.3 ATOM 源码和环境

如果使用自定义 ATOM 源码，先让用户提供宿主机目录：

```bash
read -r -p "ATOM 源码目录 ATOM_HOST_DIR: " ATOM_HOST_DIR
export ATOM_HOST_DIR
test -f "$ATOM_HOST_DIR/atom/__init__.py"
```

创建容器时额外添加：

```bash
-v "$ATOM_HOST_DIR":/workspace/ATOM
```

不挂载源码时使用镜像内 ATOM。挂载后启动脚本会把 `/workspace/ATOM` 放到 `PYTHONPATH`。
验证实际导入位置：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
PYTHONPATH=/workspace/ATOM:/workspace/Lumen-RL \
  /opt/venv/bin/python - <<PY
import atom
from atom.rollout.async_engine import AsyncLLMEngine
print("atom module:", atom.__file__)
print("AsyncLLMEngine:", AsyncLLMEngine)
PY
'
```

期望 `atom.__file__` 位于 `/workspace/ATOM/atom/`，而不是意外导入其他 site-packages 版本。

ATOM TP>1 时保留以下环境：

```bash
export LUMENRL_DISABLE_CUSTOM_AR=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
export ATOM_TORCH_COMPILE_CACHE_ROOT=/tmp/atom_torch_compile_cache
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export USE_ROCM_AITER_ROPE_BACKEND=0
```

`NoCustomARModelRunner` 会关闭 ATOM 的 HIP IPC custom all-reduce，改用 RCCL collective，避免
`ca_comm is None` 或 `hipIpcOpenMemHandle` 崩溃。这里的 RCCL 只用于 ATOM TP collective，
不是正文所述的跨节点 RDMA 权重广播。

### A.4 ATOM YAML

基准文件：

```text
examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml
```

关键配置：

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

weight_sync:
  backend: auto
  shared_folder: /shared/lumenrl_weight_sync/atom-qwen3-30b-a3b
  bucket_size_mb: 1024
  timeout_s: 600

controller:
  ray:
    enabled: true
    fuse_actor_ref: false
    actor:
      num_workers: 8

policy:
  training_backend: megatron
  generation_backend: atom
  training:
    megatron_cfg:
      tensor_parallel_size: 4
      expert_parallel_size: 8
      expert_tensor_parallel_size: 1
      use_distributed_optimizer: true
      sequence_parallel: true
      attention_backend: flash
  generation:
    atom_cfg:
      tensor_parallel_size: 2
      expert_parallel_size: 1
      kv_cache_dtype: fp8
      engine_kwargs:
        enforce_eager: false
        compilation_config:
          level: 3
    vllm_cfg:
      gpu_memory_utilization: 0.60
      max_model_len: 9216
      dtype: bfloat16
      quantization: ""
      enable_sleep_mode: true
      sleep_level: 2

moe:
  r3:
    enabled: true
    record_router_logits: true
    replay_mode: distribution
```

注意：

- `atom_cfg` 控制 ATOM engine；当前实现仍从 `vllm_cfg` 读取部分通用 generation/sleep 参数。
- `quantization: ""` 表示 BF16 权重，不启用 `per_block_fp8`。
- FP8 只用于 KV cache。Qwen3-30B-A3B 的 MoE intermediate size 为 768，在线
  `per_block_fp8` 已观测到输出乱码。
- `grpo_qwen3_30b_a3b_atom_ep8_smoke.yaml` 当前内容实际是
  `generation_backend: vllm`，不能作为 ATOM smoke 直接使用；应复制 longrun 文件并缩小参数。

ATOM smoke 建议覆盖：

```yaml
policy:
  max_total_sequence_length: 1088
  max_response_length: 1024
  train_global_batch_size: 64
  gen_batch_size: 8
  max_token_len_per_gpu: 4096

checkpointing:
  save_steps: 9999
  resume: false

logger:
  wandb_enabled: false

num_training_steps: 3
```

### A.5 启动

先准备本地同步目录：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
mkdir -p /shared/lumenrl_weight_sync/atom-qwen3-30b-a3b
rm -f /shared/lumenrl_weight_sync/atom-qwen3-30b-a3b/*.tmp
'
```

使用单独制作的 ATOM smoke YAML 启动：

```bash
source "$HOME/qwen3-rdma-node.env"
docker exec "$CONTAINER" bash -lc '
export PATH=/opt/venv/bin:$PATH
export RL_ROOT=/workspace
export DATA_ROOT=/runtime
export MODEL_PATH=/root/models/Qwen3-30B-A3B
export TRAIN_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet
export VAL_FILE=/root/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet
export MODE=smoke
export STEPS=3
export BACKEND=atom
export CONFIG_OVERRIDE=examples/GRPO/configs/grpo_qwen3_30b_a3b_atom_ep8_smoke_local.yaml
export RUN_ID=qwen3-30b-a3b-atom-smoke3
export LOG=/runtime/logs/qwen3-30b-a3b-atom-smoke3.log
export CKPT_DIR=/runtime/ckpts/qwen3-30b-a3b-atom-smoke3
export WEIGHT_SYNC_BACKEND=auto
export RESUME_OVERRIDE=false

bash /workspace/Lumen-RL/examples/GRPO/run_grpo.sh
'
```

`BACKEND=atom` 不足以修正一个内容仍为 vLLM 的 `CONFIG_OVERRIDE`；必须确认最终 YAML 中明确写有：

```yaml
policy:
  generation_backend: atom
```

### A.6 ATOM smoke 验证

启动日志应出现：

```text
ATOMReplicaManager: launched 4 colocated rollout replicas (atom_tp=2, workers=8)
Ray ATOM rollout ready: 4 colocated replicas
R3Manager: router recording ENABLED
```

TP=2×4 时权重同步应出现 safetensors export/reload 日志，而不是 RDMA 日志。每步必须确认：

```text
callbacks: step=N
```

并检查：

```bash
source "$HOME/qwen3-rdma-node.env"
LOG="$RUNTIME_HOST_DIR/logs/qwen3-30b-a3b-atom-smoke3.log"
grep -aE "ATOMReplicaManager|Ray ATOM rollout ready|reloaded weights|callbacks: step=" "$LOG" | tail -30
grep -aiE "Traceback|OutOfMemory|illegal memory access|ca_comm|=nan" "$LOG" | tail
```

通过标准：

- 连续完成至少 3 步。
- reward、loss、entropy、grad norm、KL 和 ESS 全部有限。
- R3 router-logit coverage 完整。
- 每轮训练后 ATOM 成功 reload 权重。
- 无乱码、非法显存访问、OOM 或 stale CUDA/HIP IPC handle。

ATOM smoke 通过后，才可切换到 `grpo_qwen3_30b_a3b_atom_ep8_longrun.yaml`。正式长跑仍应采用
§14 的完整 distributed optimizer checkpoint 规则；不要恢复不完整 checkpoint。

## 附录 B：已验证基线和 checkpoint 事故记录

以下数值只用于判断新部署是否明显退化，不能作为新环境的 IP 或路径配置来源。

已验证的 RDMA smoke 连续完成 3 步：

- 每步广播 61.1 GB，共 58 buckets。
- 总同步耗时 2.51–3.90 秒，有效吞吐 134–215 Gb/s。
- `rollout_corr/kl`：0.0008425、0.0008446、0.0007718。
- ESS：0.998361、0.998328、0.998397。
- 每个 TP worker 动态校验完整权重覆盖。

历史 v1 任务在保存 checkpoint 时磁盘写满。旧逻辑只保存了约 2 KiB 的
`optimizer.state_dict()` metadata，没有 Megatron distributed optimizer 的 FP32 master 和 Adam
moments；这种 checkpoint 加载模型后继续更新会出现 NaN。

当前可恢复 checkpoint 必须同时覆盖：

```text
optimizer.state_dict()
optimizer.save_parameter_state(...)
optimizer.load_parameter_state(...)
optimizer.reload_model_params()
```

完整 step checkpoint 的基线约 402 GiB，包含 8 个 model shard、8 个 optimizer metadata shard、
8 个 extra-state shard，以及 8 个约 41–45 GiB 的 optimizer parameter-state shard。严禁把只有
小型 optimizer metadata 文件的 checkpoint 视为可续训 checkpoint。

---

最后更新：2026-07-29
