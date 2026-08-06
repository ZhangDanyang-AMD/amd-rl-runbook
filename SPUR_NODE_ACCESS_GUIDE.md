# Spur 集群节点访问与作业操作指南（供 Agent 使用 · 通用版）

> 目的：让任何用户的 agent 都能连到 spur 集群、拿到/查看作业、并进入分配到的计算节点执行任务。
> 本文档**不绑定特定用户名/账号**，所有位置一律用 `$USER`、`$HOME` 或"自行探测"的方式表述。
> 关键结论：**计算节点（`crsuse2-m2m-*`）禁止直接 SSH，必须用 `spur exec <JobID>` 进入。**
> **多节点作业另有一条**：`spur exec` 只能进 **head 节点**，其余节点要用 §5c 的办法。

---

## 0. Agent 铁律（先读这一节）

1. **用户给了 JobID，就只用这个 JobID。**
   - 只对该 JobID 执行 `spur exec` / `scontrol show job` / 查日志等操作。
   - **禁止**：自己再 `sbatch` / `spur alloc` 申请新作业；**禁止**切换到 `squeue` 里看到的其它作业；
     **禁止** `scancel` 任何作业（包括用户给的这个，除非用户明确要求取消）。
   - 若该 JobID 不是 `RUNNING`（如 `PENDING` / 已结束 / 不存在 / 不属于当前用户），
     **停下来向用户报告**，不要自作主张换一个作业顶上。
   - 建议开工前先固定变量，后续命令一律引用它：
     ```bash
     JOBID=<用户给的ID>
     scontrol show job "$JOBID" | head -20     # 确认状态与节点
     ```
2. **不要直接 ssh 计算节点**，一律 `spur exec "$JOBID" bash -lc "..."`。
3. **不要污染别人的资源**：不改他人家目录、不 kill 不属于本作业的进程。

---

## 1. 环境速览

- **登录节点**：`crs-m2m-cpu-spur-0XX.crusoe.amd.com`（spur 系列登录节点均可）。
  用 `hostname -f` 确认你当前在哪台机器上。
- **调度器**：`spur`（AI-native scheduler，带 Slurm 兼容别名 `sbatch/srun/squeue/scancel/sinfo/sattach/scontrol` 等）。
- **调度器控制端**：环境变量 `SPUR_CONTROLLER_ADDR`（登录节点上由 `/etc/profile.d/spur.sh|spur.csh` 自动设置，
  当前集群值为 `http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817`）。
- **家目录 `$HOME` 是共享 NFS**，登录节点与计算节点容器内都可见，适合放代码与日志。
- **计算节点**：`crsuse2-m2m-*`，被管理员加了 `AllowUsers ubuntu root`，普通用户直接 `ssh` 一律 `Permission denied`。

### 1.1 开工自检（三行）

```bash
hostname -f                      # 确认在哪个登录节点
echo "$SPUR_CONTROLLER_ADDR"     # 非空即可；为空见 1.2
squeue -u "$USER"                # 能出结果说明链路通了
```

### 1.2 shell 差异（重要）

- 如果你**已经在登录节点上、且 agent 用的是 bash**（多数情况）：可自由使用 `2>&1`、管道等 bash 语法。
- 如果你是**从外部 `ssh host "命令"` 非交互执行**：登录 shell 可能是 **csh/tcsh**，此时
  - 设环境变量用 `setenv A B`（不是 `export A=B`）；
  - 命令串里**不要用** `2>&1` / `2>/dev/null` / 管道，csh 会报 `Ambiguous output redirect`；
  - profile 不会自动加载，需手动设控制端地址：
    ```csh
    setenv SPUR_CONTROLLER_ADDR http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817
    ```
- 注意：`echo $SHELL` 可能显示 tcsh，但 agent 实际执行命令的 shell 未必是它，以实际报错为准。

---

## 2. 探测自己的账号 / 分区 / QOS（不要照抄别人的）

```bash
# 当前用户可用的 Account / QOS / 默认值
spur accounts show user where name="$USER"

# 分区总览（PARTITION 列，带 * 的是默认分区）
spur nodes            # 等价 sinfo
```

把结果里的 `Account` 与分区名填进后续 `-A <account> -p <partition>`。
（例如本集群常见形如 `-A amd-<team>-<xxx> -p amd-spur`，QOS 通常是 `<account>-qos`，
但 `spur alloc` **不接受 `-q`**，用默认 QOS 即可。）

### 2.1 多节点之前必看：QOS 的组配额

要 2 个以上节点时，第一个会拦你的不是资源而是 **QOS 的 account 级组配额**：

```bash
spur accounts show qos            # 看 GrpTRES 那一列
```

实测（`amd-aifw-dev` 账号）：

| QOS | GrpTRES | 说明 |
|---|---|---|
| `amd-aifw-dev-qos`（默认） | `node=19` | **整个 account 全体成员共享**，实测常被队友占掉 17 个 |
| `amd-burst-qos` | `node=128` | 低优先级（priority 100 对 10000），但配额大 |

所以配额是"你们团队一共能同时用多少节点"，不是你个人的。用下面这条看当前用掉多少：

```bash
squeue -t RUNNING -o '%q %D' | tail -n +2 | awk '$1=="<你的默认qos>"{s+=$2} END{print s}'
```

剩余名额不够时，`-N 4` 会一直 `PENDING (QOSGrpNodeLimit)` ——**这不是资源不足**，
`spur nodes` 里可能还有几十台 idle。解法见 §4c。

---

## 3. 查看集群与作业

```bash
spur nodes                                     # 节点总览（分区/状态/空闲数）
spur queue                                     # 全局队列（等价 squeue）
squeue -u "$USER"                              # 只看自己的作业
scontrol show job "$JOBID"                     # 某作业详情（状态、节点、时限、账号）
squeue -u "$USER" -o '%.8i %.10L %.10l %.11M %.20S %R'   # 带剩余时间/上限的自定义列
```

---

## 4. 分配节点（**仅在用户没给 JobID 时才做**）

> 用户已经给了 JobID → 跳过整节，直接看第 5 节。

- **可以同时持有多个 8 卡作业**，没有"一个账号只能一个 8 卡"的硬上限。
- **⚠️ GPU 作业对"申请方式"很挑剔（实测）**：
  - ✅ **真实终端里的 `spur alloc`（交互）能正常拿到 8 卡节点** —— 这是可靠方式。
  - ❌ **`sbatch --wrap "..."` 申请 GPU（`-G 8` 或 `--gres gpu:8`）会一直 `PENDING (JobHoldMaxRequeue)`**；
    `ssh -tt` + 管道 stdin 拼出来的"伪交互" `spur alloc` 同样会被 hold。
    → 这不是账号并发上限，而是**非真 tty 的 GPU 分配路径在本集群不可靠**。
- 结论：**要 GPU 节点，就让真人在真终端用 `spur alloc`**，然后把 JobID 交给 agent，
  agent 之后一律用 `spur exec <JobID>` 进节点干活。

### 4a. 交互式会话（真人用，需真 tty —— GPU 首选）

```bash
spur alloc -N1 -t 1-00:00:00 -A <account> -p <partition> -G 8
```

`spur alloc`（= salloc）**必须在真实终端里跑**：不接受尾随命令、需要 tty、直接给一个交互 shell。用完 `exit` 释放。

### 4b. 批处理占位 / 跑脚本（agent 自动化用 —— 仅对 CPU 可靠）

```bash
sbatch --parsable -J myjob -A <account> -p <partition> -N1 -t 04:00:00 --wrap "sleep 14400"
```

提交后用 `squeue -u "$USER"` 等状态变为 `R`，并拿到 `NODELIST`。
（spur 的 `sbatch` 申请 GPU 用 `-G N` 而不是 `--gres gpu:N`，但 GPU 批处理常被 hold，见上。）

### 4c. 多节点（≥2 节点，真人在真终端跑）

`spur alloc` **不支持 `-q`**，所以撞上 §2.1 的配额时它无路可走。`spur run` 支持：

```bash
spur run -q amd-burst-qos -N 4 --gpus-per-node=8 \
  -t 1-00:00:00 -A <account> -p <partition> --pty bash -l
```

四个细节，每个都踩过：

- **`--gpus-per-node=8` 而不是 `-G 32`**。`-G` 的语义是"整个作业的 GPU 总数"，
  调度器可以任意摊到各节点上；只有 `--gpus-per-node` 能保证每节点满卡。
- **`--pty` 不能省**。非真 tty 申请 GPU 在本集群不可靠，会掉进 `JobHoldMaxRequeue`
  （队列里常年挂着 30+ 个这种僵尸作业，它们不占资源也永远不会跑）。
- ⚠️ **不要在已有分配的交互 shell 里跑这条**。那里有 `SLURM_JOB_ID`，`spur run` 会认为
  "我在现有分配里"，于是把命令变成那个作业的一个 **job step**——`-N 4`、`-q` 全被忽略，
  表现是打出 `dispatched to node <单个节点>` 且**不产生新 JobID**。先确认：

  ```bash
  env | grep -E 'SLURM_JOB_ID|SLURM_NNODES'    # 必须无输出
  ```
- 成功的标志是打出 `Pending job allocation <新JobID>...` 然后
  `running on crsuse2-m2m-[a,b,c,d]`。

拿到后确认卡数：

```bash
scontrol show job "$JOBID" | head -12    # 期望 NumNodes=4、TresPerNode=gpu:8/node、ReqGPUs=32
```

> `amd-burst-qos` 优先级低（100 对 10000），但 `Preempt=off`，不会被抢占；排队时会让在普通 QOS 后面。

---

## 5. 进入分配到的节点（核心）

**不要用 SSH**（会被 `AllowUsers ubuntu root` 拒绝，连从登录节点带着分配也不行）。
用 `spur exec <JobID> <命令>`，控制端会把请求代理到对应计算节点的容器里。

### 5a. 跑一次性命令（稳定，agent 首选）

```bash
JOBID=<用户给的ID>
spur exec "$JOBID" bash -lc 'hostname; whoami; pwd'
```

### 5b. 交互式 shell（真人用）

用 `spur alloc` 开交互会话即可。

> 注意：`spur exec <JobID> bash`（不带 `-c`）会"假死" —— `spur exec` 不分配 tty，
> bash 起来后没有终端、只干等 stdin。要交互请用 `spur alloc`；要执行请用 `bash -lc "..."`。

### 5c. 多节点：`spur exec` 只到 head，其余节点用 `nx.sh`

多节点作业里三条路只有一条通：

| 方式 | 结果 |
|---|---|
| `ssh <node>` | `Permission denied (publickey)`。计算节点 `AllowUsers ubuntu root`，容器里是普通用户 |
| `spur exec <JobID>` | **只到 head 节点**。它没有指定节点的参数，控制端固定代理到 head |
| `srun --jobid X --overlap -w <node> <cmd>` | 可行，**但 stdin 不是真 tty 时静默丢弃输出** |

第三条那个坑很坑：spur 0.7.0 只打一行 `spur: warning: raw mode unavailable (stdin is not a TTY)`，
然后**命令看起来什么都没发生**——`hostname` 有时有输出、`ls /` 和 `bash -c 'echo hi'` 完全没有，
很容易误判成"这台机器坏了"。加一层 `script -qec` 造 pty 就全正常。

封装成 `~/nx.sh`（脚本经由共享 NFS 的 `$HOME` 传递，所以各节点都看得到）：

```bash
cat > ~/nx.sh <<'EOF'
#!/usr/bin/env bash
# Run a bash snippet on one specific node of a spur multi-node job.
#
# spur exec <jobid> only ever reaches the job's head node, and srun job steps
# silently produce nothing unless stdin is a real tty (spur 0.7.0 prints
# "raw mode unavailable" and drops the output), hence the script(1) wrapper.
# The snippet travels through $HOME because that is shared NFS across nodes.
set -uo pipefail
: "${JOBID:?set JOBID to the spur job id}"
NODE="${1:?usage: nx.sh <node> [command...]}"
shift
D="$HOME/.nx"; mkdir -p "$D"
F="$D/cmd_${NODE}_$$_$RANDOM.sh"
if [ $# -gt 0 ]; then printf '%s\n' "$*" > "$F"; else cat > "$F"; fi
script -qec "srun --jobid $JOBID --overlap -w $NODE -N1 -n1 bash -l $F" /dev/null
rc=$?
rm -f "$F"
exit $rc
EOF
chmod +x ~/nx.sh
```

用法与自检（4 台都该回自己的 hostname 和 8 卡）：

```bash
export JOBID=<用户给的ID>
for n in $(scontrol show job "$JOBID" | grep -oE 'crsuse2-m2m-[0-9]+' | sort -u); do
  printf "%s: " "$n"
  ~/nx.sh "$n" 'hostname; rocm-smi --showid 2>/dev/null | grep -oE "GPU\[[0-9]+\]" | sort -u | wc -l'
done
```

> 输出里会夹一个 `^@`（`script` 的产物），用 `sed 's/\^@//'` 滤掉。
>
> `nx.sh` 跑的是**宿主侧**（docker host）。要进容器仍然是 `docker exec`，
> 即 `~/nx.sh <node> 'docker exec <container> bash -lc "..."'`。

### 5d. rank 与节点的对应关系

Ray / torchrun 之类的框架按节点**连续**发 rank。实测 4×8 的作业：

| 节点 | rank |
|---|---|
| head（`spur exec` 到的那台） | 0–7 |
| 第二台 | 8–15 |
| 第三台 | 16–23 |
| 第四台 | 24–31 |

排查"某个 rank 在哪台机器上"时很有用。**这是观测到的行为，不是保证**——依赖它的地方
（比如让专家并行组落在节点内）应当加强制检查。

---

## 6. 在节点容器内的常见操作 / 注意事项

- **容器内默认是 root**，工作目录 `/`；`$HOME` 为共享 NFS，跨节点可见。
  注意：容器内 `$USER`/`$HOME` 可能不同于登录节点，脚本里尽量写**绝对路径** `/home/<你的用户名>/...`。
- ⚠️ **多节点：`/mnt/m2m_nobackup` 是 node-local 的**（每台一块 28T NVMe，互相看不见）。
  模型、数据、以及任何需要所有节点读到的东西必须放共享 NFS（`$HOME` 下）；
  只有 checkpoint 这类"丢了还能重来"的大文件才适合放 node-local 盘，而且要接受
  节点故障后取不回来的风险。判断方法：
  ```bash
  # 同一路径在两台机器上看到的内容是否一致
  for n in <node1> <node2>; do printf "%s: " "$n"; ~/nx.sh "$n" 'ls /mnt/m2m_nobackup | head -3'; done
  ```
- **`ps` 报错 `Error, do this: mount -t proc proc /proc`**：容器内 `/proc` 未挂载，先挂再用：
  ```bash
  spur exec "$JOBID" bash -lc 'mount -t proc proc /proc; ps -eo pid,user,pcpu,pmem,etime,args --sort=-pcpu | head -n 20'
  ```
- **GPU 是 AMD Instinct（ROCm），没有 `nvidia-smi`**：
  ```bash
  spur exec "$JOBID" bash -lc 'rocm-smi'                # 概览
  spur exec "$JOBID" bash -lc 'rocm-smi --showpids'     # GPU 上的进程
  ```
- **健康自检一条龙**：
  ```bash
  spur exec "$JOBID" bash -lc 'hostname; uptime; free -h; df -h /; rocm-smi; mount -t proc proc /proc; rocm-smi --showpids'
  ```

---

## 7. 跑作业（示例模板）

```bash
JOBID=<用户给的ID>
PROJ=$HOME/<your_proj>
LOGDIR=$HOME/logs; mkdir -p "$LOGDIR"

# 进入代码目录并运行脚本（家目录共享，路径直接可用）
spur exec "$JOBID" bash -lc "cd $PROJ && ./run.sh"

# 后台长任务：日志写到共享家目录，便于事后查看
spur exec "$JOBID" bash -lc "cd $PROJ && nohup python train.py > $LOGDIR/train_$JOBID.log 2>&1 &"

# 查看日志
spur exec "$JOBID" bash -lc "tail -n 50 $LOGDIR/train_$JOBID.log"
```

---

## 8. 取消 / 清理

```bash
scancel "$JOBID"            # 或 spur cancel "$JOBID"
```

> **Agent 注意**：只有用户明确要求时才取消作业；用户给你的 JobID 默认是"借你用"的，不要顺手 `scancel`。
> 自己申请的占位/交互作业用完请及时取消，避免长期占用 GPU 资源。

---

## 9. 故障排查速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ssh <user>@crsuse2-m2m-XXX` → `Permission denied (publickey)` | 计算节点 `AllowUsers ubuntu root` 封锁 | 改用 `spur exec <JobID>`，别直接 ssh |
| `failed to connect to spurctld ... Connection refused` | 非登录 shell 没加载 `SPUR_CONTROLLER_ADDR` | csh：`setenv SPUR_CONTROLLER_ADDR http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817`；bash：`export SPUR_CONTROLLER_ADDR=...` |
| `spur exec <JobID> bash` 卡住无提示符 | exec 不分配 tty | 用 `bash -lc "..."` 或改用 `spur alloc` |
| `ps` 报 `mount -t proc proc /proc` | 容器 `/proc` 未挂载 | 先 `mount -t proc proc /proc` 再 `ps` |
| `nvidia-smi: command not found` | 这是 AMD ROCm 平台 | 用 `rocm-smi` |
| `Unknown option '-lc'` / `Ambiguous output redirect` | 远端是 csh，不吃 bash 语法 | 用 `setenv`，去掉 `2>/dev/null` 等重定向 |
| GPU 作业一直 `PENDING (JobHoldMaxRequeue)` | 用 `sbatch --wrap`/非真 tty 申请 GPU，在本集群不可靠（与账号并发无关） | 让真人在真实终端用 `spur alloc ... -G 8` 开节点，再把 JobID 交给 agent |
| `spur alloc` 报 `unexpected argument '-q'/'sleep'` | `spur alloc` 不支持 `-q`，也不接受尾随命令 | 去掉 `-q`；脚本占位请用 `sbatch` |
| `Invalid account or account/partition combination` | `-A`/`-p` 抄了别人的 | 用 `spur accounts show user where name="$USER"` 查自己的 |
| `-N 2` 以上一直 `PENDING (QOSGrpNodeLimit)`，但 `spur nodes` 里有大量 idle | QOS 的 **account 级**组配额用满（不是你个人的） | `spur accounts show qos` 看 `GrpTRES`；换配额大的 QOS，用 `spur run -q`（`spur alloc` 不支持 `-q`），见 §2.1 / §4c |
| `spur run` 打出 `dispatched to node <单节点>`、没有新 JobID、`-N`/`-q` 像被忽略 | 在已有分配的 shell 里跑，`SLURM_JOB_ID` 让它变成 job step | 换干净终端，先确认 `env \| grep SLURM_JOB_ID` 无输出 |
| `srun --overlap -w <node> <cmd>` 静默无输出，只有一行 `raw mode unavailable` | stdin 不是真 tty，spur 0.7.0 丢弃输出 | 用 §5c 的 `nx.sh`（`script -qec` 包一层） |
| 多节点作业里 `spur exec` 永远进同一台机器 | 它没有节点参数，固定代理到 head | 同上，非 head 节点用 `nx.sh` |
| 多节点上各机器看到的 `/mnt/m2m_nobackup` 内容不同/为空 | 那是 node-local 盘 | 共享数据放 `$HOME`（NFS），见 §6 |

---

## 10. 一句话流程

**情形 A：用户给了 JobID（最常见）**

1. `JOBID=<用户给的ID>`
2. `scontrol show job "$JOBID"` 确认是 `RUNNING`（不是就报告用户，别换作业）
3. `spur exec "$JOBID" bash -lc "……你的命令……"`
4. **不要** `scancel`，除非用户明确要求

**情形 B：用户没给 JobID**

1. `spur accounts show user where name="$USER"` 查账号/QOS，`spur nodes` 查分区
2. CPU：`sbatch --parsable -J job -A <account> -p <partition> -N1 -t 04:00:00 --wrap "sleep 14400"`
   GPU：请真人在真终端 `spur alloc -N1 -t 1-00:00:00 -A <account> -p <partition> -G 8`
3. `squeue -u "$USER"` 等 `R`，拿到 JobID / `NODELIST`
4. `spur exec "$JOBID" bash -lc "……"` → 用完 `scancel "$JOBID"`

**情形 C：多节点作业（用户给了一个 ≥2 节点的 JobID）**

1. `JOBID=<用户给的ID>`，`scontrol show job "$JOBID" | head -12` 确认 `NumNodes` 与 `TresPerNode`
2. 取节点列表：`scontrol show job "$JOBID" | grep -oE 'crsuse2-m2m-[0-9]+' | sort -u`
3. **head 节点**（`spur exec` 到的那台）：`spur exec "$JOBID" bash -lc "……"`
4. **其余节点**：`JOBID=$JOBID ~/nx.sh <node> '……'`（§5c；没有这个脚本先按那节创建）
5. 判断某个 rank 在哪台：§5d 的连续分配规律
6. 共享数据一律走 `$HOME`（NFS），别用 `/mnt/m2m_nobackup`（§6）
7. **不要** `scancel`，除非用户明确要求

（仅当你不在登录节点上时才需要：`ssh <user>@crs-m2m-cpu-spur-0XX.crusoe.amd.com`，
并在非交互命令里 `setenv SPUR_CONTROLLER_ADDR http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817`。）

---

## 11. 想跑真实的多节点训练？

本文只到"能在每台机器上执行命令"为止。再往上（多节点 Ray 集群、跨节点 NCCL、
32 卡的 DAPO smoke 实测）见 `dapo-lumenrl-4node-32gpu-runbook.md`。
