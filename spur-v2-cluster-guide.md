# SPUR v2 集群使用指南：申请 / 跑任务 / 续期 / 释放

> 适用于 v2 池子（controller `crs-m2m-cpu-spur-v2-001`，分区 `default`，节点
> `crsuse2-m2m-v2-0XX`，8× MI355X / gfx950）。
>
> **本文所有命令均于 2026-09-16 在 v2 上实测通过。** 凡未实测的地方都显式标注了，
> 请不要把标注为「未实测」的内容当结论用。
>
> ⚠️ **老集群（controller `-spur-005/006/007`、分区 `amd-spur`、节点 `crsuse2-m2m-XXX`）
> 的操作方式和 v2 多处相反**，本文不涉及。最典型的一条：老集群上非交互式申请 GPU 会被永久
> hold，必须真人终端 `spur alloc`；**v2 上 `sbatch` 直接就能拿到机器**。照老集群的习惯操作
> v2 会连着踩坑，反之亦然。

---

## 0. 三条铁律

1. **别人给你 JobID，就只用这个 JobID。** 不自己另外申请，不切到 `squeue` 里看到的别的作业，
   **不 `scancel` 任何不属于你的作业**。JobID 状态不是 `RUNNING` 就停下来问人，别自作主张换一个顶上。
2. **不要 ssh 任何 v2 主机。** 登录节点和计算节点全部拒绝，唯一入口是 `spur exec`，见 §5。
3. **自己申请的机器，用完立刻释放。** 优先用「跑完自动释放」的任务模式（§3.1），
   它天然不会漏放；真要长期持有再用占位模式（§3.2），并自己盯着释放。

---

## 1. 连到 v2：优先用 `--controller`，不要依赖环境变量

v2 是**独立的一套调度器**。`sinfo` / `squeue` 看到的永远是当前 controller 指向的那一套，
切集群只需要换 controller 地址——**不需要（也登不进）v2 自己的登录节点**，在你平时用的
登录节点上操作即可。

**登录节点的默认 controller 指向老集群**（实测默认值是 `-spur-005/006/007` 三个地址），
所以每个新会话都必须显式指向 v2。`spur` 的每个子命令和每个 Slurm 别名都支持
`--controller`，这是最稳的方式：

```bash
SPUR=http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817

sinfo    --controller "$SPUR"
squeue   --controller "$SPUR" -u "$USER"
sbatch   --controller "$SPUR" ...
scancel  --controller "$SPUR" <JobID>
spur exec --controller "$SPUR" <JobID> bash -c '...'
spur show job --controller "$SPUR" <JobID>
```

也可以设环境变量，但它更容易被登录 shell 悄悄改掉：

```bash
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817
```

⚠️ **环境变量会被登录 shell 覆盖，而且不报错。** 历史上某些登录节点有
`/etc/profile.d/spur.sh` 无条件把它写成老集群地址，于是 `bash -l` / `bash -lc` 包一层就把你
踢回老集群，`sinfo | grep v2-0` 数出 0 台，看着像「v2 一台机器都没有」，其实是在问错的集群。
**如果一定要用环境变量，本地包 shell 时用 `bash -c`，不要 `-l`；能用 `--controller` 就别用环境变量。**

> `spur exec <JobID> bash -lc '...'` 里的 `-l` 是另一回事，**没有问题**：那个 `bash` 跑在计算
> 节点上，controller 地址在派发之前就已经被客户端读走了。

⚠️ **v2 的 controller 只有 `-v2-001` 这一个地址是通的。** 实测 `-v2-005:6817` 是
`Connection refused`，别拿其它 v2 主机名当控制端。

---

## 2. 申请前先看清楚

```bash
SPUR=http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817

# 全局概览：分区 default 的 TIMELIMIT 是 infinite（没有 24h 封顶）
sinfo --controller "$SPUR"

# 你自己的作业（最常用）；%L=剩余时间 %l=时限
squeue --controller "$SPUR" -u "$USER" -o '%.8i %.22j %.2t %.12M %.12L %.14l %R'

# 关注的那几台节点的状态：idle 才可能申请得到；alloc=整机被占，mix=只占了部分资源
sinfo --controller "$SPUR" -N -o '%N|%t|%G' | grep -E 'v2-(035|03[7-9]|04[0-3])'

# 谁占着这些节点、作业名是什么（判断是不是真竞争者）
squeue --controller "$SPUR" -o '%.8i %.12u %.24j %.6D %.20R %.12T' | grep -E 'v2-0|USER'

# 单台规格与资源账
spur show node --controller "$SPUR" crsuse2-m2m-v2-035

# 历史作业的最终状态和退出码（任务模式下判断成败就靠它，见 §3.1）
sacct --controller "$SPUR"
```

单台机器的规格（实测）：

| 项 | 值 |
|---|---|
| GPU | `gpu:mi355x:8`（gfx950，288 GB/卡） |
| CPU | `CPUTot=236` |
| 内存 | 约 2.8 TB（作业里 `free -g` 看到 2751G） |
| node-local 盘 | `/mnt/m2m_nobackup`，28 T（docker 根目录在这） |
| 分区 | `default`，`TIMELIMIT=infinite`（`-t 7-00:00:00` 实测能提上去） |

### 三个反直觉的坑

⚠️⚠️ **空闲节点约 100 秒就会被别人拿走。** v2 上有多个团队在跑任务，实测有作业从提交到
起来只用了约 100 秒就抢到一台。**看到空的就尽早占上**，别假设它会等你。

⚠️⚠️ **`sinfo` 显示 `idle` 不等于你申请得到。** 2026-09-16 实测：`crsuse2-m2m-v2-050`
在 `sinfo` 和 `spur show node` 里都是 `State=IDLE`、`Reason=` 空，但提交上去直接 pending：

```
(ReqNodeNotAvail, Reserved for Kubernetes cluster)
```

节点上有**不显示在 `spur show node` 里的预留**。另外 `045`–`049` 属于 `k0s` 分区，
不是 `default`。**提交后一定要回头确认状态真的变成了 `R`**，别提交完就走。

⚠️ **反过来，队列里的高优先级 pending 作业未必是真竞争者。** 实测有优先级 11000 的作业
pending 20 小时不动，而同一时刻集群里有 8 台 `IDLE`——它们多半在**等一台特定节点**
（作业名形如 `hold-crsuse2-m2m-v2-010`）或者已经是僵尸作业。
**先看作业名和已 pending 多久，再判断抢不抢得到。**

---

## 3. 申请：两种模式，先想清楚用哪种

|  | §3.1 任务模式（推荐） | §3.2 占位模式 |
|---|---|---|
| 提交内容 | 你的任务脚本 | `sleep <很久>` |
| 结束方式 | **任务跑完自动释放** | 必须手动 `scancel` |
| 适合 | 训练、评测、批量跑、CI | 调试、交互、需要反复进容器 |
| 漏放风险 | 无 | 高，会长期空占 GPU |

**两种模式都不要带 `-A`**，见本节末尾的公共注意事项。

### 3.1 任务模式：提交任务，跑完自动释放（实测）

这是最省心的用法：把命令交给 `sbatch`，**脚本一退出，作业就结束，节点立刻回到池子里**，
你不需要（也不应该）再去 `scancel`。

```bash
SPUR=http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817
SHARED=$HOME/jobs            # 必须是共享目录，见下面的坑

mkdir -p "$SHARED/logs"
sbatch --controller "$SPUR" --parsable \
  -J "$USER-train" -p default \
  -N1 -w crsuse2-m2m-v2-035 \
  --gpus-per-node=8 --exclusive -c 64 \
  -t 1-00:00:00 \
  -o "$SHARED/logs/%j.out" -e "$SHARED/logs/%j.err" \
  "$SHARED/task.sh" arg1 arg2
```

也可以不写脚本文件，直接用 `--wrap`（和脚本文件互斥）：

```bash
sbatch --controller "$SPUR" --parsable -J "$USER-smoke" -p default \
  -N1 -c8 -t 00:30:00 -o "$SHARED/logs/%j.out" \
  --wrap 'set -euo pipefail; cd /shared/path/to/repo; python train.py'
```

实测行为（作业 6510 / 6511 / 6515）：

- 提交后约 1 秒变 `R`；`--wrap 'sleep 5'` 的作业在第 6 秒还在队列里，第 7 秒**自己消失**，
  不需要任何清理动作。
- **`-t` 是上限保险，不是预约时长。** 任务提前结束就提前释放，不会占满 `-t`。
  但 `-t` 到点会被强杀，所以给足余量。
- 作业结束后用 `sacct` 看结果，`State=COMPLETED` + `ExitCode=0:0` 才算成功：

  ```bash
  sacct --controller "$SPUR" | grep <JobID>
  #  6511  autorelease-pro  <user>   default COMPLETED  00:00:04   1  0:0
  ```

#### 任务模式的四个坑（都实测过）

⚠️⚠️ **日志必须写共享目录，写 `/tmp` 等于扔掉。** `-o /tmp/x.out` 实测在登录节点**找不到
文件**——它写在了计算节点本地的 `/tmp`。同理作业脚本里的 `$PWD`（默认继承提交时的 cwd
路径字符串）是在**计算节点上**解析的，看着一样其实是另一个盘。
**`-o` / `-e` 一律写共享路径**，用 `%j` 带上 JobID 避免多次提交互相覆盖。

⚠️ **任务脚本引用的一切文件都必须在共享存储上。** 脚本**本身**不用——实测 `sbatch` 会把脚本
内容随作业分发（在计算节点上落成 `/var/spool/spur/job<id>/spur_job.sh`），只放在登录节点
本地 `/tmp` 的脚本也能正常跑，参数也照常传进 `$1`。但脚本里 `source` / 读数据 / 写产物的
路径必须共享，否则跑到一半才炸。

⚠️⚠️ **不写 `-c` 的话你的任务只有 1 个 CPU。** 作业脚本受 cgroup 限制，实测
`-c1` → 脚本里 `nproc=1`，`-c8` → `nproc=8`。dataloader 会被这条卡死。
**提交任务模式时显式写 `-c`。**

> 这一条和只看 `spur exec` 得出的结论相反：在占位作业的容器里 `spur exec` 看到的是
> `nproc=236`（它不走作业的 cgroup 配额），所以**不能用 `spur exec` 的观测去推断任务脚本
> 里能用多少核**。
>
> ❓ **未实测**：`--exclusive` 但不写 `-c` 时，脚本里 `nproc` 是 1 还是 236。
> （想测的那台节点被 k8s 预留了，见 §2。）所以别赌，显式写 `-c`，
> 并在任务开头打一行 `nproc` 记进日志。

⚠️ **漏 `--exclusive` 会让别人的 CPU 作业落进来。** 只带 `--gpus-per-node=8` 的话你拿到 8 张卡，
但那台机器剩下的 CPU 仍是空闲的，别人的纯 CPU 作业会进来抢；节点状态是 `mix` 而不是 `alloc`。

### 3.2 占位模式：长期持有一台机器（需要手动释放）

要反复进容器调试、或者要在机器上常驻服务时才用这个。**代价是必须自己记得释放。**

```bash
SPUR=http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817
cd /tmp
sbatch --controller "$SPUR" --parsable \
  -J "$USER-hold-v2-035" -p default \
  -N1 -w crsuse2-m2m-v2-035 --gpus-per-node=8 --exclusive \
  -t 7-00:00:00 --wrap "sleep 604800"
```

提交后确认：

```bash
squeue --controller "$SPUR" -u "$USER"                 # 等 ST 变 R，拿 NODELIST
spur show job --controller "$SPUR" <JobID> | head -20  # 期望 TresPerNode=gpu:8/node
```

拿到机器后怎么进去干活见 §5，续期见 §6，释放见 §7。

### 公共注意事项（两种模式都适用）

⚠️⚠️ **不要带 `-A`。** v2 的用户表是老集群的**真子集**，有相当一部分人在老集群有账号但
v2 上没有。带 `-A` 会被硬拒：

```
Error: job submission failed
Caused by:
    code: 'Client specified an invalid argument', message: "user '<你>' has no account associations.
    Contact your cluster admin to run: sacctmgr add user name=<你> account=<account>"
```

调度器的逻辑是「**指定了 account 就必须有 association，不指定就走无 account 路径**」。
**去掉 `-A` 就能提交成功**，也不用带 `-q`（无 account 作业的 `QOS` 字段是空的，不走任何 QOS）。

⚠️ **无 account 的代价是优先级。** 这类作业 `Priority=1000`，有 account 的作业是 **11000**，
抢同一台空闲机器时你排在后面。想要优先级得找管理员补账号，调度器自己给的命令就是

```
sacctmgr add user name=<你的用户名> account=<你的团队 account>
```

但这**不是阻塞项**——不带 `-A` 现在就能提交并跑起来。

⚠️ **没有 `--test-only`**（`spur` 的 `sbatch` 不支持这个 flag）。没有试探性提交这条路，
只能真提交看结果，所以提交前先按 §2 确认目标节点状态，提交后再确认变成了 `R`。

⚠️ **别用 `spur show job` 的 `Exclusive=` 字段判断整机独占。** 实测有作业明明拿到了整机，
`spur show job` 里却是 `Exclusive=0`。**判独占看 `spur show node` 里 `CPUAlloc` 是否等于
`CPUTot`，或者 `sinfo` 里节点是 `alloc` 而不是 `mix`。**

⚠️ **作业详情里的 `CPUs/Task=1` 不能用来判断容器内的可用核数**，见 §3.1 那条 `-c` 的说明。

⚠️ **判断一台机器在不在，只认 `sinfo -n` / `spur show node`。** DNS 能解析、22 端口开着都
**不代表**调度器认得它——见过节点 DNS 正常但 `sinfo` 查无此节点、`sinfo -n` 返回 `n/a`，
几天后才注册上架。

```bash
sinfo --controller "$SPUR" -n crsuse2-m2m-v2-035
```

---

## 4. 团队独占节点池（按你自己的情况替换）

如果你的团队在 v2 上有独占节点，**申请一律从这些节点里点名（`-w <node>`）**，不去公共池子抢。
撰写本文时的那一组是：

```
crsuse2-m2m-v2-035  037  038  039  040  041  042  043
```

（没有 `036`。v2 上一共约 43 台 `v2-0XX`，只有上面这些是该团队的；`045`–`049` 在 `k0s`
分区，`050` 被 k8s 预留。**请按你实际的节点列表替换。**）

⚠️ **「团队独占」不等于「随时空闲」。** 这些机器经常被同事整片占着，跑 3–5 天的作业很常见。
2026-09-16 实测这 8 台**全是 `alloc`**。**申请前先看状态（§2）。**

---

## 5. 进节点干活（占位模式）

**唯一入口是 `spur exec`。**

```bash
spur exec --controller "$SPUR" <JobID> bash -c '你的命令'
```

- ❌ **v2 上 SSH 一律不通**：登录节点和计算节点都是 `Permission denied (publickey,password)`。
  计算节点被拒是因为管理员限制了 `AllowUsers`，**登录节点也被拒**，
  所以操作 v2 只能是「在你平时的登录节点上 + 指定 v2 的 controller」。
- ⚠️ `spur exec` **按作业属主鉴权**，进不了别人的作业：
  `user <你> cannot exec into job owned by <别人>`。节点被同事占着时只能等。
- ⚠️ `spur exec <JobID> bash`（不带 `-c`）会**假死**——它不分配 tty，bash 起来后干等 stdin。
  要执行就 `bash -c "..."`，要交互得真人开 `spur alloc`。
- ⚠️ **容器内的 `HOME` 不是你的家目录**：2026-09-16 实测是 `/opt/spur`，`PWD=/`。
  **脚本里不要依赖 `$HOME`，一律写绝对路径。**
  （历史上这个值是 `/root/spur` 且不可写，每条命令都会打印一句
  `bash: /root/spur/.bash_profile: Permission denied`；现在不再出现，出现了也无害可忽略。）
- 用 docker 前先给它一个可写的配置目录：

  ```bash
  mkdir -p /tmp/.docker; export DOCKER_CONFIG=/tmp/.docker
  ```

健康自检（GPU 是 AMD ROCm，**没有 `nvidia-smi`**）：

```bash
spur exec --controller "$SPUR" <JobID> bash -c \
  'hostname; nproc; uptime; free -h; rocm-smi'
```

`ps` 实测现在可以直接用；如果报 `Error, do this: mount -t proc proc /proc`，
就先 `mount -t proc proc /proc` 再 `ps`。

❌ **多节点作业在 v2 上未实测。** 老集群上 `spur exec` 只能进 head 节点、其余节点要用
`srun --overlap` 包一层，v2 上这条路没验证过。

---

## 6. 续期：原地改 `TimeLimit`，不要换作业

作业快到期时，**一条命令就能延长，节点一秒都不脱手**，不需要新作业、不需要 `scancel`：

```bash
spur scontrol update --controller "$SPUR" JobId=<JobID> TimeLimit=30-00:00:00
squeue --controller "$SPUR" -u "$USER" -o '%.8i %.12L %.14l'   # 确认真的生效了
```

实测（某作业剩 2 小时 14 分时执行）：`TimeLimit` 从 `7-00:00:00` 变成 `30-00:00:00`，
`EndTime` 往后推了 23 天，作业继续 `RUNNING`。**这是续期的首选。**

- ⚠️ **本地敲 `scontrol` 是 `command not found`。** 装的是 `spur`，Slurm 命令是它的别名，
  必须走 `spur scontrol ...`（`spur --help` 末尾列着这批别名）。
- ⚠️ **抬高时限通常需要管理员权限，别假设它一直好用。** 实测无 account 的作业也改成了，
  原因未查明。**每次改完都用 `squeue` 确认 `TIME_LIMIT`/`TIME_LEFT`，别改完就走。**
- ⚠️ **只往大改。** 往小改会立刻缩短作业寿命，且没有撤销。

### 要换节点：先提交新的，再取消旧的

**⚠️ 这个顺序不能反。** 先 `scancel` 再提交，中间那段空窗期会把节点暴露出去，而 v2 上空闲
节点约 100 秒就可能被别的团队抢走（§2）。正确做法是让新作业先排到目标节点上，**然后**取消
旧的——实测这样接管 4 秒内完成，节点一次都没有离开手里。

⚠️ **别把换节点当成续期的办法。** 独占节点经常全 `alloc`，这时候没有节点可换；重新排队还要
跟 `Priority=11000` 的作业抢（无 account 作业是 1000，见 §3）。而且 `/mnt/m2m_nobackup` 是
node-local，换节点等于重下模型重建产物。

---

## 7. 释放

**任务模式（§3.1）不需要做任何事**——脚本退出就自动释放了。以下只针对占位模式：

```bash
scancel --controller "$SPUR" <JobID>     # 或 spur cancel --controller "$SPUR" <JobID>
```

- ⚠️ **只有作业属主本人、或明确被要求时才取消作业。** 别人给你的 JobID 默认是「借你用」的，
  不要顺手 `scancel`。
- **释放前先清容器**，否则下一个人会以为卡还被占着：

  ```bash
  docker rm -f <name>       # 然后用宿主机 rocm-smi 确认显存回到约 298 MB
  ```

  这条真踩过：任务跑完后每卡仍占约 90.9 GB，会把下一个人的 KV cache 预算压低。
- 自己申请的占位作业用完请及时取消，别长期占着 GPU。

---

## 8. 存储：哪些跨节点，哪些不跨

- ✅ **家目录 / 共享 NFS 与老集群是同一份**，实测文件逐个对得上。**环境不用在 v2 上重建**，
  登录节点上的 checkout 直接可见。**任务脚本、日志、checkpoint、产物都放这里。**
- ⚠️ **`/mnt/m2m_nobackup` 是 node-local**（docker 根目录在这），换节点就要重建产物。
- ⚠️⚠️ **`/tmp` 是 node-local**，而且计算节点的 `/tmp` 和登录节点的 `/tmp` 是两个盘，
  路径字符串一样容易骗人。**别用 `/tmp` 传递任何你之后还要看的东西。**
- ⚠️ **node-local 盘不保证跨作业存活，不只是跨节点。** 见过 `/mnt/m2m_nobackup` 被清过一次，
  容器和几十 GB 权重全没了，而共享目录里的代码和日志都在。

---

## 9. 可靠性：作业会提前消失

⚠️ **别信 `EndTime`。** 见过作业声称 5 天后到期，实际提前 5 天就没了，日志断在中途且无报错。
**每次动手前先 `squeue --controller "$SPUR" -u "$USER"` 确认作业还在。**

❌ **抢占行为未实测。** v2 上两个 QOS 的 `PreemptMode` 都是 `off`，但无 account 作业
**不走那两个 QOS**，所以**不要读成「v2 上不会被抢占」**。
**长任务一律自己存 checkpoint，并放在共享存储上（§8）。**

---

## 10. 故障速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `sinfo \| grep v2-0` 数出 0 台，像是「v2 没有机器」 | controller 指向了老集群（默认就是老集群；`bash -l`/`bash -lc` 也可能把环境变量改回去） | 用 `--controller "$SPUR"`；见 §1 |
| `failed to connect to spurctld ... Connection refused` | 地址指向了 `-v2-005` 等（v2 只有 `-v2-001` 通），或没设 controller | 固定用 `http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817`；见 §1 |
| `job submission failed ... has no account associations` | **指定了 `-A` 就必须有 association**，而 v2 用户表是老集群真子集 | **去掉 `-A` 重新提交**（代价是 `Priority=1000`）；见 §3 |
| 提交后一直 `PD (ReqNodeNotAvail, Reserved for Kubernetes cluster)` | 节点有**不显示在 `spur show node` 里的预留**（`sinfo` 仍是 `IDLE`） | 换节点；`045`–`049` 在 `k0s` 分区、`050` 被 k8s 预留；见 §2 |
| 拿到 8 卡了，但节点是 `mix` 不是 `alloc`，别人的 CPU 作业还能落进来 | 提交时**漏了 `--exclusive`** | 占位/训练作业一律加 `--exclusive`；见 §3 |
| `spur show job` 里 `Exclusive=0`，怀疑没拿到整机 | **这个字段不可信** | 看 `spur show node` 的 `CPUAlloc` 是否 = `CPUTot`，或 `sinfo` 是 `alloc`；见 §3 |
| 任务跑得极慢，dataloader 卡住 | **没写 `-c`**，作业脚本被 cgroup 限成 1 核（实测 `-c1`→`nproc=1`） | 显式写 `-c <核数>`；不要用 `spur exec` 看到的 `nproc=236` 去推断；见 §3.1 |
| `-o` 指定的日志文件在登录节点找不到 | 写到了**计算节点本地** `/tmp` | `-o`/`-e` 写共享路径，配 `%j`；见 §3.1 / §8 |
| 任务跑到一半报文件不存在 | 脚本引用的路径是 node-local（脚本本身会被分发，引用的文件不会） | 数据/依赖放共享存储；见 §3.1 / §8 |
| 任务模式的作业跑完了，不知道成没成 | 作业已自动离队，`squeue` 里当然没有 | `sacct --controller "$SPUR"` 看 `State` + `ExitCode`；见 §3.1 |
| 作业快到期，想延长 | `-t` 到点就被杀 | `spur scontrol update JobId=<id> TimeLimit=30-00:00:00`，**不用换作业**；见 §6 |
| `scontrol: command not found` | 装的是 `spur`，Slurm 命令是它的别名 | 走 `spur scontrol ...`；见 §6 |
| `sbatch --test-only` 报 `unexpected argument` | `spur` 的 `sbatch` 不支持 | 没有试探性提交。先看节点状态再真提交，提交后确认变 `R`；见 §3 |
| 节点 DNS 能解析、22 端口开着，但 `sinfo` 查无此节点 | **机器还没注册进调度器** | 只认 `sinfo -n` / `spur show node`；见 §3 |
| `ssh crs-m2m-cpu-spur-v2-00X` / `ssh crsuse2-m2m-v2-XXX` → `Permission denied` | v2 的**登录节点和计算节点都拒绝 SSH** | 别登 v2，用 `spur exec`；见 §5 |
| `spur exec` 报 `cannot exec into job owned by <别人>` | 按**作业属主**鉴权 | 只能进自己的作业，被占着只能等；见 §5 |
| `spur exec <JobID> bash` 卡住没提示符 | 不分配 tty | 用 `bash -c "..."`；要交互找真人 `spur alloc`；见 §5 |
| 容器里 `$HOME` 不对 / 相对路径找不到文件 | 容器内 `HOME=/opt/spur`、`PWD=/` | 脚本一律写绝对路径；见 §5 |
| `ps` 报 `Error, do this: mount -t proc proc /proc` | `/proc` 未挂载 | 先 `mount -t proc proc /proc`；见 §5 |
| `nvidia-smi: command not found` | 这是 AMD ROCm 平台 | 用 `rocm-smi`；见 §5 |
| 作业跑着突然消失、日志断在中途且无报错 | 作业**会提前消失**；抢占行为未实测 | 每次动手前 `squeue` 确认；长任务存 checkpoint 到共享存储；见 §9 |
| 换了节点后产物都不见了 | `/mnt/m2m_nobackup` 是 **node-local** | 要跨节点看见的放共享存储；见 §8 |

---

## 11. 速查卡

```bash
SPUR=http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817
SHARED=$HOME/jobs   # 换成你的共享目录
```

**情形 A：别人给了你 JobID（最常见）**

1. `spur show job --controller "$SPUR" <JobID>` 确认 `RUNNING`
   （不是就报告，别换作业）
2. `spur exec --controller "$SPUR" <JobID> bash -c '...'` 干活，**路径全写绝对路径**（§5）
3. **不要 `scancel`**，除非属主明确要求（§0 / §7）

**情形 B：跑一个任务，跑完自动释放（推荐）**

1. 任务脚本和数据放共享目录（§8）
2. 挑节点：`sinfo --controller "$SPUR" -N -o '%N|%t|%G' | grep -E 'v2-0'`（§2）
   ——**看到空的尽早占，约 100 秒就可能被抢走**
3. 提交：

   ```bash
   sbatch --controller "$SPUR" --parsable -J "$USER-job" -p default \
     -N1 -w <node> --gpus-per-node=8 --exclusive -c 64 -t 1-00:00:00 \
     -o "$SHARED/logs/%j.out" -e "$SHARED/logs/%j.err" "$SHARED/task.sh"
   ```

   ——⚠️ **不带 `-A`**、⚠️ **别漏 `-c` 和 `--exclusive`**、⚠️ **日志写共享路径**（§3.1）
4. `squeue --controller "$SPUR" -u "$USER"` 确认变成 `R`（不是 `PD`）
5. 跑完作业**自动离队，节点自动释放**；用
   `sacct --controller "$SPUR" | grep <JobID>` 看 `COMPLETED` + `0:0`（§3.1）

**情形 C：长期持有一台机器（要自己释放）**

1. 同上第 2 步挑节点
2. `sbatch --controller "$SPUR" --parsable -J "$USER-hold-<节点号>" -p default -N1 -w <node> --gpus-per-node=8 --exclusive -t 7-00:00:00 --wrap "sleep 604800"`（§3.2）
3. `spur exec --controller "$SPUR" <JobID> bash -c '...'` 干活（§5）
4. 快到期了：`spur scontrol update --controller "$SPUR" JobId=<JobID> TimeLimit=30-00:00:00`，
   然后 `squeue` 确认生效（§6）
5. 用完：先 `docker rm -f <name>` 清容器，再
   `scancel --controller "$SPUR" <JobID>`（§7）
