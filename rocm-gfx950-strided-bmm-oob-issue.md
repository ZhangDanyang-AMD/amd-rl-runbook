# Issue：gfx950 / ROCm 7.0 上 `torch.bmm` 对交错批视图越界读

> 一句话：`torch.bmm` 的输入是"批间在内存里交错"的非连续视图时（batch stride 小于单个矩阵的跨度），
> 在 MI355X (gfx950) / ROCm 7.0 / torch 2.10 上会越界访问。单独跑是 `Memory access fault` 直接把进程
> 打挂，跑在模型里则是安静地产出 `inf` / `NaN`。
>
> 命中它的真实代码：`transformers` 的 `DeepseekV4GroupedLinear.forward`（DeepSeek-V4 的
> `self_attn.o_a_proj`）。**序列长度一进 2040 以上就随时可能中招**——实测 2040 / 2043 / 2044 /
> 2048 / 4096 全崩，1792 和 2052 恰好没事（取决于 kernel tile 选择，不是安全区）。
> 4k 回复长度的 RL 训练正好落在这个区间里。
>
> 修法是给 `bmm` 的两个操作数各加一个 `.contiguous()`。
>
> 本文自足：从零复现只需要一张 gfx950 卡和下面两段代码，不依赖任何 DeepSeek 权重。

---

## 0. 结论速查

| 项 | 值 |
|---|---|
| 硬件 | AMD Instinct MI355X，`gfx950` |
| 软件 | `torch 2.10.0+rocm7.0`（镜像 `vllm/vime-rocm`），ROCm 7.0 |
| 触发条件 | `torch.bmm(A, B)`，A 是 `(M, G, K).transpose(0, 1)` 这种批间交错的视图，且 `M ≳ 2040`（不是每个 `M` 都崩，见第 1 节表） |
| 单独复现的现象 | `Memory access fault by GPU node-N on address 0x...` / `HSA_STATUS_ERROR_EXCEPTION`，进程 core dump |
| 在模型里的现象 | 不崩，输出里混进 `inf` 和 `NaN`，随后污染整个前向 |
| 是否确定性 | 是。同一 `M` 重复 4 次结果一致，与输入数值内容无关 |
| 两个 BLAS 后端 | hipBLASLt 和 rocBLAS（`TORCH_BLAS_PREFER_HIPBLASLT=0`）**都中招** |
| 规避 | 两个操作数各加 `.contiguous()`，或改用 `einsum` / `matmul` |
| 规避的前提 | 用 `device_map` 加载时，补丁**必须打在 `from_pretrained` 之前**，否则静默失效，见 §4.2 |

⚠️ **复现前先看第 5 节**：崩溃会往 `$HOME` 写 GPU core dump，单个 50–80 GB，很容易把共享 NFS 写满。

---

## 1. 最小复现（不需要任何模型权重）

```python
# repro_min.py —— 约 30 行，纯 PyTorch
import os
import torch

DEV = "cuda:0"
DT = torch.bfloat16
G, K, N = 8, 4096, 1024          # DeepSeek-V4-Flash: o_groups=8, 4096 -> 1024
M = int(os.environ.get("M", "2048"))
MODE = os.environ.get("MODE", "asis")   # asis | contig

g = torch.Generator(device=DEV).manual_seed(0)
x = (torch.randn(1, M, G, K, generator=g, device=DEV, dtype=torch.float32) * 0.3).to(DT)
w = (torch.randn(G * N, K, generator=g, device=DEV, dtype=torch.float32) * 0.02).to(DT)

# 下面这四行是 transformers DeepseekV4GroupedLinear.forward 的原样搬运
wv = w.view(G, -1, K).transpose(1, 2)          # (G, K, N)，非连续
xv = x.reshape(-1, G, K).transpose(0, 1)       # (G, M, K)，非连续，batch stride = K
if MODE == "contig":
    wv, xv = wv.contiguous(), xv.contiguous()
y = torch.bmm(xv, wv).transpose(0, 1).reshape(1, M, G, N)
torch.cuda.synchronize()

ref = torch.einsum("bsgk,gnk->bsgn", x.float(), w.view(G, N, K).float())
print(f"M={M} mode={MODE} "
      f"nan={int(torch.isnan(y).sum())} inf={int(torch.isinf(y).sum())} "
      f"maxerr={(y.float() - ref).abs().max().item():.4g}")
```

上面这段是把实际跑过的 `~/dsv4/probe_rocm_grouped_bmm.py` 压缩成单文件的版本，
`bmm` 前后的四行与 transformers 里的实现逐字一致。下面表格里的数字来自那个探针。

跑法（**每个 `M` 必须单开一个进程**，因为崩溃会直接杀掉进程，串在一起会掩盖后面的用例）：

```bash
for M in 8 1024 1792 2040 2043 2044 2048 2052 4096; do
  M=$M MODE=asis  timeout 300 python3 repro_min.py || echo "M=$M CRASHED"
done
M=4096 MODE=contig python3 repro_min.py
```

### 实测结果

| `M` | `MODE=asis` | `MODE=contig` |
|---|---|---|
| 8 | ok（maxerr 0.0039） | ok |
| 1024 | ok | ok |
| 1792 | ok | — |
| 2040 | **Memory access fault** | — |
| 2043 | **Memory access fault** | — |
| 2044 | **Memory access fault** | — |
| 2048 | **Memory access fault** | ok（maxerr 0.0042） |
| 2052 | ok | — |
| 4096 | **Memory access fault** | ok（maxerr 0.0076） |
| 8192 | 未测 | ok（maxerr 0.0075） |

崩溃时的原样输出：

```
Memory access fault by GPU node-7 (Agent handle: 0x325c66f0) on address 0x7c163f00a000. Reason: Unknown.
```

偶尔换一种形态：

```
:0:rocdevice.cpp :3676: ... Callback: Queue 0x... aborting with error : HSA_STATUS_ERROR_EXCEPTION:
An HSAIL operation resulted in a hardware exception. code: 0x1016
```

⚠️ **`M=1792` 和 `M=2052` 通过不代表安全。** 越界与否取决于 kernel/tile 的选择，而 tile 选择随 `M` 变。
"我在长度 X 上试过没事"是运气，不是结论。要么加 `.contiguous()`，要么把整个区间都扫一遍。

---

## 2. 端到端复现（需要 DeepSeek-V4 权重）

如果手上有 `DeepSeek-V4-Flash-Base`（原生格式即可，`transformers ≥ 5.13.1` 能直读），可以在真实模型上看到
同一个 bug 的"安静"形态。用一个 2 层的裁剪副本就够（把 `config.json` 的 `num_hidden_layers` 改成 2，
只保留 `layers.0` / `layers.1` 和顶层张量）。

```python
# repro_model.py
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

MODEL = os.environ["DSV4_PATH"]
PATCH = os.environ.get("PATCH", "0") == "1"

if PATCH:
    def fixed(self, x):
        input_shape, hidden_dim = x.shape[:-2], x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2).contiguous()
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1).contiguous()
        return torch.bmm(x, w).transpose(0, 1).reshape(*input_shape, self.n_groups, -1)
    M.DeepseekV4GroupedLinear.forward = fixed

cfg = AutoConfig.from_pretrained(MODEL)
qc = {k: v for k, v in (getattr(cfg, "quantization_config", {}) or {}).items()
      if k not in ("quant_method", "fmt")}
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="cuda:0",
    quantization_config=FineGrainedFP8Config(dequantize=True, **qc),
).eval()

g = torch.Generator().manual_seed(0)
full = torch.randint(0, cfg.vocab_size, (1, 4096), generator=g)
for n in (1024, 2048, 4096):
    with torch.no_grad():
        lg = model(input_ids=full[:, :n].to(model.device), use_cache=True).logits.float()
    print(f"patch={PATCH} len={n} finite={bool(torch.isfinite(lg).all())} "
          f"nonfinite={int((~torch.isfinite(lg)).sum())}")
```

实测（`patch=True` 是本文原样跑出来的；`patch=False` 那三行来自同一批实验里未打补丁的
探针，输出格式略有不同，这里按同一模板重排）：

```
patch=False len=1024 finite=True                            <- 干净
patch=False len=2048 finite=False   2048/2048 个位置全是 NaN，first_bad=0
patch=False len=4096 finite=False

patch=True  len=1024 finite=True  nonfinite=0  absmax=39
patch=True  len=2048 finite=True  nonfinite=0  absmax=39
patch=True  len=4096 finite=True  nonfinite=0  absmax=41
```

注意 `first_bad=0`：连第 0 个位置都是 NaN。因果模型里 position 0 只依赖 token 0，
所以"改变序列长度会让 position 0 的输出变成 NaN"本身就说明有跨位置的内存污染，
而不是数值不稳定。

注意模型里**不崩**，只是出 `NaN`。这比崩掉更危险——训练时它会安静地毁掉一个 step。

### 全量 43 层上的确认

上面用 2 层裁剪副本是为了省时间。在完整的 `DeepSeek-V4-Flash-Base`（43 层、284B、8 卡
`device_map="auto"`、BF16 dequantize、加载约 346 s）上同样成立：

```
未打补丁          len=4096  finite=0  nonfinite=529530880     <- 4096 × 129280，整个 logits 全废
补丁在加载前打    len=8/128/512/1024/1536/2048/3072/4096 全部 finite，absmax 稳定在 31–32
```

⚠️ **补丁必须在 `from_pretrained` 之前打**，见 §4.2。

### 怎么确认 NaN 就是从这里来的

逐层打点（`hidden_states` 形状 `[B, S, hc_mult, D]`）可以看到，`len=2048` 时 layer 0 的
注意力 **内部数学是干净的**，`inf` 和 `NaN` 出现在注意力的尾巴上：

```
L0 input_ln(1, 2048, 4096)      nan=0       inf=0       absmax=0.17578
L0 attn_out(1, 2048, 4096)      nan=7964333 inf=342355  absmax=7.8092e+37   <- 这里开始坏
L0 after_attn_site(...)         nan=32096160 inf=1192032
```

而单独 hook `eager_attention_forward` 内部（`qk` / `mask` / `softmax` / `attn_out`）全部是 finite。
`attn_out` 是 `DeepseekV4Attention.forward` **模块的**返回值，也就是过了
`apply_rotary_pos_emb → o_a_proj → o_b_proj` 之后的东西。`o_a_proj` 就是 `DeepseekV4GroupedLinear`。

`absmax = 7.8e37` 紧贴 bf16 上限（3.39e38），是典型的"读到垃圾内存后 GEMM 累加溢出"。

---

## 3. 根因分析

出问题的代码（`transformers/models/deepseek_v4/modeling_deepseek_v4.py`，`DeepseekV4GroupedLinear.forward`）：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    return y.reshape(*input_shape, self.n_groups, -1)
```

关键在 `x`。`x.reshape(-1, G, K)` 得到 `(M, G, K)` 连续张量，`.transpose(0, 1)` 之后是 `(G, M, K)`，
stride 为 `(K, G*K, 1)`：

- **batch stride = `K` = 4096**
- 但单个矩阵 `(M, K)` 在内存里跨越 `M * G * K` 个元素

也就是说，8 个 batch 的数据在内存里是**交错**的，而不是一段接一段。合法但不常见的 layout。
张量总共只有 `M*G*K` 个元素，最后一个 batch 从偏移 `7*K` 开始，正常读只会读到末尾；
一旦 kernel 按"batch stride ≥ 矩阵大小"的常规假设去预取或分块，就会读到张量之外。

`w` 同样是非连续的（`(G, K, N)`，stride `(N*K, 1, K)`）。**我只对比了"两个都非连续"与"两个都连续"，
没有单独区分是哪一个操作数触发的**——这是留给上游定位的部分。

### 已排除的解释

| 假设 | 排除依据 |
|---|---|
| 普通 bf16 GEMM 在 gfx950 上有问题 | 同尺寸的 `F.linear`（K/N = 8192/4096、4096/2048、2048/4096、4096/1024、1024/32768、4096/512），M 取 1024–4096，全部干净 |
| 连续输入的 `bmm` 有问题 | `bmm` 8×(M×4096)@(4096×1024) 连续版本，M 取 1024–4096，全部干净 |
| hipBLASLt 特有 | `TORCH_BLAS_PREFER_HIPBLASLT=0` 走 rocBLAS，一样崩 |
| 数值溢出 / 输入内容 | 与 token 内容无关：random / 常量 / 真实文本三种输入结果完全一致；输入 absmax 只有个位数 |
| 不确定性 / 竞态 | 同一 `M` 重复 4 次，结果逐次一致 |
| DeepSeek 的稀疏注意力（CSA/HCA/indexer） | 只含 `sliding_attention` 层、完全没有 compressor 的 2 层模型同样中招 |
| Hyper-Connection 的批量小矩阵乘 | `[B,S,4,4] @ [B,S,4,4096]`，S 取 8–8192，bf16/fp32 都干净 |
| transformers 的权重映射错了 | key 双向差集为空，张量 absmax 与 checkpoint 一致；且最小复现完全不涉及权重 |

---

## 4. 修法

### 4.1 本地补丁（推荐，一行的事）

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2).contiguous()
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1).contiguous()
    y = torch.bmm(x, w).transpose(0, 1)
    return y.reshape(*input_shape, self.n_groups, -1)
```

运行时打补丁（不改镜像里的文件）：

```python
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
M.DeepseekV4GroupedLinear.forward = fixed
```

`w` 的 `.contiguous()` 每次前向都会拷一遍权重，值得缓存；`x` 那次拷贝是必要开销。

### 4.2 ⚠️ 用 `device_map` 加载时，补丁必须打在 `from_pretrained` 之前

这个坑很容易让人误判"补丁没用"。`device_map="auto"`（以及任何非 `None` 的 device_map）
会让 accelerate 调 `add_hook_to_module`，它的做法是：

```python
old_forward = module.forward                                  # 此刻绑定的实现被捕获
module.forward = functools.partial(new_forward, module)       # 写成【实例】属性
```

于是每个 module 实例上都多了一个自己的 `forward` 属性，**遮蔽掉类上的那个**。之后再改
`DeepseekV4GroupedLinear.forward` 就完全不生效了，而且没有任何报错或警告。

实测（完整 43 层模型，8 卡 `device_map="auto"`）：

```
instance shadows `forward`: True
  vars(instance)['forward'] = functools.partial(<function add_hook_to_module.<locals>.new_forward ...>,
                                                DeepseekV4GroupedLinear(in_features=4096, out_features=8192, bias=False))
  has accelerate hook       = True

step 1: 未打补丁                       finite=0  nonfinite=529530880
step 2: 加载【之后】打类级补丁          finite=0  nonfinite=529530880   <- 补丁是哑的
（另一次运行）加载【之前】打类级补丁     8 → 4096 全部 finite
```

所以顺序必须是：

```python
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
M.DeepseekV4GroupedLinear.forward = fixed          # 先打

model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")   # 再加载
```

FSDP2 / DDP 那种不装 accelerate hook 的路径没有这个限制，但统一按"先打补丁再加载"写不会错。

### 4.3 影响范围与自查

任何在 gfx950 上对非连续视图调 `torch.bmm` 的代码都值得查一遍。快速自查：

```python
import torch
_orig_bmm = torch.bmm

def _checked_bmm(a, b, *args, **kwargs):
    for name, t in (("A", a), ("B", b)):
        if not t.is_contiguous() and t.stride(0) < t.shape[1] * t.stride(1):
            print(f"[bmm] 交错批视图 {name}: shape={tuple(t.shape)} stride={t.stride()}")
    return _orig_bmm(a, b, *args, **kwargs)

torch.bmm = _checked_bmm
```

已确认**不**受影响的路线（它们不走 torch 的 `bmm`，各有自己的 kernel）：

- vLLM 托管 DSv4 做 rollout（`moe_backend=triton`、TP=8，4096 长度已验证）
- SGLang / miles + Megatron 那条线

---

## 5. ⚠️ 复现时的两个坑

**1）GPU core dump 会写爆 `$HOME`。** 每次 `Memory access fault` 会在**进程 cwd** 落一个
`gpucore.<pid>`，实测单个 50–80 GB。本文的复现过程一共产生了 10 个、共 215 GB，
把一个共享 NFS 卷从 997 GB 可用打到 611 GB。

可靠的规避办法是把 cwd 换到节点本地大盘（core 跟着 cwd 走，实测如此）：

```bash
mkdir -p /mnt/<node-local-scratch>/cores && cd /mnt/<node-local-scratch>/cores
python3 /path/to/repro_min.py
rm -f gpucore.*
```

`HSA_ENABLE_COREDUMP=0` 据说能直接关掉生成，但本文**没有实测过**，别指望它兜底。
跑完务必 `ls -la gpucore.*` 确认一下。

**2）一次崩溃会污染同进程后续的所有调用。** 第一次越界之后，同一进程里再跑本来正常的尺寸也会出
`NaN`。所以扫描 `M` 时**一个尺寸一个进程**，否则会得出"所有尺寸都坏"的错误结论。
同理，多个尺寸并行扫时给每个进程分不同的 GPU。

---

## 6. 上游报告要点

如果要提给 PyTorch 或 ROCm：

- **标题**：`torch.bmm` reads out of bounds on gfx950 (ROCm 7.0) when batch dim comes from a
  transposed view with batch stride smaller than the matrix extent
- **复现**：第 1 节的 `repro_min.py`，无外部依赖
- **关键信息**：`A` 的 stride 是 `(4096, 32768, 1)`、shape `(8, M, 4096)`；`M ≥ 2040` 触发；
  hipBLASLt 与 rocBLAS 均复现；`.contiguous()` 后消失；等价的 `einsum` 结果正确
- **待补**：单独区分 A / B 哪一个是触发方；`M` 的确切边界（本文只扫到 1792 ok / 2040 崩）；
  fp32 与 fp16 下是否同样（本文只测了 bf16）；其他 `G` / `K` / `N` 组合
- **上游位置**：`transformers` 的 `DeepseekV4GroupedLinear` 是现成的受害者，
  V4-Flash（`o_groups=8`）和 V4-Pro（`o_groups=16`）都走这条路径

---

## 7. 复现环境与原始日志

发现于 4 节点 32 卡 MI355X（`crsuse2-m2m-[008,068,100,204]`，spur JobID 38564，已结束），
在另一批节点上复现并做了全量 43 层确认（`crsuse2-m2m-[029,172,206,272]`，JobID 43970）——
**换机重现过，不是单台机器的偶发**。拿新机器的办法见 `SPUR_NODE_ACCESS_GUIDE.md`；
容器与环境脚本见 `dapo-lumenrl-4node-32gpu-runbook.md`。镜像 `vllm/vime-rocm` 里的实测版本：

```
torch 2.10.0+rocm7.0 · transformers 5.13.1 · vllm 0.22.1rc1.dev392+g43914dd74.rocm702
```

当时的探针与日志（在 NFS 上，节点没了也还在）：

| 内容 | 位置 |
|---|---|
| 最小复现 | `~/dsv4/probe_rocm_grouped_bmm.py`（`BMM_MODE=asis\|contig`、`BMM_SEQS=...`） |
| 补丁端到端验证（裁剪模型） | `~/dsv4/probe_dsv4_grouped_fix.py` |
| 全量 43 层长度扫描 | `~/dsv4/probe_dsv4_full_sweep.py` |
| 补丁时机（accelerate hook） | `~/dsv4/probe_dsv4_patch_timing.py` |
| 逐层打点 | `~/dsv4/probe_dsv4_layer_trace.py` |
| 注意力内部打点 | `~/dsv4/probe_dsv4_attn_trace.py` |
| 排除项：普通 GEMM / HC 批量乘 | `~/dsv4/probe_rocm_gemm_nan.py`、`probe_rocm_hc_matmul.py` |
| 日志 | `~/logs/4node/dsv4_grouped_bmm.log`、`dsv4_bmm_threshold.log`、`dsv4_bmm_fix.log`、`dsv4_layer_trace.log`、`dsv4_gemm_nan.log`、`dsv4_hc_matmul.log`、`dsv4_full43_sweep.log`、`dsv4_patch_timing.log` |

相关文档：`deepseek-v4-lumenrl-fsdp-handoff.md`（DSv4 接入 LumenRL 的主线，这个 bug 是它路上的一块石头）。
