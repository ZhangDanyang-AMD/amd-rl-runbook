# Kimi K3 DSpark Draft Distillation — LumenRL + vLLM (MI350, 8×GPU Sequential)

## Overview

Train a DSpark speculative decoding draft model for Kimi K3 using LumenRL with vLLM teacher inference on 8× AMD MI350 GPUs.

**Key difference from prior SDDD recipes (GPT-OSS-120B, Kimi-K2.5):** K3's ~1TB MoE model (93 layers, 896 experts, hidden_size=7168) requires TP=8 for inference. This means we cannot use the 4+4 GPU split (4 train / 4 infer) — instead we use **8-GPU sequential: infer all → train all**.

| Property | Value |
|----------|-------|
| Target model | moonshotai/Kimi-K3 (93 layers, 896 experts, 7168 hidden, MLA+KDA) |
| Draft model | Inferact/Kimi-K3-DSpark (5 layers, MLA-only, Markov+confidence heads) |
| Draft type | `dspark` (NOT eagle3 — different architecture and loss) |
| Hardware | 8× AMD MI350 (gfx950, 288GB HBM per GPU) |
| GPU split | Sequential: TP=8 inference → 8-GPU FSDP2 training |
| Inference | vLLM 0.19.1 + extract_hidden_states patches, TP=8, MXFP4 MoE |
| Training | LumenRL FSDP2, BF16, Lumen+AITER kernels |
| Transfer | Mooncake TCP (hidden states from infer phase written to /dev/shm) |
| Dataset | lightseekorg/kimi-mtp-dataset + open-perfectblend (on-policy regeneration) |

## Architecture

### Sequential 8-GPU Workflow

Unlike the 4+4 disaggregated pipeline used for smaller models, K3 requires all 8 GPUs for inference (TP=8). The workflow is:

```
┌─────────────────────────────────────────────────────────┐
│  Phase A: Prefill / Generate + Extract Hidden States    │
│  ─────────────────────────────────────────────────────  │
│  GPUs 0-7: vLLM (TP=8, MXFP4 MoE, FP8 KV cache)      │
│    → For each batch of prompts:                         │
│      1. Generate responses (greedy, max 8192 tokens)    │
│      2. Re-prefill full sequence (prompt + response)    │
│      3. Extract hidden states at layers [2,23,47,71,89] │
│      4. Extract last hidden state (layer 93, post-norm) │
│      5. Write to /dev/shm (SHM files, not Mooncake)    │
│    → Repeat until all dataset batches processed         │
│    → Shutdown vLLM, release GPU memory                  │
│                                                         │
│  Phase B: Train DSpark Draft Model                      │
│  ─────────────────────────────────────────────────────  │
│  GPUs 0-7: torchrun FSDP2 (8 processes)                │
│    → Load cached hidden states from /dev/shm            │
│    → Train DSpark: CE + TV + BCE losses                 │
│    → Markov head + confidence head                      │
│    → Save checkpoint                                    │
│    → Repeat for N epochs                                │
└─────────────────────────────────────────────────────────┘
```

### Target Model: Kimi K3

| Parameter | Value |
|-----------|-------|
| model_type | kimi_k3 (wraps kimi_linear text backbone) |
| hidden_size | 7168 |
| intermediate_size | 33792 |
| num_hidden_layers | 93 |
| num_attention_heads | 96 |
| num_key_value_heads | 96 |
| vocab_size | 163840 |
| max_position_embeddings | 1048576 (1M tokens) |
| hidden_act | situ (SiTU-GLU) |
| MoE: num_experts | 896 |
| MoE: experts_per_token | 16 |
| MoE: shared_experts | 2 |
| MoE: expert_hidden_size | 3584 (routed), 3072 (shared) |
| MLA: q_lora_rank | 1536 |
| MLA: kv_lora_rank | 512 |
| MLA: qk_nope_head_dim | 128 |
| MLA: qk_rope_head_dim | 64 |
| MLA: v_head_dim | 128 |
| Hybrid attention | KDA×3 + MLA×1 repeating (69 KDA + 24 MLA layers) |
| Quantization | MXFP4 compressed-tensors (routed experts only) |
| BOS/EOS/PAD | 163584 / 163586 / 163839 |

### Draft Model: DSpark (Inferact/Kimi-K3-DSpark)

| Parameter | Value |
|-----------|-------|
| model_type | k3_dspark |
| architectures | K3DSparkModel |
| hidden_size | 7168 (matches target) |
| intermediate_size | 14336 |
| num_hidden_layers | 5 (draft backbone) |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| vocab_size / draft_vocab_size | 163840 |
| attention_type | MLA (dual-source KV, low-rank Q/KV compression) |
| mla_use_nope | false |
| mla_use_output_gate | false |
| target_layer_ids | [2, 23, 47, 71, 89] |
| num_target_layers | 5 |
| markov_rank | 256 |
| markov_head_type | vanilla |
| enable_confidence_head | true |
| confidence_head_with_markov | true |
| rope_type | yarn |
| rope_theta | 50000.0 |
| rope_scaling_factor | 32.0 |
| original_max_position_embeddings | 32768 |
| beta_fast / beta_slow | 32 / 1 |
| mscale / mscale_all_dim | 1.0 / 1.0 |
| Model size | ~7.12 GB (bf16) |
| torch_dtype | bfloat16 |

### DSpark vs Eagle3: Key Differences

| Dimension | Eagle3 (GPT-OSS / Kimi-K2.5) | DSpark (Kimi K3) |
|-----------|-------------------------------|-------------------|
| Backbone | 1 layer autoregressive | 5 layers parallel |
| Attention type | GQA (standard Q/K/V) | MLA (low-rank Q/KV compression) |
| KV source | Self-attention only | Dual-source KV (context + draft) |
| Target layers | 3 aux (early/mid/late) | 5 aux ([2,23,47,71,89]) |
| Markov head | None | VanillaMarkov (rank=256) |
| Confidence head | None | AcceptRatePredictor (BCE) |
| Loss | Forward KL (soft CE) | CE (0.1) + TV (0.9) + BCE (1.0) |
| Block size | N/A (autoregressive TTT) | 7 tokens (parallel) |
| Anchors | N/A | 512 per sequence |
| Frozen components | embed_tokens, lm_head | embed_tokens, lm_head |

### GQA vs MLA: Training vs Inference Architecture

**Training uses MLA (Multi-head Latent Attention):**

LumenRL's DSpark draft model uses MLA with dual-source KV attention, matching the HuggingFace `Inferact/Kimi-K3-DSpark` config.json:

| MLA Parameter | Value | Description |
|---------------|-------|-------------|
| q_lora_rank | 1536 | Q low-rank compression dimension |
| kv_lora_rank | 512 | KV low-rank compression dimension |
| qk_nope_head_dim | 128 | Q/K non-positional head dimension |
| qk_rope_head_dim | 64 | Q/K rotary positional head dimension |
| v_head_dim | 128 | V head dimension |

MLA Q projection path:
```
hidden [B,T,7168] → q_a_proj [7168,1536] → LayerNorm → q_b_proj [1536,64×192] → split(nope=128, rope=64)
```

MLA KV projection path (shared between context and draft):
```
hidden [B,T,7168] → kv_a_proj_with_mqa [7168,576] → split(compressed=512, k_rope=64)
                                                          ↓
                                                     LayerNorm
                                                          ↓
                                                   kv_b_proj [512,64×256] → split(k_nope=128, v=128)
```

Dual-source KV flow:
```
Q: draft-only query                                    → [B, 64, block_len, 192]
K: cat(context_K_nope, draft_K_nope) + cat(ctx_rope, draft_rope) → [B, 64, ctx+block, 192]
V: cat(context_V, draft_V)                             → [B, 64, ctx+block, 128]
```

**TorchSpec reference uses GQA (Grouped-Query Attention):**

TorchSpec's DFlash/DSpark uses standard GQA projections (`q_proj`, `k_proj`, `v_proj`), not MLA. The GQA implementation is simpler but produces different weight tensors.

**Why this difference doesn't break alignment:**

1. **Weight export is MLA-native**: The export script (`export_dspark_hf.py`) maps LumenRL's MLA weight keys directly to HF `Inferact/Kimi-K3-DSpark` keys. No conversion needed.
2. **vLLM loads MLA weights**: vLLM's `K3DSparkModel` speculative decoder expects MLA-format weights (q_a_proj, kv_a_proj_with_mqa, etc.), not GQA weights.
3. **The HF reference model is MLA**: `Inferact/Kimi-K3-DSpark` config.json specifies MLA dims. TorchSpec uses GQA internally for training but must convert to MLA for deployment — our approach skips that conversion step.
4. **Mathematically equivalent**: Both GQA and MLA can represent the same function class for this draft model size. The trained model's predictive capacity is not limited by the projection choice.

## Code Versions & Dependencies

### Git Commits

| Component | Branch/Commit | Notes |
|-----------|--------------|-------|
| LumenRL | `main` @ HEAD | Training framework |
| Lumen | `third_party/Lumen` @ HEAD | AITER-patched FSDP2, BF16 kernels |
| AITER | `third_party/aiter` @ HEAD | GPU kernels (gfx950 ASM, CK, Triton) |
| ATOM | `third_party/ATOM` @ HEAD | vLLM plugin for MXFP4 online quant |
| vLLM | v0.19.1 + patches | `extract_hidden_states` + KDA support |
| Mooncake | `f2853a80` | Transfer engine (source-built HIP, TCP) |
| triton_kernels | ROCm/triton release/internal/3.5.x | MoE routing/swiglu/matmul_ogs |

### Docker Image

Base: `vllm/vllm-openai-rocm:v0.19.1`

Build command:
```bash
cd /home/danyzhan/Lumen-RL
docker buildx build \
    -f examples/Kimi_K3_SDDD_MI350_vllm/Dockerfile.train \
    -t kimi_k3_dspark_vllm_train:latest .
```

Image contents:
- vLLM 0.19.1 with `extract_hidden_states` + KDA hybrid attention patches
- ATOM (vLLM plugin, MXFP4 online quantization)
- AITER (gfx950 GPU kernels)
- Lumen (FSDP2 training with AITER)
- LumenRL (training framework)
- Mooncake Transfer Engine (source-built HIP, TCP)
- triton_kernels (ROCm MoE routing)

### Python Dependencies

```
omegaconf ray accelerate wandb pyzmq qwen-vl-utils
datasets transformers huggingface_hub safetensors pydantic openai numba
```

## Dataset & Data Processing

### Training Data

Aligned with Inferact/Kimi-K3-DSpark training recipe:

| Dataset | Source | Samples | Categories |
|---------|--------|---------|------------|
| lightseekorg/kimi-mtp-dataset | HuggingFace | ~477K | perfectblend + mixed (VL, CN, tool, agent, writing) |
| CohereLabs/aya_dataset | HuggingFace | ~50K | Multilingual |
| Nemotron SFT/RL | NVIDIA | ~100K | Chat, code, math, structured output, safety |

**Total: ~600K+ prompts** (responses regenerated on-policy by K3)

### Data Processing Pipeline

1. **Download prompts**:
   ```bash
   huggingface-cli download lightseekorg/kimi-mtp-dataset --local-dir /dev/shm/kimi-mtp-dataset
   ```

2. **Split into subsets** (if using two-phase training):
   ```bash
   python examples/Kimi_K3_SDDD_MI350_vllm/split_dataset.py
   ```
   - Phase 1 (perfectblend): ~296K samples — foundation training
   - Phase 2 (mixed): ~181K samples — domain diversity

3. **On-policy response generation**:
   - vLLM serves K3 with TP=8 on all 8 GPUs
   - Greedy decode (temperature=0), max 8192 tokens
   - Responses are K3's own outputs (non-thinking mode)

4. **Hidden state extraction** (during prefill):
   - vLLM `extract_hidden_states` speculative mode captures:
     - `aux_hidden_states`: layers [2, 23, 47, 71, 89] — 5 layers × 7168 = 35840 per token
     - `last_hidden_states`: post-norm output of layer 93 — 7168 per token
   - Stored to /dev/shm as `.bin` files (mmap-backed)

5. **Chat template**: Kimi-K3 uses the same tokenizer as K2.5 but with extended special tokens:
   - `mask_token_id`: 163837 (for DSpark block masking)
   - Chat template: `kimi-k3` (extends kimi-k25 with KDA-specific handling)

### DSpark-Specific Data Processing

Unlike Eagle3 (which processes full sequences), DSpark uses **anchor-based block training with dual-source KV**:

1. For each sequence, randomly sample `num_anchors=512` positions as anchors
2. At each anchor, construct a block of `block_size=7` positions:
   - Position 0: anchor token embedding (or mask_token for invalid anchors)
   - Positions 1-6: mask token embedding (id=163837)
3. **Dual-source KV attention**: draft queries attend to BOTH:
   - Context: full-sequence target hidden states (from all positions before anchor)
   - Self: draft block tokens (bidirectional within block, no cross-block)
4. Target hidden states from the 5 teacher layers provide auxiliary features (fused via `fc` linear + RMSNorm)
5. Loss is computed only on assistant-turn tokens (loss mask), with contiguous prefix truncation via `cumprod`

## Training Configuration

### Hyperparameters (aligned with Inferact/Kimi-K3-DSpark)

| Parameter | Value | Source |
|-----------|-------|--------|
| block_size | 7 | DSpark config |
| num_draft_layers | 5 | config.json |
| target_layer_ids | [2, 23, 47, 71, 89] | config.json |
| num_anchors | 512 | DSpark default |
| markov_rank | 256 | config.json |
| markov_head_type | vanilla | config.json |
| enable_confidence_head | true | config.json |
| confidence_head_with_markov | true | config.json |
| learning_rate | 6e-4 | DSpark train config |
| warmup_ratio | 0.04 | DSpark train config |
| weight_decay | 0.0 | DSpark train config |
| max_grad_norm | 0.5 | TorchSpec default, aligned |
| lr_decay_style | cosine | DSpark train config |
| train_global_batch_size | 512 | DSpark train config |
| train_micro_batch_size | 1 | Memory-limited |
| precision | bf16 | DSpark train config |
| num_train_epochs | 2 | DSpark model card |
| loss: ce_alpha | 0.1 | DSpark loss config |
| loss: l1_alpha (TV) | 0.9 | DSpark loss config |
| loss: confidence_alpha | 1.0 | DSpark loss config |
| loss: decay_gamma | 4.0 | DSpark loss config |

### Loss Function

**Total loss: L = 0.1 × L_ce + 0.9 × L_tv + 1.0 × L_conf**

Each loss component is weighted by position decay: `w_k = exp(-k / γ)` where γ=4.0.

1. **Cross-Entropy Loss (α=0.1)**: `L_ce = Σ_k w_k × CE(draft_logits_k, target_token_k)`
2. **Total Variation Loss (α=0.9)**: `L_tv = Σ_k w_k × |softmax(draft_k) - softmax(target_k)|₁`
   - target_logits reconstructed via `lm_head(last_hidden_states)` — no need for teacher online
3. **Confidence Loss (α=1.0)**: `L_conf = Σ_k w_k × BCE(confidence_k, acceptance_rate_k)`
   - acceptance_rate = `1 - 0.5 × |p_draft - p_target|₁`

### Sequential Execution Config

```yaml
# Key change: 8-GPU sequential mode
cluster:
  gpus_per_node: 8      # All 8 GPUs for both phases
  num_nodes: 1

algorithm:
  teacher:
    inference_backend: vllm
    tensor_parallel_size: 8    # TP=8 (full node)
    gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
    generate_mode: generate    # On-policy: generate then extract
    generate_max_tokens: 8192
    generate_temperature: 0.0
```

The trainer handles sequential mode automatically:
1. Launches vLLM subprocess with TP=8 on all GPUs
2. Processes all dataset batches → writes hidden states to /dev/shm
3. Shuts down vLLM, releases GPU memory
4. Launches torchrun with 8 processes on all GPUs
5. Trains from cached hidden states

## vLLM Inference for Kimi K3 (TP=8)

### Why TP=8?

K3 has 93 layers × 7168 hidden_size with 896 experts. In BF16:
- Embedding + LM head: ~2.3 GB
- MLA attention params (KDA + gated): ~40 GB
- MoE experts (896 × 3584 × 3072 × 2): ~38 GB per layer → compressed with MXFP4 to ~15 GB per layer
- **Total with MXFP4**: ~900+ GB → requires 8× MI350 (288 GB each)

### vLLM Configuration

```python
engine_kwargs = dict(
    model="moonshotai/Kimi-K3",
    tensor_parallel_size=8,
    trust_remote_code=True,
    distributed_executor_backend="mp",
    max_model_len=20000,
    max_num_batched_tokens=40000,
    max_num_seqs=128,
    gpu_memory_utilization=0.90,
    kv_cache_dtype="fp8_e4m3",
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [2, 23, 47, 71, 89]
            }
        },
    },
)
```

### KDA Hybrid Attention Support

K3 uses KDA (Key-Decoupled Attention) for 69 of 93 layers. vLLM handles this via:
- **FlashKDA** for prefill (CK kernel)
- Fused CUDA/HIP kernel for decode
- Hybrid KV cache: paged blocks for MLA layers + compact recurrent-state blocks for KDA layers
- Patches in `docker/patches/vllm/v0.19.1/` enable this

### Hidden State Extraction

vLLM's `extract_hidden_states` mode hooks into the model's forward pass:
1. After each specified layer, captures the hidden state tensor
2. For DSpark, captures 5 intermediate layers + 1 final (post-norm)
3. Hidden states are stored via `MooncakeHiddenStatesConnector` (TCP)
4. In sequential mode: written to /dev/shm as mmap files instead

Output per sequence:
| Tensor | Shape | Dtype | Size per 8K-token seq |
|--------|-------|-------|-----------------------|
| aux_hidden_states | [seq_len, 5×7168] | bf16 | ~572 MB |
| last_hidden_states | [seq_len, 7168] | bf16 | ~114 MB |
| input_ids | [seq_len] | int64 | ~64 KB |
| attention_mask | [seq_len] | int64 | ~64 KB |
| loss_mask | [seq_len] | int64 | ~64 KB |

## Step-by-Step Execution

### Prerequisites

```bash
# 1. Download K3 model (MXFP4 quantized weights — ~500GB)
huggingface-cli download moonshotai/Kimi-K3 --local-dir /dev/shm/Kimi-K3

# 2. Download training dataset
huggingface-cli download lightseekorg/kimi-mtp-dataset --local-dir /dev/shm/kimi-mtp-dataset

# 3. Build Docker image
cd /home/danyzhan/Lumen-RL
docker buildx build \
    -f examples/Kimi_K3_SDDD_MI350_vllm/Dockerfile.train \
    -t kimi_k3_dspark_vllm_train:latest .
```

### Training Launch

```bash
# Full training (Docker)
bash examples/Kimi_K3_SDDD_MI350_vllm/run_docker.sh

# Smoke test (Docker, 5 steps)
bash examples/Kimi_K3_SDDD_MI350_vllm/run_docker.sh --smoke-test

# Bare-metal (inside container)
bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh
```

### Expected Timeline

| Phase | Duration (8×MI350) | Steps |
|-------|-------------------|-------|
| Dataset download | ~2-4 hours | — |
| Docker build | ~30 min | — |
| Inference (per epoch) | ~8-12 hours | ~600K sequences |
| Training (per epoch) | ~4-6 hours | ~1172 steps (600K/512) |
| Total (2 epochs) | ~24-36 hours | — |

## Debug & Smoke Test

### Smoke Test (5 steps, validates full pipeline)

```bash
# Inside Docker container:
bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh --smoke-test
```

What it validates:
1. vLLM launches with TP=8 and K3 model loads successfully
2. `extract_hidden_states` produces correct shapes
3. Hidden states are written to /dev/shm
4. vLLM shutdown and GPU memory release
5. Draft model (DSpark) builds correctly
6. FSDP2 wrapping works across 8 GPUs
7. Forward pass produces valid losses (CE + TV + BCE)
8. Backward pass and optimizer step succeed
9. Checkpoint save/load works

### Debug Checklist

#### vLLM Won't Start
```bash
# Check GPU visibility
rocm-smi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -c "import torch; print(torch.cuda.device_count())"

# Test vLLM standalone (no LumenRL)
python -c "
from vllm import LLM
llm = LLM(model='/dev/shm/Kimi-K3', tensor_parallel_size=8, trust_remote_code=True, max_model_len=2048)
print('vLLM OK')
"
```

#### OOM During Inference
```bash
# Reduce batch size and max_model_len
# In config: gpu_memory_utilization: 0.85 (from 0.90)
# In config: max_num_batched_tokens: 20000 (from 40000)
```

#### Hidden State Shape Mismatch
```bash
# Verify extracted shapes
python -c "
import torch, os
slot_dir = '/dev/shm/_teacher_0'
meta = torch.load(f'{slot_dir}/meta.pt', weights_only=True)
for k, v in meta.items():
    print(f'{k}: {v}')
# Expected: aux_hidden_states: [batch, seq_len, 35840]  (5×7168)
#           last_hidden_states: [batch, seq_len, 7168]
"
```

#### Training Loss NaN/Inf
```bash
# Check gradient norms
export LUMENRL_LOG_LEVEL=DEBUG
# Look for: "grad_norm=..." in logs
# If grad_norm is inf: reduce learning_rate or max_grad_norm

# Verify lm_head loaded correctly
python -c "
import torch
w = torch.load('/dev/shm/lumenrl_vllm_lm_head_*.pt')
print(f'lm_head shape: {w.shape}, dtype: {w.dtype}')
print(f'range: [{w.min():.4f}, {w.max():.4f}]')
"
```

#### AITER JIT Deadlock
```bash
# Clear stale build artifacts
find /tmp -name '*.lock' -path '*aiter*' -delete
rm -rf /root/aiter/aiter/jit/build
```

#### Mooncake TCP Connection Failure
```bash
# Check if mooncake_master is running
pgrep -f mooncake_master

# Verify TCP connectivity
ss -tlnp | grep -E '(1234[0-9]|8080)'

# Force TCP mode (not RDMA)
export MOONCAKE_PROTOCOL=tcp
```

### Log Analysis

Key log patterns to watch:

```bash
# Successful teacher ready
grep "status.*ready" $LOG_FILE

# Step progress
grep "callbacks: step=" $LOG_FILE | tail -5

# Loss values (should decrease)
grep "loss=" $LOG_FILE | tail -10

# GPU memory usage
grep "gpu_memory" $LOG_FILE

# Completion marker
grep "SpecDistillTrainer.train finished" $LOG_FILE
```

### Monitoring GPU Health

```bash
# Real-time GPU monitoring
watch -n 2 rocm-smi

# Check for GPU errors
dmesg | grep -i "gpu\|amdgpu\|kfd" | tail -20

# VRAM usage per GPU
rocm-smi --showmeminfo vram
```

## Output & Checkpoints

| Path | Description |
|------|-------------|
| `/dev/shm/checkpoints/kimi_k3_dspark_vllm/` | Training checkpoints (draft model weights) |
| `/dev/shm/_teacher_*/` | Cached hidden states (ephemeral, deleted after training) |
| `output/Kimi_K3_SDDD/LumenRL/kimi-k3-dspark-vllm-mi350.log` | Training log |

### Checkpoint Format

```
checkpoint_NNNNN.pt
├── step: int
└── state_dict:
    ├── model_state_dict:
    │   ├── fc.weight                    # [7168, 35840] — 5-layer fusion projection
    │   ├── hidden_norm.weight           # [7168] — post-fusion RMSNorm
    │   ├── layers.{0-4}.self_attn.q_a_proj/q_a_layernorm/q_b_proj  # MLA Q path
    │   ├── layers.{0-4}.self_attn.kv_a_proj_with_mqa/kv_a_layernorm/kv_b_proj  # MLA KV path
    │   ├── layers.{0-4}.self_attn.o_proj  # Output projection
    │   ├── layers.{0-4}.mlp.*           # Draft FFN params
    │   ├── layers.{0-4}.input_layernorm.weight
    │   ├── layers.{0-4}.post_attention_layernorm.weight
    │   ├── norm.weight                  # Final RMSNorm
    │   ├── markov_head.markov_w1.weight # [163840, 256] — Markov embedding
    │   ├── markov_head.markov_w2.weight # [163840, 256] — Markov projection
    │   └── confidence_head.proj.*       # Confidence prediction head
    ├── optimizer_state_dict: ...
    └── scheduler_state_dict: ...
```

### Converting Checkpoint to HuggingFace Format

After training, convert the LumenRL checkpoint to HF-compatible safetensors:
```bash
python output/export_dspark_hf.py \
    --checkpoint /dev/shm/checkpoints/kimi_k3_dspark_vllm/checkpoint_BEST.pt \
    --target-model /dev/shm/Kimi-K3 \
    --output-dir /dev/shm/Kimi-K3-DSpark-trained

# Weight key mapping (MLA-native, no conversion needed):
#   fc.weight → fc.weight
#   layers.*.self_attn.q_a_proj.weight → model.layers.*.self_attn.q_a_proj.weight
#   layers.*.self_attn.kv_a_proj_with_mqa.weight → model.layers.*.self_attn.kv_a_proj_with_mqa.weight
#   markov_head.markov_w1.weight → model.markov_head.markov_w1.weight
#   confidence_head.proj.* → model.confidence_head.proj.*
#   + frozen embed_tokens and lm_head copied from base K3 model
```

## Batch-Alternating Architecture

The batch-alternating mode is the core innovation for K3 SDDD. Since K3 requires TP=8, all 8 GPUs alternate between teacher inference and draft training.

### Phase A/B Loop

```
for round in range(num_rounds):
    Phase A (rank 0 only):
      1. vLLM engine starts with TP=8 on GPUs 0-7
      2. For each batch (up to cache_batches=200):
         - Prefill dataset sequences
         - Extract hidden states at layers [2,23,47,71,89]
         - Write to NVMe cache: {cache_dir}/round_{r}/batch_{b}/
      3. Shutdown vLLM engine → free GPU memory
      barrier()

    Phase B (all ranks):
      1. Load draft model from CPU → GPU
      2. Re-establish composable replicate for gradient sync
      3. For each cached batch:
         - Load from disk → GPU
         - Forward + backward (DSpark loss)
         - Optimizer step
      4. Offload draft to CPU (model + optimizer + Adam state)
      5. Clean up round cache from disk
      barrier()

    Restart vLLM for next round
```

### NVMe Cache Format

Each batch stored in `{cache_dir}/round_{r}/batch_{b}/`:
- `hidden_states.bin` — `[B, T, 5*7168]` bf16 (~5.5 GB for B=8, T=8192)
- `token_embeds.bin` — `[B, T, 7168]` bf16
- `last_hidden_states.bin` — `[B, T, 7168]` bf16
- `input_ids.bin`, `attention_mask.bin`, `loss_mask.bin` — int64
- `meta.pt` — shape metadata

Total per round: ~200 batches × ~5.6 GB ≈ **1.1 TB**. Ensure cache_dir has sufficient NVMe space.

### CPU Offload Details

After Phase B, the draft model (~1.3 GB) + FP32 master params + Adam states are moved to CPU:
- `model.to("cpu")` moves all parameters
- `fp32_params`, `fp32_grads` moved to CPU
- Adam `state` dict (exp_avg, exp_avg_sq) moved to CPU
- Composable replicate hooks removed before offload

On GPU reload:
- `model.to(device)` creates new GPU tensor objects
- `self._optimizer.model_params` is rebuilt from `model.parameters()`
- FP32 params/grads/Adam state moved back to GPU
- Composable replicate re-established

**Critical:** `nn.Module.to()` creates new tensor objects. The BF16Optimizer's `model_params` list must be rebuilt to point to the new GPU-resident parameters, otherwise the optimizer step copies gradients to stale CPU references.

## Benchmarking

### Using the Benchmark Script

```bash
# Export trained checkpoint to HF format first
python examples/Kimi_K3_SDDD_MI350_vllm/export_dspark_hf.py \
    --checkpoint /dev/shm/checkpoints/kimi_k3_dspark_vllm/checkpoint_59630.pt \
    --target-model /dev/shm/Kimi-K3 \
    --output-dir /dev/shm/Kimi-K3-DSpark-trained

# Run full benchmark (DSpark speculative decoding vs baseline)
bash output/benchmark_dspark.sh \
    --target-model /dev/shm/Kimi-K3 \
    --draft-model /dev/shm/Kimi-K3-DSpark-trained \
    --tp 8

# Baseline comparison (no speculative decoding)
bash output/benchmark_dspark.sh \
    --target-model /dev/shm/Kimi-K3 \
    --baseline \
    --tp 8
```

The benchmark script runs:
1. **MT-Bench**: Multi-turn conversational quality (8 categories)
2. **SPEED-Bench throughput_16k**: Long-context throughput measurement

### Target Metrics

| Metric | Target | Source |
|--------|--------|--------|
| Mean accept length | ≥3.0 tokens | DSpark paper |
| Throughput speedup | ≥2.5x (vs no spec decode) | DSpark paper |
| Single-stream tok/s | ≥300 (on GB300) | vLLM blog |

### Interpreting Results

- **Accept length < 2.5**: Draft model underfitting — check training loss convergence, may need more steps
- **Accept length > 4.0**: Excellent — consider reducing block_size for efficiency
- **Speedup < 2x**: Draft model overhead too high — check draft latency, may need fewer layers

## Dashboard

### Setup

```bash
# One-time: verify dashboard directory exists
ls /home/danyzhan/Lumen-RL/dashboards/SDDD/Kimi_K3_SDDD_MI350/

# Manual update
bash /home/danyzhan/Lumen-RL/dashboards/SDDD/Kimi_K3_SDDD_MI350/update_dashboard.sh

# Cron setup (every 5 minutes)
crontab -e
# Add: */5 * * * * bash /home/danyzhan/Lumen-RL/dashboards/SDDD/Kimi_K3_SDDD_MI350/update_dashboard.sh >> /tmp/k3_dashboard.log 2>&1
```

### Dashboard Contents

The dashboard (`dashboard.html`) shows 10 charts:
1. **Training & Eval Loss** — composite DSpark loss over time
2. **Train Accuracy by Position (%)** — 7 positions (block_size=7), MA smoothed
3. **Accept Length (Train vs Eval)** — cumulative product of per-position accuracies
4. **DSpark Loss Components** — CE, TV, and confidence loss individually
5. **Eval Accuracy by Position (%)** — eval-time per-position accuracy
6. **Gradient Norm** — monitors training stability
7. **Step Time (ms)** — wall clock per step
8. **Train Loss by Position** — per-position loss curves
9. **Eval Loss by Position** — eval-time per-position loss
10. **Learning Rate** — cosine schedule visualization

Round boundaries (every `cache_batches=200` steps) shown as dotted vertical lines.

### Changing the Docker Container Name

Edit `container` variable at line 7 of `update_dashboard.sh`:
```python
container = "kimi_k3_dspark_v1"  # ← change this
```

### Data Persistence

Metric history persists in `data.json`. The script incrementally appends new steps, so restarting the container won't lose history. To reset, delete `data.json` and re-run.

## Troubleshooting

### vLLM Restart Failures (Between Rounds)

**Symptom:** Phase A hangs or crashes on round > 0 when vLLM engine restarts.

```bash
# Check GPU memory is actually freed
rocm-smi --showmeminfo vram

# Force GPU memory cleanup
python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Check for zombie vLLM worker processes
pgrep -af "vllm\|ray\|Worker_TP"
kill -9 $(pgrep -f "Worker_TP")  # Kill orphaned workers
```

**Root cause:** vLLM's multiproc_executor spawns grandchild processes (EngineCore, Worker_TP*) that may survive shutdown. The trainer's `VllmTeacherEngine.shutdown()` sends SIGTERM to the process group, but race conditions can leave orphans.

**Fix:** Ensure `cleanup()` is called on error paths. The Mooncake master stays alive intentionally — only the vLLM engine process is killed between rounds.

### GPU Memory Leaks Between Phases

**Symptom:** OOM on Phase B (draft training) despite draft being small (~1.3 GB).

```bash
# After Phase A shutdown, before Phase B:
rocm-smi --showmeminfo vram
# Each GPU should show < 1GB used (just PyTorch CUDA context)

# If memory is high, check for:
# 1. Leaked CUDAGraph captures
torch.cuda.empty_cache()

# 2. Undisposed Mooncake buffers
# Mooncake master stays alive but its GPU buffers should be on CPU
```

**Prevention:** The trainer calls `torch.cuda.empty_cache(); gc.collect()` after engine shutdown and before draft loading. If OOM persists, add `torch.cuda.reset_peak_memory_stats()` for diagnosis.

### Adam State Corruption After CPU Offload

**Symptom:** Loss spikes or accuracy drops sharply at round boundaries.

**Diagnosis:**
```bash
# Check if optimizer state survived offload
grep "Draft model reloaded to GPU" $LOG_FILE
grep "grad_norm=" $LOG_FILE | awk -F'grad_norm=' '{print $2}' | head -5
# Grad norm should be smooth across round boundaries
```

**Root causes:**
1. `model_params` not rebuilt after `model.to(device)` → optimizer step writes to stale CPU tensors. **Fixed in current code.**
2. Composable replicate double-wrapping → gradient all-reduce applied twice. **Fixed in current code.**
3. FP32 master params lost during offload → optimizer restarts from random. Should not happen with current implementation.

### NVMe Space Exhaustion

**Symptom:** `OSError: No space left on device` during Phase A.

```bash
# Check cache directory usage
du -sh /tmp/teacher_cache/round_*/
df -h /tmp/

# Each round uses ~1.1 TB (200 batches × ~5.6 GB)
# Only one round's cache exists at a time (cleaned after Phase B)
```

**Fix:** Ensure `cache_dir` is on a filesystem with >1.2 TB free space. If using `/tmp`, consider mounting a dedicated NVMe volume. You can also reduce `cache_batches` (e.g., from 200 to 100) at the cost of more frequent vLLM restarts.

### OOM with Large anchor_num

**Symptom:** CUDA OOM during DSpark forward pass (Phase B).

**Diagnosis:** DSpark allocates `[B, A, block_size, H]` tensors where A=anchor_num=512, block_size=7, H=7168. This is `B × 512 × 7 × 7168 × 2 bytes = B × 50 MB`. With B=8, that's 400 MB per intermediate tensor — reasonable, but multiple such tensors exist simultaneously.

**Fix:** Reduce `anchor_num` in the YAML config (e.g., from 512 to 256 or 128). This reduces memory but also reduces training signal per sequence.

## Smoke Test Validation

### Running the Smoke Test

```bash
# Inside Docker container:
bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh --smoke-test

# Uses configs/smoke_test.yaml: 5 steps, cache_batches=5
```

### Expected Output Patterns

1. **Phase A start:**
   ```
   Round 0 Phase A: prefilling 5 batches [step 0..5)
   ```

2. **vLLM ready:**
   ```
   VllmTeacherEngine ready
   ```

3. **Hidden state shapes (verify on first batch):**
   ```
   hidden_states: [8, T, 35840]  # 5 × 7168
   last_hidden_states: [8, T, 7168]
   ```

4. **Phase B start:**
   ```
   Round 0 Phase B: training on 5 cached batches
   Draft model reloaded to GPU
   ```

5. **Training metrics (first steps will have random-init loss):**
   ```
   callbacks: step=0 grad_norm=X.XX loss=X.XX lr=6e-4
     ce_loss=X.XX tv_loss=X.XX conf_loss=X.XX
     step_0_acc=0.XX ... step_6_acc=0.XX
   ```

6. **Completion:**
   ```
   Batch-alternating training finished after 5 steps.
   SpecDistillTrainer.cleanup complete.
   ```

### Validation Checks

After a successful smoke test, verify:
- [ ] All 5 steps completed without errors
- [ ] Loss values are finite (not NaN/Inf)
- [ ] Gradient norms are finite and reasonable (typically 0.1-10.0 for first steps)
- [ ] Per-position accuracy is between 0 and 1 (random init: ~1/vocab_size)
- [ ] Cache directory was cleaned up: `ls /tmp/teacher_cache_smoke/` should be empty
- [ ] GPU memory was released between phases: `rocm-smi` shows low VRAM between Phase A end and Phase B start

## Long-Running Training Monitoring

### Expected Training Curves

| Step Range | Expected Behavior |
|------------|-------------------|
| 0-1000 | Loss drops rapidly, accuracy rises from ~0% to ~20-30% at pos 0 |
| 1000-5000 | Steady improvement, pos 0 acc reaches ~40-50% |
| 5000-20000 | Gradual improvement, pos 0 acc reaches ~55-65%, diminishing returns |
| 20000-59630 | Plateau, pos 0 acc stabilizes at ~60-70%, higher positions slowly improve |

### Intervention Triggers

| Condition | Action |
|-----------|--------|
| Loss NaN for > 5 consecutive steps | Kill, reduce lr by 2x, resume from last checkpoint |
| Grad norm > 100 consistently | Reduce max_grad_norm from 0.5 to 0.25 |
| Pos 0 accuracy < 20% after 2000 steps | Check data pipeline, verify hidden state shapes |
| Pos 0 accuracy > 50% but pos 6 < 5% | Normal — higher positions are harder |
| Step time > 10s consistently | Check NVMe I/O, `iostat -x 1` |
| Accept length regression (eval) | Check for data distribution shift or optimizer state issues |
| OOM mid-training | Reduce anchor_num (512 → 256) or train_micro_batch_size |
| vLLM fails to restart on round > 0 | Kill orphaned workers, check GPU memory, restart manually |

### Checkpoint Export Workflow

When training converges (accept_len ≥ 3.0):

```bash
# 1. Find best checkpoint
ls -la /dev/shm/checkpoints/kimi_k3_dspark_vllm/

# 2. Export to HF format
python examples/Kimi_K3_SDDD_MI350_vllm/export_dspark_hf.py \
    --checkpoint /dev/shm/checkpoints/kimi_k3_dspark_vllm/checkpoint_BEST.pt \
    --target-model /dev/shm/Kimi-K3 \
    --output-dir /dev/shm/Kimi-K3-DSpark-trained

# 3. Validate with benchmark
bash output/benchmark_dspark.sh \
    --target-model /dev/shm/Kimi-K3 \
    --draft-model /dev/shm/Kimi-K3-DSpark-trained \
    --tp 8

# 4. Upload to HuggingFace (if metrics are good)
huggingface-cli upload <org>/Kimi-K3-DSpark-v1 /dev/shm/Kimi-K3-DSpark-trained
```

## Debugging Against TorchSpec Reference Implementation

TorchSpec (`/home/danyzhan/TorchSpec`) is the canonical reference implementation for DSpark training. When debugging training issues, compare LumenRL's behavior against TorchSpec to identify misalignments.

### Reference Files

| LumenRL | TorchSpec (reference) | What it covers |
|---------|----------------------|----------------|
| `lumenrl/models/dspark.py` | `torchspec/models/dspark.py` | Loss computation, label construction, eval mask |
| `lumenrl/models/dspark.py` (DSparkModel) | `torchspec/models/draft/dspark.py` | Markov head, confidence head APIs |
| `lumenrl/trainer/spec_distill_trainer.py` | `torchspec/training/dspark_trainer.py` | Trainer hooks, hyperparameter defaults |
| `lumenrl/models/dspark.py` (_sample_anchors) | `torchspec/models/dflash.py` (_sample_anchor_positions) | Anchor sampling with block_keep_mask |
| `lumenrl/models/dspark.py` (DSparkMLAAttention) | `torchspec/models/draft/dflash.py` (DFlashAttention) | Dual-source KV injection (MLA vs GQA projections) |

### Key Alignment Points to Check

When loss doesn't converge or accuracy is wrong, check these in order:

**1. Label indices (most common issue)**
```python
# TorchSpec: slot k predicts input_ids[anchor + k + 1]
label_offsets = torch.arange(1, block_size + 1)  # [1, 2, ..., 7]
label_indices = anchor_positions + label_offsets

# LumenRL must NOT pre-shift input_ids. Forward receives raw input_ids
# and constructs labels internally with the same offset.
```

**2. Target hidden state alignment**
```python
# TorchSpec: target hidden at position (anchor+k) predicts token at (anchor+k+1)
tgt_idx = (safe_label_indices - 1).clamp(min=0)  # = anchor + [0, 1, ..., 6]
aligned_hidden = torch.gather(last_hidden_states, 1, tgt_idx)
target_logits = F.linear(aligned_hidden, lm_head_weight)

# Common bug: using anchor+k+1 instead of anchor+k for hidden state gathering
```

**3. Prev token IDs (Markov head)**
```python
# TorchSpec: slot 0 prev = anchor token, slot k prev = input_ids[anchor+k]
anchor_token_ids = input_ids[anchor]
prev_token_ids = cat([anchor_token_ids, target_ids[:, :, :-1]])

# Must NOT use shifted input_ids for prev token construction
```

**4. L1 loss (no 0.5 factor)**
```python
# TorchSpec: pure L1 distance (NOT total variation)
l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)

# Acceptance rate uses 0.5 factor:
accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
```

**5. Confidence head BCE**
```python
# TorchSpec: binary_cross_entropy_with_logits (numerically stable)
conf_bce = F.binary_cross_entropy_with_logits(confidence_pred, accept_rate)

# NOT: sigmoid(pred) → binary_cross_entropy (numerically unstable for extreme values)
```

**6. Loss reduction**
```python
# TorchSpec: pooled global-mean with cross-rank denominator
global_den = all_reduce(local_den, SUM) + 1e-6
loss = (ce_num/global_den + l1_num/global_den + conf_num/global_den) * world_size

# LumenRL: local mean (acceptable for single-node replicate, but differs
# from TorchSpec's global-mean under multi-node FSDP)
```

**7. Eval mask (contiguous prefix truncation)**
```python
# TorchSpec: cumprod ensures a gap truncates rest of block
eval_bool = (block_kept & in_bounds & supervised).cumprod(dim=-1).bool()

# Without cumprod, positions after a gap incorrectly contribute to loss
```

### Quick Comparison Script

```python
"""Compare LumenRL and TorchSpec loss computation on synthetic data.

NOTE: LumenRL uses MLA projections, TorchSpec uses GQA. Direct weight
transfer is not possible — compare loss computation logic, anchor sampling,
label construction, and loss reduction separately. For end-to-end weight
validation, export LumenRL checkpoint to HF format and load in vLLM.
"""
import torch
import sys
sys.path.insert(0, "/home/danyzhan/TorchSpec")
sys.path.insert(0, "/home/danyzhan/Lumen-RL")

from torchspec.models.dspark import DSparkModel as TSModel
from lumenrl.models.dspark import DSparkModel as LRModel

# Compare: label_indices, target hidden indices, anchor sampling,
# loss decay weights, Markov head prev_token construction.
# For attention: verify dual-source KV pattern matches (same concat order,
# same mask logic) even though projection weights differ (MLA vs GQA).
```

### Architecture Alignment: MLA Dual-Source KV (Aligned with TorchSpec)

Both LumenRL and TorchSpec use **dual-source KV injection**:
- Q comes from draft tokens only
- K/V come from BOTH context (full-sequence target hidden states) AND draft (self) via shared KV projections
- Context and draft KV are concatenated along the sequence dimension before attention
- Block-sparse attention mask: intra-block bidirectional, cross-block forbidden, context visible before anchor

**Projection difference:** TorchSpec uses GQA projections (`q_proj`, `k_proj`, `v_proj`) while LumenRL uses MLA projections (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`). This difference only affects the internal low-rank factorization of the weight matrices — the attention pattern and information flow are identical. LumenRL's MLA approach produces weights directly compatible with HF `Inferact/Kimi-K3-DSpark` format, so export requires no weight conversion.

### Training Hyperparameter Reference

| Parameter | TorchSpec default | LumenRL YAML | Notes |
|-----------|-------------------|--------------|-------|
| max_grad_norm | 0.5 | 0.5 | Aligned |
| warmup_ratio | 0.015 | 0.04 | Higher warmup for large-LR K3 training |
| learning_rate | 1e-4 | 6e-4 | K3-specific (larger draft → faster LR) |
| ce_loss_alpha | 0.1 | 0.1 | Aligned |
| l1_loss_alpha | 0.9 | 0.9 | Aligned |
| confidence_head_alpha | 1.0 | 1.0 | Aligned |
| loss_decay_gamma | 4.0 | 4.0 | Aligned |
| num_anchors | 512 | 512 | Aligned |
| block_size | 7 | 7 | Aligned |

## Reference

- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — Draft model config & weights
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — Target model
- [DSpark: Confidence-Scheduled Speculative Decoding (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [vLLM K3 Day-0 Support Blog](https://vllm.ai/blog/2026-07-27-k3)
- [TorchSpec: Speculative Decoding Training at Scale](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset)
- [TorchSpec source (local)](file:///home/danyzhan/TorchSpec) — Reference implementation for debugging
