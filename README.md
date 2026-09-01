# transformer-from-scratch-numpy

A decoder-only transformer and its backpropagation, implemented from scratch in
NumPy following the DeepSeek-V3 architecture. No PyTorch, no TensorFlow, no
autograd — every gradient in this repository is derived by hand and written out
explicitly.

## Why

Modern frameworks compute gradients for you. That is convenient, but it hides
the mechanism. This project rebuilds a working language model one layer at a
time, deriving the backward pass for each component, to understand what those
frameworks actually do.

## Architecture

Follows DeepSeek-V3 rather than the original 2017 transformer:

| Component | Choice | Instead of |
|---|---|---|
| Normalization | RMSNorm (pre-norm) | LayerNorm (post-norm) |
| Positional encoding | RoPE | Learned / sinusoidal embeddings |
| Feed-forward | SwiGLU | ReLU MLP |
| Attention | Causal MHA + MLA (KV compression) | Standard MHA |
| FFN routing | DeepSeekMoE (shared + routed experts) | Dense FFN |
| Load balancing | Auxiliary-loss-free (bias-based) | Auxiliary loss |
| Optimizer | AdamW | SGD |

Decoder-only, causal masking throughout.

## Repository layout

Each file in `steps/` is standalone and runnable. They are ordered so the
repository reads as a build log.

| File | Contents |
|---|---|
| `01_backprop_mlp.py` | Two-layer MLP. Forward, hand-derived backward, softmax + cross-entropy, training loop. |
| `02_attention.py` | Single-head causal self-attention. Q/K/V, scaled dot-product, causal mask, backward. |
| `03_multihead.py` | Multi-head attention with head splitting/merging and output projection. |
| `04_rmsnorm.py` | RMSNorm forward and backward. |
| `05_rope.py` | Rotary position embeddings. Rotation is norm-preserving, verified in the output. |
| `06_swiglu.py` | SwiGLU feed-forward network with SiLU gating. |
| `07_block.py` | Full pre-norm transformer block: RMSNorm to RoPE MHA to residual to RMSNorm to SwiGLU to residual. |
| `08_tokenizer_embedding.py` | Character tokenizer, embedding table (scatter-add backward), LM head. |
| `09_adamw.py` | AdamW with momentum, adaptive step sizes, bias correction and decoupled weight decay. |
| `10_train.py` | Everything assembled. Trains a 2-layer, 85K-parameter model on text. |
| `11_generate.py` | Autoregressive sampling with temperature and top-k. |
| `12_mla_moe.py` | Multi-head Latent Attention and DeepSeekMoE with bias-based load balancing. |

## Running

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python steps/10_train.py        # train
python steps/11_generate.py     # train, then generate text
python steps/12_mla_moe.py      # MLA + MoE
```

Only dependency: NumPy.

## Results

Training (`10_train.py`), 85,312 parameters, 2 layers, `d_model=64`, 4 heads:

```
step    0  loss 3.1923      (random baseline: log(24) = 3.178)
step  600  loss 0.3615
step 3000  loss 0.1174
```

MLA + MoE (`12_mla_moe.py`):

```
KV cache: 8 values/token vs 64 for standard attention  (8x reduction)
expert load: [0.19 0.25 0.50 0.06] -> [0.25 0.25 0.25 0.25]
```

The router initially collapses onto one expert; the bias-based balancing
mechanism spreads the load evenly without touching the loss function.

## Correctness

Every hand-derived gradient was verified against central-difference numerical
differentiation during development, with relative errors below `1e-7`. The
verification helpers were removed from the final files to keep them focused; the
training curves above are the remaining evidence that the backward passes are
correct — a single sign or transpose error anywhere would prevent the loss from
converging.

## Scope and limitations

- Trained on a few hundred characters of repeated text. The model memorizes it;
  this is intentional and sufficient to validate the architecture and gradients.
- No KV cache during generation — full context is recomputed per token.
- MLA applies RoPE after the up-projection. DeepSeek-V3 uses a decoupled RoPE
  head because the rotation cannot be absorbed into the up-projection matrix;
  the simplification here is mathematically correct but gives up part of the
  cache benefit.
- Single sequence per step, no batching.

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
- Shazeer, *GLU Variants Improve Transformer* (2020)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019)
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)