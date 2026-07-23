# HILDA

Minimal PyTorch implementation of HILDA, a hierarchical latent generator for
long lifecycle trajectories.

```text
observed cycles -> cycle VAE -> block-causal latent transport
                -> cycle decoder -> generated future cycles
```

The released configuration uses:

- 192 values per cycle, a 64-dimensional VAE latent, and a 2-layer MLP codec.
- 64-cycle blocks with detached clean history and bidirectional attention only
  inside the current block.
- Endpoint (`x1`) prediction on a linear noise-to-latent path.
- A `1 -> 2 -> 4 -> all` generated-history curriculum and scheduled sampling
  from 0.10 to 0.50.
- A frozen reference codec and a frozen differentiable SOH estimator. Gradients
  pass through the estimator to HILDA, but estimator parameters never update.

## Install

```bash
pip install -e .
```

## Minimal use

```python
import torch
from hilda import HILDA, HILDAConfig

model = HILDA(HILDAConfig())
prefix = torch.randn(1, 20, 192)  # normalized complete cycles
generated = model.generate(prefix, target_length=200, flow_steps=8)
print(generated.shape)  # (1, 200, 192)
```

For training, pass normalized waveform tensors, SOH labels, valid lengths, a
frozen copy of the pretrained codec, and a pretrained differentiable SOH
network to `HILDATrainer`. Data adapters, downstream estimators, and checkpoints
are intentionally not included.
