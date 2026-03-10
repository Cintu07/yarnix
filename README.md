# Yarnix

A character-level language model built on phase-rotation neural architecture.

## What is Yarnix?

Yarnix is a language model that stores memory as **angles on a circle** instead of numbers. Angles don't decay — 90° stays 90° after a million timesteps. This gives the model theoretically infinite context.

The architecture uses a **dual-state recurrent cell** with:
- **Rich feature state** (GRU-style gates) for expressive recurrence
- **Phase angle state** (bounded to [0, 2π]) for infinite memory
- **Winding counter** (integer rotation count, gradient-detached) for context tracking
- **Multi-clock bands** — 4 timescale bands with different persistence values:
  - Ultra-fast (0.50) — character/word patterns
  - Fast (0.80) — phrase/sentence structure
  - Slow (0.95) — paragraph/topic tracking
  - Ultra-slow (0.999) — chapter-level memory

## Results

Trained on TinyShakespeare (1M characters), CPU only:

| Metric | Value |
|--------|-------|
| Parameters | 2.3M |
| Best Val Loss | 1.56 |
| Training Time | ~6 hours (CPU) |

### Generated Text (Epoch 14)
```
MENENIUS:
Sir, I see what do you grow.
Insued me in my father's gone, he loves us to the law.

First Senator:
You have would follow the colours of the
```

### Generated Text (Epoch 22)
```
MISIR:
I will send thee better in perfect loyalty.

KING RICHARD II:
Thanks.

GLOUCESTER:
I have deserve no more affect one and by
thy pock upon thy d
```

## Architecture

```
Input (char) → Embedding → [YarnixCell Layer 1] → [YarnixCell Layer 2] → Readout MLP → Output (logits)
                                   ↕                        ↕
                            h_state (GRU)             h_state (GRU)
                            local_angle               local_angle
                            winding_count              winding_count
```

Each `YarnixCell` contains:
- 3 GRU-style gates (reset, update, candidate)
- Phase accumulation with multi-clock persistence bands
- Quantization sieve (snaps to π/4 grid for stability)
- Harmonic phase reader (cos/sin at harmonics 1, 2, 4, 8)
- Cross-band mixer MLP (lets timescales communicate)

## Usage

### Training
```bash
python get_data.py          # Download TinyShakespeare
python language_model.py    # Train the model
```

### Requirements
```
torch>=2.0
numpy
```

## Files

| File | Description |
|------|-------------|
| `yarnix_cell.py` | Core architecture — YarnixCellV4 and YarnixModelV4 |
| `language_model.py` | Training script and YarnixLM wrapper |
| `config.py` | Global configuration |
| `get_data.py` | Dataset downloader |

## Author

**Pavan Kalyan** ([@Cintu07](https://github.com/Cintu07))
