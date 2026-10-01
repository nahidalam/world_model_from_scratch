# 4.8 Train a Small World Model

The model from Sections 4.5 to 4.7 has random weights. Training sets them. We first train on eight clips to check that the model can learn them. Then we train on 512 clips so the model learns motion it has not seen.

## Training Data

Each clip belongs to one split: training, validation, or test. The model trains on the training split. We evaluate it on the test split, which holds clips the model has not seen. We generate 512 training, 64 validation, and 64 test clips:

```bash
python scripts/chapter_04_train.py prepare \
  --output outputs/chapter_04/excavator/data \
  --device cuda \
  --train-clips 512 \
  --val-clips 64 \
  --test-clips 64
```

The VAE stays fixed, so we encode each clip once and cache its `[16, 5, 16, 16]` latent with its bucket positions.

## Training Step

Each update computes `training_loss` from Section 4.6 on 32 clips and takes one AdamW step. The 32 clips arrive as eight microbatches of four, which keeps memory use low. `train` runs the updates:

```python
import torch
from torch import Tensor, nn

from world_models.complete_small_world import Config, WorldModel, training_loss


def train(model: WorldModel, latents: Tensor, *, steps: int, batch_size: int = 4,
          accumulation: int = 8, learning_rate: float = 3e-4, seed: int = 17) -> list[float]:
    """Run `steps` optimizer updates of `batch_size * accumulation` clips each."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(seed + 1)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            batch = latents[torch.randint(len(latents), (batch_size,), generator=generator)]
            loss = training_loss(model, batch.to(device), generator)
            (loss / accumulation).backward()
            total += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(total / accumulation)
    return losses


tiny = Config(channels=3, frames=5, height=4, width=4, heads=2, head_dim=16,
              blocks=2, context_dim=16, lora_rank=8, mlp_ratio=2.0)
torch.manual_seed(0)
model = WorldModel(tiny)
latents = torch.randn(4, 3, 5, 4, 4)
losses = train(model, latents, steps=30, batch_size=2, accumulation=1, learning_rate=3e-3)
print(losses[-1] < losses[0])
```

```
True
```

A tiny model on random latents shows the loss falling on a CPU. The commands below train on the excavator clips with [`chapter_04_train.py`](https://github.com/nahidalam/world_model_from_scratch_private/tree/main/scripts/chapter_04_train.py). It computes the same loss and adds logging, checkpoints, and mixed precision. `--batch-size 4 --accumulation 8` sets the 32 clips per update, and `--steps` counts updates.

## Overfitting Test

Before the full run, we check that the model can learn eight clips. When it reproduces clips it has seen, the data, loss, and sampler work together. We prepare eight clips per split and train the small model on them:

```bash
python scripts/chapter_04_train.py prepare \
  --output outputs/chapter_04/excavator/data_probe \
  --device cuda \
  --train-clips 8 \
  --val-clips 8 \
  --test-clips 8
```

```bash
python scripts/chapter_04_train.py train \
  --data outputs/chapter_04/excavator/data_probe \
  --output outputs/chapter_04/excavator/overfit \
  --preset small \
  --device cuda \
  --steps 2000 \
  --batch-size 2 \
  --accumulation 4 \
  --overfit-clips 8
```

Then we sample the same eight clips:

```bash
python scripts/chapter_04_train.py sample \
  --data outputs/chapter_04/excavator/data_probe \
  --checkpoint outputs/chapter_04/excavator/overfit/checkpoint.pt \
  --output outputs/chapter_04/excavator/overfit_evaluation \
  --device cuda \
  --split train \
  --examples 8 \
  --sampling-steps 30
```

After 2,000 updates, the generated bucket is 0.44 pixels from its true position on average. Holding the last observed frame gives 10.62 pixels. The [training configuration](https://github.com/nahidalam/world_model_from_scratch_private/tree/main/assets/chapter_04/training/overfit_training.json) and [evaluation report](https://github.com/nahidalam/world_model_from_scratch_private/tree/main/assets/chapter_04/training/overfit_evaluation.json) contain the settings and the result for each clip.

## Full Training

We train the base model on all 512 clips for 2,000 updates at learning rate `3e-4`:

```bash
python scripts/chapter_04_train.py train \
  --data outputs/chapter_04/excavator/data \
  --output outputs/chapter_04/excavator/train \
  --preset base \
  --device cuda \
  --steps 2000 \
  --batch-size 4 \
  --accumulation 8
```

![Training velocity loss and fixed-noise validation loss over 2,000 updates.](../../.gitbook/assets/loss_curve.png)

_Figure 4.8: Training loss and validation loss over 2,000 updates._

Training loss falls from `2.386` to `0.0102`, and validation loss falls from `2.372` to `0.0286`. The validation clips stay out of training, so the falling validation loss shows the model learning motion beyond the clips it saw. The [training configuration](https://github.com/nahidalam/world_model_from_scratch_private/tree/main/assets/chapter_04/training/training.json) records the settings.

Training reserved under 1 GiB of GPU memory. On a smaller GPU, use `--preset small`, a microbatch of two with 16 accumulation steps, or `--checkpoint-blocks`, which recomputes activations instead of storing them.

## Resume

The checkpoint stores the model, optimizer, and random state, so a resumed run continues exactly where it stopped. To continue to 3,000 updates, keep the other arguments and raise `--steps`:

```bash
python scripts/chapter_04_train.py train \
  --data outputs/chapter_04/excavator/data \
  --output outputs/chapter_04/excavator/train \
  --preset base \
  --device cuda \
  --steps 3000 \
  --batch-size 4 \
  --accumulation 8 \
  --resume outputs/chapter_04/excavator/train/checkpoint.pt
```

Next, we evaluate the model on unseen motion.
