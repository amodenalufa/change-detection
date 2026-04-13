# Change Detection

## Quick Start

conda activate diffusion_model
python train.py

## Configuration

Edit `configs/server_config.py` to modify default settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 16 | Increase to 32-48 for dual GPU |
| `EPOCHS` | 100 | Full training epochs |
| `LEARNING_RATE` | 0.001 | Scale with batch size |
| `OPTIMIZER` | adamw | AdamW optimizer |
| `SCHEDULER` | cosine_warmup | Cosine annealing |
| `USE_MULTI_GPU` | True | Enable DataParallel |
| `AMP_ENABLED` | True | Automatic Mixed Precision |

## Monitoring Training

watch -n 1 nvidia-smi

## Expected Results (LEVIR-CD)

With baseline:
val - Loss: 0.2371 | OA: 0.9844 | IoU: 0.6810 | F1: 0.8102 | P: 0.8301 | R: 0.7913

## Output Structure

```
results/
└── exp_name/
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── checkpoint_epoch_XX.pth
    ├── logs/
    │   └── events.out.tfevents.*
    └── config.json
```
