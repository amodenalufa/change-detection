# Change Detection Baseline - Server Version

Optimized for server training with dual RTX 3090 GPUs (24GB x 2).

## Hardware Requirements

- GPU: NVIDIA GPU with 12GB+ VRAM (tested on dual RTX 3090 24GB)
- CUDA: 11.3+
- RAM: 32GB+ recommended

## Quick Start

### 1. Training

```bash
# Using default config
bash run_train.sh /path/to/LEVIR-CD

# Or with custom experiment name
bash run_train.sh /path/to/LEVIR-CD my_experiment

# Or with full command line arguments
python train.py \
    --data_root /path/to/LEVIR-CD \
    --dataset levir-cd \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.002 \
    --optimizer adamw \
    --scheduler cosine \
    --pretrained \
    --use_multi_gpu \
    --amp_enabled \
    --num_workers 8 \
    --exp_name baseline_exp
```

### 2. Evaluation

```bash
bash run_eval.sh /path/to/LEVIR-CD /path/to/checkpoint.pth
```

### 3. Single Image Inference

```bash
python eval.py \
    --checkpoint /path/to/checkpoint.pth \
    --img_t1 /path/to/t1.png \
    --img_t2 /path/to/t2.png \
    --output_path ./result.png
```

## Configuration

Edit `configs/server_config.py` to modify default settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 16 | Increase to 32-48 for dual GPU |
| `EPOCHS` | 100 | Full training epochs |
| `LEARNING_RATE` | 0.002 | Scale with batch size |
| `OPTIMIZER` | adamw | AdamW optimizer |
| `SCHEDULER` | cosine | Cosine annealing |
| `USE_MULTI_GPU` | True | Enable DataParallel |
| `AMP_ENABLED` | True | Automatic Mixed Precision |

## Multi-GPU Training

The code automatically uses `nn.DataParallel` when `use_multi_gpu` is set and multiple GPUs are detected.

To use specific GPUs:
```bash
export CUDA_VISIBLE_DEVICES=0,1
python train.py ...
```

To use single GPU only:
```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --device cuda:0
```

## Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir ./results --port 6006

# Then open http://localhost:6006 in browser
```

## Expected Results (LEVIR-CD)

With ResNet50-V2 baseline:
- mIoU: ~85-88%
- F1: ~88-90%
- OA: ~98%

Training time: ~2-3 hours for 100 epochs on dual RTX 3090.

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

## Resuming Training

```bash
python train.py \
    --data_root /path/to/LEVIR-CD \
    --resume /path/to/checkpoint.pth
```
