# ImageNet ResNet-50 Training Guide

This guide shows you how to use the training scripts with your ImageNet dataset on EC2.

## Quick Start

### 1. Validate Your Dataset
First, make sure your ImageNet dataset is properly structured:

```bash
python main.py validate-dataset --data-dir /mnt/imagenet
```

This will check:
- Dataset structure and paths
- Number of training/validation samples
- Class distribution
- Image formats

### 2. Start Training
Begin training with default settings:

```bash
python main.py train --data-dir /mnt/imagenet --epochs 90
```

### 3. Monitor Training
Training will automatically:
- Save checkpoints every 10 epochs
- Save the best model based on validation accuracy
- Log training history to JSON
- Display progress with tqdm

## Advanced Usage

### Custom Training Parameters
```bash
python main.py train \
    --data-dir /mnt/imagenet \
    --epochs 90 \
    --batch-size 64 \
    --lr 0.1 \
    --num-workers 8 \
    --save-dir ./my_checkpoints
```

### Resume Training
If training is interrupted, resume from a checkpoint:

```bash
python main.py train \
    --data-dir /mnt/imagenet \
    --resume ./checkpoints/checkpoint_epoch_30.pth
```

### Evaluate a Trained Model
Test your model on validation/test sets:

```bash
python main.py evaluate \
    --model-path ./checkpoints/best_model.pth \
    --data-dir /mnt/imagenet
```

### Plot Training History
Visualize training progress:

```bash
python main.py plot --save-dir ./checkpoints
```

### Manage Checkpoints
List available checkpoints:
```bash
python main.py list-checkpoints
```

Clean up old checkpoints (keep latest 3 + best):
```bash
python main.py cleanup --keep-latest 3
```

## Offline Training

Since you're working on EC2 without internet, make sure you have all dependencies installed in your virtual environment:

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install torch torchvision numpy pillow tqdm matplotlib

# Run training
python main.py train --data-dir /mnt/imagenet
```

## File Structure

Your project should look like this:
```
Imagenet/
├── main.py                 # Main training script
├── config.py              # Configuration settings
├── src/
│   ├── model.py           # ResNet-50 model definition
│   ├── train.py           # Original training functions
│   └── imagenet_train.py  # ImageNet-specific training
├── checkpoints/           # Saved models and checkpoints
│   ├── best_model.pth
│   ├── checkpoint_epoch_10.pth
│   └── training_history.json
└── venv/                  # Virtual environment
```

## Expected Training Time

With default settings on a typical EC2 instance:
- **Batch size 32**: ~2-3 hours per epoch
- **Batch size 64**: ~1-2 hours per epoch
- **Total for 90 epochs**: 3-7 days depending on hardware

## Tips for EC2 Training

1. **Use larger batch sizes** if you have enough GPU memory
2. **Increase num_workers** for faster data loading (but not more than CPU cores)
3. **Monitor GPU memory usage** with `nvidia-smi`
4. **Use tmux/screen** to keep training running if you disconnect
5. **Save checkpoints frequently** in case of interruptions

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch-size 16`
- Reduce number of workers: `--num-workers 2`

### Slow Training
- Increase batch size if memory allows
- Increase number of workers
- Use mixed precision training (future enhancement)

### Dataset Issues
- Run `validate-dataset` command first
- Check file permissions on `/mnt/imagenet`
- Ensure ImageNet structure is correct

## Example Commands for Your Setup

```bash
# Full training run
python main.py train \
    --data-dir /mnt/imagenet \
    --epochs 90 \
    --batch-size 32 \
    --num-workers 4 \
    --save-dir ./checkpoints

# Quick test run (5 epochs)
python main.py train \
    --data-dir /mnt/imagenet \
    --epochs 5 \
    --batch-size 16

# Resume from checkpoint
python main.py train \
    --data-dir /mnt/imagenet \
    --resume ./checkpoints/checkpoint_epoch_20.pth
```
