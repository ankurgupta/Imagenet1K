# Running Training Without Staying Connected

You **DO NOT** need to stay logged into EC2 for days! Here are several ways to run training in the background:

## 🚀 Method 1: Detached Training Script (Recommended)

Use the provided script that handles everything automatically:

```bash
python run_detached_training.py
```

This script will:
- ✅ Start training in the background
- ✅ Survive SSH disconnections
- ✅ Save logs to files
- ✅ Give you monitoring commands

## 🔧 Method 2: Manual nohup Command

If you prefer manual control:

```bash
# Start training in background
nohup python main.py train --data-dir /mnt/imagenet --epochs 90 > training.log 2>&1 &

# Note the process ID (PID) that's printed
echo $! > training.pid
```

## 🖥️ Method 3: Using tmux (Terminal Multiplexer)

Install and use tmux for persistent sessions:

```bash
# Install tmux (if not already installed)
sudo yum install tmux  # Amazon Linux
# or
sudo apt install tmux  # Ubuntu

# Start a new tmux session
tmux new-session -d -s training

# Run training in the tmux session
tmux send-keys -t training "python main.py train --data-dir /mnt/imagenet --epochs 90" Enter

# Detach from session (training continues)
tmux detach

# Later, reattach to see progress
tmux attach -t training
```

## 📊 Monitoring Your Training

### Check if training is running:
```bash
ps aux | grep "python main.py"
```

### Monitor training progress:
```bash
# If using detached script
tail -f logs/training_*.log

# If using manual nohup
tail -f training.log

# If using tmux
tmux attach -t training
```

### Check GPU usage:
```bash
nvidia-smi
```

## 🛑 Stopping Training

### If using detached script:
```bash
python run_detached_training.py
# Choose option 4: Stop all training processes
```

### If using manual method:
```bash
# Find the process ID
ps aux | grep "python main.py"

# Kill the process
kill <PID>

# Or kill all training processes
pkill -f "python main.py train"
```

### If using tmux:
```bash
# Attach to session
tmux attach -t training

# Press Ctrl+C to stop training
# Then exit tmux
exit
```

## 📁 File Structure After Training

Your directory will look like this:
```
Imagenet/
├── logs/
│   ├── training_20241201_143022.log    # Training output
│   └── training_20241201_143022.err    # Error logs
├── checkpoints/
│   ├── best_model.pth                  # Best model
│   ├── checkpoint_epoch_10.pth         # Regular checkpoints
│   ├── checkpoint_epoch_20.pth
│   └── training_history.json           # Training metrics
└── training.pid                        # Process ID (if using manual method)
```

## 🔄 Resuming Training

If training stops unexpectedly, you can resume:

```bash
# Find the latest checkpoint
ls -la checkpoints/checkpoint_epoch_*.pth

# Resume from checkpoint
python main.py train \
    --data-dir /mnt/imagenet \
    --resume checkpoints/checkpoint_epoch_30.pth
```

## 💡 Pro Tips

1. **Always validate dataset first:**
   ```bash
   python main.py validate-dataset --data-dir /mnt/imagenet
   ```

2. **Test with a few epochs first:**
   ```bash
   python main.py train --data-dir /mnt/imagenet --epochs 5
   ```

3. **Monitor disk space:**
   ```bash
   df -h /mnt/imagenet  # Check dataset disk space
   df -h .              # Check project disk space
   ```

4. **Set up log rotation** (optional):
   ```bash
   # Install logrotate to prevent huge log files
   sudo yum install logrotate
   ```

## 🚨 Troubleshooting

### Training stops unexpectedly:
- Check logs: `tail -f logs/training_*.log`
- Check system resources: `htop` or `top`
- Check GPU memory: `nvidia-smi`

### Out of memory errors:
- Reduce batch size: `--batch-size 16`
- Reduce workers: `--num-workers 2`

### SSH connection drops:
- Use the detached training script
- Or use tmux/screen for persistent sessions

## 📱 Example Workflow

```bash
# 1. Connect to EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# 2. Navigate to project
cd /path/to/Imagenet

# 3. Activate environment
source venv/bin/activate

# 4. Start detached training
python run_detached_training.py
# Choose option 1: Start detached training

# 5. Disconnect from SSH (training continues!)

# 6. Later, reconnect to check progress
ssh -i your-key.pem ec2-user@your-ec2-ip
cd /path/to/Imagenet
tail -f logs/training_*.log
```

**You can safely disconnect and reconnect anytime!** 🎉
