#!/usr/bin/env python3
"""
Detached Training Script for EC2
This script runs training in the background and handles disconnections
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def run_training_detached():
    """Run training in a way that survives SSH disconnections."""
    
    print("🚀 Starting Detached ImageNet Training")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: main.py not found. Please run from project root.")
        sys.exit(1)
    
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    error_file = log_dir / f"training_{timestamp}.err"
    
    print(f"📝 Logs will be saved to:")
    print(f"   Output: {log_file}")
    print(f"   Errors: {error_file}")
    
    # Training command
    cmd = [
        "python", "main.py", "train",
        "--data-dir", "/mnt/imagenet",
        "--epochs", "90",
        "--batch-size", "32",
        "--num-workers", "4",
        "--save-dir", "./checkpoints"
    ]
    
    print(f"\n🎯 Training Command:")
    print(f"   {' '.join(cmd)}")
    
    print(f"\n💡 To monitor training:")
    print(f"   tail -f {log_file}")
    print(f"   tail -f {error_file}")
    
    print(f"\n💡 To stop training:")
    print(f"   pkill -f 'python main.py train'")
    print(f"   or find the process: ps aux | grep 'python main.py'")
    
    # Ask for confirmation
    print(f"\n⚠️  This will start training that may run for 3-7 days!")
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("Training cancelled.")
        return
    
    print(f"\n🚀 Starting training...")
    print(f"   You can safely disconnect from SSH now!")
    print(f"   Training will continue in the background.")
    
    try:
        # Start training with nohup to survive SSH disconnection
        with open(log_file, 'w') as out, open(error_file, 'w') as err:
            # Use nohup to make process survive SSH disconnection
            process = subprocess.Popen(
                ["nohup"] + cmd,
                stdout=out,
                stderr=err,
                preexec_fn=os.setsid  # Create new process group
            )
        
        print(f"✅ Training started with PID: {process.pid}")
        print(f"📝 Process group ID: {os.getpgid(process.pid)}")
        
        # Give it a moment to start
        time.sleep(2)
        
        if process.poll() is None:
            print(f"\n🎉 Training is running successfully!")
            print(f"   Process ID: {process.pid}")
            print(f"   You can now safely disconnect from SSH")
            print(f"\n📋 To monitor later:")
            print(f"   ssh back into your EC2 instance")
            print(f"   tail -f {log_file}")
            print(f"   ps aux | grep 'python main.py'")
        else:
            print(f"❌ Training failed to start. Check {error_file}")
            
    except Exception as e:
        print(f"❌ Error starting training: {e}")

def show_running_training():
    """Show currently running training processes."""
    print("🔍 Checking for running training processes...")
    
    try:
        # Find Python training processes
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.split('\n')
        training_processes = []
        
        for line in lines:
            if 'python main.py train' in line and 'grep' not in line:
                training_processes.append(line)
        
        if training_processes:
            print(f"\n📊 Found {len(training_processes)} training process(es):")
            for i, process in enumerate(training_processes, 1):
                print(f"   {i}. {process}")
        else:
            print("   No training processes found.")
            
        # Check for log files
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("training_*.log"))
            if log_files:
                print(f"\n📝 Recent log files:")
                for log_file in sorted(log_files)[-3:]:  # Show last 3
                    size = log_file.stat().st_size
                    mtime = time.ctime(log_file.stat().st_mtime)
                    print(f"   {log_file.name} ({size} bytes, {mtime})")
        
    except Exception as e:
        print(f"❌ Error checking processes: {e}")

def stop_training():
    """Stop running training processes."""
    print("🛑 Stopping training processes...")
    
    try:
        # Find and kill training processes
        result = subprocess.run(
            ["pkill", "-f", "python main.py train"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Training processes stopped.")
        else:
            print("ℹ️  No training processes found to stop.")
            
    except Exception as e:
        print(f"❌ Error stopping training: {e}")

def monitor_training():
    """Monitor the latest training log."""
    log_dir = Path("logs")
    if not log_dir.exists():
        print("❌ No logs directory found.")
        return
    
    log_files = list(log_dir.glob("training_*.log"))
    if not log_files:
        print("❌ No training log files found.")
        return
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Monitoring latest log: {latest_log}")
    print("   Press Ctrl+C to stop monitoring")
    
    try:
        subprocess.run(["tail", "-f", str(latest_log)])
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

def main():
    print("🎯 ImageNet Detached Training Manager")
    print("="*60)
    
    while True:
        print("\n📋 Available Commands:")
        print("1. Start detached training (survives SSH disconnect)")
        print("2. Show running training processes")
        print("3. Monitor latest training log")
        print("4. Stop all training processes")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            run_training_detached()
        elif choice == "2":
            show_running_training()
        elif choice == "3":
            monitor_training()
        elif choice == "4":
            stop_training()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
