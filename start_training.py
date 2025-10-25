#!/usr/bin/env python3
"""
Quick start script for ImageNet ResNet-50 training
This script provides easy commands to start training on your EC2 instance
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    print("🎯 ImageNet ResNet-50 Training Setup")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: main.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("❌ Error: Virtual environment not found. Please create it first:")
        print("   python -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -e .")
        sys.exit(1)
    
    print("✅ Project structure looks good!")
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("📋 Available Commands:")
        print("="*60)
        print("1. Validate ImageNet dataset")
        print("2. Start training (90 epochs)")
        print("3. Start training (5 epochs - quick test)")
        print("4. Resume training from checkpoint")
        print("5. Evaluate trained model")
        print("6. Plot training history")
        print("7. List checkpoints")
        print("8. Clean up old checkpoints")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            run_command(
                "python main.py validate-dataset --data-dir /mnt/imagenet",
                "Validating ImageNet Dataset"
            )
        
        elif choice == "2":
            print("\n⚠️  This will start full training (90 epochs, ~3-7 days)")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                run_command(
                    "python main.py train --data-dir /mnt/imagenet --epochs 90 --batch-size 32",
                    "Starting Full Training (90 epochs)"
                )
            else:
                print("Training cancelled.")
        
        elif choice == "3":
            run_command(
                "python main.py train --data-dir /mnt/imagenet --epochs 5 --batch-size 16",
                "Starting Quick Test Training (5 epochs)"
            )
        
        elif choice == "4":
            # List available checkpoints first
            print("\nAvailable checkpoints:")
            run_command("python main.py list-checkpoints", "Listing Checkpoints")
            
            checkpoint = input("\nEnter checkpoint path (or press Enter to cancel): ").strip()
            if checkpoint:
                run_command(
                    f"python main.py train --data-dir /mnt/imagenet --resume {checkpoint}",
                    f"Resuming Training from {checkpoint}"
                )
        
        elif choice == "5":
            model_path = input("Enter model path (e.g., ./checkpoints/best_model.pth): ").strip()
            if model_path:
                run_command(
                    f"python main.py evaluate --model-path {model_path} --data-dir /mnt/imagenet",
                    f"Evaluating Model: {model_path}"
                )
        
        elif choice == "6":
            run_command(
                "python main.py plot",
                "Plotting Training History"
            )
        
        elif choice == "7":
            run_command(
                "python main.py list-checkpoints",
                "Listing Available Checkpoints"
            )
        
        elif choice == "8":
            run_command(
                "python main.py cleanup --keep-latest 3",
                "Cleaning Up Old Checkpoints"
            )
        
        elif choice == "9":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-9.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
