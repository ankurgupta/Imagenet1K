#!/usr/bin/env python3
"""
ImageNet ResNet-50 Training and Management Script

This script provides a comprehensive interface for:
- Training ResNet-50 on ImageNet dataset
- Dataset validation and statistics
- Model evaluation and testing
- Training monitoring and checkpointing
- Model management utilities

Usage:
    python main.py train --data-dir /mnt/imagenet --epochs 90
    python main.py validate-dataset --data-dir /mnt/imagenet
    python main.py evaluate --model-path checkpoints/best_model.pth --data-dir /mnt/imagenet
    python main.py resume --checkpoint checkpoints/checkpoint_epoch_30.pth
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Import our modules
from src.model import ResNet50
from src.imagenet_train import (
    imagenet_transforms, train_epoch, validate, 
    save_checkpoint, load_checkpoint, mixup_data, mixup_criterion
)


class ImageNetManager:
    """Main class for managing ImageNet training and evaluation."""
    
    def __init__(self, data_dir="/mnt/imagenet", save_dir="./checkpoints", device="auto"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Data directory: {self.data_dir}")
        print(f"Save directory: {self.save_dir}")
    
    def validate_dataset(self):
        """Validate ImageNet dataset structure and provide statistics."""
        print("Validating ImageNet dataset...")
        
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "validation"
        test_dir = self.data_dir / "test"
        
        # Check directories exist
        if not train_dir.exists():
            raise ValueError(f"Training directory not found: {train_dir}")
        if not val_dir.exists():
            raise ValueError(f"Validation directory not found: {val_dir}")
        
        print(f"✓ Training directory: {train_dir}")
        print(f"✓ Validation directory: {val_dir}")
        if test_dir.exists():
            print(f"✓ Test directory: {test_dir}")
        
        # Load datasets to get statistics
        train_dataset = datasets.ImageFolder(train_dir)
        val_dataset = datasets.ImageFolder(val_dir)
        
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        print(f"Number of classes: {len(train_dataset.classes):,}")
        
        # Check class distribution
        train_class_counts = defaultdict(int)
        for _, class_idx in train_dataset.samples:
            train_class_counts[class_idx] += 1
        
        val_class_counts = defaultdict(int)
        for _, class_idx in val_dataset.samples:
            val_class_counts[class_idx] += 1
        
        print(f"\nClass Distribution (first 10 classes):")
        for i in range(min(10, len(train_dataset.classes))):
            class_name = train_dataset.classes[i]
            train_count = train_class_counts[i]
            val_count = val_class_counts[i]
            print(f"  {class_name}: Train={train_count}, Val={val_count}")
        
        # Check for missing classes in validation
        missing_in_val = set(train_class_counts.keys()) - set(val_class_counts.keys())
        if missing_in_val:
            print(f"\n⚠️  Warning: {len(missing_in_val)} classes missing in validation set")
        
        # Check image formats
        print(f"\nChecking image formats...")
        sample_paths = [train_dataset.samples[i][0] for i in range(min(100, len(train_dataset.samples)))]
        extensions = set()
        for path in sample_paths:
            extensions.add(Path(path).suffix.lower())
        print(f"Image formats found: {', '.join(sorted(extensions))}")
        
        return {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'num_classes': len(train_dataset.classes),
            'train_class_counts': dict(train_class_counts),
            'val_class_counts': dict(val_class_counts),
            'image_formats': list(extensions)
        }
    
    def train(self, epochs=90, batch_size=32, lr=0.1, num_workers=4, 
              use_mixup=True, resume_from=None):
        """Train ResNet-50 on ImageNet."""
        print(f"Starting ImageNet ResNet-50 training...")
        
        # Validate dataset first
        dataset_stats = self.validate_dataset()
        
        # Create datasets
        train_dataset = datasets.ImageFolder(
            self.data_dir / "train",
            transform=imagenet_transforms(augment=True)
        )
        val_dataset = datasets.ImageFolder(
            self.data_dir / "validation",
            transform=imagenet_transforms(augment=False)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Create model
        model = ResNet50(num_classes=len(train_dataset.classes))
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_acc = 0.0
        if resume_from:
            start_epoch, best_acc = load_checkpoint(
                model, optimizer, scheduler, resume_from
            )
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [], 'epoch_times': []
        }
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Use mixup: {use_mixup}")
        print(f"  Device: {self.device}")
        print(f"  Start epoch: {start_epoch + 1}")
        
        # Training loop
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, 
                self.device, epoch, use_mixup
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, self.device)
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epoch_times'].append(epoch_time)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Epoch Time: {epoch_time:.1f}s")
            
            # Save checkpoint if best accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = self.save_dir / 'best_model.pth'
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, str(best_path))
                print(f"🎉 New best accuracy: {best_acc:.2f}%")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, str(checkpoint_path))
            
            # Save training history
            history_path = self.save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        print(f"\n🎯 Training completed!")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Checkpoints saved in: {self.save_dir}")
        
        return history
    
    def evaluate(self, model_path, batch_size=32, num_workers=4):
        """Evaluate a trained model on validation and test sets."""
        print(f"Evaluating model: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model = ResNet50(num_classes=1000)  # ImageNet has 1000 classes
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
        print(f"Best accuracy during training: {checkpoint['best_acc']:.2f}%")
        
        # Create datasets
        val_dataset = datasets.ImageFolder(
            self.data_dir / "validation",
            transform=imagenet_transforms(augment=False)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Evaluate on validation set
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = validate(model, val_loader, criterion, self.device)
        
        print(f"\n📊 Evaluation Results:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Evaluate on test set if available
        test_dir = self.data_dir / "test"
        if test_dir.exists():
            test_dataset = datasets.ImageFolder(
                test_dir,
                transform=imagenet_transforms(augment=False)
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            test_loss, test_acc = validate(model, test_loader, criterion, self.device)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%")
        
        return {'val_loss': val_loss, 'val_acc': val_acc}
    
    def plot_training_history(self, history_path=None):
        """Plot training history."""
        if history_path is None:
            history_path = self.save_dir / 'training_history.json'
        
        if not os.path.exists(history_path):
            print(f"Training history not found: {history_path}")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc')
        axes[0, 1].plot(history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Epoch time plot
        axes[1, 1].plot(history['epoch_times'])
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {plot_path}")
        plt.show()
    
    def list_checkpoints(self):
        """List available checkpoints."""
        checkpoints = list(self.save_dir.glob("*.pth"))
        if not checkpoints:
            print("No checkpoints found.")
            return
        
        print("Available checkpoints:")
        for checkpoint in sorted(checkpoints):
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            print(f"  {checkpoint.name} ({size_mb:.1f} MB)")
    
    def cleanup_checkpoints(self, keep_best=True, keep_latest=3):
        """Clean up old checkpoints, keeping only the best and latest ones."""
        checkpoints = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoints) <= keep_latest:
            print("No checkpoints to clean up.")
            return
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Keep latest checkpoints
        to_keep = checkpoints[-keep_latest:]
        to_remove = checkpoints[:-keep_latest]
        
        for checkpoint in to_remove:
            print(f"Removing: {checkpoint.name}")
            checkpoint.unlink()
        
        print(f"Kept {len(to_keep)} latest checkpoints")
        if keep_best and (self.save_dir / 'best_model.pth').exists():
            print("Kept best model checkpoint")


def main():
    parser = argparse.ArgumentParser(description='ImageNet ResNet-50 Training Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-dir', type=str, default='/mnt/imagenet',
                            help='Path to ImageNet dataset')
    train_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                            help='Directory to save checkpoints')
    train_parser.add_argument('--epochs', type=int, default=90,
                            help='Number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for training')
    train_parser.add_argument('--lr', type=float, default=0.1,
                            help='Initial learning rate')
    train_parser.add_argument('--num-workers', type=int, default=4,
                            help='Number of data loading workers')
    train_parser.add_argument('--no-mixup', action='store_true',
                            help='Disable mixup augmentation')
    train_parser.add_argument('--resume', type=str, default=None,
                            help='Path to checkpoint to resume from')
    train_parser.add_argument('--device', type=str, default='auto',
                            help='Device to use (auto, cpu, cuda)')
    
    # Validate dataset command
    validate_parser = subparsers.add_parser('validate-dataset', help='Validate dataset structure')
    validate_parser.add_argument('--data-dir', type=str, default='/mnt/imagenet',
                               help='Path to ImageNet dataset')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to model checkpoint')
    eval_parser.add_argument('--data-dir', type=str, default='/mnt/imagenet',
                           help='Path to ImageNet dataset')
    eval_parser.add_argument('--batch-size', type=int, default=32,
                           help='Batch size for evaluation')
    eval_parser.add_argument('--device', type=str, default='auto',
                           help='Device to use (auto, cpu, cuda)')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot training history')
    plot_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                           help='Directory containing training history')
    plot_parser.add_argument('--history-path', type=str, default=None,
                           help='Path to training history JSON file')
    
    # List checkpoints command
    list_parser = subparsers.add_parser('list-checkpoints', help='List available checkpoints')
    list_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                           help='Directory containing checkpoints')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cleanup_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                              help='Directory containing checkpoints')
    cleanup_parser.add_argument('--keep-latest', type=int, default=3,
                              help='Number of latest checkpoints to keep')
    cleanup_parser.add_argument('--no-keep-best', action='store_true',
                              help='Do not keep best model checkpoint')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        manager = ImageNetManager(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            device=args.device
        )
        manager.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            use_mixup=not args.no_mixup,
            resume_from=args.resume
        )
    
    elif args.command == 'validate-dataset':
        manager = ImageNetManager(data_dir=args.data_dir)
        manager.validate_dataset()
    
    elif args.command == 'evaluate':
        manager = ImageNetManager(
            data_dir=args.data_dir,
            device=args.device
        )
        manager.evaluate(
            model_path=args.model_path,
            batch_size=args.batch_size
        )
    
    elif args.command == 'plot':
        manager = ImageNetManager(save_dir=args.save_dir)
        manager.plot_training_history(args.history_path)
    
    elif args.command == 'list-checkpoints':
        manager = ImageNetManager(save_dir=args.save_dir)
        manager.list_checkpoints()
    
    elif args.command == 'cleanup':
        manager = ImageNetManager(save_dir=args.save_dir)
        manager.cleanup_checkpoints(
            keep_best=not args.no_keep_best,
            keep_latest=args.keep_latest
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
