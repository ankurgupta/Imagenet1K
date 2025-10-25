"""
Configuration file for ImageNet ResNet-50 training
"""

# Dataset paths
IMAGENET_PATHS = {
    'train': '/mnt/imagenet/train',
    'validation': '/mnt/imagenet/validation', 
    'test': '/mnt/imagenet/test'
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 90,
    'batch_size': 32,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'use_mixup': True,
    'mixup_alpha': 0.2,
    'grad_clip': 1.0
}

# Model configuration
MODEL_CONFIG = {
    'num_classes': 1000,  # ImageNet has 1000 classes
    'input_size': 224,
    'pretrained': False
}

# Data augmentation
AUGMENTATION_CONFIG = {
    'train': {
        'random_resized_crop': 224,
        'random_horizontal_flip': True,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2
        },
        'random_affine': {
            'degrees': 15,
            'translate': (0.1, 0.1)
        }
    },
    'val': {
        'resize': 256,
        'center_crop': 224
    }
}

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Checkpoint configuration
CHECKPOINT_CONFIG = {
    'save_dir': './checkpoints',
    'save_every': 10,  # Save checkpoint every N epochs
    'keep_latest': 3,  # Keep latest N checkpoints
    'keep_best': True
}

# Learning rate schedule
LR_SCHEDULE = {
    'type': 'step',
    'step_size': 30,
    'gamma': 0.1
}

# Device configuration
DEVICE_CONFIG = {
    'auto': True,  # Automatically detect CUDA
    'mixed_precision': False,  # Use automatic mixed precision
    'compile': False  # Use torch.compile (PyTorch 2.0+)
}
