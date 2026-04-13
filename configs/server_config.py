"""
Server Configuration for Change Detection Baseline
Optimized for single RTX 3090 (24GB)
"""

import os


class Config:
    """Base configuration class for server training"""

    # Data
    DATASET = 'levir-cd'
    DATA_ROOT = './data/LEVIR-CD'
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_WORKERS = 8

    # Model
    NUM_CLASSES = 2
    PRETRAINED = True
    BILINEAR = True
    FUSION = 'concat'

    # Training
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    OPTIMIZER = 'adamw'
    SCHEDULER = 'cosine_warmup'
    WARMUP_EPOCHS = 5
    MILESTONES = [40, 70, 90]
    GAMMA = 0.1

    # Loss
    #CE_WEIGHT = 1.0
    FOCAL_WEIGHT = 1.0
    DICE_WEIGHT = 1.0

    # Hardware
    DEVICE = 'cuda:0'
    AMP_ENABLED = True

    # Output
    OUTPUT_DIR = './results'
    SAVE_FREQ = 10
    RESUME = None

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_')
                and not callable(v)
                and not isinstance(v, classmethod)
                and not isinstance(v, staticmethod)
                and k.isupper()
        }

    @classmethod
    def update(cls, **kwargs):
        """Update config values"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


class LEVIRCDConfig(Config):
    """LEVIR-CD specific config"""
    DATASET = 'levir-cd'
    DATA_ROOT = './data/LEVIR-CD'


CONFIGS = {
    'server': Config,
    'levir-cd': LEVIRCDConfig,
}


def get_config(name='server'):
    """Get config by name"""
    return CONFIGS.get(name, Config)