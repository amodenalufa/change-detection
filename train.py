"""
Training script for change detection baseline model
Optimized for server with dual RTX 3090 GPUs
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from data.dataset import get_dataloader
#from utils.losses import ChangeDetectionLoss
from utils.losses import FocalDiceLoss
from utils.metrics import MetricsTracker


class Trainer:
    """Trainer class for change detection model"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        # Enable cuDNN benchmark for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[GPU] {gpu_name} | Memory: {gpu_mem:.1f}GB")

        # Automatic Mixed Precision (AMP)
        self.use_amp = config.get('amp_enabled', True)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("[AMP] Automatic Mixed Precision enabled")

        # Create output directories
        self.exp_dir = os.path.join(config['output_dir'], config['exp_name'])
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)

        # Create model
        self.model = self._create_model()

        # Create dataloaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')

        # Loss function
        #self.criterion = ChangeDetectionLoss(
        #    ce_weight=config.get('ce_weight', 1.0),
        #    dice_weight=config.get('dice_weight', 1.0),
        #    num_classes=config['num_classes']
        #)
        self.criterion = FocalDiceLoss(
            focal_weight=1.0,
            dice_weight=1.0,
            alpha=0.6,
            gamma=2.0,
            num_classes=config['num_classes']
        )

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_iou = 0.0
        self.best_epoch = 0
        self.global_step = 0

        # Resume from checkpoint if specified
        if config.get('resume'):
            self.load_checkpoint(config['resume'])
            
        # Early Stopping
        self.patience = config.get('patience', 20)
        self.min_delta = config.get('min_delta', 0.001)
        self.early_stop_counter = 0
        self.best_val_iou = 0.0
        self.early_stop = False

    def _create_model(self):
        """Create and initialize model"""
        model = create_model(self.config)

        # Multi-GPU support
        if self.config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
            print(f"[Multi-GPU] Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        model = model.to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"[Model] Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        return model

    def _create_dataloader(self, split):
        """Create dataloader for specified split"""
        return get_dataloader(
            dataset_name=self.config['dataset'],
            data_root=self.config['data_root'],
            split=split,
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            augmentation=(split == 'train'),
            num_workers=self.config['num_workers']
        )

    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.0001)

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create scheduler with warmup and cosine annealing."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        epochs = self.config['epochs']
        warmup_epochs = self.config.get('warmup_epochs', 0)
        base_lr = self.config['learning_rate']
    
        if scheduler_name == 'cosine_warmup':
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup: 0 -> 1
                    return (epoch + 1) / warmup_epochs
                else:
                    # Cosine decay: 1 -> 0
                    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif scheduler_name == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'multistep':
            milestones = self.config.get('milestones', [40, 70, 90])
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
            )
        else:
            return None

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        #ce_loss_sum = 0.0
        focal_loss_sum = 0.0
        dice_loss_sum = 0.0
        metrics_tracker = MetricsTracker(num_classes=self.config['num_classes'])

        for batch_idx, batch in enumerate(self.train_loader):
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            label = batch['label'].to(self.device)

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(img_t1, img_t2)
                    loss, loss_dict = self.criterion(output, label)
            else:
                output = self.model(img_t1, img_t2)
                loss, loss_dict = self.criterion(output, label)

            # Backward pass with AMP
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Accumulate statistics
            total_loss += loss.item()
            #ce_loss_sum += loss_dict['ce']
            focal_loss_sum += loss_dict['focal']
            dice_loss_sum += loss_dict['dice']
            metrics_tracker.update(output, label)

            self.global_step += 1

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
                #self.writer.add_scalar('Train/ce_loss', loss_dict['ce'], self.global_step)
                self.writer.add_scalar('Train/focal_loss', loss_dict['focal'], self.global_step)
                self.writer.add_scalar('Train/dice_loss', loss_dict['dice'], self.global_step)

        # Calculate average metrics
        avg_loss = total_loss / len(self.train_loader)
        #avg_ce = ce_loss_sum / len(self.train_loader)
        avg_focal = focal_loss_sum / len(self.train_loader)
        avg_dice = dice_loss_sum / len(self.train_loader)
        metrics = metrics_tracker.get_metrics()

        return {
            'loss': avg_loss,
            #'ce_loss': avg_ce,
            'focal_loss': avg_focal,
            'dice_loss': avg_dice,
            **metrics
        }

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()

        total_loss = 0.0
        #ce_loss_sum = 0.0
        focal_loss_sum = 0.0
        dice_loss_sum = 0.0
        metrics_tracker = MetricsTracker(num_classes=self.config['num_classes'])

        for batch in self.val_loader:
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            label = batch['label'].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(img_t1, img_t2)
                    loss, loss_dict = self.criterion(output, label)
            else:
                output = self.model(img_t1, img_t2)
                loss, loss_dict = self.criterion(output, label)

            # Accumulate statistics
            total_loss += loss.item()
            #ce_loss_sum += loss_dict['ce']
            focal_loss_sum += loss_dict['focal']
            dice_loss_sum += loss_dict['dice']
            metrics_tracker.update(output, label)

        # Calculate average metrics
        avg_loss = total_loss / len(self.val_loader)
        #avg_ce = ce_loss_sum / len(self.val_loader)
        avg_focal = focal_loss_sum / len(self.val_loader)
        avg_dice = dice_loss_sum / len(self.val_loader)
        metrics = metrics_tracker.get_metrics()

        return {
            'loss': avg_loss,
            #'ce_loss': avg_ce,
            'focal_loss': avg_focal,
            'dice_loss': avg_dice,
            **metrics
        }

    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"[Save] Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """Load checkpoint"""
        print(f"[Load] Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state (handle DataParallel)
        state_dict = checkpoint['model_state_dict']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_iou = checkpoint['best_iou']
        self.best_epoch = checkpoint['best_epoch']

        print(f"[Load] Resumed from epoch {self.current_epoch}, best IoU: {self.best_iou:.4f}")

    def log_metrics(self, train_metrics, val_metrics):
        """Log metrics to TensorBoard"""
        epoch = self.current_epoch

        # Loss
        self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)

        # Metrics
        self.writer.add_scalar('Metrics/train_mIoU', train_metrics['mIoU'], epoch)
        self.writer.add_scalar('Metrics/val_mIoU', val_metrics['mIoU'], epoch)
        self.writer.add_scalar('Metrics/train_IoU', train_metrics['IoU_change'], epoch)
        self.writer.add_scalar('Metrics/val_IoU', val_metrics['IoU_change'], epoch)
        self.writer.add_scalar('Metrics/train_Precision', train_metrics['Precision_change'], epoch)
        self.writer.add_scalar('Metrics/val_Precision', val_metrics['Precision_change'], epoch)
        self.writer.add_scalar('Metrics/train_Recall', train_metrics['Recall_change'], epoch)
        self.writer.add_scalar('Metrics/val_Recall', val_metrics['Recall_change'], epoch)
        self.writer.add_scalar('Metrics/train_OA', train_metrics['OA'], epoch)
        self.writer.add_scalar('Metrics/val_OA', val_metrics['OA'], epoch)
        self.writer.add_scalar('Metrics/train_F1', train_metrics['F1_change'], epoch)
        self.writer.add_scalar('Metrics/val_F1', val_metrics['F1_change'], epoch)

        # Learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Train/learning_rate', lr, epoch)

    def train(self):
        """Main training loop with early stopping"""
        print("\n" + "=" * 60)
        print(f"Start Training: {self.config['exp_name']}")
        print(f"Epochs: {self.config['epochs']} | Batch size: {self.config['batch_size']}")
        print(f"Early Stopping: patience={self.patience}, min_delta={self.min_delta}")
        print("=" * 60 + "\n")

        start_epoch = self.current_epoch
        num_epochs = self.config['epochs']

        for epoch in range(start_epoch, num_epochs):
            if self.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
                
            self.current_epoch = epoch
            epoch_start = time.time()

            print("-" * 40)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['IoU_change'])
                else:
                    self.scheduler.step()

            # Log metrics
            self.log_metrics(train_metrics, val_metrics)

            # Print results
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Time: {epoch_time:.1f}s")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | OA: {train_metrics['OA']:.4f} | "
                  f"IoU: {train_metrics['IoU_change']:.4f} | F1: {train_metrics['F1_change']:.4f} | "
                  f"P: {train_metrics['Precision_change']:.4f} | R: {train_metrics['Recall_change']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | OA: {val_metrics['OA']:.4f} | "
                  f"IoU: {val_metrics['IoU_change']:.4f} | F1: {val_metrics['F1_change']:.4f} | "
                  f"P: {val_metrics['Precision_change']:.4f} | R: {val_metrics['Recall_change']:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

            # Early stopping logic (based on Change IoU)
            current_iou = val_metrics['IoU_change']
            if current_iou > self.best_val_iou + self.min_delta:
                # Improved
                self.best_val_iou = current_iou
                self.best_epoch = epoch
                self.early_stop_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"[Best] New best model! IoU: {self.best_val_iou:.4f}")
            else:
                # No improvement
                self.early_stop_counter += 1
                print(f"[EarlyStop] Counter: {self.early_stop_counter}/{self.patience}")
                
                if self.early_stop_counter >= self.patience:
                    self.early_stop = True
                    print(f"[EarlyStop] Triggered! No improvement for {self.patience} epochs.")

        # Training completed summary
        if self.early_stop:
            print("\n" + "=" * 60)
            print("Training Stopped Early (Early Stopping)")
        else:
            print("\n" + "=" * 60)
            print("Training Completed!")
            
        print(f"Best Epoch: {self.best_epoch + 1}")
        print(f"Best IoU: {self.best_val_iou:.4f}")
        print("=" * 60 + "\n")

        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Change Detection Training (Server)')

    # Data
    parser.add_argument('--data_root', type=str, required=False, help='Data root directory')
    parser.add_argument('--dataset', type=str, default='levir-cd', help='Dataset name')

    # Model
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--fusion', type=str, default='concat', choices=['diff', 'concat', 'sum'])

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'step', 'multistep', 'cosine', 'plateau'])

    # Loss
    #parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--focal_weight', type=float, default=1.0)
    parser.add_argument('--dice_weight', type=float, default=1.0)

    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use DataParallel for multi-GPU')
    parser.add_argument('--amp_enabled', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--num_workers', type=int, default=8)

    # Output
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--exp_name', type=str, default='exp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=10)

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    config = vars(args)

    # Override with config file if exists
    if os.path.exists('configs/server_config.py'):
        from configs.server_config import Config as ServerConfig
        for key, value in ServerConfig.to_dict().items():
            if value is not None:
                config[key.lower()] = value

    print(f"[Config Check] epochs={config.get('epochs')}, scheduler={config.get('scheduler')}, "
          f"warmup_epochs={config.get('warmup_epochs', 0)}, lr={config.get('learning_rate')}")
          
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()