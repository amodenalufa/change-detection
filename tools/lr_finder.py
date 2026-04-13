"""
Learning Rate Range Test to find optimal learning rate.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import get_dataloader
from models import create_model


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}
        self.best_loss = None
        self.initial_state = None
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100, smooth_f=0.05):
        self.initial_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        # Calculate lr multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        avg_loss = 0.0
        batch_num = 0
        
        self.model.train()
        train_iter = iter(train_loader)
        
        for iteration in range(num_iter):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                
            batch_num += 1
            
            # Forward pass
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            label = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(img_t1, img_t2)
            loss, _ = self.criterion(output, label)
            
            # Check for explosion
            if torch.isnan(loss) or loss.item() > 1e6:
                print(f"Stopping early at iter {iteration}, loss exploded: {loss.item()}")
                break
                
            # Record
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            if self.best_loss is None or avg_loss < self.best_loss:
                self.best_loss = avg_loss
                
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Update lr
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        print(f"Best loss: {self.best_loss:.4f} at lr ~ {self.history['lr'][np.argmin(self.history['loss'])]}")
        
    def plot(self, skip_start=10, skip_end=5, save_path='lr_finder.png'):
        lrs = self.history['lr'][skip_start:-skip_end]
        losses = self.history['loss'][skip_start:-skip_end]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.grid(True, alpha=0.3)
        
        # Find steepest descent
        min_grad_idx = np.gradient(losses).argmin()
        steepest_lr = lrs[min_grad_idx]
        plt.axvline(steepest_lr, color='r', linestyle='--', label=f'Steepest: {steepest_lr:.2e}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
        return steepest_lr
        
    def reset(self):
        """Reset model to initial state."""
        if self.initial_state:
            self.model.load_state_dict(self.initial_state)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/LEVIR-CD')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--start_lr', type=float, default=1e-7)
    parser.add_argument('--end_lr', type=float, default=1)
    parser.add_argument('--num_iter', type=int, default=200)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create simple config
    config = {
        'model': 'base',
        'num_classes': 2,
        'pretrained': True,
        'fusion': 'diff'
    }
    
    model = create_model(config).to(device)
    
    # Use a single optimizer for LR finder
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    
    from utils.losses import ChangeDetectionLoss
    criterion = ChangeDetectionLoss(ce_weight=1.0, dice_weight=1.0, num_classes=2)
    
    train_loader = get_dataloader(
        dataset_name='levir-cd',
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        img_size=args.img_size,
        augmentation=True,
        num_workers=4
    )
    
    finder = LRFinder(model, optimizer, criterion, device)
    finder.range_test(train_loader, start_lr=args.start_lr, end_lr=args.end_lr, num_iter=args.num_iter)
    steepest_lr = finder.plot()
    print(f"\nSuggested initial LR: {steepest_lr / 10:.2e} (steepest / 10)")
    finder.reset()
    
if __name__ == '__main__':
    main()