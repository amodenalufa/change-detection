"""
Visualize data augmentation effects for change detection dataset.
Check alignment between T1, T2 and label after augmentation.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import ChangeDetectionDataset


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def concat_visualization(t1, t2, label):
    h, w = t1.shape[1], t1.shape[2]
    t1_np = t1.permute(1, 2, 0).numpy()
    t2_np = t2.permute(1, 2, 0).numpy()
    label_np = label.numpy()
    combined = np.concatenate([t1_np, t2_np], axis=1)
    label_overlay = np.zeros_like(combined)
    label_overlay[:, :, 0] = np.concatenate([label_np, label_np], axis=1) * 0.5
    vis = np.clip(combined + label_overlay, 0, 1)
    vis[:, w-1:w+1, :] = 1.0
    return vis


def visualize(data_root, split='train', num_samples=4, aug_times=3, save_path='aug_check.png'):
    dataset_orig = ChangeDetectionDataset(
        data_root=data_root,
        split=split,
        img_size=256,
        normalize=True,
        augmentation=False
    )
    
    dataset_aug = ChangeDetectionDataset(
        data_root=data_root,
        split=split,
        img_size=256,
        normalize=True,
        augmentation=True
    )
    
    fig, axes = plt.subplots(num_samples, 1 + aug_times, figsize=(4*(1+aug_times), 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        sample = dataset_orig[i]
        t1 = denormalize(sample['img_t1'])
        t2 = denormalize(sample['img_t2'])
        label = sample['label']
        vis = concat_visualization(t1, t2, label)
        axes[i, 0].imshow(vis)
        axes[i, 0].set_title(f'Original {i}', fontsize=9)
        axes[i, 0].axis('off')
        
        for j in range(aug_times):
            sample_aug = dataset_aug[i]
            t1_aug = denormalize(sample_aug['img_t1'])
            t2_aug = denormalize(sample_aug['img_t2'])
            label_aug = sample_aug['label']
            vis_aug = concat_visualization(t1_aug, t2_aug, label_aug)
            diff = torch.abs(t1_aug - t2_aug).mean().item()
            axes[i, j+1].imshow(vis_aug)
            axes[i, j+1].set_title(f'Aug{j+1} D:{diff:.3f}', fontsize=9)
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/LEVIR-CD')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--aug_times', type=int, default=3)
    parser.add_argument('--save', type=str, default='aug_check.png')
    args = parser.parse_args()
    
    visualize(args.data_root, args.split, args.samples, args.aug_times, args.save)