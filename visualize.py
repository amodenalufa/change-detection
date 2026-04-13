"""
Inference Visualization for Change Detection Model
Generates 4-panel comparison: [T1] [T2] [GT] [Prediction]
"""

import os
import sys
import argparse
import glob
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model


class ModelLoader:
    """Load trained model from checkpoint"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, path):
        """Initialize model architecture and load weights"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        # Model configuration (must match training config)
        config = {
            'num_classes': 2,
            'pretrained': False,  # No need for pretrained weights during inference
            'bilinear': True,
            'fusion': 'concat'    # Must match training: 'diff', 'concat', or 'sum'
        }
        
        # Create model
        model = create_model(config)
        model = model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both DataParallel and single GPU checkpoints
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.eval()
        
        print(f"[Load] Model loaded from {path}")
        
        return model
    
    @torch.no_grad()
    def predict(self, img_t1, img_t2):
        """
        Run inference on image pair
        Args:
            img_t1: Tensor [1, 3, H, W] or [3, H, W]
            img_t2: Tensor [1, 3, H, W] or [3, H, W]
        Returns:
            prob: Change probability map [H, W] numpy array
            mask: Binary prediction [H, W] numpy array
        """
        # Add batch dimension if needed
        if img_t1.dim() == 3:
            img_t1 = img_t1.unsqueeze(0)
        if img_t2.dim() == 3:
            img_t2 = img_t2.unsqueeze(0)
            
        img_t1 = img_t1.to(self.device)
        img_t2 = img_t2.to(self.device)
        
        # Forward pass
        logits = self.model(img_t1, img_t2)
        
        # Convert to probability and prediction
        probs = torch.softmax(logits, dim=1)
        change_prob = probs[0, 1]  # Probability of change class
        change_mask = torch.argmax(logits, dim=1)[0]
        
        return change_prob.cpu().numpy(), change_mask.cpu().numpy()


class ChangeDetectionVisualizer:
    """Visualization toolkit for change detection results"""
    
    def __init__(self, model_loader, save_dir='./visualization_results'):
        self.model = model_loader
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Preprocessing pipeline (must match training)
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
    def load_image(self, path):
        """Load and preprocess image"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert('RGB')
        return img
    
    def load_label(self, path):
        """Load ground truth mask"""
        if path is None or not os.path.exists(path):
            return None
        label = Image.open(path).convert('L')
        label = label.resize((256, 256), Image.NEAREST)
        return np.array(label) > 0
    
    def process_pair(self, t1_path, t2_path, gt_path=None):
        """
        Process a single image pair
        Returns original images (for display) and predictions
        """
        # Load images
        img_t1_pil = self.load_image(t1_path)
        img_t2_pil = self.load_image(t2_path)
        gt_mask = self.load_label(gt_path)
        
        # Save original for visualization
        t1_display = np.array(img_t1_pil.resize((256, 256)))
        t2_display = np.array(img_t2_pil.resize((256, 256)))
        
        # Preprocess for model
        t1_tensor = self.transform(img_t1_pil)
        t2_tensor = self.transform(img_t2_pil)
        
        # Predict
        prob_map, pred_mask = self.model.predict(t1_tensor, t2_tensor)
        
        return {
            't1': t1_display,
            't2': t2_display,
            'gt': gt_mask,
            'pred': pred_mask,
            'prob': prob_map
        }
    
    def visualize(self, results, save_name='result.png', threshold=0.5):
        """
        Create 4-panel visualization
        Args:
            results: dict with keys 't1', 't2', 'gt', 'pred', 'prob'
            save_name: output filename
            threshold: binarization threshold for probability map
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # T1 Image (Before)
        axes[0].imshow(results['t1'])
        axes[0].set_title('T1 Image (Before)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # T2 Image (After)
        axes[1].imshow(results['t2'])
        axes[1].set_title('T2 Image (After)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Ground Truth
        if results['gt'] is not None:
            axes[2].imshow(results['gt'], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Ground Truth', fontsize=12, fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, 'No GT\nAvailable', 
                        ha='center', va='center', fontsize=14)
            axes[2].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Prediction
        axes[3].imshow(results['pred'], cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f'Prediction (>{threshold})', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[Save] Visualization saved to {save_path}")
        
        # plt.show()
        plt.close()
    
    def visualize_with_heatmap(self, results, save_name='result_heatmap.png'):
        """Optional: Create visualization with probability heatmap"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(results['t1'])
        axes[0].set_title('T1 Image')
        axes[0].axis('off')
        
        axes[1].imshow(results['t2'])
        axes[1].set_title('T2 Image')
        axes[1].axis('off')
        
        # Probability heatmap
        im = axes[2].imshow(results['prob'], cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Change Probability')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[Save] Heatmap saved to {save_path}")
        plt.close()


def main():
    """Main execution function"""
    
    # Path to best model checkpoint (modify this)
    CHECKPOINT_PATH = './results/exp_20260412_201052/checkpoints/best_model.pth'
    
    # Input image paths (modify these for manual input)
    T1_PATH = './data/LEVIR-CD/test/A/test_1.png'  # Modify: path to time 1 image
    T2_PATH = './data/LEVIR-CD/test/B/test_1.png'  # Modify: path to time 2 image
    GT_PATH = './data/LEVIR-CD/test/label/test_1.png'  # Modify: path to ground truth
    
    # Output settings
    OUTPUT_DIR = './visualization/0412'
    SAVE_NAME = 'result_1.png'
    
    # Device
    DEVICE = 'cuda'  # or 'cpu'
    
    # Initialize model
    model_loader = ModelLoader(CHECKPOINT_PATH, device=DEVICE)
    
    # Initialize visualizer
    visualizer = ChangeDetectionVisualizer(model_loader, save_dir=OUTPUT_DIR)
    
    # Process single image pair
    print(f"\n[Process] Processing image pair...")
    print(f"  T1: {T1_PATH}")
    print(f"  T2: {T2_PATH}")
    if GT_PATH:
        print(f"  GT: {GT_PATH}")
    
    results = visualizer.process_pair(T1_PATH, T2_PATH, GT_PATH)
    
    # Generate visualization
    print(f"\n[Visualize] Generating comparison figure...")
    visualizer.visualize(results, save_name=SAVE_NAME)
    
    # Optional: Also save heatmap version
    visualizer.visualize_with_heatmap(results, save_name=SAVE_NAME.replace('.png', '_heatmap.png'))
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


def batch_visualize():
    """
    Batch processing mode: visualize first N samples from validation/test set
    Uncomment and modify to use batch mode
    """
    CHECKPOINT_PATH = './results/exp_20260412_201052/checkpoints/best_model.pth'
    DATA_ROOT = './data/LEVIR-CD'
    SPLIT = 'test'  # 'val' or 'test'
    NUM_SAMPLES = 5
    OUTPUT_DIR = './visualization/0412'
    DEVICE = 'cuda'
    
    model_loader = ModelLoader(CHECKPOINT_PATH, device=DEVICE)
    visualizer = ChangeDetectionVisualizer(model_loader, save_dir=OUTPUT_DIR)
    
    # Get image list
    t1_dir = os.path.join(DATA_ROOT, SPLIT, 'A')
    t2_dir = os.path.join(DATA_ROOT, SPLIT, 'B')
    label_dir = os.path.join(DATA_ROOT, SPLIT, 'label')
    
    image_files = sorted(glob.glob(os.path.join(t1_dir, '*.png')))[:NUM_SAMPLES]
    
    print(f"[Batch] Processing {len(image_files)} samples from {SPLIT} set...")
    
    for idx, t1_path in enumerate(image_files):
        filename = os.path.basename(t1_path)
        t2_path = os.path.join(t2_dir, filename)
        gt_path = os.path.join(label_dir, filename)
        
        if not os.path.exists(t2_path):
            continue
            
        print(f"[{idx+1}/{len(image_files)}] {filename}")
        
        results = visualizer.process_pair(t1_path, t2_path, gt_path)
        visualizer.visualize(results, save_name=f'sample_{idx:03d}_{filename}')
    
    print(f"[Batch] All results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    # Run single image visualization (modify paths in main())
    # main()
    
    # Uncomment below to run batch visualization instead
    batch_visualize()