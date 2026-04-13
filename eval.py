"""
Evaluation script for change detection model
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from data.dataset import get_dataloader
from utils.metrics import change_detection_metrics, MetricsTracker


class Evaluator:
    """Evaluator class for change detection model"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.output_dir = config.get('output_dir')
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Create model
        self.model = self._create_model()

        # Load checkpoint
        self.load_checkpoint(config['checkpoint'])

    def _create_model(self):
        """Create model"""
        model = create_model(self.config)

        # Multi-GPU support
        if self.config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(self.device)
        return model

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"[Load] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.best_epoch = checkpoint.get('epoch', 0)
            self.best_miou = checkpoint.get('best_miou', 0)
            print(f"[Load] Epoch: {self.best_epoch}, Best mIoU: {self.best_miou:.4f}")
        else:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            print("[Load] Loaded model weights only")

    @torch.no_grad()
    def evaluate(self, split='test'):
        """Evaluate on specified split"""
        self.model.eval()

        # Create dataloader
        dataloader = get_dataloader(
            dataset_name=self.config['dataset'],
            data_root=self.config['data_root'],
            split=split,
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            augmentation=False,
            num_workers=self.config['num_workers'],
            shuffle=False
        )

        print(f"\n[Eval] Evaluating on {split} set ({len(dataloader.dataset)} samples)")

        # Metrics tracker
        metrics_tracker = MetricsTracker(num_classes=self.config['num_classes'])
        all_results = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            label = batch['label'].to(self.device)
            img_names = batch['img_name']

            # Forward pass
            output = self.model(img_t1, img_t2)

            # Update metrics
            metrics_tracker.update(output, label)

            # Save predictions if needed
            if self.output_dir:
                preds = torch.argmax(output, dim=1).cpu().numpy()
                for i, name in enumerate(img_names):
                    result = {
                        'name': name,
                        'pred': preds[i],
                        'label': label[i].cpu().numpy()
                    }
                    all_results.append(result)

        # Get final metrics
        metrics = metrics_tracker.get_metrics()
        return metrics, all_results

    def save_predictions(self, results):
        """Save prediction results"""
        if not self.output_dir:
            return

        pred_dir = os.path.join(self.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        print(f"\n[Save] Saving predictions to {pred_dir}")

        for result in tqdm(results, desc="Saving predictions"):
            name = result['name']
            pred = result['pred']

            # Convert to image
            pred_img = (pred * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_img)

            # Save
            save_path = os.path.join(pred_dir, name)
            pred_pil.save(save_path)

        print(f"[Save] Saved {len(results)} predictions")

    def print_metrics(self, metrics, title="Evaluation Results"):
        """Print evaluation metrics with IoU, Precision, Recall"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Overall Accuracy (OA):    {metrics['OA']:.4f}")
        print(f"Mean IoU (mIoU):          {metrics['mIoU']:.4f}")
        print(f"IoU (Change):             {metrics['IoU_change']:.4f}")
        print(f"IoU (No Change):          {metrics['IoU_nochange']:.4f}")
        print(f"F1 Score (Change):        {metrics['F1_change']:.4f}")
        print(f"Precision (Change):       {metrics['Precision_change']:.4f}")
        print(f"Recall (Change):          {metrics['Recall_change']:.4f}")
        print(f"Kappa Coefficient:        {metrics['Kappa']:.4f}")
        print(f"{'='*60}\n")

    def save_metrics(self, metrics, filename='metrics.json'):
        """Save evaluation metrics to file"""
        if not self.output_dir:
            return

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"[Save] Metrics saved to {filepath}")

    @torch.no_grad()
    def inference(self, img_t1_path, img_t2_path):
        """Inference on single image pair"""
        from torchvision import transforms

        self.model.eval()

        # Load images
        img_t1 = Image.open(img_t1_path).convert('RGB')
        img_t2 = Image.open(img_t2_path).convert('RGB')

        original_size = img_t1.size

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_t1 = transform(img_t1).unsqueeze(0).to(self.device)
        img_t2 = transform(img_t2).unsqueeze(0).to(self.device)

        # Inference
        output = self.model(img_t1, img_t2)

        # Get predictions
        probs = torch.softmax(output, dim=1)
        change_prob = probs[0, 1].cpu().numpy()
        change_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

        # Resize to original size
        change_prob = Image.fromarray((change_prob * 255).astype(np.uint8))
        change_prob = change_prob.resize(original_size, Image.BILINEAR)
        change_prob = np.array(change_prob) / 255.0

        change_mask = Image.fromarray((change_mask * 255).astype(np.uint8))
        change_mask = change_mask.resize(original_size, Image.NEAREST)
        change_mask = np.array(change_mask) > 127

        return change_mask, change_prob


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Change Detection Evaluation')

    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='levir-cd')
    parser.add_argument('--split', type=str, default='test')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--fusion', type=str, default='concat', choices=['diff', 'concat', 'sum'])

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_pred', action='store_true')

    # Single image inference
    parser.add_argument('--img_t1', type=str, default=None)
    parser.add_argument('--img_t2', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    config = vars(args)

    evaluator = Evaluator(config)

    # Single image inference mode
    if args.img_t1 and args.img_t2:
        print(f"\n[Inference] T1: {args.img_t1}")
        print(f"[Inference] T2: {args.img_t2}")

        change_mask, change_prob = evaluator.inference(args.img_t1, args.img_t2)

        if args.output_path:
            mask_img = Image.fromarray((change_mask * 255).astype(np.uint8))
            mask_path = args.output_path.replace('.png', '_mask.png')
            mask_img.save(mask_path)
            print(f"[Save] Mask saved to {mask_path}")

            prob_img = Image.fromarray((change_prob * 255).astype(np.uint8))
            prob_path = args.output_path.replace('.png', '_prob.png')
            prob_img.save(prob_path)
            print(f"[Save] Probability saved to {prob_path}")

        print(f"\n[Result] Changed pixels: {change_mask.sum()}")
        print(f"[Result] Change ratio: {change_mask.mean()*100:.2f}%")

    # Batch evaluation mode
    else:
        metrics, results = evaluator.evaluate(args.split)

        evaluator.print_metrics(metrics, f"{args.split.upper()} Set Results")

        if args.output_dir:
            evaluator.save_metrics(metrics, f'{args.split}_metrics.json')

        if args.save_pred and args.output_dir:
            evaluator.save_predictions(results)

        print("\n[Done] Evaluation completed!")


if __name__ == '__main__':
    main()