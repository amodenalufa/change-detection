"""
评估指标模块
包含变化检测常用的评估指标
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pixel_accuracy(pred, target):
    """
    像素准确率
    Args:
        pred: 预测结果 [B, H, W] 或 [B, C, H, W]
        target: 真实标签 [B, H, W]
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    correct = (pred == target).sum().item()
    total = target.numel()
    
    return correct / total if total > 0 else 0


def intersection_and_union(pred, target, num_classes=2):
    """
    计算交并比
    Args:
        pred: 预测结果 [B, H, W] 或 [B, C, H, W]
        target: 真实标签 [B, H, W]
        num_classes: 类别数
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    # 展平
    pred = pred.flatten()
    target = target.flatten()
    
    # 计算交集和并集
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    
    for i in range(num_classes):
        pred_i = (pred == i)
        target_i = (target == i)
        
        intersection[i] = (pred_i & target_i).sum().item()
        union[i] = (pred_i | target_i).sum().item()
    
    return intersection, union


def iou_score(pred, target, num_classes=2, ignore_index=None):
    """
    IoU (Intersection over Union) / Jaccard Index
    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数
        ignore_index: 忽略的类别索引
    """
    intersection, union = intersection_and_union(pred, target, num_classes)
    
    iou_per_class = intersection / (union + 1e-8)
    
    if ignore_index is not None:
        mask = torch.ones(num_classes, dtype=torch.bool)
        mask[ignore_index] = False
        iou_per_class = iou_per_class[mask]
    
    return iou_per_class.mean().item()


def dice_score(pred, target, num_classes=2, smooth=1e-8):
    """
    Dice系数 / F1-Score
    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数
        smooth: 平滑因子
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    # 展平
    pred = pred.flatten()
    target = target.flatten()
    
    dice_per_class = torch.zeros(num_classes, device=pred.device)
    
    for i in range(num_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        
        intersection = (pred_i * target_i).sum()
        dice_per_class[i] = (2. * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)
    
    return dice_per_class.mean().item()


def precision_recall_f1(pred, target, num_classes=2):
    """
    计算精确率、召回率、F1分数
    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    # 转换为numpy
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_np, pred_np, labels=list(range(num_classes)), average=None, zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_mean': precision.mean(),
        'recall_mean': recall.mean(),
        'f1_mean': f1.mean()
    }


def change_detection_metrics(pred, target):
    """
    变化检测专用指标
    专注于变化类别的检测性能
    Args:
        pred: 预测结果 [B, H, W] 或 [B, C, H, W]
        target: 真实标签 [B, H, W]
    Returns:
        dict: 包含各种指标的字典
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    # 展平
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 转换为numpy用于sklearn
    pred_np = pred_flat.cpu().numpy()
    target_np = target_flat.cpu().numpy()
    
    # 混淆矩阵
    cm = confusion_matrix(target_np, pred_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 计算指标
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-8)  # Overall Accuracy
    
    # 变化类别指标（class 1）
    precision_change = tp / (tp + fp + 1e-8)
    recall_change = tp / (tp + fn + 1e-8)
    f1_change = 2 * precision_change * recall_change / (precision_change + recall_change + 1e-8)
    
    # IoU for change class
    iou_change = tp / (tp + fp + fn + 1e-8)
    
    # 未变化类别指标（class 0）
    precision_nochange = tn / (tn + fn + 1e-8)
    recall_nochange = tn / (tn + fp + 1e-8)
    f1_nochange = 2 * precision_nochange * recall_nochange / (precision_nochange + recall_nochange + 1e-8)
    
    # Mean IoU
    iou_nochange = tn / (tn + fn + fp + 1e-8)
    miou = (iou_change + iou_nochange) / 2
    
    # Kappa系数
    total = tp + tn + fp + fn
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total)
    kappa = (po - pe) / (1 - pe + 1e-8)
    
    return {
        'OA': oa,
        'mIoU': miou,
        'IoU_change': iou_change,
        'IoU_nochange': iou_nochange,
        'F1_change': f1_change,
        'F1_nochange': f1_nochange,
        'Precision_change': precision_change,
        'Recall_change': recall_change,
        'Precision_nochange': precision_nochange,
        'Recall_nochange': recall_nochange,
        'Kappa': kappa,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


class MetricsTracker:
    """
    指标追踪器，用于累积多个batch的指标
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total_tp = 0
        self.total_tn = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, pred, target):
        """
        更新累积指标
        Args:
            pred: 预测结果 [B, H, W] 或 [B, C, H, W]
            target: 真实标签 [B, H, W]
        """
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        # 展平
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        pred_np = pred_flat.cpu().numpy()
        target_np = target_flat.cpu().numpy()
        
        # 混淆矩阵
        cm = confusion_matrix(target_np, pred_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        self.total_tp += tp
        self.total_tn += tn
        self.total_fp += fp
        self.total_fn += fn
        self.total_pixels += len(target_np)
        self.correct_pixels += (pred_np == target_np).sum()
    
    def get_metrics(self):
        """获取累积后的指标"""
        tp, tn, fp, fn = self.total_tp, self.total_tn, self.total_fp, self.total_fn
        
        oa = (tp + tn) / (self.total_pixels + 1e-8)
        iou_change = tp / (tp + fp + fn + 1e-8)
        iou_nochange = tn / (tn + fn + fp + 1e-8)
        miou = (iou_change + iou_nochange) / 2
        
        precision_change = tp / (tp + fp + 1e-8)
        recall_change = tp / (tp + fn + 1e-8)
        f1_change = 2 * precision_change * recall_change / (precision_change + recall_change + 1e-8)
        
        total = tp + tn + fp + fn
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total)
        kappa = (po - pe) / (1 - pe + 1e-8)
        
        return {
            'OA': oa,
            'mIoU': miou,
            'IoU_change': iou_change,
            'F1_change': f1_change,
            'Precision_change': precision_change,
            'Recall_change': recall_change,
            'Kappa': kappa
        }


if __name__ == '__main__':
    # 测试代码
    print("Testing metrics...")
    
    # 创建模拟数据
    pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    target = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]])
    
    # 测试各项指标
    print(f"\nPixel Accuracy: {pixel_accuracy(pred, target):.4f}")
    print(f"IoU Score: {iou_score(pred, target):.4f}")
    print(f"Dice Score: {dice_score(pred, target):.4f}")
    
    # 测试变化检测专用指标
    metrics = change_detection_metrics(pred, target)
    print(f"\nChange Detection Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试MetricsTracker
    tracker = MetricsTracker()
    tracker.update(pred, target)
    tracker.update(pred, target)
    
    cum_metrics = tracker.get_metrics()
    print(f"\nCumulative Metrics:")
    for key, value in cum_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMetrics tests passed!")
