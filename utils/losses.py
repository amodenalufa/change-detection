"""
损失函数模块
包含变化检测常用的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失
    """
    def __init__(self, weight=None, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W]
        """
        return self.ce(pred, target)


class DiceLoss(nn.Module):
    """
    Dice损失
    适用于类别不平衡的分割任务
    """
    def __init__(self, smooth=1e-8, num_classes=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]（未经过softmax）
            target: 真实标签 [B, H, W]
        """
        # 将预测转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 将target转换为one-hot编码
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # 展平
        pred_flat = pred.view(pred.size(0), self.num_classes, -1)
        target_flat = target_one_hot.view(target_one_hot.size(0), self.num_classes, -1)
        
        # 计算Dice系数
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回Dice损失（1 - Dice）
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal损失
    解决类别不平衡问题，关注难分类样本
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W]
        """
        # 计算交叉熵
        ce_loss = self.ce(pred, target)
        
        # 计算预测概率
        pred_prob = F.softmax(pred, dim=1)
        pt = torch.gather(pred_prob, 1, target.unsqueeze(1)).squeeze(1)
        
        # 计算Focal权重
        focal_weight = (1 - pt) ** self.gamma
        
        # Focal损失
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky损失
    通过调整alpha和beta参数来平衡精确率和召回率
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-8, num_classes=2):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 假阳性的权重
        self.beta = beta    # 假阴性的权重
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W]
        """
        # 将预测转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 将target转换为one-hot编码
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # 展平
        pred_flat = pred.view(pred.size(0), self.num_classes, -1)
        target_flat = target_one_hot.view(target_one_hot.size(0), self.num_classes, -1)
        
        # 计算TP, FP, FN
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Tversky指数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()
        

class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss 组合
    Focal 解决类别不平衡，Dice 直接优化 IoU
    """
    def __init__(self, focal_weight=1.0, dice_weight=1.0, alpha=0.5, gamma=2.0, num_classes=2):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
        self.dice_loss = DiceLoss(num_classes=num_classes)
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total = self.focal_weight * focal + self.dice_weight * dice
        
        return total, {'focal': focal.item(), 'dice': dice.item(), 'total': total.item()}


class ComboLoss(nn.Module):
    """
    组合损失
    结合多种损失函数
    """
    def __init__(self, weights=None, num_classes=2):
        super(ComboLoss, self).__init__()
        
        if weights is None:
            weights = {'ce': 1.0, 'dice': 1.0}
        
        self.weights = weights
        self.num_classes = num_classes
        
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W]
        """
        loss = 0
        
        if 'ce' in self.weights:
            loss += self.weights['ce'] * self.ce_loss(pred, target)
        
        if 'dice' in self.weights:
            loss += self.weights['dice'] * self.dice_loss(pred, target)
        
        return loss


class ChangeDetectionLoss(nn.Module):
    """
    变化检测专用损失
    默认使用CE + Dice的组合
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, num_classes=2):
        super(ChangeDetectionLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W]
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        loss = self.ce_weight * ce + self.dice_weight * dice
        
        return loss, {'ce': ce.item(), 'dice': dice.item()}


def get_loss_function(loss_name, **kwargs):
    """
    根据名称获取损失函数
    Args:
        loss_name: 损失函数名称
        **kwargs: 额外参数
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'ce' or loss_name == 'crossentropy':
        return CrossEntropyLoss(**kwargs)
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_name == 'combo':
        return ComboLoss(**kwargs)
    elif loss_name == 'cd':
        return ChangeDetectionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")