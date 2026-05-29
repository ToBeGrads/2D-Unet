import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# LOSS FUNCTIONS
# Shared between 2D U-Net and Swin-UNet.
#
# All losses ignore the background class (index 0).
# Class weights are passed only to the CE component;
# the region-based components handle imbalance structurally.
#
# ==========================================


class DiceLoss(nn.Module):
    """
    Soft Dice loss averaged over foreground classes only (STN=1, RN=2).
    Used as the region-based component of CEDiceLoss.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality   = torch.sum(probs + targets_one_hot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Ignore background (index 0); average over STN (1) and RN (2)
        return 1.0 - (dice_score[1] + dice_score[2]) / 2.0


class CEDiceLoss(nn.Module):
    """
    Hybrid Cross-Entropy + Dice loss.
    L = a * L_CE  +  b * L_Dice
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(weight=class_weights)
        self.dice        = DiceLoss()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return (
            self.ce_weight   * self.ce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


class FocalTverskyLoss(nn.Module):
    """
    Hybrid Cross-Entropy + Focal Tversky loss.
    L = a * L_CE  +  b * L_FT
    where L_FT = (1 - TI)^gamma,  gamma < 1 focuses on hard examples.

    Tversky index:
        TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    Setting beta > alpha biases towards higher recall (fewer missed structures).

    """
    def __init__(
        self,
        class_weights=None,
        alpha=0.3,       
        beta=0.7,        
        gamma=0.75,      
        smooth=1e-5,
        ce_weight=0.3,
        ft_weight=0.7,
    ):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss(weight=class_weights)
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.smooth    = smooth
        self.ce_weight = ce_weight
        self.ft_weight = ft_weight

    def forward(self, logits, targets):
        # --- Cross-Entropy component ---
        ce_loss = self.ce(logits, targets)

        # --- Focal Tversky component ---
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        dims = (0, 2, 3)
        TP = torch.sum(probs * targets_one_hot,             dim=dims)
        FP = torch.sum(probs * (1.0 - targets_one_hot),    dim=dims)
        FN = torch.sum((1.0 - probs) * targets_one_hot,    dim=dims)

        tversky       = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)

        # Foreground classes only: STN (index 1) and RN (index 2)
        ft_loss = (focal_tversky[1] + focal_tversky[2]) / 2.0

        return self.ce_weight * ce_loss + self.ft_weight * ft_loss




class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss: Focal CE  +  Focal Tversky.
    L = L_FocalCE  +  L_FT

    Focal CE:      L_FocalCE = (1 - p_t)^gamma * L_CE
    Focal Tversky: L_FT      = (1 - TI)^gamma
                               averaged over foreground classes only.

    """
    def __init__(self, class_weights=None, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # --- Focal Cross-Entropy ---
        ce_loss  = self.ce(logits, targets)          # shape: [B, H, W]
        pt       = torch.exp(-ce_loss)
        focal_ce = ((1.0 - pt) ** self.gamma * ce_loss).mean()

        # --- Focal Tversky ---
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        dims = (0, 2, 3)
        TP = torch.sum(probs * targets_one_hot,          dim=dims)
        FP = torch.sum(probs * (1.0 - targets_one_hot), dim=dims)
        FN = torch.sum((1.0 - probs) * targets_one_hot, dim=dims)

        tversky       = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)

        # Foreground classes only: STN (index 1) and RN (index 2)
        ft_loss = (focal_tversky[1] + focal_tversky[2]) / 2.0

        return focal_ce + ft_loss