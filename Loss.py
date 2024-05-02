import torch
from monai.losses import DiceLoss, FocalLoss
from torch.nn import functional as F


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(DiceFocalLoss, self).__init__()
        self.dice_loss = DiceLoss(sigmoid=True, batch=True)
        self.ce_loss = FocalLoss(include_background=True, alpha=0.7, weight=[0.7])
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, output, target):
        dice_loss = self.dice_loss(output, target)
        ce_loss = self.ce_loss(output, target)
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss
    

class MultiScaleDiceFocalLoss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super().__init__()
        self.base_loss = DiceFocalLoss(weight_dice=weight_dice, weight_ce=weight_ce)

    def forward(self, input, target):
        # input is a list of output tensors from the model
        # target is the ground truth tensor
        assert isinstance(input, list), "input must be a list of tensors"
        total_loss = 0
        for output in input:
            # resize target to match the size of output
            target_resized = F.interpolate(target, size=output.shape[2:], mode='trilinear', align_corners=False)
            total_loss += self.base_loss(output, target_resized)
        return total_loss / len(input)

