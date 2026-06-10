import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Simple combined loss for deterministic segmentation models
class CombinedBCEDiceLoss(nn.Module):
    """
    A simple combined loss for a standard segmentation model.
    Loss = BCE_Loss + Dice_Loss
    """
    def __init__(self):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return loss_bce + loss_dice
    
# Loss for the probabilistic segmentation model using ELBO
class ELBOLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(ELBOLoss, self).__init__()
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = DiceLoss()

    def forward(self, model_outputs):
        logits, (mu_prior, log_var_prior), (mu_post, log_var_post), gt_mask = model_outputs
        
        # Reconstruction Loss
        loss_bce = self.bce_loss(logits, gt_mask)
        loss_dice = self.dice_loss(logits, gt_mask)
        reconstruction_loss = loss_bce + loss_dice

        # KL Divergence
        prior_dist = Normal(mu_prior, torch.exp(0.5 * log_var_prior))
        posterior_dist = Normal(mu_post, torch.exp(0.5 * log_var_post))
        kl_div = kl_divergence(posterior_dist, prior_dist).sum(dim=1)
        kl_loss = kl_div.mean()
        
        # 3. Final ELBO Loss
        elbo_loss = reconstruction_loss + (self.beta * kl_loss)
        
        return elbo_loss, reconstruction_loss, kl_loss
