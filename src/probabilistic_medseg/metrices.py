import torch
import numpy as np

def compute_dice(probs, targets, threshold=0.5, epsilon=1e-6):
    """
    Computes Dice Coefficient.
    Dice = 2 * (A ∩ B) / (|A| + |B|)
    """
    # Binarize predictions
    preds = (probs > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()

def compute_iou(probs, targets, threshold=0.5, epsilon=1e-6):
    """
    Computes Intersection over Union (Jaccard Index).
    IoU = (A ∩ B) / (A ∪ B)
    """
    preds = (probs > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection # A + B - (A ∩ B)
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()

def compute_brier_score(probs, targets):
    """
    Computes Brier Score (Mean Squared Error for probabilities).
    Lower is better. Perfect = 0.0.
    """
    # Flatten to [N_pixels] using reshape
    probs = probs.reshape(-1)
    targets = targets.reshape(-1).float()
    
    return torch.mean((probs - targets) ** 2).item()

class ECE(torch.nn.Module):
    """
    Expected Calibration Error (ECE).
    Measures the gap between confidence and accuracy.
    Lower is better.
    """
    def __init__(self, n_bins=15):
        super(ECE, self).__init__()
        self.n_bins = n_bins

    def forward(self, probs, targets):
        # Flatten to [N_pixels] using reshape (FIXED)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)
        
        for i in range(self.n_bins):
            # Find pixels falling into this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                # Accuracy in this bin (Average Ground Truth)
                accuracy_in_bin = targets[in_bin].float().mean()
                # Confidence in this bin (Average Prob)
                avg_confidence_in_bin = probs[in_bin].mean()
                
                # |Acc - Conf| * P(in_bin)
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece.item()