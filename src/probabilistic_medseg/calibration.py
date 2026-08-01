import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

from data import dataloader
from model import ProbabilisticUNet, DeterministicUNET
from metrices import compute_brier_score, ECE 

def get_reliability_data(probs, targets, n_bins=10):
    """
    Extracts the individual bin data needed to draw a Reliability Diagram.
    (Your ECE class in metrices.py calculates the final number; this gets the plot points).
    """
    probs = probs.reshape(-1)
    targets = targets.reshape(-1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            
    return bin_accuracies, bin_confidences

def plot_reliability_diagram(mc_acc, mc_conf, prob_acc, prob_conf, save_path):
    """Draws the Reliability Diagram."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='gray')
    
    plt.plot(mc_conf, mc_acc, marker='o', label='MC Dropout (Epistemic)', color='blue', linewidth=2)
    plt.plot(prob_conf, prob_acc, marker='s', label='Prob U-Net (Aleatoric)', color='red', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
    plt.ylabel('True Fraction of Positives (Accuracy)', fontsize=12)
    plt.title('Reliability Diagram: Probabilistic vs MC Dropout', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    PROB_MODEL_PATH = "best_prob_unet_model_final.pth"
    MC_MODEL_PATH = "best_mc_dropout_model.pth"
    
    VAL_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
    VAL_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"
    
    os.makedirs("evaluation_results", exist_ok=True)

    _, val_loader = dataloader(VAL_IMAGES_DIR, VAL_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR, batch_size=1, num_workers=0)
    
    prob_model = ProbabilisticUNet(in_channels=3, out_channels=1, init_features=32, latent_dim=6).to(DEVICE)
    prob_model.load_state_dict(torch.load(PROB_MODEL_PATH, map_location=DEVICE))
    prob_model.eval()
    
    mc_model = DeterministicUNET(in_channels=3, out_channels=1, init_features=32).to(DEVICE)
    mc_model.load_state_dict(torch.load(MC_MODEL_PATH, map_location=DEVICE))
    
    all_prob_probs, all_mc_probs, all_targets = [], [], []

    print("Gathering predictions for Calibration Metrics...")
    with torch.no_grad():
        for i, (images, mask_sets) in enumerate(val_loader):
            images = images.to(DEVICE)
            gt_mask = mask_sets[:, 0, :, :, :].to(DEVICE)
            
            # Prob U-Net
            prob_samples = [torch.sigmoid(prob_model(images)) for _ in range(10)]
            prob_mean = torch.stack(prob_samples).mean(dim=0)
            
            # MC Dropout
            mc_model.train() # Active dropout
            mc_samples = [torch.sigmoid(mc_model(images)) for _ in range(10)]
            mc_mean = torch.stack(mc_samples).mean(dim=0)
            
            # Move to CPU numpy for the plot, but keep tensors for your metrices.py
            all_prob_probs.append(prob_mean.cpu())
            all_mc_probs.append(mc_mean.cpu())
            all_targets.append((gt_mask > 0.5).float().cpu()) # Binarize GT
            
            if (i+1) % 25 == 0: print(f"Processed {i+1} images...")

    # Concat all data
    prob_tensor = torch.cat(all_prob_probs)
    mc_tensor = torch.cat(all_mc_probs)
    target_tensor = torch.cat(all_targets)

    # Calculate Metrics using YOUR metrices.py
    ece_calculator = ECE(n_bins=10)
    
    print("\n=== CALIBRATION RESULTS ===")
    
    prob_brier = compute_brier_score(prob_tensor, target_tensor)
    prob_ece = ece_calculator(prob_tensor, target_tensor)
    print(f"Probabilistic U-Net -> Brier Score: {prob_brier:.4f} | ECE: {prob_ece:.4f}")
    
    mc_brier = compute_brier_score(mc_tensor, target_tensor)
    mc_ece = ece_calculator(mc_tensor, target_tensor)
    print(f"MC Dropout          -> Brier Score: {mc_brier:.4f} | ECE: {mc_ece:.4f}")

    # Draw Reliability Diagram 
    # Convert to numpy for matplotlib
    prob_acc, prob_conf = get_reliability_data(prob_tensor.numpy(), target_tensor.numpy())
    mc_acc, mc_conf = get_reliability_data(mc_tensor.numpy(), target_tensor.numpy())
    
    plot_reliability_diagram(mc_acc, mc_conf, prob_acc, prob_conf, "evaluation_results/reliability_diagram.png")
    print("\nSaved Reliability Diagram to evaluation_results/reliability_diagram.png")