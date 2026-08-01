import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

from data import dataloader
from model import ProbabilisticUNet, UNETEncoder, UNETDecoder, AxisAlignedConvGaussian, FeatureCombiner, downConv_block, upConv_block, DeterministicUNET
from loss import DiceLoss 


def unnormalize_image(image_tensor):
    """Helper to convert normalized tensor back to viewable image."""
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    return np.clip(image, 0, 1)

def save_single_map(image, gt_mask, mean_prediction, uncertainty, save_path, title_unc):
    """Saves the 1x4 grid for a single model (Aleatoric or Epistemic)"""
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(mean_prediction, cmap='gray')
    axes[2].set_title("Mean Prediction")
    axes[2].axis('off')
    
    im = axes[3].imshow(uncertainty, cmap='hot')
    axes[3].set_title(title_unc)
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def comparison_map(image, gt_mask, aleatoric_map, epistemic_map, save_path):
    """Saves a 1x4 grid comparing both uncertainties side-by-side."""
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    im_a = axes[2].imshow(aleatoric_map, cmap='hot')
    axes[2].set_title("Aleatoric (Data Noise)")
    axes[2].axis('off')
    fig.colorbar(im_a, ax=axes[2], fraction=0.046, pad=0.04)
    
    im_e = axes[3].imshow(epistemic_map, cmap='hot')
    axes[3].set_title("Epistemic (Model Ignorance)")
    axes[3].axis('off')
    fig.colorbar(im_e, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def calculate_dice(mean_probs, gt_mask):
    """Calculates Dice score safely using binarized predictions."""
    binary_pred = (mean_probs > 0.5).float()
    intersection = (binary_pred * gt_mask).sum()
    union = binary_pred.sum() + gt_mask.sum()
    return ((2. * intersection + 1e-7) / (union + 1e-7)).item()


# Main Execution
if __name__ == "__main__":
    
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1 
    NUM_WORKERS = 0
    NUM_SAMPLES = 10 
    
    # Paths
    PROB_MODEL_PATH = "best_prob_unet_model_final.pth" 
    MC_MODEL_PATH = "best_mc_dropout_model.pth" 
    
    # Data Paths
    VAL_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
    VAL_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"

    # Output Directories
    DIR_ALEATORIC = "evaluation_results/aleatoric_uncertainty"
    DIR_EPISTEMIC = "evaluation_results/epistemic_uncertainty"
    DIR_COMPARE = "evaluation_results/compare_uncertainties"
    
    for d in [DIR_ALEATORIC, DIR_EPISTEMIC, DIR_COMPARE]:
        os.makedirs(d, exist_ok=True)

    print(f"Starting Uncertainty Evaluation on {DEVICE}")
    
    # Load Data
    _, val_loader = dataloader( 
        train_data_dir=VAL_IMAGES_DIR,
        train_mask_dir=VAL_MASKS_DIR,
        val_data_dir=VAL_IMAGES_DIR,
        val_mask_dir=VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # 3. Load Models
    prob_model = ProbabilisticUNet(in_channels=3, out_channels=1, init_features=32, latent_dim=6).to(DEVICE)
    mc_dropout_model = DeterministicUNET(in_channels=3, out_channels=1, init_features=32).to(DEVICE)

    prob_model.load_state_dict(torch.load(PROB_MODEL_PATH, map_location=DEVICE))
    mc_dropout_model.load_state_dict(torch.load(MC_MODEL_PATH, map_location=DEVICE))

    prob_model.eval()

    # Evaluation Loop
    prob_dice_scores = []
    mc_dice_scores = []
    
    with torch.no_grad():
        for i, (images, mask_sets) in enumerate(val_loader):
            images = images.to(DEVICE)
            gt_mask = mask_sets[:, 0, :, :, :].to(DEVICE)
            
            # PROB U-NET (Aleatoric)
            prob_samples = []
            for _ in range(NUM_SAMPLES):
                logits = prob_model(images)
                prob_samples.append(torch.sigmoid(logits))
            
            prob_samples = torch.stack(prob_samples)
            prob_mean = prob_samples.mean(dim=0)
            prob_unc = prob_samples.var(dim=0)
            prob_dice_scores.append(calculate_dice(prob_mean, gt_mask))
            
            # MC DROPOUT (Epistemic)
            mc_dropout_model.train() # IN TRAIN MODE FOR DROPOUT!
            mc_samples = []
            for _ in range(NUM_SAMPLES):
                logits = mc_dropout_model(images)
                mc_samples.append(torch.sigmoid(logits))
                
            mc_samples = torch.stack(mc_samples)
            mc_mean = mc_samples.mean(dim=0)
            mc_unc = mc_samples.var(dim=0)
            mc_dice_scores.append(calculate_dice(mc_mean, gt_mask))

            # SAVE VISUALIZATIONS (First 10 images only)
            if i < 10:
                img_viz = unnormalize_image(images[0])
                gt_viz = gt_mask[0].cpu().numpy()[0]
                
                # Save Aleatoric
                save_single_map(
                    img_viz, gt_viz, prob_mean[0].cpu().numpy()[0], prob_unc[0].cpu().numpy()[0],
                    os.path.join(DIR_ALEATORIC, f"aleatoric_{i}.png"), "Aleatoric Uncertainty"
                )
                
                # Save Epistemic
                save_single_map(
                    img_viz, gt_viz, mc_mean[0].cpu().numpy()[0], mc_unc[0].cpu().numpy()[0],
                    os.path.join(DIR_EPISTEMIC, f"epistemic_{i}.png"), "Epistemic Uncertainty"
                )
                
                # Save Comparison
                comparison_map(
                    img_viz, gt_viz, prob_unc[0].cpu().numpy()[0], mc_unc[0].cpu().numpy()[0],
                    os.path.join(DIR_COMPARE, f"compare_{i}.png")
                )
            
            if (i+1) % 25 == 0:
                print(f"Processed {i+1}/{len(val_loader)} images...")

    # Final Results
    print("\nEVALUATION COMPLETE")
    print(f"Average Prob U-Net Dice: {np.mean(prob_dice_scores):.4f}")
    print(f"Average MC Dropout Dice: {np.mean(mc_dice_scores):.4f}")
    print("All visualizations saved in 'evaluation_results/'")