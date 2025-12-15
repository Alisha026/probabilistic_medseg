import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --- This is critical for running on a headless server ---
import matplotlib
matplotlib.use('Agg')

# --- Import all your custom files ---
from data import dataloader
from model import ProbabilisticUNet, UNETEncoder, UNETDecoder, AxisAlignedConvGaussian, FeatureCombiner, downConv_block, upConv_block
from loss import DiceLoss # We need this for the metric

# --- Helper function for saving images ---
def save_evaluation_maps(image, gt_mask, mean_prediction, uncertainty_map, save_path):
    """Saves a 1x4 grid of images for analysis."""
    
    # Move tensors to CPU and convert to numpy
    image = image.cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask.cpu().numpy()[0]
    mean_prediction = torch.sigmoid(mean_prediction).cpu().numpy()[0]
    uncertainty_map = uncertainty_map.cpu().numpy()[0]
    
    # Un-normalize the image (approximate)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    axes[2].imshow(mean_prediction, cmap='gray')
    axes[2].set_title("Mean Prediction")
    axes[2].axis('off')
    
    im = axes[3].imshow(uncertainty_map, cmap='hot')
    axes[3].set_title("Aleatoric Uncertainty (Variance)")
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# --- Main Evaluation Script ---
if __name__ == "__main__":
    
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1 # We evaluate one image at a time
    NUM_WORKERS = 0
    LATENT_DIM = 6
    INIT_FEATURES = 32
    NUM_SAMPLES = 10 # Number of samples to get uncertainty
    
    MODEL_PATH = "best_prob_unet_model.pth"
    OUTPUT_DIR = "evaluation_results"
    
    # --- Data Paths ---
    # We only need the validation data
    VAL_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
    VAL_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"

    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Get DataLoader (Validation only) ---
    _, val_loader = dataloader(
        train_data_dir=VAL_IMAGES_DIR, # Dummy path, not used
        train_mask_dir=VAL_MASKS_DIR,  # Dummy path, not used
        val_data_dir=VAL_IMAGES_DIR,
        val_mask_dir=VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Validation DataLoader created. Batches: {len(val_loader)}")

    # --- 2. Load Model ---
    model = ProbabilisticUNet(
        in_channels=3, 
        out_channels=1, 
        init_features=INIT_FEATURES, 
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to evaluation mode (CRITICAL)
    
    print(f"Model loaded from {MODEL_PATH}")

    # --- 3. Evaluation Loop ---
    dice_loss_fn = DiceLoss()
    all_dice_scores = []
    
    with torch.no_grad():
        for i, (images, mask_sets) in enumerate(val_loader):
            images = images.to(DEVICE)
            gt_mask = mask_sets[:, 0, :, :, :].to(DEVICE)

            # --- Sample N times from the PRIOR ---
            all_samples = []
            for _ in range(NUM_SAMPLES):
                # We pass *only* the image. The model is in eval() mode,
                # so it samples 'z' from the 'prior_net' automatically.
                logits = model(images)
                all_samples.append(torch.sigmoid(logits))
            
            # Stack all samples: [NUM_SAMPLES, B, 1, H, W]
            all_samples = torch.stack(all_samples)

            # --- Calculate Metrics ---
            
            # 1. Get the average prediction
            # [NUM_SAMPLES, 1, 1, H, W] -> [1, 1, H, W]
            mean_probs = all_samples.mean(dim=0)
            
            # We need logits for the DiceLoss, so invert sigmoid
            mean_logits = torch.log(mean_probs / (1 - mean_probs + 1e-7))
            
            dice_score = (1 - dice_loss_fn(mean_logits, gt_mask)).item()
            all_dice_scores.append(dice_score)

            # --- Calculate Uncertainty Map (Step 6) ---
            # We calculate the variance across the samples
            uncertainty_map = all_samples.var(dim=0) # Shape [1, 1, H, W]

            # --- Save Visualization (for the first 10 images) ---
            if i < 10:
                save_path = os.path.join(OUTPUT_DIR, f"eval_result_{i}.png")
                print(f"  Saving visualization {i} to {save_path}...")
                save_evaluation_maps(
                    images[0], 
                    gt_mask[0], 
                    mean_logits[0], 
                    uncertainty_map[0],
                    save_path
                )
            
            if (i+1) % 25 == 0:
                print(f"  Processed {i+1}/{len(val_loader)} images...")

    # --- 4. Print Final Results ---
    final_avg_dice = sum(all_dice_scores) / len(all_dice_scores)
    print("\n--- Evaluation Complete ---")
    print(f"Average Validation Dice Score (over {NUM_SAMPLES} samples): {final_avg_dice:.4f}")