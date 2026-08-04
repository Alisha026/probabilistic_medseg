import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from data import dataloader
from model import ProbabilisticUNet, DeterministicUNET, downConv_block_withDropout
from metrices import compute_dice, compute_iou, compute_brier_score, ECE

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROB_MODEL_PATH = "best_prob_unet_model_final.pth"
MC_MODEL_PATH = "best_mc_dropout_model.pth"
INIT_FEATURES = 32
LATENT_DIM = 6
NUM_SAMPLES = 10  # N=10 samples for uncertainty
BATCH_SIZE = 1    # Process one image at a time for metrics

# Data Paths
TEST_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
TEST_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"

def get_model_predictions(model, image, model_type="probabilistic"):
    """
    Runs N samples and returns Mean Prediction.
    """
    if model_type == "mc_dropout":
        model.train() # Keep dropout ON
    else:
        model.eval() # ProbNet uses eval() to sample from prior
        
    with torch.no_grad():
        samples = []
        for _ in range(NUM_SAMPLES):
            logits = model(image)
            probs = torch.sigmoid(logits)
            samples.append(probs)
        
        # Stack: [N, B, 1, H, W]
        samples_tensor = torch.stack(samples)
        
        # Mean Prediction
        # Shape becomes [B, 1, H, W] - preserving channel dim to match mask
        mean_probs = samples_tensor.mean(dim=0)
        
    return mean_probs

if __name__ == "__main__":
    
    # Data
    _, test_loader = dataloader(
        train_data_dir=TEST_IMAGES_DIR, 
        train_mask_dir=TEST_MASKS_DIR,  
        val_data_dir=TEST_IMAGES_DIR,   # Using Val set as Test
        val_mask_dir=TEST_MASKS_DIR,    
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    print(f"Comparison DataLoader created with {len(test_loader)} images.")
    
    # Models
    print("Loading Models")
    
    # Probabilistic U-Net
    prob_model = ProbabilisticUNet(3, 1, LATENT_DIM, INIT_FEATURES).to(DEVICE)
    try:
        prob_model.load_state_dict(torch.load(PROB_MODEL_PATH, map_location=DEVICE))
        print("Probabilistic U-Net loaded.")
        prob_loaded = True
    except FileNotFoundError:
        print(f"{PROB_MODEL_PATH} not found.")
        prob_loaded = False

    # MC Dropout U-Net
    mc_model = DeterministicUNET(3, 1, INIT_FEATURES).to(DEVICE)
    try:
        mc_model.load_state_dict(torch.load(MC_MODEL_PATH, map_location=DEVICE))
        print("MC Dropout U-Net loaded.")
        mc_loaded = True
    except FileNotFoundError:
        print(f"{MC_MODEL_PATH} not found.")
        mc_loaded = False

    if not (prob_loaded and mc_loaded):
        print("Both models are needed for comparison. Exiting.")
        exit()

    # Evaluation Loop
    # Initialize dictionary lists for all metrics
    results_prob = {"dice": [], "iou": [], "brier": [], "ece": []}
    results_mc = {"dice": [], "iou": [], "brier": [], "ece": []}
    ece_fn = ECE(n_bins=10).to(DEVICE)

    print("Starting Comparison Loop (Metrics Only)...")
    
    for i, (image, mask_sets) in enumerate(test_loader):
        image = image.to(DEVICE)
        gt_mask = mask_sets[:, 0, :, :, :].to(DEVICE) # Single GT mask
        
        # Ensure Ground Truth is Binary (0/1) for Metrics
        # Fixes Dice score logic if masks are not strictly binary
        gt_mask_binary = (gt_mask > 0.5).float()
        
        # Run Probabilistic U-Net
        mean_prob = get_model_predictions(prob_model, image, "probabilistic")
        results_prob["dice"].append(compute_dice(mean_prob, gt_mask_binary))
        results_prob["iou"].append(compute_iou(mean_prob, gt_mask_binary))
        results_prob["brier"].append(compute_brier_score(mean_prob, gt_mask_binary))
        results_prob["ece"].append(ece_fn(mean_prob, gt_mask_binary))
        
        # Run MC Dropout U-Net
        mean_mc = get_model_predictions(mc_model, image, "mc_dropout")
        results_mc["dice"].append(compute_dice(mean_mc, gt_mask_binary))
        results_mc["iou"].append(compute_iou(mean_mc, gt_mask_binary))
        results_mc["brier"].append(compute_brier_score(mean_mc, gt_mask_binary))
        results_mc["ece"].append(ece_fn(mean_mc, gt_mask_binary))

    # Process and Print Summary Stats
    # Average the metrics
    results_prob = {k: np.mean(v) for k, v in results_prob.items()}
    results_mc = {k: np.mean(v) for k, v in results_mc.items()}

    print("\n" + "="*50)
    print(f"{'METRIC':<15} | {'PROB U-NET':<20} | {'MC DROPOUT':<20}")
    print("-" * 60)
    
    metrics = ["Dice Score", "IoU", "Brier Score (↓)", "ECE (↓)"]
    keys = ["dice", "iou", "brier", "ece"]
    
    for m, k in zip(metrics, keys):
        val_prob = f"{results_prob[k]:.4f}"
        val_mc = f"{results_mc[k]:.4f}"
        print(f"{m:<15} | {val_prob:<20} | {val_mc:<20}")
    print("="*50)