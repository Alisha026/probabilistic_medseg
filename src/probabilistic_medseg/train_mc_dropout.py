import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os

# --- Import your custom files ---
from data import dataloader
# Import your new DeterministicUNET and the helper blocks
from model import DeterministicUNET, downConv_block_withDropout, upConv_block, downConv_block
# We can reuse the DiceLoss from our other project
from loss import DiceLoss, CombinedBCEDiceLoss

# ----------------------- TRAINING & VALIDATION FUNCTIONS -----------------------
# --- 2. TRAINING & VALIDATION FUNCTIONS ---
# -----------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train() # Keep model in training mode (for dropout)
    total_loss = 0.0

    for images, mask_sets in loader:
        images = images.to(device)
        # We only need one ground truth mask
        gt_mask = mask_sets[:, 0, :, :, :].to(device)

        # --- Forward pass ---
        logits = model(images)
        
        # --- Calculate loss ---
        loss = loss_fn(logits, gt_mask)
        
        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

def validate(model, loader, device, num_samples=10):
    # --- CRITICAL ---
    # We keep the model in .train() mode to activate dropout,
    # even during validation. This is the "MC" in "MC Dropout".
    model.train() 
    
    all_dice_scores = []
    dice_loss_fn = DiceLoss() # Use DiceLoss for the metric
    
    with torch.no_grad(): # Still use no_grad to save memory
        for images, mask_sets in loader:
            images = images.to(device)
            gt_mask = mask_sets[:, 0, :, :, :].to(device)

            # --- Sample N times ---
            mc_predictions = []
            for _ in range(num_samples):
                # Each forward pass will have different dropout masks
                logits = model(images)
                mc_predictions.append(torch.sigmoid(logits))
            
            # Stack samples and average them
            # (B, N, C, H, W) -> (B, C, H, W)
            avg_probs = torch.stack(mc_predictions).mean(dim=0)
            
            # Calculate Dice score on the averaged probability map
            # We need logits for our DiceLoss, so we invert sigmoid
            avg_logits = torch.log(avg_probs / (1 - avg_probs + 1e-7))
            
            dice_score = (1 - dice_loss_fn(avg_logits, gt_mask))
            all_dice_scores.append(dice_score.item())

    avg_dice = sum(all_dice_scores) / len(all_dice_scores)
    return avg_dice

# -----------------------------------------------------------------
# --- 3. MAIN SCRIPT ---
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    BATCH_SIZE = 4
    NUM_WORKERS = 0 # Start with 0 for debugging
    INIT_FEATURES = 32
    
    # --- Your Data Paths (Unchanged) ---
    TRAIN_IMAGES_DIR = "data/raw/ISIC2018_Task1/train/images/ISIC2018_Task1-2_Training_Input"
    TRAIN_MASKS_DIR = "data/raw/ISIC2018_Task1/train/masks/ISIC2018_Task1_Training_GroundTruth"
    VAL_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
    VAL_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"

    print(f"--- Starting MC Dropout Training on {DEVICE} ---")
    
    # --- 1. Get DataLoaders ---
    train_loader, val_loader = dataloader(
        train_data_dir=TRAIN_IMAGES_DIR,
        train_mask_dir=TRAIN_MASKS_DIR,
        val_data_dir=VAL_IMAGES_DIR,
        val_mask_dir=VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"DataLoaders created. Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 2. Get Model (DeterministicUNET) ---
    model = DeterministicUNET(
        in_channels=3, 
        out_channels=1, 
        init_features=INIT_FEATURES
    ).to(DEVICE)
    print(f"Model DeterministicUNET created.")

    # --- 3. Get Optimizer and Loss (Simple Loss) ---
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CombinedBCEDiceLoss()

    # --- 4. Training Loop ---
    best_val_dice = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        print(f"  Train: Loss = {train_loss:.4f}")
        
        val_dice = validate(model, val_loader, DEVICE)
        print(f"  Val:   Dice Score = {val_dice:.4f}")
        
        # Save the best model with a new name
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            # --- SAVE WITH A NEW NAME ---
            torch.save(model.state_dict(), "best_mc_dropout_model.pth")
            print(f"  -> Saved new best model with Dice: {best_val_dice:.4f}")
            
    print("\n--- Training Complete ---")
    print(f"Best validation Dice score: {best_val_dice:.4f}")