import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from data import dataloader
from model import DeterministicUNET, downConv_block_withDropout, upConv_block, downConv_block
from loss import DiceLoss, CombinedBCEDiceLoss

# Training function for MC Dropout model
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train() 
    total_loss = 0.0

    for images, mask_sets in loader:
        images = images.to(device)
        # We only need one ground truth mask
        gt_mask = mask_sets[:, 0, :, :, :].to(device)

        # Forward pass
        logits = model(images)
        
        # Calculate loss
        loss = loss_fn(logits, gt_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    
    return avg_loss

def validate(model, loader, device, writer, epoch, num_samples=10):
    # We keep the model in .train() mode to activate dropout,
    # even during validation.
    model.train() 
    
    all_dice_scores = []
    dice_loss_fn = DiceLoss() 
    
    with torch.no_grad(): 
        for batch_idx, (images, mask_sets) in enumerate(loader):
            images = images.to(device)
            gt_mask = mask_sets[:, 0, :, :, :].to(device)

            # Sample N times
            mc_predictions = []
            for _ in range(num_samples):
                # Each forward pass will have different dropout masks
                logits = model(images)
                mc_predictions.append(torch.sigmoid(logits))
            
            # Stack samples and average them
            mc_stack = torch.stack(mc_predictions)  # (N, B, C, H, W)

            mean_prediction = mc_stack.mean(dim=0)
            variance_map = mc_stack.var(dim=0)
            
            # Calculate Dice score on the averaged probability map
            # We need logits for our DiceLoss, so we invert sigmoid
            avg_logits = torch.log(mean_prediction / (1 - mean_prediction + 1e-7))
            dice_score = (1 - dice_loss_fn(avg_logits, gt_mask))
            all_dice_scores.append(dice_score.item())
            
            # Log ONLY first batch to TensorBoard
            if batch_idx == 0:
                writer.add_image("Val/Input", images[0], epoch)
                writer.add_image("Val/GT_Mask", gt_mask[0], epoch)
                writer.add_image("Val/Mean_Prediction", mean_prediction[0], epoch)
                writer.add_image("Val/Epistemic_Uncertainty", variance_map[0], epoch)


    avg_dice = sum(all_dice_scores) / len(all_dice_scores)
    return avg_dice

# MAIN SCRIPT
if __name__ == "__main__":
    
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    EPOCHS = 300
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    INIT_FEATURES = 32
    
    # Your Data Paths
    TRAIN_IMAGES_DIR = "data/raw/ISIC2018_Task1/train/images/ISIC2018_Task1-2_Training_Input"
    TRAIN_MASKS_DIR = "data/raw/ISIC2018_Task1/train/masks/ISIC2018_Task1_Training_GroundTruth"
    VAL_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
    VAL_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"
    
    writer = SummaryWriter(log_dir="runs/unet_mc_dropout")


    print(f"Starting MC Dropout Training on {DEVICE}")
    
    # DataLoaders
    train_loader, val_loader = dataloader(
        train_data_dir=TRAIN_IMAGES_DIR,
        train_mask_dir=TRAIN_MASKS_DIR,
        val_data_dir=VAL_IMAGES_DIR,
        val_mask_dir=VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"DataLoaders created. Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model (DeterministicUNET)
    model = DeterministicUNET(
        in_channels=3, 
        out_channels=1, 
        init_features=INIT_FEATURES
    ).to(DEVICE)
    print(f"Model DeterministicUNET created.")

    # Optimizer and Loss (Simple Loss)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CombinedBCEDiceLoss()

    # Training Loop
    best_val_dice = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        print(f"Train: Loss = {train_loss:.4f}")
        
        # TensorBoard: training loss
        writer.add_scalar("Loss/Train", train_loss, epoch) 
    
        val_dice = validate(model, val_loader, DEVICE, writer, epoch)
        print(f"Val:   Dice Score = {val_dice:.4f}")
        
        # TensorBoard: validation Dice score 
        writer.add_scalar("Dice/Val", val_dice, epoch)
        
        # Save the best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_mc_dropout_model.pth")
            print(f"Saved new best model with Dice: {best_val_dice:.4f}")
            
    print("\nTraining Complete")
    print(f"Best validation Dice score: {best_val_dice:.4f}")