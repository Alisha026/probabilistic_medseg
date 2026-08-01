import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import os 
from data import dataloader 
from model import ProbabilisticUNet, UNETEncoder, UNETDecoder, AxisAlignedConvGaussian, FeatureCombiner, downConv_block, upConv_block
from loss import ELBOLoss, DiceLoss
from torch.utils.tensorboard import SummaryWriter

print(torch.cuda.get_device_name(0))

# Training Function 
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for images, mask_sets in loader:
        images = images.to(device)
        mask_sets = mask_sets.to(device) # Shape [B, 5, 1, H, W]

        # Forward pass
        model_outputs = model(images, mask_sets)
        
        # Calculate loss
        loss, recon_loss, kl_loss = loss_fn(model_outputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(loader)
    avg_recon_loss = total_recon_loss / len(loader)
    avg_kl_loss = total_kl_loss / len(loader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

# Validation Function
def validate(model, loader, device):
    model.eval()
    all_dice_scores = []
    dice_loss_fn = DiceLoss() 
    
    with torch.no_grad():
        for images, mask_sets in loader:
            images = images.to(device)
            gt_mask = mask_sets[:, 0, :, :, :].to(device) 

            # We pass only the image to sample from the prior
            logits = model(images)
            
            # Calculate Dice score (1 - DiceLoss)
            dice_score = (1 - dice_loss_fn(logits, gt_mask))
            all_dice_scores.append(dice_score.item())

    avg_dice = sum(all_dice_scores) / len(all_dice_scores)
    return avg_dice

# Main
if __name__ == "__main__":
    
    # Configuration and Hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    EPOCHS = 300
    BATCH_SIZE = 8 
    WARMUP_EPOCHS = 50 # We will linearly increase beta from 0 to MAX_BETA during these warmup epochs, then keep it constant for the rest of training
    NUM_WORKERS = 4 
    LATENT_DIM = 6
    MAX_BETA = 0.01 # we will do the kL Anneleaning manually by adjusting this value during training
    INIT_FEATURES = 32 

    # Data Paths
    TRAIN_IMAGES_DIR = "data/raw/ISIC2018_Task1/train/images/ISIC2018_Task1-2_Training_Input"
    TRAIN_MASKS_DIR = "data/raw/ISIC2018_Task1/train/masks/ISIC2018_Task1_Training_GroundTruth"
    VAL_IMAGES_DIR = "data/raw/ISIC2018_Task1/val/images/ISIC2018_Task1-2_Validation_Input"
    VAL_MASKS_DIR = "data/raw/ISIC2018_Task1/val/masks/ISIC2018_Task1_Validation_GroundTruth"
    
    log_dir = f'runs/prob_unet_lr{LEARNING_RATE}_bs{BATCH_SIZE}_beta{MAX_BETA}'
    writer = SummaryWriter(log_dir)
    print(f"Starting Training on {DEVICE} ---")
    print(f"TensorBoard logs will be saved to {log_dir} ---")

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

    # Model
    model = ProbabilisticUNet(
        in_channels=3, 
        out_channels=1, 
        init_features=INIT_FEATURES, 
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    # Optimizer and Loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = ELBOLoss(beta=MAX_BETA)

    # Training Loop
    best_val_dice = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS} ---")
        
        current_beta = MAX_BETA * min(1.0, epoch / WARMUP_EPOCHS) # Linear warmup of beta
        loss_fn.beta = current_beta # Update the beta value in the loss function

        train_loss, recon_loss, kl_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        print(f"Train: ELBO={train_loss:.4f} | Recon Loss={recon_loss:.4f} | KL Loss={kl_loss:.4f}")
        
        val_dice = validate(model, val_loader, DEVICE)
        print(f"Val: Dice Score = {val_dice:.4f}")
        
        # Log scalars to TensorBoard
        writer.add_scalar('Loss/Train ELBO', train_loss, epoch)
        writer.add_scalar('Loss/Train Reconstruction', recon_loss, epoch)
        writer.add_scalar('Loss/Train KL', kl_loss, epoch)
        writer.add_scalar('Metric/Validation Dice', val_dice, epoch)
        writer.add_scalar('Hyperparameters/Beta', current_beta, epoch)
        
        # Save the best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_prob_unet_model_final.pth")
            print(f"  -> Saved new best model with Dice: {best_val_dice:.4f}")
            
    print("\nTraining Complete")
    print(f"Best validation Dice score: {best_val_dice:.4f}")
    
    
    writer.close()