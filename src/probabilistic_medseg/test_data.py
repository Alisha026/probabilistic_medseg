import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
from model import ProbabilisticUNet, DeterministicUNET
matplotlib.use('Agg')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGES_DIR = "data/raw/ISIC2018_Task1/test/images/ISIC2018_Task1-2_Test_Input"
PROB_MODEL_PATH = "best_prob_unet_model_final.pth"
MC_MODEL_PATH = "best_mc_dropout_model.pth"
OUTPUT_DIR = "evaluation_results/test_set_predictions"

IMG_SIZE = (256, 256)
NUM_SAMPLES = 10 
LATENT_DIM = 6
INIT_FEATURES = 32

image_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def denormalize_image(tensor):
    """Converts normalized PyTorch tensor back to RGB numpy array."""
    img_np = tensor.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    return np.clip(img_np, 0.0, 1.0)

def predict_with_uncertainty(model, image_tensor, model_type="probabilistic", num_samples=10):
    """
    Runs N stochastic forward passes on a single unlabelled test image.
    Returns the mean predicted probability map and the variance (uncertainty) map.
    """
    if model_type == "mc_dropout":
        model.train()  # Active dropout layers for epistemic uncertainty
    else:
        model.eval()   # Probabilistic U-Net samples z ~ Prior(z|X)

    sample_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(image_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            sample_probs.append(prob)

    samples_array = np.array(sample_probs)  # Shape: (num_samples, H, W)
    
    mean_prob = np.mean(samples_array, axis=0)
    uncertainty_map = np.var(samples_array, axis=0)
    
    return mean_prob, uncertainty_map

def save_test_visualization(orig_img, prob_mean, prob_unc, mc_mean, mc_unc, img_id, save_dir):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    # Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Test Input ({img_id})", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    # Prob U-Net Mean Mask
    axes[1].imshow(prob_mean > 0.5, cmap="gray")
    axes[1].set_title("Prob U-Net Mask", fontsize=11)
    axes[1].axis("off")

    # Aleatoric Uncertainty
    im3 = axes[2].imshow(prob_unc, cmap="inferno")
    axes[2].set_title("Aleatoric Uncertainty\n(Probabilistic U-Net)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # MC Dropout Mean Mask
    axes[3].imshow(mc_mean > 0.5, cmap="gray")
    axes[3].set_title("MC Dropout Mask", fontsize=11)
    axes[3].axis("off")

    # Epistemic Uncertainty
    im5 = axes[4].imshow(mc_unc, cmap="inferno")
    axes[4].set_title("Epistemic Uncertainty\n(MC Dropout U-Net)", fontsize=10)
    axes[4].axis("off")
    plt.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"test_pred_{img_id}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    print("=" * 70)
    print("ISIC 2018 TASK 1: UNLABELLED TEST SET INFERENCE SUITE")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(TEST_IMAGES_DIR, "**", "*.jpg"), recursive=True)
    )

    if not image_paths:
        print(f"No test images found in '{TEST_IMAGES_DIR}'.")
        print("Please verify the directory path or check your test images folder.")
        return

    print(f"Found {len(image_paths)} unlabelled test images.")
    print(f"Inference Device: {DEVICE}")
    print(f"Monte Carlo Passes per image: {NUM_SAMPLES}\n")

    print("[*] Loading Probabilistic U-Net model weights...")
    prob_model = ProbabilisticUNet(
        in_channels=3, 
        out_channels=1, 
        latent_dim=LATENT_DIM, 
        init_features=INIT_FEATURES
    ).to(DEVICE)
    
    if os.path.exists(PROB_MODEL_PATH):
        prob_model.load_state_dict(torch.load(PROB_MODEL_PATH, map_location=DEVICE))
        print("Probabilistic U-Net loaded successfully.")
    else:
        print(f"Checkpoint not found at {PROB_MODEL_PATH}")

    print("Loading MC Dropout U-Net model weights...")
    mc_model = DeterministicUNET(
        in_channels=3, 
        out_channels=1, 
        init_features=INIT_FEATURES
    ).to(DEVICE)
    
    if os.path.exists(MC_MODEL_PATH):
        mc_model.load_state_dict(torch.load(MC_MODEL_PATH, map_location=DEVICE))
        print("MC Dropout U-Net loaded successfully.")
    else:
        print(f"Checkpoint not found at {MC_MODEL_PATH}")

    print("\nProcessing test images and generating uncertainty maps...")
    
    for idx, img_path in enumerate(image_paths):
        img_id = os.path.basename(img_path).replace(".jpg", "")
        
        raw_pil = Image.open(img_path).convert("RGB")
        img_tensor = image_transform(raw_pil).unsqueeze(0).to(DEVICE)
        
        # Probabilistic U-Net (Aleatoric)
        prob_mean, prob_unc = predict_with_uncertainty(
            prob_model, img_tensor, model_type="probabilistic", num_samples=NUM_SAMPLES
        )

        # Run MC Dropout U-Net (Epistemic)
        mc_mean, mc_unc = predict_with_uncertainty(
            mc_model, img_tensor, model_type="mc_dropout", num_samples=NUM_SAMPLES
        )

        # Save predicted binary mask PNG
        mask_out_dir = os.path.join(OUTPUT_DIR, "binary_masks")
        os.makedirs(mask_out_dir, exist_ok=True)
        pred_mask_pil = Image.fromarray(((prob_mean > 0.5) * 255).astype(np.uint8))
        pred_mask_pil.save(os.path.join(mask_out_dir, f"{img_id}_segmentation.png"))

        # Save qualitative visualization for sample images (first 20)
        if idx < 20:
            viz_img = denormalize_image(img_tensor)
            save_test_visualization(viz_img, prob_mean, prob_unc, mc_mean, mc_unc, img_id, OUTPUT_DIR)
            print(f"    Saved visualization {idx+1}/20: {img_id}.png")

        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_paths):
            print(f"    Processed {idx+1}/{len(image_paths)} images...")

    print("\n" + "=" * 70)
    print("Test Set Processing Complete!")
    print(f"Visual comparisons saved in: '{OUTPUT_DIR}/'")
    print(f"Predicted binary masks saved in: '{OUTPUT_DIR}/binary_masks/'")
    print("=" * 70)

if __name__ == "__main__":
    main()