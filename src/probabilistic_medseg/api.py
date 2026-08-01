import uvicorn
import torch
import numpy as np
import albumentations as A
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from albumentations.pytorch import ToTensorV2
from model import ProbabilisticUNet
import matplotlib.cm as cm

# --- Configuration ---
MODEL_PATH = "best_prob_unet_model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
INIT_FEATURES = 32
LATENT_DIM = 6
NUM_SAMPLES = 4 # Number of variations to show in the grid

# --- Load Model ---
print(f"Loading model to {DEVICE}...")
model = ProbabilisticUNet(
    in_channels=3,
    out_channels=1,
    init_features=INIT_FEATURES,
    latent_dim=LATENT_DIM
).to(DEVICE)

try:
    # Map location ensures it works even if you trained on GPU but run on CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Successfully loaded weights from {MODEL_PATH}")
except FileNotFoundError:
    print(f"WARNING: {MODEL_PATH} not found. Using random weights (for testing only).")

model.eval()

# --- Preprocessing ---
transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

app = FastAPI(
    title="Probabilistic Medical Segmentation",
    description="Upload an image to see Aleatoric Uncertainty and Segmentation Variations."
)

# --- Helper Functions ---
def preprocess_image(image_bytes):
    """Convert bytes to tensor."""
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_pil = image_pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    image_np = np.array(original_pil)
    tensor = transform(image=image_np)['image']
    return tensor.unsqueeze(0).to(DEVICE), original_pil

def tensor_to_pil_mask(tensor, threshold=0.5):
    """Convert probability tensor to B&W PIL Image."""
    mask = (tensor > threshold).cpu().numpy().astype(np.uint8)
    mask_2d = mask[0] * 255
    return Image.fromarray(mask_2d, mode='L')

def tensor_to_heatmap(tensor):
    """Convert variance tensor to Heatmap PIL Image."""
    heatmap_np = tensor.cpu().numpy()[0]
    # Normalize 0-1 for colormap
    if heatmap_np.max() > 0:
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
    
    # Apply 'inferno' colormap (Purple -> Orange -> Yellow)
    colormap = cm.get_cmap('inferno')
    heatmap_colored = colormap(heatmap_np) # RGBA
    # Drop Alpha channel and convert to uint8
    heatmap_img = Image.fromarray((heatmap_colored[:, :, :3] * 255).astype(np.uint8))
    return heatmap_img

def create_grid(original_img, samples, mean_mask, uncertainty_map):
    """
    Creates a 2-row visualization grid.
    """
    w, h = IMG_SIZE, IMG_SIZE
    grid_w = w * 4
    grid_h = h * 2
    
    grid = Image.new('RGB', (grid_w, grid_h), color='white')
    
    # Row 1: Analysis
    grid.paste(original_img, (0, 0)) # Col 1: Input
    grid.paste(mean_mask.convert("RGB"), (w, 0)) # Col 2: Best Prediction
    grid.paste(uncertainty_map, (w * 2, 0)) # Col 3: Uncertainty
    
    # Row 2: The Samples (Variations)
    for i in range(3): 
        if i < len(samples):
            grid.paste(samples[i].convert("RGB"), (i * w, h))
        
    return grid

# --- API Endpoint ---
@app.post("/segment/", response_class=StreamingResponse)
async def segment_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor, original_pil = preprocess_image(image_bytes)
    
    sample_logits = []
    
    # 1. Run Inference Loop (Probabilistic Sampling)
    with torch.no_grad():
        for _ in range(NUM_SAMPLES):
            # model.eval() automatically samples a NEW 'z' from prior each time
            logits = model(input_tensor)
            sample_logits.append(logits)
            
    # 2. Process Results
    # Convert logits to probabilities
    sample_probs = [torch.sigmoid(l) for l in sample_logits]
    
    # Generate sample images (Row 2)
    sample_images = [tensor_to_pil_mask(p[0]) for p in sample_probs]
    
    # Calculate Mean (Best Prediction)
    stacked_probs = torch.stack(sample_probs)
    mean_prob = torch.mean(stacked_probs, dim=0)
    mean_mask_img = tensor_to_pil_mask(mean_prob[0])
    
    # Calculate Variance (Uncertainty Map)
    variance = torch.var(stacked_probs, dim=0)
    uncertainty_img = tensor_to_heatmap(variance[0])
    
    # 3. Create Grid
    final_image = create_grid(original_pil, sample_images, mean_mask_img, uncertainty_img)
    
    # 4. Return
    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)