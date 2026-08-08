import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ProbabilisticUNet, DeterministicUNET

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROB_MODEL_PATH = "best_prob_unet_model_final.pth"
MC_MODEL_PATH = "best_mc_dropout_model.pth"

IMG_SIZE = (256, 256)
LATENT_DIM = 6
INIT_FEATURES = 32

image_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(
    title="Probabilistic Medical Image Segmentation",
    description="Skin lesion segmentation with Aleatoric and Epistemic Uncertainty estimation using Probabilistic U-Net and MC Dropout U-Net.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prob_model = None
mc_model = None

@app.get("/")
def root():
    """Redirects the root URL directly to the interactive documentation."""
    return RedirectResponse(url="/docs")

def load_models():
    """Loads checkpoint weights into memory during server initialization."""
    global prob_model, mc_model
    
    # Probabilistic U-Net
    prob_model = ProbabilisticUNet(
        in_channels=3,
        out_channels=1,
        latent_dim=LATENT_DIM,
        init_features=INIT_FEATURES
    ).to(DEVICE)
    
    if os.path.exists(PROB_MODEL_PATH):
        prob_model.load_state_dict(torch.load(PROB_MODEL_PATH, map_location=DEVICE))
        prob_model.eval()
        print(f"Loaded Probabilistic U-Net weights from '{PROB_MODEL_PATH}'")
    else:
        print(f"Checkpoint missing at '{PROB_MODEL_PATH}'. Running with initialized weights.")
        prob_model.eval()

    # MC Dropout U-Net
    mc_model = DeterministicUNET(
        in_channels=3,
        out_channels=1,
        init_features=INIT_FEATURES
    ).to(DEVICE)
    
    if os.path.exists(MC_MODEL_PATH):
        mc_model.load_state_dict(torch.load(MC_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded MC Dropout U-Net weights from '{MC_MODEL_PATH}'")
    else:
        print(f"Checkpoint missing at '{MC_MODEL_PATH}'. Running with initialized weights.")

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("Initializing Probabilistic MedSeg API Server")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    load_models()

def preprocess_image_bytes(image_bytes: bytes):
    """Converts uploaded raw byte stream into normalized model input tensor and PIL Image."""
    try:
        raw_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = image_transform(raw_pil).unsqueeze(0).to(DEVICE)
        return tensor, raw_pil
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def run_stochastic_inference(model, input_tensor: torch.Tensor, model_type: str, num_samples: int = 10):
    """Executes N stochastic forward passes and computes mean prediction and uncertainty map."""
    if model_type == "mc_dropout":
        model.train()  # Active dropout for epistemic uncertainty
    else:
        model.eval()   # Probabilistic U-Net samples z ~ Prior(z|X)

    sample_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            sample_probs.append(prob)

    samples_array = np.array(sample_probs)  # Shape: (num_samples, H, W)
    mean_prob = np.mean(samples_array, axis=0)
    uncertainty_map = np.var(samples_array, axis=0)

    return mean_prob, uncertainty_map

@app.get("/health")
def health_check():
    """Health check endpoint confirming API status and device availability."""
    return {
        "status": "online",
        "device": str(DEVICE),
        "prob_model_loaded": prob_model is not None,
        "mc_model_loaded": mc_model is not None
    }

@app.post("/visualize")
async def visualize_grid(
    file: UploadFile = File(...),
    num_samples: int = Query(default=10, ge=1, le=50)
):
    """Returns a single combined 5-panel PNG comparison image for direct viewing/downloading."""
    contents = await file.read()
    input_tensor, orig_pil = preprocess_image_bytes(contents)

    prob_mean, prob_unc = run_stochastic_inference(
        prob_model, input_tensor, model_type="probabilistic", num_samples=num_samples
    )
    mc_mean, mc_unc = run_stochastic_inference(
        mc_model, input_tensor, model_type="mc_dropout", num_samples=num_samples
    )

    # Generate 5-panel side-by-side plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    
    axes[0].imshow(orig_pil.resize(IMG_SIZE))
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(prob_mean > 0.5, cmap="gray")
    axes[1].set_title("Prob U-Net Mask", fontsize=11)
    axes[1].axis("off")

    im2 = axes[2].imshow(prob_unc, cmap="inferno")
    axes[2].set_title("Aleatoric Uncertainty\n(Prob U-Net)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(mc_mean > 0.5, cmap="gray")
    axes[3].set_title("MC Dropout Mask", fontsize=11)
    axes[3].axis("off")

    im4 = axes[4].imshow(mc_unc, cmap="inferno")
    axes[4].set_title("Epistemic Uncertainty\n(MC Dropout)", fontsize=10)
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)