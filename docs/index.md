# Probabilistic Medical Segmentation

**Uncertainty-aware deep learning for dermatological lesion segmentation.**

[View on GitHub](https://github.com/Alisha026/probabilistic_medseg){ .md-button .md-button--primary }


---

## Why Uncertainty Matters in Medicine

A standard segmentation model outputs a single mask and stops there — it never tells you *how sure* it is. In clinical imaging that silence is dangerous. Lesion borders in dermoscopic images (e.g. the ISIC dataset) are frequently ambiguous: different dermatologists will draw different boundaries around the same lesion, and a model trained to predict "the one correct mask" is implicitly hiding that disagreement.

This project treats segmentation as a **probabilistic inference problem** rather than a deterministic mapping. Instead of asking *"what is the mask?"*, it asks *"what is the distribution over plausible masks, and where does the model disagree with itself?"*

!!! info "Clinical Note"
    An uncertainty map is not a hedge — it's information. A dermatologist reviewing a high-uncertainty border region knows to look more closely, rather than trusting a single confident-looking (but potentially wrong) boundary line.

Two complementary sources of uncertainty are modeled explicitly:

| Type | Source | Modeled by |
|---|---|---|
| **Aleatoric** | Irreducible noise/ambiguity in the data itself (e.g. genuinely fuzzy lesion borders, inter-rater disagreement) | Probabilistic U-Net (VAE latent space) |
| **Epistemic** | Uncertainty in the model's own parameters (what the model *doesn't know* due to limited data) | Monte Carlo (MC) Dropout at test time |

See [Architecture](architecture.md) for the full technical breakdown and [Results](results.md) for quantitative comparisons between the two approaches.

---

## Project Highlights

- **Probabilistic U-Net** with a 6-dimensional VAE latent space, trained with an ELBO objective (reconstruction + β-weighted KL divergence)
- **MC Dropout** baseline with 30–50 stochastic forward passes at inference
- **Calibration-first evaluation**: Dice/IoU alongside Brier Score and Expected Calibration Error (ECE), not just overlap metrics
- **Simulated multi-rater pipeline** using morphological erosion/dilation to model inter-annotator boundary variability on ISIC
- **FastAPI backend** serving live predictions with uncertainty heatmaps, deployed on Render
- **Static interactive dashboard** on GitHub Pages

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Alisha026/probabilistic_medseg.git
cd probabilistic_medseg
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run inference locally

```bash
uvicorn src.probabilistic_medseg.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. See [API Reference](api.md) for endpoint details.

### 4. Build and preview these docs

```bash
pip install mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Where to Go Next

<div class="grid cards" markdown>

- :material-atom-variant: **[Architecture](architecture.md)**
  Probabilistic U-Net internals, VAE latent space, and MC Dropout's Bayesian interpretation.

- :material-flask: **[Methodology](methodology.md)**
  How multi-rater variability was simulated and how uncertainty decomposition was validated.

- :material-chart-bell-curve: **[Results](results.md)**
  Dice, IoU, Brier Score, ECE, and reliability diagram analysis.

- :material-api: **[API Reference](api.md)**
  FastAPI endpoints, request/response schemas.

- :material-cloud-upload: **[Deployment](deployment.md)**
  Hosting the backend on Render and the frontend on GitHub Pages.

</div>
