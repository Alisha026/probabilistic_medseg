# 🔮 Probabilistic Medical Image Segmentation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

### *Quantifying Uncertainty in Skin Lesion Diagnosis with Deep Learning*

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)

---

## 📖 Overview

Standard deep learning models are often **overconfident**. In medical imaging, knowing *what* a model doesn't know is just as important as the prediction itself.

This project implements and compares two approaches to **uncertainty estimation** for skin lesion segmentation on the **ISIC 2018 Task 1** dataset. Instead of a single deterministic mask, both models produce a distribution over plausible segmentations, giving clinicians a "heatmap of doubt" alongside the prediction.

### 🎯 Key Goals

1. **Aleatoric Uncertainty:** Modeling data ambiguity (fuzzy lesion borders, annotator disagreement) with a **Probabilistic U-Net**.
2. **Epistemic Uncertainty:** Modeling model confidence via **Monte Carlo Dropout**.
3. **Evaluation & Calibration:** Comparing both models on Dice, IoU, Brier score, and Expected Calibration Error (ECE), with reliability diagrams.
4. **Deployment:** Serving predictions through a **FastAPI** endpoint.

---

## 🧠 The Methodologies

### 1. Probabilistic U-Net

A standard U-Net combined with a **Conditional VAE**.

* **Architecture** (`src/probabilistic_medseg/model.py`): a U-Net encoder/decoder, a **Prior Net** (image → latent space), and a **Posterior Net** (image + mask → latent space), fused into the decoder via a `FeatureCombiner`.
* **How it works:** it learns a low-dimensional latent space `z` over segmentation variants. Sampling `z` multiple times at inference time yields multiple plausible masks for the same image.
* **Captures:** **aleatoric uncertainty**.
* **Training:** `train.py`, optimized with an ELBO loss (reconstruction + KL) via `loss.py`.

### 2. Monte Carlo Dropout

A deterministic U-Net (`DeterministicUNET`) with `Dropout2d` layers left **active at inference time**.

* **How it works:** run the model `N` times on the same image; each pass drops a different set of neurons, approximating an ensemble.
* **Captures:** **epistemic uncertainty**.
* **Training:** `train_mc_dropout.py`.

---

## 📂 Project Layout

```
├── src/probabilistic_medseg/
│   ├── data.py              <- ISIC dataset loading + augmentations (albumentations)
│   ├── model.py             <- Probabilistic U-Net & MC-Dropout U-Net architectures
│   ├── loss.py               <- ELBO / Dice losses
│   ├── train.py              <- Trains the Probabilistic U-Net
│   ├── train_mc_dropout.py   <- Trains the MC-Dropout U-Net
│   ├── evaluate.py           <- Generates per-model uncertainty visualizations
│   ├── compare.py            <- Head-to-head comparison of both models
│   ├── metrices.py           <- Dice, IoU, Brier score, ECE
│   ├── calibration.py        <- Reliability diagrams / calibration analysis
│   └── api.py                <- FastAPI inference server
├── data/raw/ISIC2018_Task1   <- Raw ISIC 2018 images & ground-truth masks
├── models/                   <- Saved checkpoints (best_prob_unet_model.pth, best_mc_dropout_model.pth)
├── evaluation_results/       <- Saved evaluation figures (aleatoric/epistemic/reliability)
├── comparison_results/       <- Output of compare.py
├── docs/                     <- mkdocs project (see docs/README.md)
├── run.sh / run_mc.sh        <- SLURM job scripts for training
├── compare.sh                <- SLURM job script for model comparison
└── tests/                    <- Unit tests
```

---

## ⚙️ Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

Place the ISIC 2018 Task 1 dataset under `data/raw/ISIC2018_Task1` (train/val images + masks).

---

## 🚀 Usage

Train the Probabilistic U-Net:

```bash
python src/probabilistic_medseg/train.py
```

Train the MC-Dropout U-Net:

```bash
python src/probabilistic_medseg/train_mc_dropout.py
```

Evaluate a trained model and generate uncertainty maps:

```bash
python src/probabilistic_medseg/evaluate.py
```

Compare both models (Dice, IoU, Brier score, ECE):

```bash
python src/probabilistic_medseg/compare.py
```

On an HPC cluster with SLURM, use the corresponding job scripts instead: `sbatch run.sh`, `sbatch run_mc.sh`, `sbatch compare.sh`.

Serve predictions via the API:

```bash
cd src/probabilistic_medseg
uvicorn api:app --reload
```

---

## 📊 Results

Evaluation figures (aleatoric/epistemic uncertainty maps, model comparisons, reliability diagrams) are written to `evaluation_results/` and `comparison_results/`.

---

## License

See [LICENSE](LICENSE).
