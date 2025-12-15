# 🔮 Probabilistic Medical Image Segmentation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

### *Quantifying Uncertainty in Skin Lesion Diagnosis with Deep Learning*

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)

---

## 📖 Overview

Standard deep learning models are often **overconfident**. In medical imaging, knowing *what* a model doesn't know is just as important as the prediction itself.

This project implements and compares two state-of-the-art approaches to **uncertainty estimation** for skin lesion segmentation using the ISIC 2018 dataset. We move beyond simple deterministic masks to provide clinicians with a "heatmap of doubt."

### 🎯 Key Goals

1.  **Aleatoric Uncertainty:** Modeling data ambiguity (fuzzy borders) using a **Probabilistic U-Net**.
2.  **Epistemic Uncertainty:** Modeling model confidence using **Monte Carlo Dropout**.
3.  **Deployment:** Serving predictions via a real-time **FastAPI** interface.

---

## 🧠 The Methodologies

We compare two powerful competitors in the Bayesian Deep Learning space.

### 1. The Probabilistic U-Net (Baseline)
> *"The Generative Approach"*

This model combines a standard U-Net with a **Variational Autoencoder (VAE)**.
* **Architecture:** It uses **three encoders**: a U-Net encoder, a Prior Net (Image $\rightarrow$ Latent Space), and a Posterior Net (Image + Mask $\rightarrow$ Latent Space).
* **How it works:** It learns a low-dimensional latent space $\mathbf{z}$ that encodes segmentation variants. By sampling $\mathbf{z}$ multiple times, we generate multiple plausible segmentations for a single image.
* **Captures:** **Aleatoric Uncertainty** (noise/ambiguity in the data).

### 2. Monte Carlo Dropout (Competitor)
> *"The Bayesian Approximation Approach"*

This model uses a **Deterministic U-Net** with dropout layers active during inference.
* **Architecture:** A standard U-Net with `Dropout2d` layers injected into the encoder blocks.
* **How it works:** We run the model $N$ times on the same image. Each pass drops a different set of neurons, simulating an ensemble of different models.
* **Captures:** **Epistemic Uncertainty** (lack of knowledge/training data).

---

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         probabilistic_medical_seg and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── probabilistic_medical_seg   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes probabilistic_medical_seg a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

