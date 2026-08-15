<div align="center">

# 🩻 Probabilistic Medical Segmentation

### Teaching a neural network to say *"I'm not sure"* and actually mean it.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="MkDocs Material" src="https://img.shields.io/badge/Docs-MkDocs%20Material-526CFE?style=flat-square&logo=materialformkdocs&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-black?style=flat-square">
</p>

[📖 Full Documentation](https://alisha026.github.io/probabilistic_medseg/) · [🐛 Report a Bug](https://github.com/Alisha026/probabilistic_medseg/issues) · [⭐ Star this repo](https://github.com/Alisha026/probabilistic_medseg) · [Full Report](report/report.pdf)

</div>

---

## The problem with confident machines

Ask a standard segmentation model to outline a skin lesion, and it will draw you a single, crisp, unwavering boundary. Even when the boundary itself is a matter of opinion. Two dermatologists looking at the same fuzzy, blended edge will draw two *different* lines. The model doesn't know that. It just picks one line and commits, with the same flat confidence whether the lesion is a textbook case or a genuine judgment call.

That's not a bug you can fix by training longer. It's a missing capability: **the model was never taught to have doubt.**

This project builds a segmentation system that does. Instead of collapsing every image down to one "correct" mask, it learns a *distribution* over plausible masks, then shows you exactly where it's confident, where it's guessing, and why.

<div align="center">
<img src="results/compare_uncertainties/compare_1.png" alt="Five-panel uncertainty visualization: input image, mean prediction, aleatoric uncertainty, epistemic uncertainty, and ground truth" width="900">

<sub><i>Same lesion, four different questions answered at once: what it looks like, where the image itself is ambiguous, where the model is out of its depth, and what an expert actually drew.</i></sub>
</div>

---

## Two flavors of "I don't know"

Not all uncertainty is created equal, and conflating them is how you end up solving the wrong problem. This project keeps them strictly separate:

<table>
<tr>
<td width="50%" valign="top">

### 🌫️ Aleatoric
**"The image itself is ambiguous."**

Some lesion borders blend into skin so gradually that *no amount of data* would resolve them. Even a room full of dermatologists would draw slightly different lines. This is irreducible noise baked into the data itself.

Modeled with a **Probabilistic U-Net**: a VAE-style latent space (`z ∈ ℝ⁶`) that learns the *shape of disagreement* itself, then samples from it.

</td>
<td width="50%" valign="top">

### 🤖 Epistemic
**"The model hasn't seen enough like this."**

Some uncertainty has nothing to do with the image and everything to do with the model's own gaps in experience: an atypical texture, an underrepresented case. More training data *would* fix this.

Modeled with **MC Dropout**: keep dropout switched on at inference, run 30–50 stochastic passes, and watch where the predictions disagree with each other.

</td>
</tr>
</table>

> **Why it matters clinically:** an uncertainty map isn't a disclaimer. It's a second opinion baked into the output. A high-uncertainty border tells a clinician *"look here more carefully"* before a single confident-looking line does any damage to their trust.

---

## Does it actually work better? Yes, and it's not close.

Evaluated head-to-head on 100 held-out ISIC test images:

| Metric | Probabilistic U-Net | MC Dropout | Winner |
|:--|:--:|:--:|:--:|
| **Dice** ↑ | **0.8788** | 0.8600 | 🏆 Probabilistic U-Net |
| **IoU** ↑ | **0.7960** | 0.7779 | 🏆 Probabilistic U-Net |
| **Brier Score** ↓ | **0.0427** | 0.0613 | 🏆 Probabilistic U-Net |
| **ECE** ↓ | **0.0565** | 0.0752 | 🏆 Probabilistic U-Net |

The interesting part isn't the Dice score. A ~2-point edge is nice but not headline news. The interesting part is that **Brier Score improves by ~30% and ECE by ~25%**. The Probabilistic U-Net isn't just *more accurate*, it's *more honest*: when it says 90% confident, it's a lot closer to actually being right 90% of the time.

<div align="center">
<img src="results/reliability_diagram.png" alt="Reliability diagram comparing calibration of Probabilistic U-Net vs MC Dropout" width="600">

<sub><i>The dotted diagonal is what perfect honesty looks like. One model hugs it. One doesn't.</i></sub>
</div>

📊 Full metric breakdown, calibration analysis, and figure interpretation → [**results.md**](https://alisha026.github.io/probabilistic_medseg/results/)

---

## How it's built

```
                    ┌─────────────────────┐
   skin lesion  ──▶ │   U-Net Backbone     │
     image          │  (encoder/decoder)   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
    ┌───────────────────┐             ┌───────────────────┐
    │  Probabilistic     │             │    MC Dropout      │
    │  U-Net (VAE)        │             │  (30-50 passes)     │
    │  z ∈ ℝ⁶, ELBO loss  │             │  weight-space       │
    │  → ALEATORIC map    │             │  → EPISTEMIC map    │
    └───────────────────┘             └───────────────────┘
              │                                  │
              └────────────────┬────────────────┘
                               ▼
                   mean mask + two uncertainty
                       heatmaps, side by side
```

- **Backbone:** standard U-Net encoder/decoder with skip connections
- **Probabilistic U-Net:** 6-D VAE latent space, trained with an ELBO objective: reconstruction loss (BCE/Dice) + β-weighted KL divergence
- **MC Dropout:** dropout layers kept active at test time, 30–50 stochastic forward passes per image
- **Multi-rater signal:** since true multi-annotator masks weren't available for every image, rater disagreement was **simulated** via morphological erosion/dilation on ground-truth boundaries. It's a deliberately boundary-focused noise model, because that's exactly where real inter-rater disagreement lives
- **Serving:** a FastAPI backend exposes a single `POST /predict` endpoint that returns the mean mask plus both uncertainty heatmaps for any uploaded image

🔬 Full architecture writeup with equations → [**architecture.md**](https://alisha026.github.io/probabilistic_medseg/architecture/)
🧪 Multi-rater simulation methodology → [**methodology.md**](https://alisha026.github.io/probabilistic_medseg/methodology/)

---

## Try it yourself

```bash
# Clone it
git clone https://github.com/Alisha026/probabilistic_medseg.git
cd probabilistic_medseg

# Install it
pip install -r requirements.txt

# Run it
uvicorn src.probabilistic_medseg.api:app --host 0.0.0.0 --port 8000
```

---

## Project layout

```
probabilistic_medseg/
├── src/probabilistic_medseg/   # model, training, inference, FastAPI app
├── results/                    # reliability diagram, both uncertainity with comparison as well
├── docs/                       # MkDocs Material documentation site
├── mkdocs.yml                  # docs site config
└── requirements.txt
```

---

## Why this exists

Most segmentation portfolios stop at "here's my Dice score." This one exists to make a different argument: **in medicine, a model that knows the limits of its own knowledge is more valuable than a model that's marginally more accurate but silently overconfident.** Calibration isn't a footnote metric here. It's the whole thesis.

📚 Read the complete documentation, including full derivations, evaluation protocol, and deployment plans → **[alisha026.github.io/probabilistic_medseg](https://alisha026.github.io/probabilistic_medseg/)**

---

<div align="center">
<sub>Built by <a href="https://github.com/Alisha026">Alisha026</a> · MIT Licensed</sub>
</div>