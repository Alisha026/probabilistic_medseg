# Methodology

## Dataset

The project uses the **ISIC** (International Skin Imaging Collaboration) dataset of dermoscopic images, with lesion segmentation masks as ground truth. Final evaluation results reported in [Results](results.md) are computed on a held-out test set of **100 images**.

---

## Simulating Multi-Rater Variability

Truly multi-rater datasets (where several dermatologists independently annotate the same lesion) are the gold standard for training models like the Probabilistic U-Net, since the whole point of the latent space is to capture *inter-rater* disagreement. Where multiple independent expert annotations aren't available for every image, this project **simulates rater variability directly from the single available ground-truth mask** using morphological transformations.

### Approach

For each ground-truth mask, additional plausible "rater" masks are synthesized using morphological **erosion** and **dilation** operations with varying kernel sizes:

- **Erosion** shrinks the lesion boundary inward — simulating a conservative annotator who only labels the unambiguous lesion core
- **Dilation** expands the lesion boundary outward — simulating a liberal annotator who includes surrounding ambiguous/blended tissue

By applying these operations with randomized kernel sizes and iteration counts, a small set of synthetic mask variants is generated per image, each representing a plausible alternative boundary a different human rater might have drawn.

!!! info "Clinical Note"
    This mirrors a well-documented phenomenon in dermatology: inter-rater disagreement is concentrated almost entirely at the lesion *boundary*, not the lesion *core*. Erosion/dilation directly targets that boundary region, which is exactly the property the Probabilistic U-Net's latent space is meant to learn.

### Training signal

During training of the Probabilistic U-Net, the posterior network \(q(z \mid x, y)\) is exposed to this distribution of mask variants (rather than a single fixed mask) across training iterations/epochs. This encourages the latent space \(z \in \mathbb{R}^6\) to organize itself along axes of genuine boundary variability — larger vs. smaller lesion extent, smoother vs. more irregular borders — rather than collapsing to a single deterministic mode.

### Limitations

This is a **simulated approximation** of true multi-rater disagreement, not a substitute for it. Morphological perturbation captures *boundary extent* disagreement well, but does not capture other real sources of inter-rater variance (e.g. disagreement about whether a secondary satellite lesion should be included at all). Where true multi-rater datasets (e.g. ISIC subsets with multiple annotators) are available, they are the preferred training signal, and the simulated pipeline should be treated as a data-efficient stand-in.

---

## Evaluation Protocol

- Both models (Probabilistic U-Net and MC Dropout) are evaluated on the same held-out set of **100 test images**, unseen during training
- For the Probabilistic U-Net, the predictive distribution is formed by sampling multiple \(z\) from the prior \(p(z \mid x)\) and decoding each; the **mean** prediction is used for overlap metrics, and per-pixel **variance** across samples is used for the aleatoric uncertainty map
- For MC Dropout, \(N = 30\text{–}50\) stochastic forward passes are run per image at inference; the **mean** across passes gives the point prediction, and per-pixel **variance** gives the epistemic uncertainty map
- Metrics reported: **Dice**, **IoU**, **Brier Score**, and **ECE** — see [Results](results.md) for definitions, values, and calibration analysis

## Next

- [Results](results.md) — full metric comparison and reliability diagram
- [Architecture](architecture.md) — how these masks feed into the ELBO training objective
