# Architecture

This project implements and compares two Bayesian approaches to segmentation uncertainty: a **Probabilistic U-Net** (which models aleatoric uncertainty via a learned latent space) and **MC Dropout** (which approximates epistemic uncertainty via test-time stochasticity). Both share a standard U-Net encoder/decoder backbone.

---

## 1. Backbone

Both models are built on a standard **U-Net** encoder/decoder:

- Convolutional encoder with progressive downsampling (max-pooling), doubling channel depth at each stage
- Symmetric decoder with transposed convolutions / upsampling and skip connections from corresponding encoder stages
- Skip connections preserve fine-grained spatial detail (critical for lesion boundary precision)

The backbone itself is deterministic — the *probabilistic* behavior is introduced on top of it in two different ways, described below.

---

## 2. Probabilistic U-Net (Aleatoric Uncertainty)

### Core idea

Rather than mapping an input image \(x\) directly to a single mask \(y\), the Probabilistic U-Net learns a **conditional distribution over plausible masks**, \(p(y \mid x)\), by introducing a low-dimensional latent variable \(z\):

\[
p(y \mid x) = \int p(y \mid x, z)\, p(z \mid x)\, dz
\]

At inference time, sampling different \(z\) from the learned latent distribution and decoding each sample produces a *different but plausible* segmentation mask — directly capturing the ambiguity a human annotator would face at a fuzzy lesion border.

### Latent space

- Latent dimension: \(z \in \mathbb{R}^6\)
- A **prior network** encodes the input image \(x\) into \(p(z \mid x)\), a low-dimensional Gaussian
- During training, a **posterior network** additionally encodes the ground-truth mask \(y\) into \(q(z \mid x, y)\)
- The decoder is conditioned on a sample \(z\) (broadcast and concatenated into the U-Net feature maps) to produce the final segmentation

!!! note "Why 6 dimensions?"
    A small latent space is intentional. It's large enough to capture the dominant modes of boundary/shape variability seen across simulated raters, but small enough to keep the latent distribution well-regularized and the sampled masks semantically meaningful rather than noisy.

### Training objective — ELBO

The model is trained by maximizing the Evidence Lower Bound (equivalently, minimizing its negative):

\[
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{z \sim q(z \mid x,y)}\big[-\log p(y \mid x, z)\big]}_{\text{reconstruction term}} \;+\; \beta \cdot D_{KL}\big(q(z \mid x, y) \,\|\, p(z \mid x)\big)
\]

- **Reconstruction term**: a pixel-wise segmentation loss (Binary Cross-Entropy and/or Dice Loss) between the decoded mask and ground truth, evaluated using the *posterior* sample of \(z\)
- **KL term**: pulls the posterior \(q(z \mid x, y)\) toward the prior \(p(z \mid x)\), so that at test time (when \(y\) is unavailable) sampling from the prior alone produces reasonable masks
- **\(\beta\)**: weighting coefficient balancing reconstruction fidelity against latent regularization — higher \(\beta\) encourages a more informative/diverse latent space at some cost to per-sample accuracy

At inference, only the prior network is used: \(N\) latent samples \(z_i \sim p(z \mid x)\) are drawn, each decoded into a mask, and the resulting set of masks forms an empirical approximation of \(p(y \mid x)\).

---

## 3. MC Dropout (Epistemic Uncertainty)

### Core idea

MC Dropout (Gal & Ghahramani, 2016) reinterprets dropout — normally a regularizer disabled at test time — as an approximation to **Bayesian inference over the network's weights**. Keeping dropout **active during inference** and running multiple stochastic forward passes approximates sampling from the posterior distribution over model parameters, \(p(\theta \mid \mathcal{D})\).

### Implementation

- Dropout layers are inserted at multiple stages of the U-Net (typically in the bottleneck and decoder blocks) and remain **active at test time** (`model.train()` mode for dropout layers specifically, even during evaluation)
- For a given input \(x\), the model is run \(N\) times, each pass using a different random dropout mask, producing \(N\) stochastic output masks \(\{\hat{y}_1, \dots, \hat{y}_N\}\)
- **Number of passes**: \(N = 30\text{–}50\) — chosen as a practical trade-off between a stable Monte Carlo estimate of the predictive distribution and inference latency

Because each forward pass effectively samples a different "thinned" sub-network, disagreement across passes reflects the model's own uncertainty about its learned weights — highest in regions where training data was sparse or ambiguous, rather than where the *image itself* is ambiguous.

---

## 4. Uncertainty Decomposition

The two mechanisms above are not redundant — they capture fundamentally different quantities:

!!! warning "Interpretation"
    Aleatoric and epistemic uncertainty answer different clinical questions. Aleatoric uncertainty says *"this border is inherently ambiguous, even to a human expert."* Epistemic uncertainty says *"the model itself hasn't seen enough similar cases to be confident here."* Conflating the two can lead to the wrong corrective action (e.g. collecting more data won't help with irreducible aleatoric noise).

| | Aleatoric | Epistemic |
|---|---|---|
| **Definition** | Uncertainty inherent to the data (image noise, genuine boundary ambiguity, inter-rater disagreement) | Uncertainty in the model's parameters, due to limited or unrepresentative training data |
| **Reducible by more data?** | No — irreducible | Yes — more/better data narrows it |
| **Modeled by** | Probabilistic U-Net's latent space \(z\) | MC Dropout's stochastic forward passes |
| **Computed as** | Pixel-wise variance across decoded samples from \(p(z \mid x)\) | Pixel-wise variance across the \(N\) MC Dropout forward passes |

In practice, both are visualized per-pixel as heatmaps alongside the mean predicted mask, giving a full picture: *where* the model is uncertain, and *why*. See the five-panel visualization in [Results](results.md#visual-analysis) for a concrete example on an ISIC image.

---

## Next

- [Methodology](methodology.md) — how the multi-rater training signal was constructed
- [Results](results.md) — quantitative comparison and calibration analysis
