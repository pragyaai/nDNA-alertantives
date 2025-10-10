# Spectral Curvature, Thermodynamic Length, and Belief Vector Fields 

This repository accompanies a set of experiments that measure geometric properties of hidden-state trajectories inside causal language models. The code computes spectral curvature, thermodynamic length, and belief vector fields in a way that is invariant to arbitrary orthogonal changes of basis across layers. All three quantities are built on a common scaffold that aligns neighboring layers via an orthogonal Procrustes transport, then measures displacements in an intrinsic, whitened metric. Plots are rendered directly with `matplotlib` and are not saved to disk.

The experiments target three illustrative models, namely `microsoft/DialoGPT-medium`, `gpt2`, and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The implementation runs on GPU if available and defaults to numerically stable FP32 for spectral decompositions, while allowing faster mixed-precision forward passes where appropriate. Datasets are loaded from the Hugging Face Hub via the `datasets` library.

The experiments benefit from a CUDA-capable GPU for speed, but CPU-only runs are also supported. The notebook suppresses progress bars and tokenizer warnings. Plots are shown with `plt.show()` and are not written to files.

## Method 1: introduction

At each depth $\ell$ we collect the row-centered token table $X_\ell\in\mathbb{R}^{N\times d}$ over the current batch, define the intrinsic metric $\Sigma_\ell=\tfrac{1}{N}X_\ell^\top X_\ell+\lambda I_d$ with $\lambda>0$, and align neighbors by the orthogonal Procrustes rotation $R_\ell=\mathrm{polar}(C_\ell)$ where $C_\ell=\tfrac{1}{N}X_\ell^\top X_{\ell+1}$. We then express first and second covariant differences in the frame of $\ell$ as $\Delta_\ell=X_{\ell+1}R_\ell^\top-X_\ell$ and $\Delta^2_\ell=X_{\ell+1}R_\ell^\top-2X_\ell+X_{\ell-1}R_{\ell-1}$, whiten by $\Sigma_\ell^{-1/2}$, and evaluate norms in intrinsic units. This construction is invariant to per-layer orthogonal gauge spins $X_\ell\mapsto X_\ell Q_\ell$ and yields layerwise thermodynamic length $L^{(PT)}_  \ell$, spectral curvature $\kappa_ \ell$, and class-conditioned belief vectors $\mathbf{v}^{(PT)}_\ell(c)$.

## Spectral curvature

The notebook computes spectral curvature on a small multilingual and math-heavy prompt suite. For a single forward pass, it extracts all hidden states, row-centers token tables per prompt, estimates $\Sigma_\ell^{-1/2}$ at every depth with a ridge $\lambda$, and builds the Procrustes transports $R_\ell$ from cross-covariances. It then forms the first and second covariant differences in the $\ell$-frame, whitens them, and reports

$$
\kappa_\ell = \frac{|\Delta^2_\ell,\Sigma_\ell^{-1/2}|_ F}{\big(|\Delta_ \ell,\Sigma_\ell^{-1/2}|_F^2+\varepsilon\big)^{3/2}},.
$$

Curves are plotted across all layers and, for convenience, also per prompt. The $y$ axis is shown on a log scale to make small and large curvature regimes visible on the same figure. Because lengths and curvature use the intrinsic metric, curves are comparable across layers and models without any ad hoc rescaling.

## Thermodynamic length

Thermodynamic length quantifies how far the representation cloud moves, in intrinsic units, between consecutive stacks. For each model, the notebook streams through a subset of SQuAD, accumulates second moments needed for $A=\mathbb{E}[X_\ell^\top X_\ell]$, $B=\mathbb{E}[X_{\ell+1}^\top X_{\ell+1}]$, and $C=\mathbb{E}[X_\ell^\top X_{\ell+1}]$, obtains $R_\ell=\mathrm{polar}(C)$, and evaluates the transported, whitened quadratic form

$$
L_\ell^2 = N\cdot \mathrm{tr}\left(\Sigma_\ell^{-1/2} M_\ell \Sigma_\ell^{-1/2}\right),
$$

with $M_\ell=R_\ell B R_\ell^\top - R_\ell C^\top - C R_\ell^\top + A$ and $\Sigma_\ell=A+\lambda I$. The implementation uses a single streaming pass with batchwise row-centering to remain memory friendly. Each model’s curve is then plotted, followed by an overlay across all models with a log-scale $y$ option if desired. As with curvature, the length is gauge-invariant because both the transport and whitening absorb arbitrary orthogonal spins.

## Belief vector fields 

Belief vector fields summarize the task-driven steering pressure the objective exerts at each depth. The notebook computes them as whitened, token-pooled gradients with respect to hidden states, averaged across samples:

$$
\mathbf{v}^{(PT)}_ \ell = \mathbb{E}_ x\big[\Sigma_ \ell^{-1/2} g_\ell(x) \big],
$$

where $g_\ell(x)$ is the pooled gradient of the token-level next-token log-likelihood with respect to $X_\ell$. Gradients are captured from hidden states directly by retaining their `.grad` tensors and then pooled across valid tokens before whitening. Norms $|\mathbf{v}^{(PT)}_ \ell|_ 2$ are plotted over stacks for every model, along with an overlay plot. Because gradients are expressed in intrinsic coordinates via $\Sigma_ \ell^{-1/2}$, comparisons across layers reflect real geometry rather than arbitrary coordinate scale.

The class-conditioned variant computes one belief vector field per concept $c$ and overlays all classes on a single figure per model. The notebook uses AG News with four concepts, namely World, Sports, Business, and Sci/Tech, builds a balanced subset, estimates a single $\Sigma_\ell^{-1/2}$ across all classes, and then averages whitened pooled gradients only over samples belonging to a given class:

$$
\mathbf{v}^{(PT)}_ \ell(c) = \mathbb{E}_ {x\sim P(c)} \big[\Sigma_\ell^{-1/2} g_\ell(x) \big],
$$

To ensure gradients are available at the desired internal interfaces, the code disables key-value caching and registers full backward hooks on transformer blocks, harvesting $\partial\mathcal{L}/\partial X_\ell$ on the inputs and $\partial\mathcal{L}/\partial X_{\ell+1}$ on the outputs. Token masks exclude padding from pooling. After whitening, per-class means are accumulated and the norms are plotted on a log scale. This construction is gauge-invariant and directly reflects how each layer participates in steering for a given concept.

## Concreteness experiment: philosophy versus business

To contrast abstract and non-abstract domains, the notebook builds a two-class dataset consisting of philosophy texts from the Stanford Plato corpus and business articles from AG News. It selects appropriate text columns, trims each class to a configurable cap, shuffles to mix classes for the covariance pass, and computes $\mathbf{v}^{(PT)}_\ell(c)$ for $c\in$ Philosophy, Business. The resulting per-model figures show two curves on a log scale and reveal where layers amplify or attenuate steering pressure for abstract versus concrete content. The same whitening metric is used across both classes so that differences in norms reflect class-conditioned scores rather than anisotropy of the feature cloud.

## Practical notes & Summary

The ridge $\lambda$ in $\Sigma_\ell=\tfrac{1}{N}X_\ell^\top X_\ell+\lambda I$ prevents instabilities in the inverse square root and bounds $|\Sigma_\ell^{-1/2}|_2\le \lambda^{-1/2}$. Batchwise row-centering removes mean shifts that are irrelevant to covariance and cross-covariance geometry. Mixed precision is enabled only for forward passes; all second-moment and eigendecomposition operations are done in FP32 for accuracy. Hooks-based gradient capture requires `use_cache=False` and a grad-enabled context; the code guards against accidental no-grad scopes and verifies that logits require gradients before backpropagating. When memory is tight, reducing `BATCH_SIZE`, shortening maximum sequence length, or restricting to a subset of stacks can help.

With $C_\ell=\tfrac{1}{N}X_\ell^\top X_{\ell+1}$ and $R_\ell=\mathrm{polar}(C_\ell)=U_\ell V_\ell^\top$, define transported neighbors $X_{\ell+1\to \ell}=X_{\ell+1}R_\ell^\top$ and $X_{\ell-1\to \ell}=X_{\ell-1}R_{\ell-1}$. The first and second covariant differences are $\Delta_\ell=X_{\ell+1\to \ell}-X_\ell$ and $\Delta^2_\ell=X_{\ell+1\to \ell}-2X_\ell+X_{\ell-1\to \ell}$. With $\Sigma_\ell=\tfrac{1}{N}X_\ell^\top X_\ell+\lambda I$, the intrinsic thermodynamic length and curvature are

$$
L^{(PT)}_\ell = |\Delta_\ell,\Sigma_\ell^{-1/2}|_ F
\quad\text{and}\quad
\kappa_ \ell = \frac{|\Delta^2_\ell,\Sigma_\ell^{-1/2}|_ F}{\big(|\Delta_  \ell,\Sigma_\ell^{-1/2}|_F^2+\varepsilon\big)^{3/2}},
$$

while the class-conditional belief vector is

$$
\mathbf{v}^{(PT)}_ \ell(c) = \mathbb{E}_{x\sim P(c)}\big[\Sigma_\ell^{-1/2}g_\ell(x)\big],
$$

All three are invariant to per-layer orthogonal gauge spins $X_\ell\mapsto X_\ell Q_\ell$ because the transport and whitening transform covariantly and Frobenius and Euclidean norms are preserved by orthogonal maps.

The notebook fixes random seeds for Python and PyTorch and enables high-precision matmul where available. The forward-pass autocast is used for throughput on GPU, while accumulation buffers and eigensolvers stay in FP32. Each model run clears CUDA caches between sections to reduce peak memory. This work builds on classic results for orthogonal Procrustes alignment, polar decomposition, and Fisher geometry, adapted to neural hidden-state analysis. The datasets and models are provided via the Hugging Face ecosystem. Any figures shown by the notebook are produced purely for research and educational purposes and are not saved to disk.
