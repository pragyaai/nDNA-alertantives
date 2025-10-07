# <p align="left"> ![ndna_logo_v5.jpg](https://github.com/user-attachments/assets/e093b2ec-4918-4622-a3c0-7d86d8e98db3)</p>
# <p align="center"> Neural DNA - Cartograph: Latent Semantic Genome of Foundation Models </p>

### <p align="center"> Webpage: https://pragyaai.github.io/ndna/llm/ndna/ </p>

# nDNA Calculation:

Our unified framework establishes three fundamental metrics that collectively characterize the information processing landscape of Deep Neural Network in modern LLMs.<br>

**► 1. _Spectral Curvature_ ($κℓ$)**: Quantifies geometric properties of the parameter manifold <br><br>
**► 2. _Thermodynamic Length_ ($Lℓ$)**: Measures information processing complexity via Fisher-Rao distances <br><br>
**► 3. _Belief Vector Dynamics_**: Captures epistemic confidence and uncertainty propagation <br><br>

<img width="1276" height="663" alt="image" src="https://github.com/user-attachments/assets/a4be9afd-cac8-4aba-ae3a-10216dacb4a0" />

The **Spectral metrics (αℓ, κℓ)** capture geometry, **thermodynamic length(Lℓ)** captures latent movement, and belief norm quantifies alignment strength of the model.

# Mathematical Foundation:

## **Spectral Curvature Analysis (κℓ):** A ramp up of Power-Law exponent will increase the Latent Binding in higher decoder layers, signaling sharper representational semantics.

Spectral curvature at layer $\ell$ is defined as:

$$
\kappa_\ell := \big\| \Delta^2 h_\ell \big\| = \big\| h_{\ell+1} - 2 h_\ell + h_{\ell-1} \big\|
$$

In continuous form, this corresponds to:

$$
\kappa(s) = \left\| \frac{d^2 h(s)}{ds^2} \right\|
$$

where $s$ parameterizes depth through the network.  
Our discrete $\kappa_\ell$ provides a practical, layerwise estimator.
s = parameterizes depth through the network. 

## **Thermodynamic length(Lℓ) Analysis** It is defined by Fisher geometry that quantifies semantic effort needed to move a token from layer(l) to layer (l+1).

<img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/09e72306-af42-40cf-9490-f42c9d4f7d6e" />
<img width="1095" height="205" alt="image" src="https://github.com/user-attachments/assets/32346f7d-e6fb-4f3d-9209-cd871e99bbb3" />
<br><br>

It quantifies the epistemic work performed across transformer layers, calculated as the cumulative squared gradient norm of layerwise log-likelihoods. Higher values signal internal resistance–zones of significant restructuring, belief compression, or negotiation of conflicting priors. In culturally fine-tuned models, these peaks localize to upper decoder layers, indicating intense adaptation near output-generating blocks. Within the nDNA construct, **Lℓ** helps reveal latent epistemic effort that underlies surface-level behavior. This metric thus provides a nuanced window into where and how models internally allocate effort during learning and inference. <br><br>
Mathematically it is defined as <br><br>
<img width="400" height="100" alt="image" src="https://github.com/user-attachments/assets/7b09d118-3612-4250-bcf3-636a6fc99594" />

During our experiments, we generate the following visuals to understand the internals of thermodynamics length:
<img width="1196" height="600" alt="image" src="https://github.com/user-attachments/assets/aa25dc34-ffde-48be-b0a4-86aa9c2a1e0c" />
<img width="1196" height="600" alt="image" src="https://github.com/user-attachments/assets/2ef85314-6e4a-4937-952d-3e2c81c3d816" />
<img width="1196" height="600" alt="image" src="https://github.com/user-attachments/assets/4ab935fc-ae44-481c-8181-978893c4c2b5" />
<img width="1196" height="600" alt="image" src="https://github.com/user-attachments/assets/3859345a-d6e0-494e-a44d-d7511e62b427" />
<br><br>

<img width="1047" height="517" alt="image" src="https://github.com/user-attachments/assets/e432dba5-bd48-423b-867a-7318bc3aec49" />

## **Belief Factors Calculation**

# Datasets Used

We have used **SQuAD 2.0** (https://huggingface.co/datasets/rajpurkar/squad_v2) dataset for our experiments and analysis of these metrics.

# Models Evaluated

Throughout our work, we used the following foundational LLMs to prove our hypothesis:

**► Dense transformers** (e.g., LLaMA-2 base/instruct, LLaMA-3 3B base/instruct, Gemma, Falcon, GPT-NeoX, DialoGPT-medium, GPT-2)<br>
**► Sparse mixture-of-expert** designs (e.g., Mixtral expert variants)<br>
**► Multilingual and culturally calibrated models** (e.g., Qwen2.5 base/instruct)<br>
**► Compact efficient architectures** (e.g., Phi-2, TinyLLaMA)<br>

# Installation Instructions:
