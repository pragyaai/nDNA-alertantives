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

where $s$ parameterizes depth through the network. Discrete $\kappa_\ell$ provides a practical, layerwise estimator.

## **Thermodynamic length(Lℓ) Analysis** : It is defined by Fisher geometry that quantifies semantic effort needed to move a token from layer(l) to layer (l+1).

*Thermodynamic length offers a window onto the model’s "latent energy budget" — illuminating how internal belief states reshape to meet complexity, constraint, and context.*

---

**Formulation**  
Let $p_\ell(y|x)$ denote the model’s conditional distribution at layer $\ell$ given input $x$.  
The local epistemic cost is reflected in the squared norm of the gradient of log-likelihood with respect to model parameters:

$$
\big\| \nabla_\theta \log p_\ell(x) \big\|^2
$$

This quantity measures how much the model must *adjust its parameters locally* at layer $\ell$ to improve its fit to input $x$.  
*Thermodynamic length at layer $\ell$* aggregates this cost across the dataset $\mathcal{D}$:

> **Thermodynamic length at layer $\ell$ is defined as:**
>
> $$
> \mathcal{L}_\ell := \sum_{x \in \mathcal{D}} \big\| \nabla_\theta \log p_\ell(x) \big\|^2
> = |\mathcal{D}| \, \mathbb{E}_{x \sim \mathcal{D}} \big\| \nabla_\theta \log p_\ell(x) \big\|^2
> $$

This formulation reveals that $\mathcal{L}_\ell$ captures both the *average local effort* and its scaling with dataset size.  

Furthermore, in differential geometric terms, thermodynamic length can be written as a path energy:

$$
\mathcal{L}_\ell = \int_{\gamma_\ell} 
\left\langle \frac{d h_\ell}{ds}, 
\mathcal{G}_{\mathrm{Fisher}}(h_\ell)
\frac{d h_\ell}{ds} \right\rangle ds
$$

where 

$$
h_\ell
$$ 

denotes latent trajectories at layer $\ell$,  

$$
\mathcal{G}_{\mathrm{Fisher}} is the Fisher information metric, $$ and  

$$
s is the arc length along $\gamma_\ell$. $$

Thus, $\mathcal{L}_\ell$ can be seen as an *energy integral over the belief manifold* — capturing how much *"heat"* or computational work is generated to reconcile prior belief state with new input at depth $\ell$.


<img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/09e72306-af42-40cf-9490-f42c9d4f7d6e" />
<img width="1095" height="205" alt="image" src="https://github.com/user-attachments/assets/32346f7d-e6fb-4f3d-9209-cd871e99bbb3" />
<br><br>

It quantifies the epistemic work performed across transformer layers, calculated as the cumulative squared gradient norm of layerwise log-likelihoods. Higher values signal internal resistance–zones of significant restructuring, belief compression, or negotiation of conflicting priors. In culturally fine-tuned models, these peaks localize to upper decoder layers, indicating intense adaptation near output-generating blocks. Within the nDNA construct, **Lℓ** helps reveal latent epistemic effort that underlies surface-level behavior. This metric thus provides a nuanced window into where and how models internally allocate effort during learning and inference. <br><br>

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

## **► Dense transformers** <br>
##### 1. LLaMA-2 base (https://huggingface.co/docs/transformers/en/model_doc/llama2)<br>
##### 2. LLaMA-2 instruct (https://huggingface.co/upstage/Llama-2-70b-instruct)<br>
##### 3. LLaMA-3 3B base (https://huggingface.co/meta-llama/Llama-3.2-3B)<br>
##### 4. LLaMA-3 3B instruct (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
##### 5. Gemma 7B (https://huggingface.co/google/gemma-7b)<br> 
##### 6. Falcon (https://huggingface.co/docs/transformers/en/model_doc/falcon#transformers.FalconModel)<br>
##### 7. GPT-2 (https://huggingface.co/openai-community/gpt2)<br>
##### 8. GPT-NeoX (https://huggingface.co/EleutherAI/gpt-neox-20b)<br>
##### 9. DialoGPT-medium (https://huggingface.co/microsoft/DialoGPT-medium))<br><br>
## **► Sparse mixture-of-expert** <br>
##### 1. Mixtral expert variants (https://huggingface.co/mistralai/Mixtral-8x7B-v0.1))<br><br>
## **► Multilingual and culturally calibrated models**<br>
##### 1. Qwen2.5 base (https://huggingface.co/Qwen/Qwen2.5-72B)<br>
##### 2. Qwen2.5 instruct (https://huggingface.co/unsloth/Qwen2.5-7B-Instruct))<br><br>
## **► Compact efficient architectures** <br>
##### 1. Phi-2 base (https://huggingface.co/microsoft/phi-2)
##### 2. Phi-2 instruct (https://huggingface.co/venkycs/phi-2-instruct)
##### 3. TinyLLama (https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0))<br><br>

# Installation Instructions:
