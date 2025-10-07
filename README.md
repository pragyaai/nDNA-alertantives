# <p align="left"> ![ndna_logo_v5.jpg](https://github.com/user-attachments/assets/e093b2ec-4918-4622-a3c0-7d86d8e98db3)</p>
# <p align="center"> Neural DNA - Cartograph: Latent Semantic Genome of Foundation Models </p>

### <p align="center"> Webpage: https://pragyaai.github.io/ndna/llm/ndna/ </p>

# Our Goals:
**► 1. How alignment and instruction tuning inscribe persistent latent signatures, distinguishing inherited traits from semantic mutations.<br><br>
► 2. Whether architectural form (dense vs. MoE) yields distinct geometric adaptation patterns (e.g., localized vs. distributed reconfiguration).<br><br>
► 3. Whether compact models preserve latent genomic complexity or collapse toward lower-dimensional manifolds with flattened nDNA signatures.**

# Introduction and Core Framework:

Our unified framework is based on three pillars that establishes three fundamental metrics that collectively characterize the information processing landscape of Deep Neural Network in modern LLMs.<br>

**► 1. _Spectral Curvature_ ($κℓ$)**: Quantifies geometric properties of the parameter manifold <br><br>
**► 2. _Thermodynamic Length_ ($Lℓ$)**: Measures information processing complexity via Fisher-Rao distances <br><br>
**► 3. _Belief Vector Dynamics_**: Captures epistemic confidence and uncertainty propagation <br><br>

<img width="1276" height="663" alt="image" src="https://github.com/user-attachments/assets/a4be9afd-cac8-4aba-ae3a-10216dacb4a0" />

The **Spectral metrics (αℓ, κℓ)** capture geometry, **thermodynamic length(Lℓ)** captures latent movement, and belief norm quantifies alignment strength of the model.

# Mathematical Foundation:

**Spectral Curvature Analysis (κℓ):**
                        
# <p align="center"> <img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/cf9d0f56-c8e5-4dbb-aea8-3a667003d4e1" /> </p>
where 
 s = parameterizes depth through the network. 
 Our discrete **(κℓ)** provides a practical, layerwise estimator

**Thermodynamic length(Lℓ)** at layers is defined as the amount of semantic effort needed to move a token from layer(l) to layer (L+1)

<img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/09e72306-af42-40cf-9490-f42c9d4f7d6e" />
<img width="1095" height="205" alt="image" src="https://github.com/user-attachments/assets/32346f7d-e6fb-4f3d-9209-cd871e99bbb3" />

# Datasets Used

We have used **SQuAD 2.0** (https://huggingface.co/datasets/rajpurkar/squad_v2) dataset for our experiments and analysis of these metrics.

# Models Evaluated

Throughout our work, we used the following foundational LLMs to prove our hypothesis:

**► Dense transformers** (e.g., LLaMA-2 base/instruct, LLaMA-3 3B base/instruct, Gemma, Falcon, GPT-NeoX, DialoGPT-medium, GPT-2)<br>
**► Sparse mixture-of-expert** designs (e.g., Mixtral expert variants)<br>
**► Multilingual and culturally calibrated models** (e.g., Qwen2.5 base/instruct)<br>
**► Compact efficient architectures** (e.g., Phi-2, TinyLLaMA)<br>


# Thermodynamic length(Lℓ) : 
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

# Installation Instructions:
