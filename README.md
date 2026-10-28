# BiSpikeNet-V2: A Binary Spiking Neural Network for Tea Pest and Disease Recognition in Complex Agricultural Environments

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)
![Bit-Width](https://img.shields.io/badge/Bit--Width-1W1A-brightgreen.svg)
![Params](https://img.shields.io/badge/Params-4.80M-purple.svg)
![FLOPs](https://img.shields.io/badge/FLOPs-117.8M-red.svg)

**Official PyTorch Implementation of BiSpikeNet-V2**  
*Yalin Hu, Shengbing Che\**  
College of Computer and Mathematics, Central South University of Forestry and Technology, Changsha 410004, China

</div>

---

## 📖 Overview

**BiSpikeNet-V2** is a lightweight 1-bit neuromorphic vision network designed for *in-situ* tea pest and disease recognition under complex open-field interferences (e.g., direct sunlight glare, heavy canopy shadows, and background weed clutter). Guided by the **Feature Attribution Diagnosis and Targeted Reconstruction (FADTR)** paradigm, our framework integrates `ATan-PLIF`, `SF-STA`, and `LRR-Block` to compress parameters by **77.4% (4.80 M)** and equivalent FLOPs by **62.7% (117.8 M)** while maintaining robust cross-environment representation.

---

## 🌿 DECo-Split Protocol: Environment Decoupling & Zero-Leakage Split

To prevent cross-set data leakage from continuous field bursts and evaluate robustness across diverse physical imaging conditions, we propose the **Foreground-Decoupled Environment Domain Clustering and Zero-Leakage Orthogonal Split (`DECo-Split`)** protocol:

1. **Foreground-Background Spatial Decoupling**: Uses bounding box masks (with a 4-pixel padding) to strip out foreground leaf/lesion regions, isolating pure background pixels for physical imaging analysis (automatically falling back to global statistics if background area $< 8\%$).
2. **69-D Multi-Space Physical Feature Extraction**: Extracts 7 categories of low-level imaging features across HSV, Lab, and RGB spaces, including channel mean/std statistics (15-D), 16-bin normalized histograms (48-D), green canopy index (1-D), luminance & dynamic contrast (2-D), Canny edge density (1-D), Laplacian high-frequency variance (1-D), and valid background ratio (1-D).
3. **Homologous Isolation & Manifold Clustering**: Binds homologous images into independent source groups using exact **MD5** deduplication and **64-bit dHash** (Hamming distance $\le 2$) union-find grouping, strictly guaranteeing zero cross-set leakage ($\vert{}\text{Train} \cap \text{Val}\vert{} = 0, \vert{}\text{Train} \cap \text{Test}\vert{} = 0, \vert{}\text{Val} \cap \text{Test}\vert{} = 0$). PCA (retaining 95% variance) and K-Means ($K=5$) then decouple the complex habitats into 5 physical environment domains (`Env 0` to `Env 4`):
   - **Env 0**: Weak-Shadow Mixed Background (Diffuse light with inter-row weeds)
   - **Env 1**: Standard Front-Lit Foliage (Uniform illumination, clear contours)
   - **Env 2**: Detached Low-Texture Control (Clean low-frequency backdrop)
   - **Env 3**: Backlit Weak-Light Low-Contrast (Heavy canopy shadows, low SNR)
   - **Env 4**: Direct-Glare High-Reflection (Strong midday specular glare & deep shadows)

<div align="center">
  <!-- [Paste Paper Fig. 3 (PCA Distribution) and Fig. 4 (Env 0-4 Samples) here] -->
  <img src="<img width="601" height="481" alt="image" src="https://github.com/user-attachments/assets/79c2424a-b65f-47ad-b7ab-f20bf7d58d46" />
" width="46%" alt="PCA Distribution"/>
  <img src="<img width="447" height="323" alt="image" src="https://github.com/user-attachments/assets/87b705fd-ea8a-40e1-bcdf-8e2edbb4dae0" />
" width="48%" alt="Environment Samples"/>
  <br>
  <em><b>Fig. 1.</b> Left: PCA manifold projection of the 5 physical background domains. Right: Representative field samples decoupled across Env 0 – Env 4.</em>
</div>

```bash
# Run DECo-Split protocol on raw field images
python tools/deco_split.py \
    --raw_data_dir ./data/raw_images \
    --yolo_label_dir ./data/labels \
    --output_dir ./data/TeaFuza_Split \
    --padding 4 --dhash_threshold 2 --n_clusters 5
