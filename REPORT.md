# REPORT — Five-Block Hybrid Deep Learning for Multivariate PM₂.₅ Forecasting (Zhang et al., 2017)

**Course project:** Multi-layer models integrating CNN, RNN/LSTM, and Autoencoder components  
**Repository:** FiveBlockModel  
**Primary report:** This file (`REPORT.md`) · Extended manuscript: `README.md`  
**Metrics source of truth:** `outputs/ablation_tables.md` (seed 42, Optuna Trial 11)

---

## 1. Project Objective

This project designs, trains, and evaluates a **five-block hybrid deep neural network** for **one-step-ahead ($T+1$) PM₂.₅ forecasting** from 24-hour multivariate atmospheric windows. The system integrates course-mandated building blocks in a single coherent pipeline:

1. **Autoencoder (AE)** — time-distributed denoising autoencoder (DAE) with auxiliary reconstruction loss  
2. **Convolutional Neural Network (CNN)** — Conv1D feature extraction with batch normalisation and max pooling  
3. **Recurrent network (LSTM)** — bidirectional LSTM (BiLSTM) for forward–backward temporal context  
4. **Residual fusion** — skip connection from CNN feature maps into the BiLSTM representation space  
5. **Self-attention** — query-independent Bahdanau-style temporal weighting  

We further study **regularization and robustness** through L2 weight decay, dropout, batch normalisation, early stopping, compound DAE loss ($\beta$), denoising training noise, and **controlled ablation** under clean versus **synthetically corrupted auxiliary sensors**.

---

## 2. Research Questions

1. Can a serial DAE–CNN–BiLSTM–residual–attention architecture forecast Beijing hourly PM₂.₅ with competitive $R^2$ on a chronological test split?  
2. Which architectural blocks (DAE, attention, full stack vs. base) contribute most to **clean** and **noisy** test performance?  
3. How do **Optuna-tuned** hyperparameters (learning rate, dropout, BiLSTM width) compare with hand-picked Adam defaults?  
4. Does **denoising autoencoder training** improve robustness when meteorology and gas sensor channels are corrupted at deployment time?  
5. Do **self-attention weights** align with interpretable temporal patterns (e.g., rush-hour emission peaks)?

---

## 3. Dataset and Preprocessing

### 3.1 Corpus and citation

| Field | Detail |
| :--- | :--- |
| **Primary source** | Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). *Atmospheric Environment*, 172, 156–166. [DOI: 10.1016/j.atmosenv.2017.10.053](https://doi.org/10.1016/j.atmosenv.2017.10.053) |
| **Archive** | UCI ML Repository mirror — PRSA Beijing Multi-Site Air Quality (2013–2017) |
| **Station** | **Aotizhongxin** — $N = 35{,}064$ contiguous hourly records |
| **Features** | PM₂.₅, PM₁₀, SO₂, NO₂, CO, O₃, temperature, pressure, dew point, precipitation, wind speed, one-hot wind direction |
| **Implementation** | `download_and_preprocess_data()` in `main.py` |

### 3.2 Why this dataset (not MNIST / Kaggle baseline)?

- **Course bonus:** Research-paper corpus (**+15** coding bonus tier vs. Kaggle baseline).  
- **Scientific fit:** Zhang et al. document sensor noise, missing hours, and meteorological confounding—motivating the **DAE** block.  
- **Multivariate structure:** 24-hour windows over $D \approx 26$ features justify **CNN + BiLSTM + attention**, not a unimodal image classifier.

### 3.3 Preprocessing protocol (Task 1 — Student 210911030)

1. Reconstruct `datetime`; drop non-informative ID columns.  
2. **Linear interpolation** on numeric columns → `ffill` → `bfill`.  
3. **One-hot encode** wind direction (`wd`, `drop_first=True`).  
4. Reorder columns: **PM₂.₅ at index 0** (forecast target).  
5. **Chronological split:** train 70% (~24,544 h) · validation 15% (~5,260 h) · test 15% (~5,260 h).  
6. **MinMaxScaler** fit on **train only**, applied to val/test (leakage-safe).  
7. **Sliding windows:** $\mathbf{X} \in \mathbb{R}^{24 \times D} \rightarrow y_{T+1}$ via `create_sliding_windows()`.

Representative shapes after windowing: train **24,520** · val **5,236** · test **5,236** samples.

---

## 4. Model and Training Core

**Implementation:** `main.py` (`build_hybrid_model`, `train_ablation_model`, `run_ablation_studies`), `hyperparameter_tuning.py`, `visualization.py`

### 4.1 Five-block architecture (Task 2 — Student 210911026)

| Block | Layer(s) | Role |
| :---: | :--- | :--- |
| **1 — DAE** | TimeDistributed encoder/decoder (`latent_dim=16`) | Denoise multivariate windows; auxiliary $\mathcal{L}_{\mathrm{recon}}$ |
| **2 — CNN** | Conv1D(64, k=3) + BN + MaxPool1D(2) | Local multivariate temporal patterns |
| **3 — BiLSTM** | Bidirectional LSTM(64, `return_sequences=True`) + BN | Forward–backward sequence memory |
| **4 — Residual** | 1×1 Conv projection + Add + BN + ReLU | Fuse CNN skip with BiLSTM pathway |
| **5 — Attention** | `SimpleAttention` (or `GlobalAveragePooling1D` fallback) | Learn temporal importance weights |

**Forecast head (separate):** Dense(64) → BN → Dropout → Dense(32) → BN → Dropout → Dense(1).

### 4.2 Compound loss (Model A)

$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{forecast\_mse}} + \beta \cdot \mathcal{L}_{\mathrm{reconstruction\_mse}}, \quad \beta = 0.05$$

### 4.3 Why Conv1D?

Multivariate air-quality windows exhibit **local cross-channel correlations** across neighbouring hours (e.g., wind shifts and gas precursors). Conv1D extracts these localized structures before recurrence.

### 4.4 Why BiLSTM?

PM₂.₅ dynamics depend on **past and future context within the 24-hour window** (diurnal cycles). Bidirectional LSTM captures both directions while preserving per-timestep outputs for attention.

### 4.5 Why DAE?

Zhang et al. emphasise **sensor noise and missing data**. Training with light Gaussian noise on **auxiliary channels only** ($\sigma = 0.04$) follows the denoising autoencoder protocol (Vincent et al., 2010) and regularises representations for corrupted deployment inputs.

### 4.6 Regularization and optimization

| Technique | Setting | Purpose |
| :--- | :--- | :--- |
| **L2** | $\lambda = 10^{-4}$ on kernels | Weight shrinkage |
| **Dropout** | 0.5 on both forecast-head layers (Optuna) | Overfitting control |
| **Batch normalisation** | CNN, BiLSTM, dense stacks | Stabilise activations |
| **Early stopping** | patience 10, restore best weights | Limit over-training |
| **DAE denoising noise** | $\sigma = 0.04$ train (aux. channels) | Robust representations |
| **Optimizer** | Adam, $\eta = 10^{-2}$ (Optuna) | Adaptive learning rate |
| **Batch size** | 128 | Memory / convergence trade-off |

---

## 5. Methodology

We use **controlled ablation**: four model variants (A–D) share Optuna-tuned hyperparameters; only architectural flags (`use_ae`, `use_attention`, `use_residual`) change.

| Model | DAE | CNN | BiLSTM | Residual | Attention |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Full)** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **B (No DAE)** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **C (No Attn)** | ✓ | ✓ | ✓ | ✓ | ✗ |
| **D (Base)** | ✗ | ✓ | ✓ | ✓ | ✗ |

**Evaluation protocols:**

- **Clean test:** no sensor corruption.  
- **Noisy test:** Gaussian noise $\sigma = 0.12$ on scaled **auxiliary** channels (columns $1{:}$); PM₂.₅ history untouched.  
- **Noise sweep:** $\sigma \in \{0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20\}$ for Models A vs. B.

**Metrics (inverse-scaled to physical units):** MSE $(\mu\mathrm{g}/\mathrm{m}^3)^2$, MAE $(\mu\mathrm{g}/\mathrm{m}^3)$, $R^2$.

**Reproducibility:** `RANDOM_SEED = 42`; checkpoints `outputs/best_model_{A,B,C,D}.keras`.

---

## 6. Hyperparameter Optimisation (Task 3 — Student 210911051)

**Tool:** Optuna, TPE sampler, 15 trials on **Model A** (validation loss minimisation).

| Hyperparameter | Search space | Best (Trial 11) |
| :--- | :--- | :---: |
| `learning_rate` | $10^{-2}$, $10^{-3}$, $10^{-4}$ | **$10^{-2}$** |
| `dropout_rate` | 0.2, 0.3, 0.5 | **0.5** |
| `bilstm_units` | 32, 64, 128 | **64** |

| Metric | Hand-tuned ($\eta=10^{-3}$) | Optuna best |
| :--- | :---: | :---: |
| Validation loss | ~0.037 | **0.008977** |

**Artifacts:**

- `hyperparameter_tuning.py`  
- `outputs/hyperparameter_search_results.csv`  
- `outputs/best_hyperparameters.json`  
- `outputs/optimization_history.png`, `outputs/param_importances.png`  
- `outputs/hyperparameter_summary.md`

**Interpretation:** Automated search reduced validation loss by roughly **4×** versus informal defaults, confirming that hybrid stacks are sensitive to optimisation settings.

---

## 7. Experiments and Results

*All tables below: chronological test split, $T+1$ horizon, seed 42, Optuna Trial 11 config.*

### 7.1 Student 1 — Data Architect (210911030)

**Deliverable:** `210911030.md`  
**Code:** `download_and_preprocess_data()`, `create_sliding_windows()`, `create_notebook.py`

| Deliverable | Evidence |
| :--- | :--- |
| Leakage-free chronological split | 70 / 15 / 15 in `main.py` |
| Train-only MinMaxScaler | `scaler.fit_transform(train)` only |
| PRSA bundle + UCI download | `PRSA_Data/`, `DATA_URL` in `main.py` |
| Notebook pipeline | `deep_learning_project.ipynb` |

**Interpretation:** Task 1 established the preprocessing contract preserved by all later tasks; no change to split/scaler rules when BiLSTM, DAE noise, and Optuna were added.

---

### 7.2 Student 2 — Deep Learning Engineer (210911026)

**Deliverable:** `210911026.md`, `outputs/dae_noise_robustness_report.md`  
**Code:** `build_hybrid_model()`, `inject_gaussian_sensor_noise()`, `run_ablation_studies()`

#### Table 7.2a — Clean test set

| Model | MSE | MAE | $R^2$ |
| :--- | :---: | :---: | :---: |
| **A (Full)** | 728.72 | 19.89 | 0.9159 |
| **B (No DAE)** | 728.84 | 16.05 | 0.9159 |
| **C (No Attn)** | 804.56 | 16.00 | 0.9072 |
| **D (Base)** | 701.53 | 19.34 | **0.9191** |

#### Table 7.2b — Noisy test set ($\sigma = 0.12$ on auxiliary sensors)

| Model | MSE | MAE | $R^2$ |
| :--- | :---: | :---: | :---: |
| **A (Full)** | 719.79 | 19.01 | **0.9170** |
| **B (No DAE)** | 1133.56 | 22.11 | 0.8692 |
| **C (No Attn)** | 812.88 | 17.20 | 0.9062 |
| **D (Base)** | 1190.35 | 26.68 | 0.8627 |

#### Table 7.2c — Robustness (Δ$R^2$ noisy − clean)

| Model | $R^2$ clean | $R^2$ noisy | Δ$R^2$ |
| :--- | :---: | :---: | :---: |
| **A (Full)** | 0.9159 | 0.9170 | **+0.0010** |
| **B (No DAE)** | 0.9159 | 0.8692 | −0.0467 |
| **D (Base)** | 0.9191 | 0.8627 | −0.0564 |

**DAE headline (A vs. B on noisy test):** $\Delta R^2 = \mathbf{+0.0477}$ in favour of Model A.

**Interpretation:**

- **CNN + BiLSTM + residual + attention** lift noisy baseline D from $R^2 = 0.863$ to full stack A at $R^2 = 0.917$.  
- **Attention** (A vs. C): ~0.009 $R^2$ gain on clean data.  
- **DAE:** Tied with B on clean data; decisive advantage under sensor corruption—validating the denoising training protocol.

---

### 7.3 Student 3 — Optimization Expert (210911051)

**Deliverable:** `210911051.md`  
**Artifacts:** See Section 6 above.

**Top-5 Optuna trials (validation loss):**

| Trial | lr | dropout | bilstm | val_loss |
| :---: | :---: | :---: | :---: | :---: |
| 11 | 0.01 | 0.5 | 64 | **0.008977** |
| 5 | 0.01 | 0.5 | 64 | 0.010770 |
| 13 | 0.01 | 0.5 | 64 | 0.009580 |

**Interpretation:** TPE converged to `lr = 10^{-2}` with high dropout (0.5) from Trial 5 onward—manual $\eta = 10^{-3}$ was systematically suboptimal.

---

### 7.4 Student 4 — Visualization & Analytics (210911028)

**Deliverable:** `210911028.md`  
**Code:** `visualization.py`, `generate_visualizations.py`, `regenerate_all_figures.py`

| Artifact | Role |
| :--- | :--- |
| `outputs/ablation_tables.md` | Publication Tables 1–4 |
| `outputs/figure_catalog.md` | Figure index (300 DPI) |
| `outputs/prediction_scatter_plot.png` | Model A actual vs. predicted |
| `outputs/ablation_clean_vs_noisy_r2.png` | Robustness bar chart |
| `outputs/attention_hour_of_day.png` | Rush-hour attention profile |
| `outputs/noise_sweep_a_vs_b.png` | Stress-test curve |
| `outputs/ablation_loss_curves.png` | Validation convergence |

**Attention finding:** Elevated weights during **07:00–09:00** and **17:00–20:00** (Beijing rush bands), consistent with traffic-related PM₂.₅ dynamics.

**Interpretation:** Task 4 separates diagnostic plot types (scatter vs. time-series vs. heatmap) and enforces **300 DPI** export for report embedding.

---

### 7.5 Student 5 — Academic Author (190722054)

**Deliverable:** `190722054.md`, `README.md` (extended IEEE-style manuscript)  
**Role:** Dataset citation framing, methodology narrative, results synthesis, references.

| README section | Content |
| :--- | :--- |
| Abstract + Keywords | Five-block hybrid; Zhang (2017); clean/noisy metrics |
| §1–3 | Motivation, preprocessing, architecture equations |
| §4 | Implementation, Optuna, reproduction commands |
| §5–6 | Results, discussion, conclusion, future work |
| References | Zhang, Hochreiter, Bahdanau, Vincent |

**Interpretation:** `README.md` is the long-form manuscript; `REPORT.md` (this file) is the consolidated course evidence document aligned with the CIFAR-10 report template.

---

### 7.6 Noise sweep — Model A vs. Model B ($R^2$)

| Noise $\sigma$ | Model A | Model B | A − B |
| :---: | :---: | :---: | :---: |
| 0.05 | 0.9195 | 0.9049 | +0.0146 |
| 0.08 | 0.9193 | 0.8921 | +0.0272 |
| 0.10 | 0.9184 | 0.8814 | +0.0370 |
| **0.12** | **0.9170** | **0.8692** | **+0.0478** |
| 0.15 | 0.9137 | 0.8485 | +0.0652 |
| 0.18 | 0.9091 | 0.8253 | +0.0838 |
| 0.20 | 0.9052 | 0.8088 | +0.0964 |

**Source:** `outputs/noise_sweep_a_vs_b.csv`  
**Finding:** Model A wins at **all** tested noise levels.

---

## 8. Course Requirements and Bonus Alignment

### 8.1 Mandatory components

| Requirement | Evidence |
| :--- | :--- |
| CNN | Conv1D block (§4.1) |
| RNN / LSTM / GRU | Bidirectional LSTM (§4.4) |
| Autoencoder | Time-distributed DAE (§4.5) |
| Coherent deep integration | Serial five-block pipeline |
| Component justification | §4.3–4.5, README §3 |
| Dataset source + rationale | §3.1–3.2 |
| Hyperparameter tuning | §6, `hyperparameter_tuning.py` |
| Regularization explained | §4.6 |

### 8.2 Coding bonus (max 60)

| Criterion | Points | Status |
| :--- | :---: | :---: |
| Not MNIST / Fashion-MNIST | avoid −10 | Met |
| Research-paper dataset (Zhang 2017) | +15 | Met |
| Five+ distinct blocks | +15 | Met |
| Ablation study | +15 | Met |
| Conference-style repository report | +15 | Met |
| **Total** | **60** | **Eligible** |

---

## 9. Reproducibility

### Install

```bash
pip install -r requirements.txt
```

### Full pipeline (train + evaluate + figures)

```bash
python hyperparameter_tuning.py          # optional: refresh HPO (or --export-only)
python main.py                           # ~12–20 min CPU (4 models × 20 epochs)
python generate_visualizations.py        # tables + CSV-based plots
```

### Figures from checkpoints (no retrain)

```bash
python regenerate_all_figures.py
```

### Regenerate notebook

```bash
python create_notebook.py
```

**Verified run log:** `test_results.md` (2026-05-22, exit code 0 on all steps).

---

## 10. Deliverables

| Category | Files |
| :--- | :--- |
| **Core code** | `main.py`, `hyperparameter_tuning.py`, `visualization.py` |
| **Utilities** | `generate_visualizations.py`, `regenerate_all_figures.py`, `create_notebook.py` |
| **Notebook** | `deep_learning_project.ipynb` |
| **Reports** | `REPORT.md` (this file), `README.md`, `test_results.md` |
| **Per-student** | `210911030.md`, `210911026.md`, `210911051.md`, `210911028.md`, `190722054.md` |
| **Metrics** | `outputs/ablation_tables.md`, `outputs/ablation_metrics_*.csv` |
| **Models** | `outputs/best_model_A.keras` … `D.keras` |
| **HPO** | `outputs/best_hyperparameters.json`, `outputs/hyperparameter_search_results.csv` |
| **Figures** | `outputs/*.png`, `outputs/figure_catalog.md` |
| **Robustness** | `outputs/dae_noise_robustness_report.md`, `outputs/noise_sweep_a_vs_b.csv` |

---

## 11. Final Evidence Table (Method → Clean $R^2$ → Noisy $R^2$ → Interpretation)

This table ties each major method or architectural choice to **clean** and **noisy** test $R^2$, matching the course request for method comparison with robustness evidence.

| Method / Setting | Clean $R^2$ | Noisy $R^2$ ($\sigma{=}0.12$) | Main interpretation |
| :--- | :---: | :---: | :--- |
| **Model A (Full five-block + DAE)** | 0.9159 | **0.9170** | Best noisy performance; stable under corruption |
| **Model B (No DAE)** | 0.9159 | 0.8692 | Same clean score; large drop when aux. sensors noisy |
| **Model C (No attention)** | 0.9072 | 0.9062 | Attention adds ~0.009 clean $R^2$; stable under noise |
| **Model D (Base CNN+BiLSTM+residual)** | **0.9191** | 0.8627 | Best clean; largest robustness collapse (Δ$R^2 = -0.0564$) |
| **L2 regularisation** ($\lambda{=}10^{-4}$) | — | — | Applied to Conv/BiLSTM/Dense kernels |
| **Dropout** (Optuna 0.5) | — | — | Strong forecast-head regularisation |
| **Batch normalisation** | — | — | CNN, BiLSTM, dense stacks |
| **Early stopping** (patience 10) | — | — | Best val-loss checkpoint restored |
| **DAE + $\beta{=}0.05$ recon loss** | 0.9159 (A) | 0.9170 (A) | Tied clean with B; **+0.0477 noisy Δ vs. B** |
| **DAE denoising train noise** ($\sigma{=}0.04$) | — | — | Vincent-style training on auxiliary channels |
| **Optuna HPO** (Trial 11) | — | — | $\eta{=}10^{-2}$, dropout 0.5, BiLSTM 64 → val_loss 0.0090 |
| **Self-attention** (A vs. C) | +0.0087 | +0.0108 | Learned temporal weighting vs. pooling |
| **Noise sweep A vs. B** | — | A wins all $\sigma$ | See §7.6; `noise_sweep_a_vs_b.png` |

**Evidence sources:**

- `outputs/ablation_tables.md`  
- `outputs/ablation_metrics_clean.csv`, `outputs/ablation_metrics_noisy.csv`  
- `outputs/dae_noise_robustness_report.md`  
- `outputs/noise_sweep_a_vs_b.csv`  
- `outputs/ablation_clean_vs_noisy_r2.png`  
- `outputs/attention_hour_of_day.png`  
- `test_results.md`

---

## 12. Conclusion

The FiveBlockModel project satisfies the course mandate to combine **CNN**, **LSTM (BiLSTM)**, and **Autoencoder** modules in a justified, reproducible forecasting system on a **peer-reviewed research dataset** (Zhang et al., 2017). Optuna tuning and multi-layer regularisation yield $R^2 \approx 0.92$ on clean hourly PM₂.₅ forecasts. Controlled ablation shows that the **denoising autoencoder** is the critical block for **sensor-noise robustness** ($\Delta R^2 = +0.0477$ vs. the no-DAE variant), while **self-attention** improves temporal selectivity and **rush-hour interpretability**. The repository provides code, checkpoints, 300 DPI figures, per-student deliverables, and two report layers (`REPORT.md`, `README.md`) suitable for submission and presentation.

---

## References

1. Zhang, S., et al. (2017). *Atmospheric Environment*, 172, 156–166. https://doi.org/10.1016/j.atmosenv.2017.10.053  
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.  
3. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv:1409.0473*.  
4. Vincent, P., et al. (2010). Stacked denoising autoencoders. *JMLR*, 11, 3371–3408.

---

*FiveBlockModel — aligned with `outputs/ablation_tables.md` and pipeline log `test_results.md`.*
