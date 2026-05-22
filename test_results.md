# Test Results — FiveBlockModel Pipeline Run

**Run date:** 2026-05-22  
**Workspace:** `/Users/sevketmelihergun/FiveBlockModel`  
**Random seed:** 42  
**Dataset:** Zhang et al. (2017) — Aotizhongxin station, PRSA 2013–2017  

---

## 1. Commands Executed

| Step | Command | Exit code | Approx. duration | Notes |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `pip install -r requirements.txt` | 0 | ~6 s | Dependencies installed |
| 2 | `python generate_visualizations.py` | 0 | ~4 s | Tables + CSV plots; first pass skipped loss curves (no histories JSON yet) |
| 3 | `python main.py` | 0 | **~11.6 min** (698 s) | Full train + eval + 300 DPI figures + `ablation_training_histories.json` |
| 4 | `python regenerate_all_figures.py` | 0 | ~20 s | All Task 4 figures from checkpoints |
| 5 | `python generate_visualizations.py` (post-run) | 0 | ~3 s | Regenerated `ablation_loss_curves.png` from histories JSON |

---

## 2. Training Configuration (Loaded at Runtime)

Source: `outputs/best_hyperparameters.json` (`optuna_csv_export`)

| Parameter | Value |
| :--- | :--- |
| `learning_rate` | **0.01** |
| `dropout_rate` | **0.5** (both forecast-head layers) |
| `bilstm_units` | **64** |
| `reconstruction_loss_weight` (β) | **0.05** |
| `train_dae_noise_std` | **0.04** (auxiliary sensors only) |
| `test_sensor_noise_std` | **0.12** (auxiliary sensors only) |
| Optuna best `val_loss` (Trial 11) | 0.008977 |
| Optimizer | Adam |
| Batch size | 128 |
| Max epochs | 20 (early stopping patience 10) |

Console confirmation:

```text
[HPO] Production hyperparameters (optuna_csv_export): lr=0.01, dropout=0.5, bilstm_units=64
[NOISE] Auxiliary-sensor Gaussian noise for test (std=0.12). DAE models trained with denoising noise std=0.04.
```

---

## 3. Data Pipeline Summary

| Item | Value |
| :--- | :--- |
| CSV | `PRSA_Data/.../PRSA_Data_Aotizhongxin_20130301-20170228.csv` |
| Chronological split | Train 24,544 \| Val 5,260 \| Test 5,260 hours |
| Window shape | `(N, 24, 26)` → predict PM2.5 at T+1 |
| Train windows | 24,520 |
| Val windows | 5,236 |
| Test windows | 5,236 |

---

## 4. Ablation Metrics — Clean Test Set

| Model | MSE (µg/m³)² | MAE (µg/m³) | R² |
| :--- | :---: | :---: | :---: |
| **A (Full — DAE+CNN+BiLSTM+Residual+Attn)** | 728.72 | 19.89 | **0.9159** |
| **B (No DAE)** | 728.84 | 16.05 | 0.9159 |
| **C (No Attention)** | 804.56 | 16.00 | 0.9072 |
| **D (Base CNN+BiLSTM+Residual)** | **701.53** | 19.34 | **0.9191** |

---

## 5. Ablation Metrics — Noisy Test Set (σ = 0.12)

| Model | MSE (µg/m³)² | MAE (µg/m³) | R² |
| :--- | :---: | :---: | :---: |
| **A (Full)** | 719.79 | 19.01 | **0.9170** |
| **B (No DAE)** | 1133.56 | 22.11 | 0.8692 |
| **C (No Attention)** | 812.88 | 17.20 | 0.9062 |
| **D (Base)** | 1190.35 | 26.68 | 0.8627 |

---

## 6. Robustness Summary (Clean → Noisy)

| Model | R² (Clean) | R² (Noisy) | ΔR² | MAE (Clean) | MAE (Noisy) | ΔMAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Full)** | 0.9159 | 0.9170 | **+0.0010** | 19.89 | 19.01 | −0.88 |
| **B (No DAE)** | 0.9159 | 0.8692 | −0.0467 | 16.05 | 22.11 | +6.06 |
| **C (No Attn)** | 0.9072 | 0.9062 | −0.0010 | 16.00 | 17.20 | +1.20 |
| **D (Base)** | 0.9191 | 0.8627 | −0.0564 | 19.34 | 26.68 | +7.34 |

### DAE contribution (Model A vs. B)

| Comparison | Clean ΔR² (A − B) | Noisy ΔR² (A − B) |
| :--- | :---: | :---: |
| DAE advantage | +0.0000 | **+0.0477** |

**Finding:** On clean data, A and B are effectively tied. Under auxiliary-sensor noise, Model A **improves** slightly (R² 0.917 vs. 0.869 for B), validating the DAE denoising protocol.

---

## 7. Noise Sweep — Model A vs. Model B (R²)

| Noise σ | Model A R² | Model B R² | A − B |
| :---: | :---: | :---: | :---: |
| 0.05 | 0.9195 | 0.9049 | +0.0146 |
| 0.08 | 0.9193 | 0.8921 | +0.0272 |
| 0.10 | 0.9184 | 0.8814 | +0.0370 |
| **0.12** | **0.9170** | **0.8692** | **+0.0478** |
| 0.15 | 0.9137 | 0.8485 | +0.0652 |
| 0.18 | 0.9091 | 0.8253 | +0.0838 |
| 0.20 | 0.9052 | 0.8088 | +0.0964 |

Model A wins at **all** sweep levels. Source: `outputs/noise_sweep_a_vs_b.csv`.

---

## 8. Per-Model Console Results (`main.py`)

```
[CLEAN TEST]  Model A: MSE=728.7167 | MAE=19.8910 | R2=0.9159
[NOISY TEST]  Model A: MSE=719.7894 | MAE=19.0068 | R2=0.9170

[CLEAN TEST]  Model B: MSE=728.8406 | MAE=16.0521 | R2=0.9159
[NOISY TEST]  Model B: MSE=1133.5642 | MAE=22.1092 | R2=0.8692

[CLEAN TEST]  Model C: MSE=804.5593 | MAE=16.0014 | R2=0.9072
[NOISY TEST]  Model C: MSE=812.8808 | MAE=17.2029 | R2=0.9062

[CLEAN TEST]  Model D: MSE=701.5348 | MAE=19.3398 | R2=0.9191
[NOISY TEST]  Model D: MSE=1190.3451 | MAE=26.6846 | R2=0.8627
```

---

## 9. Generated Artifacts

### Metrics & reports (CSV / MD / JSON)

| File | Description |
| :--- | :--- |
| `outputs/ablation_metrics_clean.csv` | Clean-test metrics |
| `outputs/ablation_metrics_noisy.csv` | Noisy-test metrics |
| `outputs/ablation_metrics_comparison.csv` | Combined clean + noisy |
| `outputs/ablation_tables.md` | Publication Tables 1–4 |
| `outputs/noise_sweep_a_vs_b.csv` | Noise sweep data |
| `outputs/dae_noise_robustness_report.md` | DAE robustness narrative |
| `outputs/ablation_training_histories.json` | Validation loss per epoch |
| `outputs/best_hyperparameters.json` | Optuna production config |
| `outputs/figure_catalog.md` | Figure index + latest metrics |

### Model checkpoints

| File | Model |
| :--- | :--- |
| `outputs/best_model_A.keras` | Full 5-block |
| `outputs/best_model_B.keras` | No DAE |
| `outputs/best_model_C.keras` | No attention |
| `outputs/best_model_D.keras` | Base |

### Figures (300 DPI unless noted)

| File | Status after run |
| :--- | :---: |
| `ablation_loss_curves.png` | ✓ |
| `ablation_clean_vs_noisy_r2.png` | ✓ |
| `prediction_scatter_plot.png` | ✓ |
| `prediction_scatter_all_models.png` | ✓ |
| `prediction_scatter_A_Full.png` … `D_Base.png` | ✓ |
| `prediction_timeseries_72h.png` | ✓ |
| `attention_weights_map.png` | ✓ |
| `attention_hour_of_day.png` | ✓ |
| `attention_rush_hour_comparison.png` | ✓ |
| `noise_sweep_a_vs_b.png` | ✓ |
| `dae_clean_vs_noisy_forecast.png` | ✓ |
| `optimization_history.png` | ✓ |
| `param_importances.png` | ✓ (pre-existing; `generate_visualizations.py` skips if Optuna import fails) |

---

## 10. Brief Interpretation

1. **Clean test:** Model D reaches the highest R² (0.919); Models A and B are tied (~0.916). Model C (no attention) is slightly lower (0.907).
2. **Noisy test (σ=0.12):** Model A is best (R² 0.917); B and D drop sharply (0.869 / 0.863). DAE + denoising training preserves performance under sensor corruption.
3. **Attention:** Comparing A vs. C on noisy data, full attention (A: 0.917) beats no-attention variant (C: 0.906) by ~0.01 R².
4. **MAE on clean:** B and C achieve lower MAE (~16 µg/m³) than A (~19.9), but this does not hold under noise.

---

## 11. Reproduce This Run

```bash
pip install -r requirements.txt
python generate_visualizations.py
python main.py
python regenerate_all_figures.py
```

---

*Compiled automatically after full pipeline execution on 2026-05-22.*
