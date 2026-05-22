# Hyperparameter Optimization Summary (Task 3 — Student 210911051)

## Method
- **Library:** [Optuna](https://optuna.org/) with **TPE** sampler (`seed=42`)
- **Objective:** Minimize validation loss on **Model A** (full 5-block architecture)
- **Trials:** 15
- **Training protocol:** Matches `main.py` — DAE denoising noise on auxiliary sensors (`TRAIN_DAE_NOISE_STD=0.04`), β=0.05 reconstruction weight

## Search space

| Hyperparameter | Candidates |
| :--- | :--- |
| `learning_rate` | 1e-2, 1e-3, 1e-4 |
| `dropout_rate` | 0.2, 0.3, 0.5 (both forecast-head dropouts) |
| `bilstm_units` | 32, 64, 128 |

## Best trial (production configuration)

| Parameter | Value |
| :--- | :--- |
| Trial index | 11 |
| Validation loss | 0.008977 |
| `learning_rate` | 0.01 |
| `dropout_rate` | 0.5 |
| `bilstm_units` | 64 |

## Comparison to hand-picked defaults

| Setting | lr | dropout | bilstm | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Before tuning** | 1e-3 | 0.2 / 0.3 | 64 | Informal Adam default |
| **After Optuna** | 0.01 | 0.5 | 64 | **Used in `main.py`** |

Hand-tuned `lr=1e-3` trials clustered around val_loss ≈ 0.037; Optuna best reached **0.0090** (~4× lower), demonstrating that manual defaults were suboptimal.

## Integration

`python hyperparameter_tuning.py` writes `outputs/best_hyperparameters.json`.  
`python main.py` loads this file via `load_production_hyperparameters()` for all ablation models.

Re-export JSON from an existing CSV without retraining:

```bash
python hyperparameter_tuning.py --export-only
```

## Artifacts

| File | Description |
| :--- | :--- |
| `hyperparameter_search_results.csv` | Full trial log |
| `optimization_history.png` | Trial val_loss + running best |
| `param_importances.png` | Fanova parameter importance |
| `best_hyperparameters.json` | Production config for main pipeline |

## Top 5 trials (validation loss)

| Trial | lr | dropout | bilstm | val_loss |
| :---: | :---: | :---: | :---: | :---: |
| 11 | 0.01 | 0.5 | 64 | 0.008977 |
| 5 | 0.01 | 0.5 | 64 | 0.010770 |
| 13 | 0.01 | 0.5 | 64 | 0.009580 |
| 12 | 0.01 | 0.5 | 64 | 0.011360 |
| 9 | 0.01 | 0.5 | 64 | 0.012080 |
