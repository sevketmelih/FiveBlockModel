# DAE Robustness Under Synthetic Sensor Noise

Models are trained on **clean** windows. Models **with DAE** (A, C) additionally receive **denoising training noise** on auxiliary sensors (`TRAIN_DAE_NOISE_STD`).

## Protocol
- Test noise std (auxiliary channels only): `0.12`
- Training DAE noise std: `0.04`
- Reconstruction loss weight (beta): `0.05`
- Training hyperparameters: see `outputs/best_hyperparameters.json` or module defaults
- Random seed: `42`

## R² Summary

| Condition | Model A (DAE) | Model B (No DAE) | Model D (Base) | A vs B |
| :--- | :---: | :---: | :---: | :---: |
| Clean test | 0.9159 | 0.9159 | 0.9191 | +0.0000 |
| Noisy test | 0.9170 | 0.8692 | 0.8627 | +0.0477 |

**Finding:** Under auxiliary-sensor noise, Model A (DAE) outperforms Model B, empirically supporting the denoising block and resolving the clean-test AE paradox under deployment-time corruption.

## Noise Sweep (Model A vs B R²)

```
Model      Model A (Full 5-Block: DAE+CNN+BiLSTM+Residual+Attn)  Model B (No DAE - BiLSTM+Residual+Attn)
Noise_Std                                                                                               
0.05                                                     0.9195                                   0.9049
0.08                                                     0.9193                                   0.8921
0.10                                                     0.9184                                   0.8814
0.12                                                     0.9170                                   0.8692
0.15                                                     0.9137                                   0.8485
0.18                                                     0.9091                                   0.8253
0.20                                                     0.9052                                   0.8088
```


**Crossover:** Model A wins at noise levels: `0.05`, `0.08`, `0.10`, `0.12`, `0.15`, `0.18`, `0.20`.