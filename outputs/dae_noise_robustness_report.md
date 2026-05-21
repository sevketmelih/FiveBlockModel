# DAE Robustness Under Synthetic Sensor Noise

Models are trained on **clean** windows. Models **with DAE** (A, C) additionally receive **denoising training noise** on auxiliary sensors (`TRAIN_DAE_NOISE_STD`).

## Protocol
- Test noise std (auxiliary channels only): `0.12`
- Training DAE noise std: `0.04`
- Reconstruction loss weight (beta): `0.05`
- Random seed: `42`

## R² Summary

| Condition | Model A (DAE) | Model B (No DAE) | Model D (Base) | A vs B |
| :--- | :---: | :---: | :---: | :---: |
| Clean test | 0.8690 | 0.8631 | 0.8719 | +0.0059 |
| Noisy test | 0.8388 | 0.8046 | 0.7193 | +0.0343 |

**Finding:** Under auxiliary-sensor noise, Model A (DAE) outperforms Model B, empirically supporting the denoising block and resolving the clean-test AE paradox under deployment-time corruption.

## Noise Sweep (Model A vs B R²)

```
Model      Model A (Full 5-Block: DAE+CNN+BiLSTM+Residual+Attn)  Model B (No DAE - BiLSTM+Residual+Attn)
Noise_Std                                                                                               
0.05                                                     0.8649                                   0.8515
0.08                                                     0.8562                                   0.8365
0.10                                                     0.8484                                   0.8224
0.12                                                     0.8388                                   0.8046
0.15                                                     0.8208                                   0.7690
0.18                                                     0.7982                                   0.7217
0.20                                                     0.7811                                   0.6830
```


**Crossover:** Model A wins at noise levels: `0.05`, `0.08`, `0.10`, `0.12`, `0.15`, `0.18`, `0.20`.