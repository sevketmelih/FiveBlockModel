#!/usr/bin/env python3
"""
Student 3 - The Optimization Expert
Automated hyperparameter search via Optuna for the 5-Block Hybrid Model.

Arama uzayı:
  learning_rate : {1e-2, 1e-3, 1e-4}
  dropout_rate  : {0.2, 0.3, 0.5}   (dropout_1 ve dropout_2 için aynı değer)
  bilstm_units  : {32, 64, 128}

Çıktılar (outputs/ klasörüne):
  optimization_history.png          — her trial'daki val_loss + en iyi değer
  param_importances.png             — hangi parametrenin skoru en çok etkilediği
  hyperparameter_search_results.csv — tüm trial detayları
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # ekransız ortamda PNG kaydetmek için
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import optuna
from optuna.samplers import TPESampler

# ─── Proje içi import'lar ─────────────────────────────────────────────────────
from main import (
    RANDOM_SEED,
    RECONSTRUCTION_LOSS_WEIGHT,
    build_hybrid_model,
    create_sliding_windows,
    download_and_preprocess_data,
    set_global_seeds,
)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
N_TRIALS        = 15   # toplam Optuna deneme sayısı
EPOCHS_PER_TRIAL = 12  # her trial için maksimum epoch (EarlyStopping devrede)
BATCH_SIZE      = 128
WINDOW_SIZE     = 24

os.makedirs("outputs", exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Veriyi bir kez yükle (her trial'da tekrar yüklememek için) ───────────────
print("[HPO] Veri yükleniyor...")
set_global_seeds(RANDOM_SEED)
train_scaled, val_scaled, test_scaled, scaler, feature_names = (
    download_and_preprocess_data()
)
X_train, y_train = create_sliding_windows(train_scaled, window_size=WINDOW_SIZE)
X_val,   y_val   = create_sliding_windows(val_scaled,   window_size=WINDOW_SIZE)
INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
print(f"[HPO] Hazır. Girdi şekli: {INPUT_SHAPE}\n")


# ─── Optuna objective fonksiyonu ──────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """Bir trial için model kurar, eğitir ve en iyi val_loss'u döndürür."""
    set_global_seeds(RANDOM_SEED + trial.number)

    # ── Hiperparametre örnekleme ──
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    dropout_rate  = trial.suggest_categorical("dropout_rate",  [0.2, 0.3, 0.5])
    bilstm_units  = trial.suggest_categorical("bilstm_units",  [32, 64, 128])

    # ── Model oluşturma ──
    model = build_hybrid_model(
        input_shape=INPUT_SHAPE,
        use_ae=True,
        use_cnn=True,
        use_bilstm=True,
        use_residual=True,
        use_attention=True,
        bilstm_units=bilstm_units,
        dropout_1=dropout_rate,
        dropout_2=dropout_rate,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "forecast_output":      "mse",
            "reconstruction_output": "mse",
        },
        loss_weights={
            "forecast_output":      1.0,
            "reconstruction_output": RECONSTRUCTION_LOSS_WEIGHT,
        },
        metrics={"forecast_output": ["mae"]},
    )

    train_labels = {
        "forecast_output":      y_train,
        "reconstruction_output": X_train,
    }
    val_labels = {
        "forecast_output":      y_val,
        "reconstruction_output": X_val,
    }

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        train_labels,
        validation_data=(X_val, val_labels),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )

    val_loss = float(min(history.history["val_loss"]))

    print(
        f"  Trial {trial.number:>2} | "
        f"lr={learning_rate:.0e}  dropout={dropout_rate}  bilstm={bilstm_units:>3} "
        f"→ val_loss={val_loss:.6f}"
    )
    return val_loss


# ─── Grafik 1: Optimization History ──────────────────────────────────────────
def _plot_optimization_history(study: optuna.Study) -> None:
    values    = [t.value for t in study.trials if t.value is not None]
    best_vals = [min(values[: i + 1]) for i in range(len(values))]
    trial_nums = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_nums, values, color="#bc5090", s=50, zorder=3, label="Trial val_loss")
    ax.plot(trial_nums, best_vals, color="#003f5c", linewidth=2.5, label="En iyi val_loss")
    ax.set_title("Optuna: Optimizasyon Geçmişi (Validation Loss)", fontsize=15, pad=12)
    ax.set_xlabel("Trial Numarası", fontsize=12)
    ax.set_ylabel("Validation Loss (MSE)", fontsize=12)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig("outputs/optimization_history.png", dpi=300)
    plt.close(fig)
    print("[HPO] Kaydedildi: outputs/optimization_history.png")


# ─── Grafik 2: Parameter Importances ─────────────────────────────────────────
def _plot_param_importances(study: optuna.Study) -> None:
    """
    Fanova tabanlı parametre önem skorlarını hesaplar ve çubuk grafik olarak kaydeder.
    Yeterli trial yoksa (< 4) basit bir hata mesajı yazdırır ve atlar.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 4:
        print("[HPO] Yeterli tamamlanmış trial yok — param_importances atlandı.")
        return

    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=RANDOM_SEED)
        importances = optuna.importance.get_param_importances(
            study, evaluator=evaluator
        )
    except Exception as exc:
        print(f"[HPO] Parametre önemi hesaplanamadı: {exc}")
        return

    params = list(importances.keys())
    scores = list(importances.values())
    colors = ["#003f5c", "#bc5090", "#ffa600"][: len(params)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(params, scores, color=colors, edgecolor="white", height=0.5)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=11,
        )
    ax.set_title("Optuna: Hiperparametre Önemi (Fanova)", fontsize=15, pad=12)
    ax.set_xlabel("Önem Skoru", fontsize=12)
    ax.set_xlim(0, max(scores) * 1.25)
    plt.tight_layout()
    fig.savefig("outputs/param_importances.png", dpi=300)
    plt.close(fig)
    print("[HPO] Kaydedildi: outputs/param_importances.png")


# ─── Ana akış ─────────────────────────────────────────────────────────────────
def run_hyperparameter_optimization() -> optuna.Study:
    print(f"[HPO] Optuna arama başlıyor ({N_TRIALS} trial × maks {EPOCHS_PER_TRIAL} epoch)...")
    print(f"[HPO] Arama uzayı:")
    print(f"       learning_rate : [1e-2, 1e-3, 1e-4]")
    print(f"       dropout_rate  : [0.2, 0.3, 0.5]")
    print(f"       bilstm_units  : [32, 64, 128]\n")

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name="5block_hpo",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    # ── En iyi sonucu yazdır ──
    best = study.best_trial
    print("\n" + "=" * 55)
    print("[HPO] EN İYİ TRIAL")
    print("=" * 55)
    print(f"  Trial #   : {best.number}")
    print(f"  Val Loss  : {best.value:.6f}")
    print("  Parametreler:")
    for k, v in best.params.items():
        print(f"    {k:<18}: {v}")
    print("=" * 55 + "\n")

    # ── CSV kaydet ──
    df_results = study.trials_dataframe()
    df_results.to_csv("outputs/hyperparameter_search_results.csv", index=False)
    print("[HPO] Kaydedildi: outputs/hyperparameter_search_results.csv")

    # ── Grafikler ──
    _plot_optimization_history(study)
    _plot_param_importances(study)

    return study


if __name__ == "__main__":
    run_hyperparameter_optimization()
