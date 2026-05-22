#!/usr/bin/env python3
"""
Regenerate full Task 4 figure set from saved checkpoints (no retraining).

Requires: outputs/best_model_*.keras from a prior `python main.py` run.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

from main import (
    SimpleAttention,
    build_hybrid_model,
    create_sliding_windows,
    download_and_preprocess_data,
    evaluate_forecast,
    inject_gaussian_sensor_noise,
    set_global_seeds,
    DEFAULT_SENSOR_NOISE_STD,
)
from visualization import generate_all_publication_figures, regenerate_static_plots_from_csv

SCENARIOS = {
    "Model A (Full 5-Block: DAE+CNN+BiLSTM+Residual+Attn)": {
        "use_ae": True,
        "use_attention": True,
        "filepath": "outputs/best_model_A.keras",
    },
    "Model B (No DAE - BiLSTM+Residual+Attn)": {
        "use_ae": False,
        "use_attention": True,
        "filepath": "outputs/best_model_B.keras",
    },
    "Model C (No Attention - DAE+CNN+BiLSTM+Residual)": {
        "use_ae": True,
        "use_attention": False,
        "filepath": "outputs/best_model_C.keras",
    },
    "Model D (Base CNN+BiLSTM+Residual)": {
        "use_ae": False,
        "use_attention": False,
        "filepath": "outputs/best_model_D.keras",
    },
}


def _load_model(name: str, config: dict, input_shape):
    if not os.path.exists(config["filepath"]):
        raise FileNotFoundError(f"Missing checkpoint: {config['filepath']}")
    model = build_hybrid_model(
        input_shape=input_shape,
        use_ae=config["use_ae"],
        use_attention=config["use_attention"],
    )
    model.load_weights(config["filepath"])
    return model


def main():
    set_global_seeds(42)
    train_scaled, val_scaled, test_scaled, scaler, _, test_target_hours = (
        download_and_preprocess_data()
    )
    X_test, y_test = create_sliding_windows(test_scaled, window_size=24)
    X_test_noisy = inject_gaussian_sensor_noise(X_test, noise_std=DEFAULT_SENSOR_NOISE_STD)
    input_shape = (X_test.shape[1], X_test.shape[2])

    metrics_clean, metrics_noisy = [], []
    test_predictions_clean = {}
    y_true_clean = None
    attention_weights = None

    for name, config in SCENARIOS.items():
        print(f"[LOAD] {name}")
        model = _load_model(name, config, input_shape)
        clean_m, y_true_clean, y_pred = evaluate_forecast(
            model, X_test, y_test, test_scaled, scaler, config["use_ae"]
        )
        noisy_m, _, _ = evaluate_forecast(
            model, X_test_noisy, y_test, test_scaled, scaler, config["use_ae"]
        )
        metrics_clean.append({"Scenario": name, "Condition": "Clean", **clean_m})
        metrics_noisy.append(
            {"Scenario": name, "Condition": "Noisy (Gaussian)", **noisy_m}
        )
        test_predictions_clean[name] = y_pred

        if name.startswith("Model A") and config["use_attention"]:
            att_model = Model(
                inputs=model.input,
                outputs=model.get_layer("attention_layer").output,
            )
            _, attention_weights = att_model.predict(X_test, verbose=0)

    pd.DataFrame(metrics_clean).to_csv("outputs/ablation_metrics_clean.csv", index=False)
    pd.DataFrame(metrics_noisy).to_csv("outputs/ablation_metrics_noisy.csv", index=False)

    sweep_path = "outputs/noise_sweep_a_vs_b.csv"
    sweep_df = pd.read_csv(sweep_path) if os.path.exists(sweep_path) else None

    key_a = next(k for k in test_predictions_clean if k.startswith("Model A"))
    key_b = next(k for k in test_predictions_clean if k.startswith("Model B"))
    _, _, y_pred_a_noisy = evaluate_forecast(
        _load_model(key_a, SCENARIOS[key_a], input_shape),
        X_test_noisy,
        y_test,
        test_scaled,
        scaler,
        True,
    )
    _, _, y_pred_b_noisy = evaluate_forecast(
        _load_model(key_b, SCENARIOS[key_b], input_shape),
        X_test_noisy,
        y_test,
        test_scaled,
        scaler,
        False,
    )

    generate_all_publication_figures(
        histories={},
        y_true_clean=y_true_clean,
        test_predictions_clean=test_predictions_clean,
        metrics_clean=metrics_clean,
        metrics_noisy=metrics_noisy,
        attention_weights=attention_weights,
        target_hours=test_target_hours,
        sweep_df=sweep_df,
        y_pred_a_clean=test_predictions_clean[key_a],
        y_pred_a_noisy=y_pred_a_noisy,
        y_pred_b_clean=test_predictions_clean[key_b],
        y_pred_b_noisy=y_pred_b_noisy,
    )
    print("[DONE] All Task 4 figures regenerated from checkpoints.")


if __name__ == "__main__":
    main()
