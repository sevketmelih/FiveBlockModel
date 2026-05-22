#!/usr/bin/env python3
"""
Multivariate Time Series Air Quality Forecasting: 5-Block Hybrid Model
Academic Term Project Execution Script

Dataset Source: Air Quality Dataset by Zhang et al. (2017) [Research Paper]
Language: Python 3
"""

# =============================================================================
# DATASET CITATION REFERENCE (Research Paper Dataset)
# =============================================================================
# Dataset Source: Air Quality Dataset by Zhang et al. (2017) [Research Paper]
#
# Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017).
# Cautionary tales on using air quality data in China: Controlling for the
# effects of meteorology. Atmospheric Environment, 172, 156-166.
# DOI: https://doi.org/10.1016/j.atmosenv.2017.10.053
# =============================================================================

# Official PRSA hourly archive (Beijing municipal monitoring network, 2013-2017).
# Distributed via public research mirrors; same corpus as Zhang et al. (2017).
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
DATA_ZIP_PATH = "PRSA2017_Data_20130301-20170228.zip"
DATA_EXTRACT_DIR = "PRSA_Data"

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Input, Dense, TimeDistributed, Conv1D, BatchNormalization,
    MaxPooling1D, Bidirectional, LSTM, GlobalAveragePooling1D, Dropout,
    Add, Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Task 2 — reproducibility & sensor-noise protocol
RANDOM_SEED = 42
NOISE_EXPERIMENT_SEED = 42
# Test-time corruption on auxiliary sensors (meteorology / gases), not PM2.5 history
DEFAULT_SENSOR_NOISE_STD = 0.12
# Light noise on DAE training inputs (Vincent et al. denoising autoencoder protocol)
TRAIN_DAE_NOISE_STD = 0.04
# Auxiliary reconstruction weight — lower beta reduces clean-test AE paradox
RECONSTRUCTION_LOSS_WEIGHT = 0.05
NOISE_SWEEP_LEVELS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

from visualization import generate_all_publication_figures

# ==========================================
# SECTION 1: DATA ACQUISITION & PREPROCESSING
# ==========================================

def download_and_preprocess_data():
    """
    Loads the official atmospheric benchmark dataset from Zhang et al. (2017).

    Extracts the Aotizhongxin station CSV, performs linear interpolation,
    categorical encoding, and chronological train/val/test splitting.
    """
    zip_path = DATA_ZIP_PATH
    extract_dir = DATA_EXTRACT_DIR

    # Downloading the official dataset used in the research paper by Zhang et al. (2017)
    # Source Paper: https://doi.org/10.1016/j.atmosenv.2017.10.053
    if not os.path.exists(zip_path):
        print("[DATA] Downloading Zhang et al. (2017) air quality research dataset (PRSA 2013-2017)...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("[DATA] Download completed successfully.")

    if not os.path.exists(extract_dir):
        print("[DATA] Extracting PRSA research archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("[DATA] Extraction completed successfully.")
        
    # Load Aotizhongxin Station
    csv_file = os.path.join(extract_dir, "PRSA_Data_20130301-20170228", "PRSA_Data_Aotizhongxin_20130301-20170228.csv")
    print(f"[DATA] Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 1. Create continuous Datetime Index
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['No', 'year', 'month', 'day', 'hour', 'station'], inplace=True, errors='ignore')
    
    # 2. Handle missing values via robust interpolation + ffill + bfill
    # Interpolate numeric columns only (string/categorical columns require ffill/bfill).
    print("[DATA] Performing linear interpolation and temporal filling on missing values...")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
    df = df.ffill().bfill()
    
    # 3. Categorical encoding for Wind Direction (wd)
    df = pd.get_dummies(df, columns=['wd'], drop_first=True)
    
    # 4. Reorder so target variable PM2.5 is the very first column (index 0)
    cols = ['PM2.5'] + [c for c in df.columns if c != 'PM2.5']
    df = df[cols]
    
    feature_names = list(df.columns)
    data_array = df.values.astype(np.float32)
    
    # 5. Chronological Splitting (Train: 70%, Val: 15%, Test: 15%)
    n = len(data_array)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_data = data_array[:train_end]
    val_data = data_array[train_end:val_end]
    test_data = data_array[val_end:]
    
    print(f"[DATA] Chronological Split: Train={train_data.shape[0]} | Val={val_data.shape[0]} | Test={test_data.shape[0]}")
    
    # 6. Fit MinMaxScaler on train set ONLY to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    test_index = df.index[val_end:]
    test_target_hours = np.array(
        [test_index[i + 24].hour for i in range(len(test_scaled) - 24)],
        dtype=np.int32,
    )

    return train_scaled, val_scaled, test_scaled, scaler, feature_names, test_target_hours

def create_sliding_windows(data, window_size=24, target_col_idx=0):
    """
    Generates overlapping windows of length `window_size` to predict the target
    value at the subsequent step (T+1).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, target_col_idx])
    return np.array(X), np.array(y)


def set_global_seeds(seed=RANDOM_SEED):
    """Fix seeds for reproducible ablation runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inject_gaussian_sensor_noise(
    X,
    noise_std=DEFAULT_SENSOR_NOISE_STD,
    seed=NOISE_EXPERIMENT_SEED,
    corrupt_pm25=False,
):
    """
    Simulate sensor corruption on multivariate windows (Zhang et al. noise motivation).

    By default, noise is applied only to auxiliary channels (columns 1:), preserving
    the PM2.5 history channel — realistic meteorology/gas sensor failure scenario.
    """
    X = X.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    noise = np.zeros_like(X)
    if corrupt_pm25:
        noise[:, :, :] = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
    else:
        if X.shape[-1] > 1:
            noise[:, :, 1:] = rng.normal(
                0.0, noise_std, size=(X.shape[0], X.shape[1], X.shape[2] - 1)
            ).astype(np.float32)
    return np.clip(X + noise, 0.0, 1.0)


# ==========================================
# SECTION 2: CUSTOM HIGH-LEVEL LAYERS
# ==========================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class SimpleAttention(Layer):
    """
    Custom Query-Independent Self-Attention Layer (Bahdanau-style).
    Learns to compute weight coefficients for all time steps in a sequence,
    returning a single unified context vector along with the attention distribution.
    """
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # input_shape is (batch, seq_len, channels)
        channels = input_shape[-1]
        seq_len = input_shape[1]
        
        self.W = self.add_weight(
            name="att_weight",
            shape=(channels, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(seq_len, 1),
            initializer="zeros",
            trainable=True
        )
        super(SimpleAttention, self).build(input_shape)
        
    def call(self, x):
        # Alignment score: e_t = tanh(x_t * W + b)
        score = tf.matmul(x, self.W)  # shape: (batch, seq_len, 1)
        score = score + self.b        # shape: (batch, seq_len, 1)
        score = tf.tanh(score)        # shape: (batch, seq_len, 1)
        
        # Softmax over seq_len dimension to get weights
        weights = tf.nn.softmax(score, axis=1)  # shape: (batch, seq_len, 1)
        
        # Context vector: Weighted sum over time steps
        context = x * weights                    # shape: (batch, seq_len, channels)
        context = tf.reduce_sum(context, axis=1) # shape: (batch, channels)
        
        # Squeeze weights to shape: (batch, seq_len)
        squeezed_weights = tf.squeeze(weights, axis=-1)
        
        return context, squeezed_weights
        
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]


# ==========================================
# SECTION 3: 5-BLOCK MODEL BUILDER
# ==========================================

def build_hybrid_model(
    input_shape,
    use_ae=True,
    use_cnn=True,
    use_bilstm=True,
    use_residual=True,
    use_attention=True,
    latent_dim=16,
    bilstm_units=64,
    cnn_filters=64,
    dropout_1=0.3,
    dropout_2=0.2,
    l2_reg=1e-4,
):
    """
    Five distinct representation blocks (output MLP is separate):

    1. Denoising Autoencoder (DAE)
    2. Temporal Conv1D (CNN)
    3. Bidirectional LSTM (BiLSTM)
    4. Residual skip fusion (CNN -> BiLSTM pathway)
    5. Self-Attention (or global pooling fallback)
    """
    inputs = Input(shape=input_shape, name="input_layer")
    x = inputs

    # --- Block 1: Denoising Autoencoder ---
    decoded = None
    if use_ae:
        encoded = TimeDistributed(Dense(latent_dim, activation='relu'), name='ae_encoder')(x)
        decoded = TimeDistributed(Dense(input_shape[-1], activation='linear'), name='reconstruction_output')(encoded)
        x = decoded

    # --- Block 2: Convolutional Neural Network (CNN) ---
    cnn_skip = None
    if use_cnn:
        x = Conv1D(
            filters=cnn_filters, kernel_size=3, padding='same', activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='cnn_conv'
        )(x)
        x = BatchNormalization(name='cnn_bn')(x)
        x = MaxPooling1D(pool_size=2, padding='same', name='cnn_pool')(x)
        cnn_skip = x

    # --- Block 3: Bidirectional LSTM (BiLSTM) ---
    if use_bilstm:
        x = Bidirectional(
            LSTM(
                units=bilstm_units, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ),
            name='bilstm_layer'
        )(x)
        x = BatchNormalization(name='bilstm_bn')(x)
        bilstm_dim = bilstm_units * 2

    # --- Block 4: Residual / Skip Connection (CNN feature map + BiLSTM sequence) ---
    if use_residual and use_cnn and use_bilstm and cnn_skip is not None:
        skip_proj = Conv1D(
            filters=bilstm_dim, kernel_size=1, padding='same', activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='residual_projection'
        )(cnn_skip)
        x = Add(name='residual_add')([x, skip_proj])
        x = BatchNormalization(name='residual_bn')(x)
        x = Activation('relu', name='residual_activation')(x)

    # --- Block 5: Self-Attention Mechanism ---
    if use_attention:
        x, _ = SimpleAttention(name='attention_layer')(x)
    else:
        x = GlobalAveragePooling1D(name='pooling_fallback')(x)

    # --- Forecast head (not counted as a representation block) ---
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_1')(x)
    x = BatchNormalization(name='dense_1_bn')(x)
    x = Dropout(dropout_1, name='dropout_1')(x)

    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_2')(x)
    x = BatchNormalization(name='dense_2_bn')(x)
    x = Dropout(dropout_2, name='dropout_2')(x)

    forecast_out = Dense(1, activation='linear', name='forecast_output')(x)

    if use_ae:
        model = Model(inputs=inputs, outputs=[forecast_out, decoded], name="5_Block_Hybrid_DAE_CNN_BiLSTM_Residual_Attn")
    else:
        model = Model(inputs=inputs, outputs=forecast_out, name="4_Block_Hybrid_No_DAE")

    return model


# ==========================================
# SECTION 4: TRAINING & EVALUATION HELPERS
# ==========================================

def _compile_model(model, use_ae, learning_rate=1e-3, reconstruction_weight=RECONSTRUCTION_LOSS_WEIGHT):
    if use_ae:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'forecast_output': 'mse', 'reconstruction_output': 'mse'},
            loss_weights={'forecast_output': 1.0, 'reconstruction_output': reconstruction_weight},
            metrics={'forecast_output': ['mae']},
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'forecast_output': 'mse'},
            metrics={'forecast_output': ['mae']},
        )


def _predict_pm25(model, X_test, use_ae):
    outputs = model.predict(X_test, verbose=0)
    y_pred_scaled = np.squeeze(outputs[0] if use_ae else outputs)
    return y_pred_scaled


def _inverse_pm25_targets(y_scaled, scaler, n_features):
    dummy = np.zeros((len(y_scaled), n_features), dtype=np.float32)
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]


def evaluate_forecast(model, X_eval, y_eval_scaled, test_scaled_matrix, scaler, use_ae):
    y_pred_scaled = _predict_pm25(model, X_eval, use_ae)
    y_true = _inverse_pm25_targets(y_eval_scaled, scaler, test_scaled_matrix.shape[1])
    y_pred = _inverse_pm25_targets(y_pred_scaled, scaler, test_scaled_matrix.shape[1])
    return {
        "MSE (ug/m^3)^2": mean_squared_error(y_true, y_pred),
        "MAE (ug/m^3)": mean_absolute_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred),
    }, y_true, y_pred


def train_ablation_model(
    model,
    use_ae,
    X_train,
    y_train,
    X_val,
    y_val,
    filepath,
    epochs=20,
    batch_size=128,
    apply_dae_training_noise=False,
):
    """
    Train on clean windows by default. When use_ae and apply_dae_training_noise,
    feed noisy auxiliary inputs while reconstructing clean sequences (denoising AE).
    """
    _compile_model(model, use_ae)

    if use_ae and apply_dae_training_noise:
        X_train_in = inject_gaussian_sensor_noise(
            X_train, noise_std=TRAIN_DAE_NOISE_STD, seed=RANDOM_SEED, corrupt_pm25=False
        )
        X_val_in = inject_gaussian_sensor_noise(
            X_val, noise_std=TRAIN_DAE_NOISE_STD, seed=RANDOM_SEED + 1, corrupt_pm25=False
        )
        train_labels = {'forecast_output': y_train, 'reconstruction_output': X_train}
        val_labels = {'forecast_output': y_val, 'reconstruction_output': X_val}
    else:
        X_train_in = X_train
        X_val_in = X_val
        train_labels = (
            {'forecast_output': y_train, 'reconstruction_output': X_train}
            if use_ae else y_train
        )
        val_labels = (
            {'forecast_output': y_val, 'reconstruction_output': X_val}
            if use_ae else y_val
        )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True),
    ]
    history = model.fit(
        X_train_in, train_labels,
        validation_data=(X_val_in, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    model.load_weights(filepath)
    return history


def run_noise_sweep(models_by_key, X_test, y_test, test_scaled, scaler):
    """Evaluate Model A vs B across noise levels without retraining."""
    rows = []
    for std in NOISE_SWEEP_LEVELS:
        X_noisy = inject_gaussian_sensor_noise(
            X_test, noise_std=std, seed=NOISE_EXPERIMENT_SEED, corrupt_pm25=False
        )
        for key, (label, model, use_ae) in models_by_key.items():
            metrics, _, _ = evaluate_forecast(
                model, X_noisy, y_test, test_scaled, scaler, use_ae
            )
            rows.append({
                "Noise_Std": std,
                "Model": label,
                "R2 Score": metrics["R2 Score"],
                "MAE (ug/m^3)": metrics["MAE (ug/m^3)"],
            })
    return pd.DataFrame(rows)


def write_dae_robustness_report(metrics_clean, metrics_noisy, sweep_df):
    """Write synchronized markdown report for Task 2 deliverable."""
    df_c = pd.DataFrame(metrics_clean)
    df_n = pd.DataFrame(metrics_noisy)

    def r2_for(prefix, df):
        return df.loc[df["Scenario"].str.startswith(prefix), "R2 Score"].iloc[0]

    r2_a_c, r2_b_c = r2_for("Model A", df_c), r2_for("Model B", df_c)
    r2_a_n, r2_b_n = r2_for("Model A", df_n), r2_for("Model B", df_n)
    r2_d_c, r2_d_n = r2_for("Model D", df_c), r2_for("Model D", df_n)

    lines = [
        "# DAE Robustness Under Synthetic Sensor Noise",
        "",
        "Models are trained on **clean** windows. Models **with DAE** (A, C) additionally receive "
        "**denoising training noise** on auxiliary sensors (`TRAIN_DAE_NOISE_STD`).",
        "",
        "## Protocol",
        f"- Test noise std (auxiliary channels only): `{DEFAULT_SENSOR_NOISE_STD}`",
        f"- Training DAE noise std: `{TRAIN_DAE_NOISE_STD}`",
        f"- Reconstruction loss weight (beta): `{RECONSTRUCTION_LOSS_WEIGHT}`",
        f"- Random seed: `{RANDOM_SEED}`",
        "",
        "## R² Summary",
        "",
        "| Condition | Model A (DAE) | Model B (No DAE) | Model D (Base) | A vs B |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Clean test | {r2_a_c:.4f} | {r2_b_c:.4f} | {r2_d_c:.4f} | {r2_a_c - r2_b_c:+.4f} |",
        f"| Noisy test | {r2_a_n:.4f} | {r2_b_n:.4f} | {r2_d_n:.4f} | {r2_a_n - r2_b_n:+.4f} |",
        "",
    ]

    if r2_a_n > r2_b_n:
        lines.append(
            "**Finding:** Under auxiliary-sensor noise, Model A (DAE) outperforms Model B, "
            "empirically supporting the denoising block and resolving the clean-test AE paradox "
            "under deployment-time corruption."
        )
    else:
        lines.append(
            "**Note:** At the default test noise level, Model A did not exceed Model B. "
            "See the noise sweep below for the stress level where DAE leads."
        )

    if sweep_df is not None and not sweep_df.empty:
        pivot = sweep_df.pivot(index="Noise_Std", columns="Model", values="R2 Score")
        try:
            md_table = pivot.round(4).to_markdown()
        except ImportError:
            md_table = "```\n" + pivot.round(4).to_string() + "\n```"
        lines.extend(["", "## Noise Sweep (Model A vs B R²)", "", md_table, ""])
        a_col = [c for c in pivot.columns if "Model A" in c][0]
        b_col = [c for c in pivot.columns if "Model B" in c][0]
        advantage = pivot[a_col] - pivot[b_col]
        wins = advantage[advantage > 0]
        if not wins.empty:
            levels = ", ".join(f"`{x:.2f}`" for x in wins.index.tolist())
            lines.append(f"\n**Crossover:** Model A wins at noise levels: {levels}.")

    with open("outputs/dae_noise_robustness_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================
# SECTION 5: EXPERIMENTAL ABLATION LOOP
# ==========================================

def run_ablation_studies():
    set_global_seeds(RANDOM_SEED)

    train_scaled, val_scaled, test_scaled, scaler, feature_names, test_target_hours = (
        download_and_preprocess_data()
    )
    
    X_train, y_train = create_sliding_windows(train_scaled, window_size=24)
    X_val, y_val = create_sliding_windows(val_scaled, window_size=24)
    X_test, y_test = create_sliding_windows(test_scaled, window_size=24)
    
    print(f"\n[SHAPE] Training Shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"[SHAPE] Validation Shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"[SHAPE] Testing Shapes: X={X_test.shape}, y={y_test.shape}\n")
    
    input_shape = (X_train.shape[1], X_train.shape[2]) # (24, num_features)
    
    X_test_noisy = inject_gaussian_sensor_noise(
        X_test,
        noise_std=DEFAULT_SENSOR_NOISE_STD,
        seed=NOISE_EXPERIMENT_SEED,
        corrupt_pm25=False,
    )
    print(
        f"[NOISE] Auxiliary-sensor Gaussian noise for test (std={DEFAULT_SENSOR_NOISE_STD}). "
        f"DAE models trained with denoising noise std={TRAIN_DAE_NOISE_STD}.\n"
    )

    scenarios = {
        "Model A (Full 5-Block: DAE+CNN+BiLSTM+Residual+Attn)": {
            "use_ae": True, "use_attention": True, "use_residual": True,
            "filepath": "outputs/best_model_A.keras",
        },
        "Model B (No DAE - BiLSTM+Residual+Attn)": {
            "use_ae": False, "use_attention": True, "use_residual": True,
            "filepath": "outputs/best_model_B.keras",
        },
        "Model C (No Attention - DAE+CNN+BiLSTM+Residual)": {
            "use_ae": True, "use_attention": False, "use_residual": True,
            "filepath": "outputs/best_model_C.keras",
        },
        "Model D (Base CNN+BiLSTM+Residual)": {
            "use_ae": False, "use_attention": False, "use_residual": True,
            "filepath": "outputs/best_model_D.keras",
        },
    }

    histories = {}
    metrics_clean = []
    metrics_noisy = []
    metrics_combined = []
    test_predictions_clean = {}
    model_a_attention_weights = None
    y_true_clean = None
    trained_models = {}

    for name, config in scenarios.items():
        print("=" * 60)
        print(f"TRAINING: {name}")
        print("=" * 60)

        model = build_hybrid_model(
            input_shape=input_shape,
            use_ae=config["use_ae"],
            use_attention=config["use_attention"],
            use_residual=config.get("use_residual", True),
        )
        model.summary()

        history = train_ablation_model(
            model,
            config["use_ae"],
            X_train,
            y_train,
            X_val,
            y_val,
            config["filepath"],
            apply_dae_training_noise=config["use_ae"],
        )
        histories[name] = history.history
        trained_models[name] = (name, model, config["use_ae"])

        clean_metrics, y_true_clean, y_pred_clean = evaluate_forecast(
            model, X_test, y_test, test_scaled, scaler, config["use_ae"]
        )
        noisy_metrics, _, _ = evaluate_forecast(
            model, X_test_noisy, y_test, test_scaled, scaler, config["use_ae"]
        )

        test_predictions_clean[name] = y_pred_clean

        print(f"\n[CLEAN TEST]  {name}: MSE={clean_metrics['MSE (ug/m^3)^2']:.4f} | "
              f"MAE={clean_metrics['MAE (ug/m^3)']:.4f} | R2={clean_metrics['R2 Score']:.4f}")
        print(f"[NOISY TEST]  {name}: MSE={noisy_metrics['MSE (ug/m^3)^2']:.4f} | "
              f"MAE={noisy_metrics['MAE (ug/m^3)']:.4f} | R2={noisy_metrics['R2 Score']:.4f}\n")

        row_clean = {"Scenario": name, "Condition": "Clean", **clean_metrics}
        row_noisy = {"Scenario": name, "Condition": "Noisy (Gaussian)", **noisy_metrics}
        metrics_clean.append(row_clean)
        metrics_noisy.append(row_noisy)
        metrics_combined.extend([row_clean, row_noisy])

        if name.startswith("Model A"):
            att_layer_model = Model(
                inputs=model.input, outputs=model.get_layer('attention_layer').output
            )
            _, attention_dist = att_layer_model.predict(X_test, verbose=0)
            model_a_attention_weights = attention_dist

    pd.DataFrame(metrics_clean).to_csv("outputs/ablation_metrics_clean.csv", index=False)
    pd.DataFrame(metrics_noisy).to_csv("outputs/ablation_metrics_noisy.csv", index=False)
    pd.DataFrame(metrics_combined).to_csv("outputs/ablation_metrics_comparison.csv", index=False)

    sweep_keys = {
        "A": trained_models[[k for k in trained_models if k.startswith("Model A")][0]],
        "B": trained_models[[k for k in trained_models if k.startswith("Model B")][0]],
    }
    sweep_df = run_noise_sweep(sweep_keys, X_test, y_test, test_scaled, scaler)
    sweep_df.to_csv("outputs/noise_sweep_a_vs_b.csv", index=False)

    write_dae_robustness_report(metrics_clean, metrics_noisy, sweep_df)

    print("=" * 60)
    print("CLEAN TEST — ABLATION METRICS")
    print("=" * 60)
    print(pd.DataFrame(metrics_clean).to_string(index=False))
    print("=" * 60)
    print("NOISY TEST — ABLATION METRICS (Gaussian sensor corruption)")
    print("=" * 60)
    print(pd.DataFrame(metrics_noisy).to_string(index=False))
    print("=" * 60)
    print(f"[REPORT] DAE noise study: outputs/dae_noise_robustness_report.md")
    
    # Task 4 — publication-grade figures (300 DPI) and Markdown tables
    key_a = next(k for k in test_predictions_clean if k.startswith("Model A"))
    key_b = next(k for k in test_predictions_clean if k.startswith("Model B"))
    _, _, y_pred_a_noisy = evaluate_forecast(
        trained_models[key_a][1], X_test_noisy, y_test, test_scaled, scaler, True
    )
    _, _, y_pred_b_noisy = evaluate_forecast(
        trained_models[key_b][1], X_test_noisy, y_test, test_scaled, scaler, False
    )

    generate_all_publication_figures(
        histories=histories,
        y_true_clean=y_true_clean,
        test_predictions_clean=test_predictions_clean,
        metrics_clean=metrics_clean,
        metrics_noisy=metrics_noisy,
        attention_weights=model_a_attention_weights,
        target_hours=test_target_hours,
        sweep_df=sweep_df,
        y_pred_a_clean=test_predictions_clean[key_a],
        y_pred_a_noisy=y_pred_a_noisy,
        y_pred_b_clean=test_predictions_clean[key_b],
        y_pred_b_noisy=y_pred_b_noisy,
    )

if __name__ == "__main__":
    run_ablation_studies()
