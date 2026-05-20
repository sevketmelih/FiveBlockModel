#!/usr/bin/env python3
"""
Multivariate Time Series Air Quality Forecasting: 5-Block Hybrid Model
Academic Term Project Execution Script
Author: Senior AI Research Engineer
Language: Python 3
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, TimeDistributed, Conv1D, BatchNormalization, MaxPooling1D, LSTM, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Set plotting styling for publication-grade charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300
})
PALETTE = ["#003f5c", "#bc5090", "#ffa600", "#ff6361", "#58508d"]

# ==========================================
# SECTION 1: DATA ACQUISITION & PREPROCESSING
# ==========================================

def download_and_preprocess_data():
    """
    Downloads the official UCI Beijing Multi-Site Air Quality Dataset,
    extracts the Aotizhongxin station csv, performs linear interpolation,
    categorical encoding, and splits it into chronological train/val/test sets.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
    zip_path = "PRSA2017_Data_20130301-20170228.zip"
    extract_dir = "PRSA_Data"
    
    # Programmatically download
    if not os.path.exists(zip_path):
        print("[DATA] Downloading Beijing Air Quality dataset from UCI Repository...")
        urllib.request.urlretrieve(url, zip_path)
        print("[DATA] Download completed successfully.")
        
    # Extract
    if not os.path.exists(extract_dir):
        print("[DATA] Extracting zip file...")
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
    print("[DATA] Performing linear interpolation and temporal filling on missing values...")
    df = df.interpolate(method='linear').ffill().bfill()
    
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
    
    return train_scaled, val_scaled, test_scaled, scaler, feature_names

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

def build_hybrid_model(input_shape, use_ae=True, use_cnn=True, use_lstm=True, use_attention=True, latent_dim=8, l2_reg=1e-4):
    """
    Constructs a highly parameterized deep learning architecture containing up to 5 sequential blocks:
    1. Denoising Autoencoder (Jointly trained with multi-output compilation if use_ae=True)
    2. Convolutional Layer (Conv1D + BatchNorm + MaxPool1D)
    3. Recurrent Layer (LSTM with sequence return)
    4. Custom Self-Attention Layer (or GlobalAveragePooling1D fallback)
    5. Dense MLP Decoder (Dense + BN + Dropout + Output)
    """
    inputs = Input(shape=input_shape, name="input_layer")
    x = inputs
    
    # --- Block 1: Denoising Autoencoder ---
    decoded = None
    if use_ae:
        # Encoder: compress step-wise features
        encoded = TimeDistributed(Dense(latent_dim, activation='relu'), name='ae_encoder')(x)
        # Decoder: reconstruct original features
        decoded = TimeDistributed(Dense(input_shape[-1], activation='linear'), name='reconstruction_output')(encoded)
        # Pass reconstructed denoised data to subsequent blocks
        x = decoded
        
    # --- Block 2: Convolutional Neural Network (CNN) ---
    if use_cnn:
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='cnn_conv')(x)
        x = BatchNormalization(name='cnn_bn')(x)
        x = MaxPooling1D(pool_size=2, padding='same', name='cnn_pool')(x)
        
    # --- Block 3: Recurrent Neural Network (LSTM) ---
    if use_lstm:
        x = LSTM(units=64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='lstm_layer')(x)
        x = BatchNormalization(name='lstm_bn')(x)
        
    # --- Block 4: Self-Attention Mechanism ---
    attention_weights = None
    if use_attention:
        # Custom Attention extracts context vector and historical focus weights
        x, attention_weights = SimpleAttention(name='attention_layer')(x)
    else:
        # Fallback to GlobalAveragePooling1D to collapse temporal dimension cleanly
        x = GlobalAveragePooling1D(name='pooling_fallback')(x)
        
    # --- Block 5: Dense / MLP Output ---
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_1')(x)
    x = BatchNormalization(name='dense_1_bn')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='dense_2')(x)
    x = BatchNormalization(name='dense_2_bn')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    
    forecast_out = Dense(1, activation='linear', name='forecast_output')(x)
    
    # Establish inputs and outputs based on Autoencoder availability
    if use_ae:
        model = Model(inputs=inputs, outputs=[forecast_out, decoded], name="5_Block_Hybrid_AE")
    else:
        model = Model(inputs=inputs, outputs=forecast_out, name="4_Block_Hybrid_No_AE")
        
    return model


# ==========================================
# SECTION 4: EXPERIMENTAL ABLATION LOOP
# ==========================================

def run_ablation_studies():
    # 1. Load and window the dataset
    train_scaled, val_scaled, test_scaled, scaler, feature_names = download_and_preprocess_data()
    
    X_train, y_train = create_sliding_windows(train_scaled, window_size=24)
    X_val, y_val = create_sliding_windows(val_scaled, window_size=24)
    X_test, y_test = create_sliding_windows(test_scaled, window_size=24)
    
    print(f"\n[SHAPE] Training Shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"[SHAPE] Validation Shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"[SHAPE] Testing Shapes: X={X_test.shape}, y={y_test.shape}\n")
    
    input_shape = (X_train.shape[1], X_train.shape[2]) # (24, num_features)
    
    # Define the 4 ablation model configurations
    scenarios = {
        "Model A (Full Model - 5 Blocks)": {
            "use_ae": True, "use_attention": True, "filepath": "outputs/best_model_A.keras"
        },
        "Model B (No Autoencoder - 4 Blocks)": {
            "use_ae": False, "use_attention": True, "filepath": "outputs/best_model_B.keras"
        },
        "Model C (No Attention - 4 Blocks)": {
            "use_ae": True, "use_attention": False, "filepath": "outputs/best_model_C.keras"
        },
        "Model D (Base CNN+LSTM - 3 Blocks)": {
            "use_ae": False, "use_attention": False, "filepath": "outputs/best_model_D.keras"
        }
    }
    
    histories = {}
    metrics_summary = []
    test_predictions = {}
    
    # Store attention weights map from Model A for visualization
    model_a_attention_weights = None
    
    for name, config in scenarios.items():
        print("="*60)
        print(f"TRAINING: {name}")
        print("="*60)
        
        # Build model
        model = build_hybrid_model(
            input_shape=input_shape, 
            use_ae=config["use_ae"], 
            use_attention=config["use_attention"]
        )
        
        # Compile Model with joint multi-output weighting if AE is enabled
        if config["use_ae"]:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    'forecast_output': 'mse',
                    'reconstruction_output': 'mse'
                },
                loss_weights={
                    'forecast_output': 1.0,
                    'reconstruction_output': 0.2
                },
                metrics={'forecast_output': ['mae']}
            )
            
            # Map labels to respective outputs
            train_labels = {'forecast_output': y_train, 'reconstruction_output': X_train}
            val_labels = {'forecast_output': y_val, 'reconstruction_output': X_val}
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={'forecast_output': 'mse'},
                metrics={'forecast_output': ['mae']}
            )
            
            train_labels = y_train
            val_labels = y_val
            
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=config["filepath"], monitor='val_loss', save_best_only=True)
        ]
        
        # Train
        history = model.fit(
            X_train, train_labels,
            validation_data=(X_val, val_labels),
            epochs=20, # Dynamic balance for stable notebook training demonstration
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        
        histories[name] = history.history
        
        # Load best weights
        model.load_weights(config["filepath"])
        
        # Evaluate on Test Set
        outputs = model.predict(X_test)
        
        # Handle multiple outputs
        if config["use_ae"]:
            y_pred_scaled = outputs[0]  # First element is the forecast output
        else:
            y_pred_scaled = outputs     # Sole element is the forecast output
            
        # Squeeze predictions
        y_pred_scaled = np.squeeze(y_pred_scaled)
        
        # Inverse transform target variable only (index 0) to evaluate in true physical units (ug/m^3)
        # Create a dummy array with identical feature columns to transform back PM2.5
        dummy_pred = np.zeros((len(y_pred_scaled), test_scaled.shape[1]))
        dummy_pred[:, 0] = y_pred_scaled
        y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_true = np.zeros((len(y_test), test_scaled.shape[1]))
        dummy_true[:, 0] = y_test
        y_true = scaler.inverse_transform(dummy_true)[:, 0]
        
        # Save predictions
        test_predictions[name] = y_pred
        
        # Compute academic metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n[RESULTS] {name}: MSE={mse:.4f} | MAE={mae:.4f} | R2={r2:.4f}\n")
        
        metrics_summary.append({
            "Scenario": name,
            "MSE (ug/m^3)^2": mse,
            "MAE (ug/m^3)": mae,
            "R2 Score": r2
        })
        
        # Extract attention weights for Model A
        if name == "Model A (Full Model - 5 Blocks)":
            # Access SimpleAttention layer inside Model A and extract attention distribution
            att_layer_model = Model(inputs=model.input, outputs=model.get_layer('attention_layer').output)
            _, attention_dist = att_layer_model.predict(X_test)
            model_a_attention_weights = attention_dist
            
    # Save Metrics Comparison Table to CSV
    df_results = pd.DataFrame(metrics_summary)
    df_results.to_csv("outputs/ablation_metrics_comparison.csv", index=False)
    
    print("="*60)
    print("FINAL ABLATION STUDIES METRICS COMPARISON TABLE")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    # ==========================================
    # SECTION 5: PUBLICATION-GRADE VISUALIZATIONS
    # ==========================================
    
    # Plot 1: Combined Loss Curves
    plt.figure(figsize=(10, 6))
    for i, (name, hist) in enumerate(histories.items()):
        val_loss_key = 'val_loss' if 'val_loss' in hist else 'val_forecast_output_loss'
        plt.plot(hist[val_loss_key], label=f"{name} (Val)", color=PALETTE[i], linewidth=2.5)
    plt.title("Ablation Study: Validation Loss Curves Across Scenarios", pad=15)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss (MSE)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("outputs/ablation_loss_curves.png")
    plt.close()
    
    # Plot 2: Prediction Comparison (Sub-segment of Test Set)
    plt.figure(figsize=(12, 6))
    # We display a 72-hour window (3 days) of the test set for clear visualization
    plt.plot(y_true[200:272], label="Ground Truth (Real)", color="#000000", linewidth=2.5, linestyle='--')
    for i, (name, y_pred) in enumerate(test_predictions.items()):
        plt.plot(y_pred[200:272], label=name.split(" (")[0], color=PALETTE[i], linewidth=2)
    plt.title("Real vs. Forecasted PM2.5 Concentration (72-Hour Test Interval)", pad=15)
    plt.xlabel("Hours")
    plt.ylabel("PM2.5 Level (ug/m^3)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("outputs/prediction_scatter_plot.png")
    plt.close()
    
    # Plot 3: Attention Weights Heatmap (Model A) using pure Matplotlib
    if model_a_attention_weights is not None:
        plt.figure(figsize=(10, 6))
        im = plt.imshow(model_a_attention_weights[:50, :], cmap="magma", aspect="auto")
        plt.colorbar(im, label='Attention Weight')
        plt.title("Attention Weights Across Time Steps (Model A - First 50 Samples)", pad=15)
        plt.xlabel("Conv-Pooled Time Steps (Compressed Sequence Length)")
        plt.ylabel("Test Sample Index")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("outputs/attention_weights_map.png")
        plt.close()
        print("[VISUALIZATION] Generated all publication-grade plots inside outputs/ directory.")

if __name__ == "__main__":
    run_ablation_studies()
