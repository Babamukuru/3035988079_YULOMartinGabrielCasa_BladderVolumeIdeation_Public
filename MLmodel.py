#!/usr/bin/env python3
"""
Bladder Volume Prediction — Training & Evaluation

Accepts positional arguments in fixed order (for main.sh pipeline):
    csv1 csv2 csv3 csv4 bv1 bv3 bv4 bv5 bv6 bv_pvr

Trains Random Forest and LSTM models with leave-one-session-out CV.
Prints evaluation metrics (MAE, RMSE, R², classification metrics, ROC).

Usage:
    python train_and_evaluate.py \
        session1.csv session2.csv session3.csv session4.csv \
        50 200 250 300 350 30
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'Trained_Models'

CORE_FEATURES = [
    'TOI_mean', 'BVI_mean', 'OEF_proxy_mean',
    'TOI_asymmetry', 'BVI_asymmetry',
    'TOI_gradient', 'BVI_gradient',
    'd_TOI_mean_dt_smooth', 'd_BVI_mean_dt_smooth',
    'd_TOI_mean_dt_pct', 'd_BVI_mean_dt_pct',
]
CHANNEL_FEATURES = [
    'left_outer_TOI', 'right_outer_TOI', 'left_inner_TOI', 'right_inner_TOI',
    'left_outer_BVI', 'right_outer_BVI', 'left_inner_BVI', 'right_inner_BVI',
    'left_outer_HbO_value', 'right_outer_HbO_value',
    'left_inner_HbO_value', 'right_inner_HbO_value',
    'left_outer_HbR_value', 'right_outer_HbR_value',
    'left_inner_HbR_value', 'right_inner_HbR_value',
]
TIME_FEATURES = ['elapsed_time_sec']
WAVELENGTH_FEATURES = [
    'wavelength_ratio_outer', 'wavelength_ratio_inner',
    'wavelength_ratio_outer_norm', 'wavelength_ratio_inner_norm',
]


# ============================================================
# PARSE ARGUMENTS (positional, fixed order)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate bladder volume prediction models.'
    )
    parser.add_argument('csv1', help='CSV for session 1')
    parser.add_argument('csv2', help='CSV for session 2')
    parser.add_argument('csv3', help='CSV for session 3')
    parser.add_argument('csv4', help='CSV for session 4')
    parser.add_argument('bv1',   help='Bladder volume 1 (mL)')
    parser.add_argument('bv3',   help='Bladder volume 3 (mL)')
    parser.add_argument('bv4',   help='Bladder volume 4 (mL)')
    parser.add_argument('bv5',   help='Bladder volume 5 (mL)')
    parser.add_argument('bv6',   help='Bladder volume 6 (mL)')
    parser.add_argument('bv_pvr', help='PVR bladder volume (mL)')
    parser.add_argument('--output-dir', default='Trained_Models',
                       help='Output directory (default: Trained_Models)')
    return parser.parse_args()


def parse_vol(val):
    """Parse volume argument; return None if NA/empty."""
    if val is None or str(val).strip().upper() in ('NA', '', 'NONE'):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def build_session_list(args):
    """Build list of (name, csv_path, pre_vol, post_vol) from args."""
    sessions = []
    bv1   = parse_vol(args.bv1)
    bv3   = parse_vol(args.bv3)
    bv4   = parse_vol(args.bv4)
    bv5   = parse_vol(args.bv5)
    bv6   = parse_vol(args.bv6)
    bv_pvr = parse_vol(args.bv_pvr)

    csv_files = [
        (args.csv1, 'Session_1'),
        (args.csv2, 'Session_2'),
        (args.csv3, 'Session_3'),
        (args.csv4, 'Session_4'),
    ]

    # Session 1: pre = PVR if >0 else bv1; post = bv3
    path, name = csv_files[0]
    if path and path.upper() not in ('NA', ''):
        pre = bv_pvr if (bv_pvr is not None and bv_pvr > 0) else bv1
        if pre is not None and bv3 is not None:
            sessions.append((name, path, pre, bv3))

    # Session 2: pre = bv3; post = bv4
    path, name = csv_files[1]
    if path and path.upper() not in ('NA', ''):
        if bv3 is not None and bv4 is not None:
            sessions.append((name, path, bv3, bv4))

    # Session 3: pre = bv4; post = bv5
    path, name = csv_files[2]
    if path and path.upper() not in ('NA', ''):
        if bv4 is not None and bv5 is not None:
            sessions.append((name, path, bv4, bv5))

    # Session 4: pre = bv5; post = bv6
    path, name = csv_files[3]
    if path and path.upper() not in ('NA', ''):
        if bv5 is not None and bv6 is not None:
            sessions.append((name, path, bv5, bv6))

    return sessions


# ============================================================
# DATA LOADING
# ============================================================

def load_sessions(session_list):
    names, dfs, vols = [], [], []
    for name, path, pre, post in session_list:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping {name}.")
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            print(f"Warning: {path} empty. Skipping {name}.")
            continue
        names.append(name)
        dfs.append(df)
        vols.append((pre, post))
        print(f"  {name}: {len(df)} rows, pre={pre}mL, post={post}mL")
    return names, dfs, vols


def get_features(df):
    all_candidates = CORE_FEATURES + CHANNEL_FEATURES + TIME_FEATURES + WAVELENGTH_FEATURES
    return [c for c in all_candidates if c in df.columns and df[c].notna().sum() > 0]


def add_labels(df, pre, post):
    df = df.copy()
    n = len(df)
    df['label_volume_mL'] = np.linspace(pre, post, n)
    return df


# ============================================================
# MODELS
# ============================================================

class RFModel:
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.features = None

    def fit(self, X, y, features):
        self.features = features
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def save(self, path, scaler_path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)


class LSTMModel:
    def __init__(self, window_size=60, epochs=30):
        self.window_size = window_size
        self.epochs = epochs
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.features = None

    def _build(self, input_dim):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.window_size, input_dim)),
            Dropout(0.2), LSTM(32), Dropout(0.2),
            Dense(16, activation='relu'), Dense(1)
        ])
        model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
        return model

    def _sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.window_size):
            Xs.append(X[i:i+self.window_size])
            ys.append(y[i+self.window_size])
        if not Xs:
            return np.zeros((0, self.window_size, X.shape[1])), np.zeros(0)
        return np.array(Xs), np.array(ys)

    def fit(self, X, y, features):
        self.features = features
        X_s = self.scaler_X.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y.reshape(-1,1)).flatten()
        X_seq, y_seq = self._sequences(X_s, y_s)
        if len(X_seq) < 10:
            self.model = None
            return self
        self.model = self._build(X.shape[1])
        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=32,
                       validation_split=0.1, verbose=0)
        return self

    def predict(self, X):
        X_s = self.scaler_X.transform(X)
        X_seq, _ = self._sequences(X_s, np.zeros(len(X_s)))
        if len(X_seq) == 0:
            return np.full(len(X), np.nan)
        y_s = self.model.predict(X_seq, verbose=0).flatten()
        y = self.scaler_y.inverse_transform(y_s.reshape(-1,1)).flatten()
        padded = np.full(len(X), np.nan)
        padded[self.window_size:] = y
        return padded

    def save(self, path, sx_path, sy_path):
        if self.model is not None:
            self.model.save(path)
        with open(sx_path, 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(sy_path, 'wb') as f:
            pickle.dump(self.scaler_y, f)


# ============================================================
# LEAVE-ONE-SESSION-OUT
# ============================================================

def build_train_data(indices, session_dfs, session_volumes):
    X_list, y_list = [], []
    for idx in indices:
        df = add_labels(session_dfs[idx].copy(), *session_volumes[idx])
        feats = get_features(df)
        clean = df[feats + ['label_volume_mL']].dropna()
        if len(clean) == 0:
            continue
        X_list.append(clean[feats].values)
        y_list.append(clean['label_volume_mL'].values)
    if not X_list:
        return None, None, None
    return np.vstack(X_list), np.hstack(y_list), feats


def run_cv(model_class, model_name, names, dfs, vols, **kw):
    n = len(dfs)
    results = []
    for test_i in range(n):
        train_i = [j for j in range(n) if j != test_i]
        X_tr, y_tr, feats = build_train_data(train_i, dfs, vols)
        if X_tr is None:
            continue
        model = model_class(**kw).fit(X_tr, y_tr, feats)
        test_df = dfs[test_i].copy()
        avail = [f for f in feats if f in test_df.columns]
        if not avail:
            continue
        clean = test_df[avail].dropna()
        test_df.loc[clean.index, 'pred_vol'] = model.predict(clean[avail].values)
        test_df['session'] = names[test_i]
        test_df['fold'] = test_i
        test_df['model'] = model_name
        test_df['pre_vol'] = vols[test_i][0]
        test_df['post_vol'] = vols[test_i][1]
        results.append(test_df)
    return results


# ============================================================
# EVALUATION
# ============================================================

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

def classify_volume(vol, thresholds=(100, 300)):
    """Bin continuous volume into low/moderate/high."""
    if pd.isna(vol):
        return np.nan
    low, high = thresholds
    if vol < low:
        return 'low'
    elif vol < high:
        return 'moderate'
    else:
        return 'high'


def evaluate_model(predictions, model_name, thresholds=(100, 300)):
    """Print regression and classification metrics. Plot confusion matrix and ROC."""
    df = pd.concat(predictions, ignore_index=True)

    # Use first and last prediction of each session for endpoint evaluation
    endpoints = df.dropna(subset=['pred_vol']).groupby('session').agg(
        pred_start=('pred_vol', 'first'),
        pred_end=('pred_vol', 'last'),
        actual_pre=('pre_vol', 'first'),
        actual_post=('post_vol', 'first'),
    )

    # --- Regression metrics ---
    print(f"\n{'='*60}")
    print(f"REGRESSION METRICS — {model_name}")
    print(f"{'='*60}")

    # Endpoint errors
    valid = endpoints.dropna()
    if len(valid) >= 2:
        mae_start = mean_absolute_error(valid['actual_pre'], valid['pred_start'])
        mae_end   = mean_absolute_error(valid['actual_post'], valid['pred_end'])
        mae_all = (mae_start + mae_end) / 2
        rmse_start = np.sqrt(mean_squared_error(valid['actual_pre'], valid['pred_start']))
        rmse_end   = np.sqrt(mean_squared_error(valid['actual_post'], valid['pred_end']))
        r2_start = r2_score(valid['actual_pre'], valid['pred_start'])
        r2_end   = r2_score(valid['actual_post'], valid['pred_end'])

        print(f"  Start volume — MAE: {mae_start:.1f} mL, RMSE: {rmse_start:.1f} mL, R²: {r2_start:.3f}")
        print(f"  End volume   — MAE: {mae_end:.1f} mL, RMSE: {rmse_end:.1f} mL, R²: {r2_end:.3f}")
        print(f"  Combined     — MAE: {mae_all:.1f} mL")

    # --- Classification metrics ---
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION METRICS — {model_name}")
    print(f"  Thresholds: <{thresholds[0]} mL = low, "
          f"{thresholds[0]}–{thresholds[1]} mL = moderate, >{thresholds[1]} mL = high")
    print(f"{'='*60}")

    # Create classification labels from endpoint predictions
    valid['actual_class'] = valid['actual_post'].apply(lambda x: classify_volume(x, thresholds))
    valid['pred_class']   = valid['pred_end'].apply(lambda x: classify_volume(x, thresholds))
    class_data = valid.dropna(subset=['actual_class', 'pred_class'])

    if len(class_data) >= 3:
        y_true = class_data['actual_class']
        y_pred = class_data['pred_class']
        labels = ['low', 'moderate', 'high']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f} (weighted)")
        print(f"  Recall:    {rec:.3f} (weighted)")
        print(f"  F1-score:  {f1:.3f} (weighted)")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"              low  mod  high")
        for i, label in enumerate(labels):
            print(f"    {label:8s}  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}")

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Blues')
        ax.set_title(f'{model_name} — Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_confusion.png'), dpi=120)
        plt.show()

        # --- ROC curve (binary: low vs not-low) ---
        # Binarize: 1 = high, 0 = not high
        y_bin = (y_true == 'high').astype(int)
        # Use predicted_end as score for ROC
        scores = valid.loc[class_data.index, 'pred_end']
        if len(np.unique(y_bin)) >= 2:
            auc = roc_auc_score(y_bin, scores)
            fpr, tpr, _ = roc_curve(y_bin, scores)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} — ROC (high vs not-high)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_roc.png'), dpi=120)
            plt.show()

            print(f"\n  ROC AUC (high vs not-high): {auc:.3f}")
    else:
        print("  Not enough classification data for metrics.")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session_list = build_session_list(args)
    if len(session_list) < 2:
        print("ERROR: Need at least 2 valid sessions for leave-one-out.")
        return

    print(f"\nParsed {len(session_list)} sessions:")
    names, dfs, vols = load_sessions(session_list)
    print(f"Loaded {len(names)} valid sessions.")

    feats = get_features(dfs[0])
    print(f"\n{len(feats)} features available.")

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("TRAINING: Random Forest")
    print("=" * 60)
    rf_preds = run_cv(RFModel, "RF", names, dfs, vols)
    evaluate_model(rf_preds, "RF")

    # Save RF model (trained on all sessions) for deployment
    X_all, y_all, _ = build_train_data(list(range(len(dfs))), dfs, vols)
    final_rf = RFModel().fit(X_all, y_all, feats)
    final_rf.save(
        os.path.join(OUTPUT_DIR, 'rf_model.pkl'),
        os.path.join(OUTPUT_DIR, 'rf_scaler.pkl')
    )
    pd.DataFrame({'feature': feats}).to_csv(
        os.path.join(OUTPUT_DIR, 'feature_list.csv'), index=False
    )
    print(f"Saved RF model to {OUTPUT_DIR}/")

    # --- LSTM (optional) ---
    try:
        import tensorflow
        print("\n" + "=" * 60)
        print("TRAINING: LSTM")
        print("=" * 60)
        lstm_preds = run_cv(LSTMModel, "LSTM", names, dfs, vols, window_size=60, epochs=30)
        evaluate_model(lstm_preds, "LSTM")

        # Save LSTM model
        final_lstm = LSTMModel(window_size=60, epochs=30).fit(X_all, y_all, feats)
        final_lstm.save(
            os.path.join(OUTPUT_DIR, 'lstm_model.keras'),
            os.path.join(OUTPUT_DIR, 'lstm_scaler_X.pkl'),
            os.path.join(OUTPUT_DIR, 'lstm_scaler_y.pkl')
        )
        print(f"Saved LSTM model to {OUTPUT_DIR}/")
    except ImportError:
        print("\nTensorFlow not installed. Skipping LSTM.")
        print("Install: pip install tensorflow")

    print("\nDone.")


if __name__ == "__main__":
    main()