#!/usr/bin/env python3
"""
Bladder Volume Prediction — scikit-learn Pipeline Version

Replaces the manual feature selection + model training in MLmodel.py
with a proper scikit-learn pipeline that prevents data leakage.

Accepts the same arguments as MLmodel.py:
    csv1 csv2 csv3 csv4 bv1 bv3 bv4 bv5 bv6 bv_pvr

Usage:
    python MLmodel_sklearn.py \
        session1_features.csv session2_features.csv session3_features.csv session4_features.csv \
        50 200 250 300 350 30
"""

import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

OUTPUT_DIR = 'Trained_Models'


# ============================================================
# PARSE ARGUMENTS (same as MLmodel.py)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train bladder volume models with scikit-learn pipeline.'
    )
    parser.add_argument('csv1', help='CSV for session 1 (from y_features.py)')
    parser.add_argument('csv2', help='CSV for session 2')
    parser.add_argument('csv3', help='CSV for session 3')
    parser.add_argument('csv4', help='CSV for session 4')
    parser.add_argument('bv1',   help='Bladder volume 1 (mL)')
    parser.add_argument('bv3',   help='Bladder volume 3 (mL)')
    parser.add_argument('bv4',   help='Bladder volume 4 (mL)')
    parser.add_argument('bv5',   help='Bladder volume 5 (mL)')
    parser.add_argument('bv6',   help='Bladder volume 6 (mL)')
    parser.add_argument('bv_pvr', help='PVR bladder volume (mL)')
    parser.add_argument('--pid', default='unknown',
                       help='Patient ID for output file naming')
    parser.add_argument('--output-dir', default='Trained_Models_Scikit',
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

    path, name = csv_files[0]
    if path and path.upper() not in ('NA', ''):
        pre = bv_pvr if (bv_pvr is not None and bv_pvr > 0) else bv1
        if pre is not None and bv3 is not None:
            sessions.append((name, path, pre, bv3))

    path, name = csv_files[1]
    if path and path.upper() not in ('NA', ''):
        if bv3 is not None and bv4 is not None:
            sessions.append((name, path, bv3, bv4))

    path, name = csv_files[2]
    if path and path.upper() not in ('NA', ''):
        if bv4 is not None and bv5 is not None:
            sessions.append((name, path, bv4, bv5))

    path, name = csv_files[3]
    if path and path.upper() not in ('NA', ''):
        if bv5 is not None and bv6 is not None:
            sessions.append((name, path, bv5, bv6))

    return sessions


# ============================================================
# DATA LOADING
# ============================================================

def load_and_prepare_data(session_list):
    """
    Load feature CSVs and prepare X (features) and y (labels).
    Uses linear ramp labeling between pre and post volumes.
    
    Returns:
        X_all : np.array - all feature data stacked
        y_all : np.array - all labels stacked
        groups : np.array - session identifier for each row (for LOSO CV)
        feature_names : list - column names of features used
        session_info : list of dicts - metadata per session
    """
    X_list, y_list, groups_list = [], [], []
    session_info = []
    feature_names = None
    
    for session_idx, (name, path, pre_vol, post_vol) in enumerate(session_list):
        if not os.path.exists(path):
            print(f"  Warning: {path} not found. Skipping {name}.")
            continue
        
        df = pd.read_csv(path)
        print(f"  {name}: {len(df)} rows, pre={pre_vol}mL, post={post_vol}mL")
        
        # Identify feature columns (exclude metadata columns)
        exclude_cols = ['window_id', 'prediction_time_sec', 'timestamp_dt',
                       'elapsed_time_sec', 'elapsed_time_min',
                       'Bladder_Filling_Index', 'Bladder_Filling_Index_Smooth',
                       'Estimated_Bladder_Volume_mL', 
                       'TOI_norm', 'BVI_norm', 'OEF_norm', 'Asymmetry_norm', 'dTOI_norm']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols 
                       and df[c].dtype in ['float64', 'int64']
                       and df[c].notna().sum() > 0]
        
        if feature_names is None:
            feature_names = feature_cols
        else:
            # Use intersection of features across sessions
            feature_names = [f for f in feature_names if f in feature_cols]
        
        # Create linear ramp labels
        n = len(df)
        labels = np.linspace(pre_vol, post_vol, n)
        
        # Extract feature matrix
        X = df[feature_names].values
        X_list.append(X)
        y_list.append(labels)
        groups_list.extend([session_idx] * n)
        
        session_info.append({
            'name': name,
            'n_rows': n,
            'pre_vol': pre_vol,
            'post_vol': post_vol
        })
    
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    groups = np.array(groups_list)
    
    print(f"\n  Total: {X_all.shape[0]} rows, {len(feature_names)} features")
    print(f"  Volume range: {y_all.min():.1f} - {y_all.max():.1f} mL")
    
    return X_all, y_all, groups, feature_names, session_info


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(n_features, max_features=40):
    """
    Build a scikit-learn pipeline with:
    1. StandardScaler - normalize features
    2. Feature selection - keep top N most important features
    3. RandomForestRegressor - the actual model
    
    The pipeline ensures feature selection happens inside CV folds,
    preventing data leakage.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_select', SelectFromModel(
            RandomForestRegressor(n_estimators=100, max_depth=10, 
                                  random_state=42, n_jobs=-1),
            max_features=min(max_features, n_features)
        )),
        ('model', RandomForestRegressor(
            n_estimators=200, max_depth=15, 
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ))
    ])
    
    return pipeline


# ============================================================
# LEAVE-ONE-SESSION-OUT CROSS-VALIDATION
# ============================================================

def run_loso_cv(pipeline, X, y, groups, session_info):
    """
    Leave-one-session-out cross-validation.
    Each session is held out once, model trains on remaining sessions,
    predicts on held-out session.
    """
    logo = LeaveOneGroupOut()
    predictions = np.full_like(y, np.nan)
    
    print("\n" + "=" * 60)
    print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_session = groups[test_idx[0]]
        session_name = session_info[test_session]['name']
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit pipeline on training sessions
        pipeline.fit(X_train, y_train)
        
        # Predict on held-out session
        preds = pipeline.predict(X_test)
        predictions[test_idx] = preds
        
        # Quick metrics for this fold
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"  {session_name}: MAE={mae:.1f}mL, R²={r2:.3f}")
    
    return predictions


# ============================================================
# EVALUATION (same as MLmodel.py)
# ============================================================

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


def evaluate_results(y_true, y_pred, session_info, model_name, pid):
    """Print regression and classification metrics."""
    
    # --- Regression metrics ---
    print(f"\n{'='*60}")
    print(f"REGRESSION METRICS — {model_name}")
    print(f"{'='*60}")
    
    valid_mask = ~np.isnan(y_pred)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    r2 = r2_score(y_true_valid, y_pred_valid)
    
    print(f"  MAE: {mae:.1f} mL")
    print(f"  RMSE: {rmse:.1f} mL")
    print(f"  R²: {r2:.3f}")
    
    # --- Endpoint metrics per session ---
    print(f"\n  Per-session endpoint errors:")
    start_idx = 0
    for info in session_info:
        end_idx = start_idx + info['n_rows']
        if start_idx < len(y_true):
            actual_pre = y_true[start_idx]
            actual_post = y_true[min(end_idx-1, len(y_true)-1)]
            pred_pre = y_pred[start_idx] if not np.isnan(y_pred[start_idx]) else np.nan
            pred_post = y_pred[min(end_idx-1, len(y_true)-1)] if not np.isnan(y_pred[min(end_idx-1, len(y_true)-1)]) else np.nan
            print(f"    {info['name']}: Start err={abs(actual_pre-pred_pre):.1f}mL, "
                  f"End err={abs(actual_post-pred_post):.1f}mL")
        start_idx = end_idx
    
    # --- Classification metrics (on endpoints only) ---
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION METRICS — {model_name}")
    print(f"  Thresholds: <100 mL = low, 100–300 mL = moderate, >300 mL = high")
    print(f"{'='*60}")
    
    # Get endpoint predictions per session
    actual_classes = []
    pred_classes = []
    start_idx = 0
    for info in session_info:
        end_idx = start_idx + info['n_rows']
        if end_idx <= len(y_true):
            actual_post = y_true[end_idx - 1]
            pred_post = y_pred[end_idx - 1]
            if not np.isnan(pred_post):
                actual_classes.append(classify_volume(actual_post))
                pred_classes.append(classify_volume(pred_post))
        start_idx = end_idx
    
    if len(actual_classes) >= 3:
        y_true_cls = np.array(actual_classes)
        y_pred_cls = np.array(pred_classes)
        labels = ['low', 'moderate', 'high']
        
        acc = accuracy_score(y_true_cls, y_pred_cls)
        prec = precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
        rec  = recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
        f1   = f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
        
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f} (weighted)")
        print(f"  Recall:    {rec:.3f} (weighted)")
        print(f"  F1-score:  {f1:.3f} (weighted)")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"              low  mod  high")
        for i, label in enumerate(labels):
            print(f"    {label:8s}  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}")
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Blues')
        ax.set_title(f'{model_name} — Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'pid{pid}_{model_name}_confusion.png'), 
                   dpi=120, bbox_inches='tight')
        plt.close()
    
    return mae, r2


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pid = args.pid
    
    session_list = build_session_list(args)
    if len(session_list) < 2:
        print("ERROR: Need at least 2 valid sessions.")
        return
    
    print(f"\nParsed {len(session_list)} sessions:")
    X, y, groups, feature_names, session_info = load_and_prepare_data(session_list)
    
    # ============================================
    # Build and evaluate pipeline with LOSO CV
    # ============================================
    pipeline = build_pipeline(len(feature_names), max_features=40)
    
    print("\n" + "=" * 60)
    print("Pipeline structure:")
    print("=" * 60)
    print(pipeline)
    
    # Cross-validated predictions
    cv_predictions = run_loso_cv(pipeline, X, y, groups, session_info)
    
    # Evaluate
    evaluate_results(y, cv_predictions, session_info, "RF_Pipeline", pid)
    
    # ============================================
    # Train final model on ALL data for deployment
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING: Final model on all sessions")
    print("=" * 60)
    
    pipeline.fit(X, y)
    
    train_preds = pipeline.predict(X)
    train_mae = mean_absolute_error(y, train_preds)
    print(f"  Training MAE: {train_mae:.1f} mL")
    
    # Get selected feature names
    try:
        selector = pipeline.named_steps['feature_select']
        selected_mask = selector.get_support()
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        print(f"  Selected {len(selected_features)}/{len(feature_names)} features")
        print(f"  Top 5: {selected_features[:5]}")
        
        # Save feature list
        pd.DataFrame({'feature': selected_features}).to_csv(
            os.path.join(OUTPUT_DIR, f'pid{pid}_selected_features.csv'), index=False
        )
    except:
        pass
    
    # Save pipeline
    pipeline_path = os.path.join(OUTPUT_DIR, f'pid{pid}_pipeline.pkl')
    joblib.dump(pipeline, pipeline_path)
    print(f"\n✓ Pipeline saved to: {pipeline_path}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()