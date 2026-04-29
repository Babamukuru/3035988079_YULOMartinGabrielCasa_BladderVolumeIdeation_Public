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

OUTPUT_DIR = 'Trained_Models_Scikit'


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
    X_list, y_list, groups_list = [], [], []
    session_info = []
    feature_names = None
    
    for session_idx, (name, path, pre_vol, post_vol) in enumerate(session_list):
        if not os.path.exists(path):
            print(f"  Warning: {path} not found. Skipping {name}.")
            continue
        
        # Skip if volumes are NaN
        if pd.isna(pre_vol) or pd.isna(post_vol):
            print(f"  Skipping {name}: NaN volume")
            continue
        
        df = pd.read_csv(path)
        print(f"  {name}: {len(df)} rows, pre={pre_vol}mL, post={post_vol}mL")
        
        # Identify feature columns
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
            feature_names = [f for f in feature_names if f in feature_cols]
        
        # --- FORWARD FILL NaN in features ---
        df[feature_names] = df[feature_names].fillna(method='ffill')
        df[feature_names] = df[feature_names].fillna(method='bfill')  # handle NaN at start
        
        # Drop any rows still NaN (should be none after ffill+bfill)
        before = len(df)
        df = df.dropna(subset=feature_names)
        after = len(df)
        if before > after:
            print(f"    Dropped {before - after} rows still NaN after forward fill")
        
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
    
    if not X_list:
        return None, None, None, None, None
    
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
    
def plot_predicted_vs_actual(session_info, y_true, y_pred, output_dir, model_name, pid='all'):
    """
    Bar chart comparing predicted vs actual bladder volumes per session.
    Uses endpoint predictions (first and last of each session).
    """
    # Collect endpoint data
    sessions = []
    actual_pre = []
    actual_post = []
    pred_pre = []
    pred_post = []
    
    start_idx = 0
    for info in session_info:
        end_idx = start_idx + info['n_rows']
        if end_idx <= len(y_true):
            ap = y_true[start_idx]
            a_post = y_true[end_idx - 1]
            pp = y_pred[start_idx] if not np.isnan(y_pred[start_idx]) else np.nan
            p_post = y_pred[end_idx - 1] if not np.isnan(y_pred[end_idx - 1]) else np.nan
            
            if not np.isnan(pp) and not np.isnan(p_post):
                label = f"{info['name']}" if pid == 'all' else f"{info['name']}"
                if 'pid' in info:
                    label = f"P{info['pid']} {info['name']}"
                
                sessions.append(label)
                actual_pre.append(ap)
                actual_post.append(a_post)
                pred_pre.append(pp)
                pred_post.append(p_post)
        start_idx = end_idx
    
    if not sessions:
        return None
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(sessions)*0.8), 5))
    
    # LEFT: Pre (start) volumes
    ax = axes[0]
    x = np.arange(len(sessions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, actual_pre, width, label='Actual', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, pred_pre, width, label='Predicted', color='#F44336', alpha=0.8)
    
    ax.set_ylabel('Bladder Volume (mL)')
    ax.set_title('Start Volumes (Pre)')
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # RIGHT: Post (end) volumes
    ax = axes[1]
    bars1 = ax.bar(x - width/2, actual_post, width, label='Actual', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, pred_post, width, label='Predicted', color='#F44336', alpha=0.8)
    
    ax.set_ylabel('Bladder Volume (mL)')
    ax.set_title('End Volumes (Post)')
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Predicted vs Actual Bladder Volume — {model_name} (PID {pid})', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'pid{pid}_{model_name}_pred_vs_actual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def save_volume_table(session_info, y_true, y_pred, output_dir, model_name, pid='all'):
    """
    Save a CSV table of predicted vs actual volumes.
    """
    rows = []
    start_idx = 0
    
    for info in session_info:
        end_idx = start_idx + info['n_rows']
        if end_idx <= len(y_true):
            ap = y_true[start_idx]
            a_post = y_true[end_idx - 1]
            pp = y_pred[start_idx] if not np.isnan(y_pred[start_idx]) else np.nan
            p_post = y_pred[end_idx - 1] if not np.isnan(y_pred[end_idx - 1]) else np.nan
            
            if not np.isnan(pp) and not np.isnan(p_post):
                rows.append({
                    'Patient': info.get('pid', pid),
                    'Session': info['name'],
                    'Actual_Pre_mL': round(ap, 1),
                    'Pred_Pre_mL': round(pp, 1),
                    'Pre_Error_mL': round(abs(ap - pp), 1),
                    'Actual_Post_mL': round(a_post, 1),
                    'Pred_Post_mL': round(p_post, 1),
                    'Post_Error_mL': round(abs(a_post - p_post), 1),
                    'Volume_Change_Actual': round(a_post - ap, 1),
                    'Volume_Change_Pred': round(p_post - pp, 1)
                })
        start_idx = end_idx
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'pid{pid}_{model_name}_volume_table.csv')
    df.to_csv(csv_path, index=False)
    
    # Also print as formatted table
    print(f"\n  {'='*70}")
    print(f"  PREDICTED vs ACTUAL VOLUMES — {model_name} (PID {pid})")
    print(f"  {'='*70}")
    print(f"  {'Session':<20} {'Actual Pre':>10} {'Pred Pre':>10} {'Error':>8}  |  {'Actual Post':>11} {'Pred Post':>11} {'Error':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}  |  {'-'*11} {'-'*11} {'-'*8}")
    
    total_error = 0
    n_err = 0
    for _, row in df.iterrows():
        label = f"P{row['Patient']} {row['Session']}" if pid == 'all' else row['Session']
        print(f"  {label:<20} {row['Actual_Pre_mL']:>10.1f} {row['Pred_Pre_mL']:>10.1f} {row['Pre_Error_mL']:>8.1f}  |  {row['Actual_Post_mL']:>11.1f} {row['Pred_Post_mL']:>11.1f} {row['Post_Error_mL']:>8.1f}")
        total_error += row['Pre_Error_mL'] + row['Post_Error_mL']
        n_err += 2
    
    if n_err > 0:
        print(f"  {'='*70}")
        print(f"  Mean absolute error: {total_error/n_err:.1f} mL")
    
    return csv_path


def evaluate_results(y_true, y_pred, session_info, model_name, pid):
    """Print regression and classification metrics. Save ROC curve and metrics file."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_file = os.path.join(OUTPUT_DIR, f'pid{pid}_{model_name}_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"EVALUATION METRICS — {model_name}\n")
        f.write(f"Patient ID: {pid}\n")
        f.write(f"{'='*60}\n\n")
        
        # --- Regression metrics ---
        header = f"{'='*60}\nREGRESSION METRICS — {model_name}\n{'='*60}"
        print(header)
        f.write(header + "\n")
        
        valid_mask = ~np.isnan(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        r2 = r2_score(y_true_valid, y_pred_valid)
        
        reg_metrics = (f"  MAE: {mae:.1f} mL\n"
                      f"  RMSE: {rmse:.1f} mL\n"
                      f"  R²: {r2:.3f}\n")
        print(reg_metrics)
        f.write(reg_metrics + "\n")
        
        # --- Per-session endpoint errors ---
        endpoint_header = "  Per-session endpoint errors:"
        print(endpoint_header)
        f.write(endpoint_header + "\n")
        
        start_idx = 0
        endpoint_data = []
        for info in session_info:
            end_idx = start_idx + info['n_rows']
            if start_idx < len(y_true):
                actual_pre = y_true[start_idx]
                actual_post = y_true[min(end_idx-1, len(y_true)-1)]
                pred_pre = y_pred[start_idx] if not np.isnan(y_pred[start_idx]) else np.nan
                pred_post = y_pred[min(end_idx-1, len(y_true)-1)] if not np.isnan(y_pred[min(end_idx-1, len(y_true)-1)]) else np.nan
                err_start = abs(actual_pre-pred_pre)
                err_end = abs(actual_post-pred_post)
                
                line = (f"    {info['name']}: "
                       f"Start err={err_start:.1f}mL, End err={err_end:.1f}mL")
                print(line)
                f.write(line + "\n")
                
                endpoint_data.append({
                    'session': info['name'],
                    'actual_pre': actual_pre,
                    'actual_post': actual_post,
                    'pred_pre': pred_pre,
                    'pred_post': pred_post,
                    'err_start': err_start,
                    'err_end': err_end
                })
            start_idx = end_idx
        
        # Save endpoint CSV
        pd.DataFrame(endpoint_data).to_csv(
            os.path.join(OUTPUT_DIR, f'pid{pid}_{model_name}_endpoints.csv'),
            index=False
        )
        
        # --- Classification metrics ---
        cls_header = (f"\n{'='*60}\n"
                     f"CLASSIFICATION METRICS — {model_name}\n"
                     f"  Thresholds: <50 mL = low, 50–200 mL = moderate, >200 mL = high\n"
                     f"{'='*60}")
        print(cls_header)
        f.write("\n" + cls_header + "\n")
        
        # Get endpoint predictions per session
        actual_classes = []
        pred_classes = []
        pred_scores = []  # for ROC
        actual_binary = []  # for ROC
        start_idx = 0
        for info in session_info:
            end_idx = start_idx + info['n_rows']
            if end_idx <= len(y_true):
                actual_post = y_true[end_idx - 1]
                pred_post = y_pred[end_idx - 1]
                if not np.isnan(pred_post):
                    actual_classes.append(classify_volume(actual_post))
                    pred_classes.append(classify_volume(pred_post))
                    pred_scores.append(pred_post)
                    actual_binary.append(1 if actual_post > 300 else 0)  # high vs not-high
            start_idx = end_idx
        
        if len(actual_classes) >= 3:
            y_true_cls = np.array(actual_classes)
            y_pred_cls = np.array(pred_classes)
            labels = ['low', 'moderate', 'high']
            
            acc = accuracy_score(y_true_cls, y_pred_cls)
            prec = precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
            rec  = recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
            f1score = f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
            
            cls_metrics = (f"  Accuracy:  {acc:.3f}\n"
                          f"  Precision: {prec:.3f} (weighted)\n"
                          f"  Recall:    {rec:.3f} (weighted)\n"
                          f"  F1-score:  {f1score:.3f} (weighted)\n")
            print(cls_metrics)
            f.write(cls_metrics)
            
            # Confusion matrix
            cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
            cm_str = (f"\n  Confusion Matrix (rows=actual, cols=predicted):\n"
                     f"              low  mod  high\n")
            for i, label in enumerate(labels):
                cm_str += f"    {label:8s}  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}\n"
            print(cm_str)
            f.write(cm_str)
            
            # Plot and save confusion matrix
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Blues')
            ax.set_title(f'{model_name} — Confusion Matrix (PID {pid})')
            plt.tight_layout()
            cm_path = os.path.join(OUTPUT_DIR, f'pid{pid}_{model_name}_confusion.png')
            plt.savefig(cm_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Confusion matrix saved to: {cm_path}")
            f.write(f"\n  Confusion matrix saved to: {cm_path}\n")
            
            # --- ROC CURVE ---
            pred_scores = np.array(pred_scores)
            actual_binary = np.array(actual_binary)
            
            if len(np.unique(actual_binary)) >= 2:
                auc = roc_auc_score(actual_binary, pred_scores)
                fpr, tpr, thresholds = roc_curve(actual_binary, pred_scores)
                
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{model_name} — ROC: High vs Not-High (PID {pid})')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                roc_path = os.path.join(OUTPUT_DIR, f'pid{pid}_{model_name}_roc.png')
                plt.savefig(roc_path, dpi=120, bbox_inches='tight')
                plt.close()
                print(f"  ✓ ROC curve saved to: {roc_path}")
                
                roc_metrics = (f"\n  ROC AUC (high vs not-high): {auc:.3f}\n")
                print(roc_metrics)
                f.write(roc_metrics)
                f.write(f"  ROC curve saved to: {roc_path}\n")
            else:
                f.write("\n  ROC: Not enough class diversity for ROC curve\n")
        else:
            no_cls = "  Not enough classification data for metrics (need ≥3 endpoints)"
            print(no_cls)
            f.write(no_cls + "\n")

        vol_plot_path = plot_predicted_vs_actual(
    session_info, y_true, y_pred, OUTPUT_DIR, model_name, pid
)
        if vol_plot_path:
            f.write(f"\n  Volume comparison plot: {vol_plot_path}\n")

        # Volume table
        vol_table_path = save_volume_table(
            session_info, y_true, y_pred, OUTPUT_DIR, model_name, pid
        )
        if vol_table_path:
            f.write(f"  Volume table: {vol_table_path}\n")
        
        # Summary footer
        footer = (f"\n{'='*60}\n"
                 f"SUMMARY\n"
                 f"{'='*60}\n"
                 f"  Regression MAE: {mae:.1f} mL\n"
                 f"  Regression R²:  {r2:.3f}\n"
                 f"  Samples evaluated: {len(y_true_valid)}\n"
                 f"  Sessions: {len(session_info)}\n")
        print(footer)
        f.write(footer)
    
    print(f"\n✓ Metrics saved to: {metrics_file}")
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