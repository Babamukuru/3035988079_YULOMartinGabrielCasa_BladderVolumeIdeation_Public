#!/usr/bin/env python3
"""
Bladder Volume Prediction — Train on ALL patients together

Accepts multiple patients' data via a simple CSV config file.
Each row = one patient, same format as your main.sh loop.

Usage:
    python MLmodel_all_patients.py patients_config.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

OUTPUT_DIR = 'FinalModel'


# ============================================================
# PARSE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train one RF model on ALL patients combined.'
    )
    parser.add_argument('config_csv', 
                       help='CSV with columns: patient_id, csv1, csv2, csv3, csv4, bv1, bv3, bv4, bv5, bv6, bv_pvr')
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


# ============================================================
# DATA LOADING
# ============================================================

def build_all_sessions(config_df):
    """
    Read config CSV and build list of all sessions across all patients.
    Each session gets a unique group ID for cross-validation.
    
    Returns:
        all_sessions : list of (patient_id, session_name, csv_path, pre_vol, post_vol, group_id)
    """
    all_sessions = []
    group_id = 0
    
    for _, row in config_df.iterrows():
        pid = row['patient_id']
        
        bv1   = parse_vol(row.get('bv1'))
        bv3   = parse_vol(row.get('bv3'))
        bv4   = parse_vol(row.get('bv4'))
        bv5   = parse_vol(row.get('bv5'))
        bv6   = parse_vol(row.get('bv6'))
        bv_pvr = parse_vol(row.get('bv_pvr'))
        
        csv_files = [
            (row.get('csv1'), f'P{pid}_S1'),
            (row.get('csv2'), f'P{pid}_S2'),
            (row.get('csv3'), f'P{pid}_S3'),
            (row.get('csv4'), f'P{pid}_S4'),
        ]
        
        # Session 1: pre = PVR if >0 else bv1; post = bv3
        path, name = csv_files[0]
        if pd.notna(path) and str(path).strip().upper() not in ('NA', ''):
            pre = bv_pvr if (bv_pvr is not None and bv_pvr > 0) else bv1
            if pre is not None and bv3 is not None:
                all_sessions.append((pid, name, path, pre, bv3, group_id))
                group_id += 1
        
        # Session 2: pre = bv3; post = bv4
        path, name = csv_files[1]
        if pd.notna(path) and str(path).strip().upper() not in ('NA', ''):
            if bv3 is not None and bv4 is not None:
                all_sessions.append((pid, name, path, bv3, bv4, group_id))
                group_id += 1
        
        # Session 3: pre = bv4; post = bv5
        path, name = csv_files[2]
        if pd.notna(path) and str(path).strip().upper() not in ('NA', ''):
            if bv4 is not None and bv5 is not None:
                all_sessions.append((pid, name, path, bv4, bv5, group_id))
                group_id += 1
        
        # Session 4: pre = bv5; post = bv6
        path, name = csv_files[3]
        if pd.notna(path) and str(path).strip().upper() not in ('NA', ''):
            if bv5 is not None and bv6 is not None:
                all_sessions.append((pid, name, path, bv5, bv6, group_id))
                group_id += 1
    
    return all_sessions


def load_and_prepare_all_data(all_sessions):
    print(f"\nLoading {len(all_sessions)} sessions...")
    
    # First pass: find common features across ALL sessions
    all_feature_sets = []
    valid_sessions = []  # Track sessions that pass all checks
    
    for pid, name, path, pre, post, gid in all_sessions:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        
        # Skip if volumes are NaN
        if pd.isna(pre) or pd.isna(post):
            print(f"  SKIPPING {name} (PID {pid}): NaN volume (pre={pre}, post={post})")
            continue
        
        df = pd.read_csv(path)
        
        exclude_cols = ['window_id', 'prediction_time_sec', 'timestamp_dt',
                       'elapsed_time_sec', 'elapsed_time_min',
                       'Bladder_Filling_Index', 'Bladder_Filling_Index_Smooth',
                       'Estimated_Bladder_Volume_mL',
                       'TOI_norm', 'BVI_norm', 'OEF_norm', 
                       'Asymmetry_norm', 'dTOI_norm']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols
                       and df[c].dtype in ['float64', 'int64']
                       and df[c].notna().sum() > 0]
        all_feature_sets.append(set(feature_cols))
        valid_sessions.append((pid, name, path, pre, post, gid))

            # HANDLE EMPTY CASE
    if not all_feature_sets:
        print("ERROR: No valid sessions with features found!")
        return None, None, None, None, None
    
    # Only keep features present in ALL sessions
    common_features = list(set.intersection(*all_feature_sets))
    print(f"  Common features across all sessions: {len(common_features)}")
    
    # Second pass: build X, y, groups
    X_list, y_list, groups_list = [], [], []
    session_info = []
    
    for pid, name, path, pre, post, gid in valid_sessions:
        df = pd.read_csv(path)
        
        # Use only common features
        available = [f for f in common_features if f in df.columns]
        
        # --- FORWARD FILL NaN in features ---
        df[available] = df[available].fillna(method='ffill')
        df[available] = df[available].fillna(method='bfill')
        
        # Drop remaining NaN
        before = len(df)
        df = df.dropna(subset=available)
        after = len(df)
        if before > after:
            print(f"    {name}: Dropped {before - after} rows still NaN after forward fill")
        
        n = len(df)
        labels = np.linspace(pre, post, n)
        
        X = df[available].values
        X_list.append(X)
        y_list.append(labels)
        groups_list.extend([gid] * n)
        
        session_info.append({
            'pid': pid,
            'name': name,
            'n_rows': n,
            'pre_vol': pre,
            'post_vol': post,
            'group_id': gid
        })
        
        print(f"  {name} (PID {pid}): {n} rows, pre={pre}mL, post={post}mL")
    
    if not X_list:
        print("ERROR: No valid sessions remaining after filtering")
        return None, None, None, None, None
    
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    groups = np.array(groups_list)
    
    print(f"\n  Total: {X_all.shape[0]} rows, {len(common_features)} features")
    print(f"  Volume range: {y_all.min():.1f} - {y_all.max():.1f} mL")
    print(f"  Unique sessions: {len(np.unique(groups))}")
    
    return X_all, y_all, groups, common_features, session_info


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(n_features, max_features=40):
    """Build pipeline with imputer, scaler, feature selection, RF."""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
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
# LEAVE-ONE-SESSION-OUT CV
# ============================================================

def run_loso_cv(pipeline, X, y, groups, session_info):
    """Leave-one-session-out cross-validation."""
    logo = LeaveOneGroupOut()
    predictions = np.full_like(y, np.nan)
    fold_metrics = []
    
    print("\n" + "=" * 60)
    print("LEAVE-ONE-SESSION-OUT CV (all patients)")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_gid = groups[test_idx[0]]
        # Find session info for this group
        sinfo = next(s for s in session_info if s['group_id'] == test_gid)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        predictions[test_idx] = preds
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        fold_metrics.append({
            'fold': fold,
            'session': sinfo['name'],
            'pid': sinfo['pid'],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'mae': mae,
            'r2': r2
        })
        print(f"  Fold {fold}: {sinfo['name']} (PID {sinfo['pid']}) | "
              f"Train={len(X_train)}, Test={len(X_test)} | "
              f"MAE={mae:.1f}mL, R²={r2:.3f}")
    
    # Save fold metrics
    pd.DataFrame(fold_metrics).to_csv(
        os.path.join(OUTPUT_DIR, 'cv_fold_metrics.csv'), index=False
    )
    
    return predictions


# ============================================================
# EVALUATION
# ============================================================

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

def classify_volume(vol, thresholds=(50, 200)):
    if pd.isna(vol):
        return np.nan
    low, high = thresholds
    if vol < low:
        return 'low'
    elif vol < high:
        return 'moderate'
    else:
        return 'high'


def evaluate_and_save(y_true, y_pred, session_info):
    """Evaluate and save all metrics, plots, and CSVs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_name = "RF_AllPatients"
    metrics_file = os.path.join(OUTPUT_DIR, f'{model_name}_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"COMBINED MODEL — ALL PATIENTS\n")
        f.write(f"Sessions: {len(session_info)}\n")
        f.write(f"{'='*60}\n\n")
        
        # --- Regression ---
        valid_mask = ~np.isnan(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        r2 = r2_score(y_true_valid, y_pred_valid)
        
        reg_text = (f"REGRESSION METRICS\n"
                   f"  MAE: {mae:.1f} mL\n"
                   f"  RMSE: {rmse:.1f} mL\n"
                   f"  R²: {r2:.3f}\n\n")
        print(reg_text)
        f.write(reg_text)
        
        # Per-session endpoint errors
        f.write("PER-SESSION ENDPOINT ERRORS\n")
        start_idx = 0
        endpoint_data = []
        for info in session_info:
            end_idx = start_idx + info['n_rows']
            if end_idx <= len(y_true):
                actual_pre = y_true[start_idx]
                actual_post = y_true[end_idx - 1]
                pred_pre = y_pred[start_idx]
                pred_post = y_pred[end_idx - 1]
                if not np.isnan(pred_pre) and not np.isnan(pred_post):
                    err_start = abs(actual_pre - pred_pre)
                    err_end = abs(actual_post - pred_post)
                    line = (f"  {info['name']} (PID {info['pid']}): "
                           f"Start err={err_start:.1f}mL, End err={err_end:.1f}mL")
                    f.write(line + "\n")
                    endpoint_data.append({
                        'pid': info['pid'],
                        'session': info['name'],
                        'actual_pre': actual_pre,
                        'actual_post': actual_post,
                        'pred_pre': pred_pre,
                        'pred_post': pred_post,
                        'err_start': err_start,
                        'err_end': err_end
                    })
            start_idx = end_idx
        
        pd.DataFrame(endpoint_data).to_csv(
            os.path.join(OUTPUT_DIR, f'{model_name}_endpoints.csv'), index=False
        )
        
        # --- Classification on endpoints ---
        f.write("\nCLASSIFICATION (endpoints only)\n")
        f.write("Thresholds: <50=low, 50-200=mod, >200=high\n\n")
        
        actual_classes, pred_classes, scores_list, binary_list = [], [], [], []
        start_idx = 0
        for info in session_info:
            end_idx = start_idx + info['n_rows']
            if end_idx <= len(y_true):
                ap = y_true[end_idx - 1]
                pp = y_pred[end_idx - 1]
                if not np.isnan(pp):
                    actual_classes.append(classify_volume(ap))
                    pred_classes.append(classify_volume(pp))
                    scores_list.append(pp)
                    binary_list.append(1 if ap > 300 else 0)
            start_idx = end_idx
        
        if len(actual_classes) >= 3:
            y_true_cls = np.array(actual_classes)
            y_pred_cls = np.array(pred_classes)
            labels = ['low', 'moderate', 'high']
            
            acc = accuracy_score(y_true_cls, y_pred_cls)
            prec = precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
            rec  = recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
            f1s  = f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
            
            cls_text = (f"  Accuracy:  {acc:.3f}\n"
                       f"  Precision: {prec:.3f}\n"
                       f"  Recall:    {rec:.3f}\n"
                       f"  F1-score:  {f1s:.3f}\n")
            print(cls_text)
            f.write(cls_text)
            
            cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
            f.write("\n  Confusion Matrix:\n")
            f.write(f"              low  mod  high\n")
            for i, label in enumerate(labels):
                f.write(f"    {label:8s}  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}\n")
            
            # Confusion matrix plot
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Blues')
            ax.set_title(f'{model_name} — Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_confusion.png'), dpi=120)
            plt.close()
            
            # ROC curve
            if len(np.unique(binary_list)) >= 2:
                auc = roc_auc_score(binary_list, scores_list)
                fpr, tpr, _ = roc_curve(binary_list, scores_list)
                
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{model_name} — ROC (High vs Not-High)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_roc.png'), dpi=120)
                plt.close()
                f.write(f"\n  ROC AUC (high vs not-high): {auc:.3f}\n")
        pid='all'
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

        # Summary
        f.write(f"\n{'='*60}\n")
        f.write(f"SUMMARY\n")
        f.write(f"  MAE: {mae:.1f} mL\n")
        f.write(f"  R²:  {r2:.3f}\n")
        f.write(f"  Total samples: {len(y_true_valid)}\n")
        f.write(f"  Sessions: {len(session_info)}\n")
    
    print(f"\n✓ Metrics saved to: {metrics_file}")
    return mae, r2



# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read config
    config_df = pd.read_csv(args.config_csv)
    print(f"Loaded {len(config_df)} patients from config")
    
    # Build all sessions
    all_sessions = build_all_sessions(config_df)
    print(f"Built {len(all_sessions)} sessions")
    
    if len(all_sessions) < 3:
        print("ERROR: Need at least 3 sessions for LOSO CV")
        return
    
    # Load data
    X, y, groups, feature_names, session_info = load_and_prepare_all_data(all_sessions)
    
    # Build pipeline
    pipeline = build_pipeline(len(feature_names), max_features=40)
    print(f"\nPipeline:\n{pipeline}")
    
    # Cross-validate
    cv_preds = run_loso_cv(pipeline, X, y, groups, session_info)
    
    # Evaluate
    evaluate_and_save(y, cv_preds, session_info)
    
    # Train final model on ALL data
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL (all data)")
    print("=" * 60)
    pipeline.fit(X, y)
    train_mae = mean_absolute_error(y, pipeline.predict(X))
    print(f"  Training MAE: {train_mae:.1f} mL")
    
    # Save selected features
    try:
        selector = pipeline.named_steps['feature_select']
        selected = [f for f, m in zip(feature_names, selector.get_support()) if m]
        print(f"  Selected {len(selected)}/{len(feature_names)} features")
        pd.DataFrame({'feature': selected}).to_csv(
            os.path.join(OUTPUT_DIR, 'selected_features.csv'), index=False
        )
    except:
        pass
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'rf_model_all_patients.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()