#!/usr/bin/env python3
"""
Simple Feature Visualization - Works with ANY feature CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_visualizations(input_file, output_dir):
    """Generate plots from a single feature CSV file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Visualizations for: {Path(input_file).name}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Get numeric columns (exclude metadata)
    exclude = ['window_id', 'timestamp', 'elapsed_time_sec', 'elapsed_time_min', 
               'prediction_time_sec', 'index']
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    
    if not numeric_cols:
        print("ERROR: No numeric feature columns found!")
        return False
    
    print(f"  Found {len(numeric_cols)} numeric feature columns")
    
    # Take top features by variance for plotting
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(10).index.tolist()
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Time series of top 3 features
    ax = axes[0, 0]
    for i, feat in enumerate(top_features[:3]):
        ax.plot(df[feat].values[:500], label=feat[:20], alpha=0.7)
    ax.set_title('Feature Trends (first 500 samples)')
    ax.set_xlabel('Sample')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution histograms
    ax = axes[0, 1]
    for feat in top_features[:4]:
        ax.hist(df[feat].dropna(), bins=50, alpha=0.5, label=feat[:15])
    ax.set_title('Feature Distributions')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 3. Boxplot of top features
    ax = axes[0, 2]
    box_data = [df[feat].dropna().values for feat in top_features[:6]]
    ax.boxplot(box_data, labels=[f[:10] for f in top_features[:6]], rot=45)
    ax.set_title('Boxplots - Top 6 Features')
    ax.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    ax = axes[1, 0]
    corr = df[top_features[:8]].corr()
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features[:8])))
    ax.set_xticklabels([f[:10] for f in top_features[:8]], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top_features[:8])))
    ax.set_yticklabels([f[:10] for f in top_features[:8]], fontsize=7)
    ax.set_title('Correlation Matrix')
    plt.colorbar(im, ax=ax)
    
    # 5. PCA
    ax = axes[1, 1]
    # Take first 1000 rows for PCA (faster)
    sample_df = df[numeric_cols].dropna().head(1000)
    if len(sample_df) > 10:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(sample_df)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=np.arange(len(pca_result)), 
                           cmap='viridis', alpha=0.6, s=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA - First 1000 samples')
        plt.colorbar(scatter, ax=ax, label='Sample order')
    ax.grid(True, alpha=0.3)
    
    # 6. Feature variance ranking
    ax = axes[1, 2]
    top_var = variances.head(10)
    ax.barh(range(len(top_var)), top_var.values)
    ax.set_yticks(range(len(top_var)))
    ax.set_yticklabels([f[:25] for f in top_var.index], fontsize=8)
    ax.set_xlabel('Variance')
    ax.set_title('Top 10 Features by Variance')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature Analysis - {Path(input_file).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_analysis_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save variance ranking to CSV
    variance_df = pd.DataFrame({'Feature': top_var.index, 'Variance': top_var.values})
    variance_df.to_csv(os.path.join(output_dir, 'feature_variance_ranking.csv'), index=False)
    
    print(f"\n✓ Saved: feature_analysis_summary.png")
    print(f"✓ Saved: feature_variance_ranking.csv")
    print(f"\n{'='*60}")
    print(f"Done! Results in: {output_dir}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate feature visualizations')
    parser.add_argument('input_file', help='Feature CSV file')
    parser.add_argument('--output_dir', '-o', default='Feature_Selector_Plots3/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    success = generate_visualizations(args.input_file, args.output_dir)
    sys.exit(0 if success else 1)