import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: COMPUTE TOI, BVI, OEF PROXY FOR EACH CHANNEL
# ============================================================================

def compute_fNIRS_derived_metrics(df):
    """
    Compute Tissue Oxygenation Index (TOI), Blood Volume Index (BVI),
    and Oxygen Extraction Fraction (OEF) proxy for each sensor channel.
    
    Parameters:
    -----------
    df : pandas DataFrame with columns containing HbO_value and HbR_value
    
    Returns:
    --------
    df : DataFrame with new metric columns added
    """
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Define channel pairs
    channels = []
    possible_channels = [
        ('left_outer', 'left_outer_HbO_value', 'left_outer_HbR_value'),
        ('right_outer', 'right_outer_HbO_value', 'right_outer_HbR_value'),
        ('left_inner', 'left_inner_HbO_value', 'left_inner_HbR_value'),
        ('right_inner', 'right_inner_HbO_value', 'right_inner_HbR_value')
    ]
    for prefix, hbo_col, hbr_col in possible_channels:
        if hbo_col in df.columns and hbr_col in df.columns:
            channels.append((prefix, hbo_col, hbr_col))
    
    if not channels:
        raise ValueError("No valid channel columns found in dataframe")
    
    for prefix, hbo_col, hbr_col in channels:
        # Tissue Oxygenation Index (TOI) - percentage
        df[f'{prefix}_TOI'] = (df[hbo_col] / (df[hbo_col] + df[hbr_col] + 1e-10)) * 100
        
        # Blood Volume Index (BVI) - total hemoglobin proxy
        df[f'{prefix}_BVI'] = df[hbo_col] + df[hbr_col]
        
        # Oxygen Extraction Fraction (OEF) proxy - proportion
        df[f'{prefix}_OEF_proxy'] = df[hbr_col] / (df[hbo_col] + df[hbr_col] + 1e-10)
    
    # Also compute for wavelength ratios if available
    if 'wavelength_ratio_outer' in df.columns:
        df['wavelength_ratio_outer_norm'] = df['wavelength_ratio_outer'] / df['wavelength_ratio_outer'].mean()
    
    if 'wavelength_ratio_inner' in df.columns:
        df['wavelength_ratio_inner_norm'] = df['wavelength_ratio_inner'] / df['wavelength_ratio_inner'].mean()
    
    # --- DYNAMIC COMPOSITE INDICES ---
    toi_cols = [f'{p}_TOI' for p, _, _ in channels]
    bvi_cols = [f'{p}_BVI' for p, _, _ in channels]
    oef_cols = [f'{p}_OEF_proxy' for p, _, _ in channels]
    
    df['TOI_mean'] = df[toi_cols].mean(axis=1)
    df['BVI_mean'] = df[bvi_cols].mean(axis=1)
    df['OEF_proxy_mean'] = df[oef_cols].mean(axis=1)
    
    # --- DYNAMIC ASYMMETRY ---
    # Separate outer and inner channels based on what's available
    outer_prefixes = [p for p, _, _ in channels if 'outer' in p]
    inner_prefixes = [p for p, _, _ in channels if 'inner' in p]
    left_prefixes = [p for p, _, _ in channels if 'left' in p]
    right_prefixes = [p for p, _, _ in channels if 'right' in p]
    
    # TOI asymmetry (only if we have both left and right)
    if left_prefixes and right_prefixes:
        left_toi = df[[f'{p}_TOI' for p in left_prefixes]].mean(axis=1)
        right_toi = df[[f'{p}_TOI' for p in right_prefixes]].mean(axis=1)
        df['TOI_asymmetry'] = left_toi - right_toi
    
    if left_prefixes and right_prefixes:
        left_bvi = df[[f'{p}_BVI' for p in left_prefixes]].mean(axis=1)
        right_bvi = df[[f'{p}_BVI' for p in right_prefixes]].mean(axis=1)
        df['BVI_asymmetry'] = left_bvi - right_bvi
    
    # Inner-outer gradients (only if we have both)
    if outer_prefixes and inner_prefixes:
        outer_toi = df[[f'{p}_TOI' for p in outer_prefixes]].mean(axis=1)
        inner_toi = df[[f'{p}_TOI' for p in inner_prefixes]].mean(axis=1)
        df['TOI_gradient'] = outer_toi - inner_toi
    
    if outer_prefixes and inner_prefixes:
        outer_bvi = df[[f'{p}_BVI' for p in outer_prefixes]].mean(axis=1)
        inner_bvi = df[[f'{p}_BVI' for p in inner_prefixes]].mean(axis=1)
        df['BVI_gradient'] = outer_bvi - inner_bvi
    
    return df, channels


# ============================================================================
# PART 2: COMPUTE RATES OF CHANGE
# ============================================================================

def compute_rates_of_change(df, channels, metrics=None, smooth_window=5):
    """
    Compute rate of change (first derivative) for specified metrics.
    Optionally apply Savitzky-Golay smoothing first for noisy signals.
    
    Parameters:
    -----------
    df : pandas DataFrame (should already have derived metrics)
    metrics : list of metric column names to compute derivatives for
    smooth_window : window size for Savitzky-Golay filter (odd number, 5 is default)
    
    Returns:
    --------
    df : DataFrame with rate of change columns added
    """
    df = df.copy()
    
    if metrics is None:
        metrics = ['TOI_mean', 'BVI_mean', 'OEF_proxy_mean']
        # Add metrics that actually exist
        for extra in ['TOI_asymmetry', 'BVI_asymmetry']:
            if extra in df.columns:
                metrics.append(extra)
        for p, _, _ in channels:
            for suffix in ['_TOI']:  # add more if needed
                col = f'{p}{suffix}'
                if col in df.columns:
                    metrics.append(col)
    
    for metric in metrics:
        if metric in df.columns:
            # Raw rate of change
            df[f'd_{metric}_dt'] = df[metric].diff()
            
            # Smoothed rate of change (more robust to noise)
            if len(df) > smooth_window:
                try:
                    smoothed = savgol_filter(df[metric].values, 
                                            window_length=smooth_window, 
                                            polyorder=2)
                    df[f'{metric}_smoothed'] = smoothed
                    df[f'd_{metric}_dt_smooth'] = np.gradient(smoothed)
                except:
                    # Fallback to simple gradient if smoothing fails
                    df[f'd_{metric}_dt_smooth'] = np.gradient(df[metric].values)
            else:
                df[f'd_{metric}_dt_smooth'] = np.gradient(df[metric].values)
            
            # Normalized rate of change (percentage change per sample)
            df[f'd_{metric}_dt_pct'] = df[f'd_{metric}_dt'] / (df[metric].abs() + 1e-10) * 100
    
    return df


# ============================================================================
# PART 3: CREATE BLADDER VOLUME ESTIMATION INDEX
# ============================================================================

def compute_bladder_filling_index(df, pre_wear_volume=None, post_wear_volume=None):
    """
    Compute a continuous Bladder Filling Index based on fNIRS metrics.
    
    The index is normalized to [0, 1] range where:
    - 0 = minimum observed value (likely empty bladder state)
    - 1 = maximum observed value (likely full bladder state)
    
    If pre/post volumes are provided, can map index to mL estimates.
    
    Parameters:
    -----------
    df : pandas DataFrame with derived metrics
    pre_wear_volume : float, optional - bladder volume at session start (mL)
    post_wear_volume : float, optional - bladder volume at session end (mL)
    
    Returns:
    --------
    df : DataFrame with Filling Index added
    filling_index_formula : str - description of how index was computed
    """
    df = df.copy()
    
    # Component 1: TOI change (decreased TOI may indicate sympathetic activation)
    # Normalize and invert (lower TOI → higher filling index)
    toi_min = df['TOI_mean'].min()
    toi_max = df['TOI_mean'].max()
    df['TOI_norm'] = 1 - ((df['TOI_mean'] - toi_min) / (toi_max - toi_min + 1e-10))
    
    # Component 2: BVI change (blood volume shifts)
    bvi_min = df['BVI_mean'].min()
    bvi_max = df['BVI_mean'].max()
    df['BVI_norm'] = (df['BVI_mean'] - bvi_min) / (bvi_max - bvi_min + 1e-10)
    
    # Component 3: OEF proxy (oxygen extraction)
    oef_min = df['OEF_proxy_mean'].min()
    oef_max = df['OEF_proxy_mean'].max()
    df['OEF_norm'] = (df['OEF_proxy_mean'] - oef_min) / (oef_max - oef_min + 1e-10)
    
    # Component 4: Asymmetry magnitude (absolute asymmetry may increase with distension)
    df['asymmetry_magnitude'] = np.abs(df['TOI_asymmetry'])
    asym_min = df['asymmetry_magnitude'].min()
    asym_max = df['asymmetry_magnitude'].max()
    df['Asymmetry_norm'] = (df['asymmetry_magnitude'] - asym_min) / (asym_max - asym_min + 1e-10)
    
    # Component 5: Rate of change (dynamic filling signal)
    if 'd_TOI_mean_dt_smooth' in df.columns:
        dtoi_min = df['d_TOI_mean_dt_smooth'].min()
        dtoi_max = df['d_TOI_mean_dt_smooth'].max()
        df['dTOI_norm'] = (df['d_TOI_mean_dt_smooth'] - dtoi_min) / (dtoi_max - dtoi_min + 1e-10)
    else:
        df['dTOI_norm'] = 0.5
    
    # Combined Filling Index (equal weights - can be optimized with validation data)
    df['Bladder_Filling_Index'] = (
        0.25 * df['TOI_norm'] +
        0.20 * df['BVI_norm'] +
        0.20 * df['OEF_norm'] +
        0.20 * df['Asymmetry_norm'] +
        0.15 * df['dTOI_norm']
    )
    
    # Smooth the index
    if len(df) > 10:
        df['Bladder_Filling_Index_Smooth'] = savgol_filter(
            df['Bladder_Filling_Index'].values,
            window_length=min(11, len(df) if len(df) % 2 == 1 else len(df) - 1),
            polyorder=2
        )
    else:
        df['Bladder_Filling_Index_Smooth'] = df['Bladder_Filling_Index']
    
    # If volume endpoints are provided, map to mL
    if pre_wear_volume is not None and post_wear_volume is not None:
        # Assume linear relationship between index and volume
        idx_start = df['Bladder_Filling_Index_Smooth'].iloc[0]
        idx_end = df['Bladder_Filling_Index_Smooth'].iloc[-1]
        
        # Linear mapping
        slope = (post_wear_volume - pre_wear_volume) / (idx_end - idx_start + 1e-10)
        intercept = pre_wear_volume - slope * idx_start
        
        df['Estimated_Bladder_Volume_mL'] = slope * df['Bladder_Filling_Index_Smooth'] + intercept
        
        # Ensure values are within reasonable bounds
        min_vol = min(pre_wear_volume, post_wear_volume) * 0.8
        max_vol = max(pre_wear_volume, post_wear_volume) * 1.2
        df['Estimated_Bladder_Volume_mL'] = df['Estimated_Bladder_Volume_mL'].clip(min_vol, max_vol)
    
    formula = "Filling Index = 0.25*TOI_norm + 0.20*BVI_norm + 0.20*OEF_norm + 0.20*Asymmetry_norm + 0.15*dTOI_norm"
    
    return df, formula

def add_elapsed_time_feature(df, time_col='prediction_time_sec'):
    """
    Add elapsed time in seconds from session start.
    This is the single most predictive feature from Fechner et al. (2023).
    
    If you have a real timestamp column, uses that.
    If not, uses index assuming approximately regular sampling.
    """
    if time_col in df.columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df['timestamp_dt'] = pd.to_datetime(df[time_col])
        else:
            df['timestamp_dt'] = df[time_col]
        
        df['elapsed_time_sec'] = (df['timestamp_dt'] - df['timestamp_dt'].iloc[0]).dt.total_seconds()
    else:
        # Fallback: assume index represents sequential samples
        # Estimate sampling rate from your data
        print("Warning: No timestamp column found. Using row index as proxy for time.")
        print("This is NOT ideal — elapsed time should be real seconds.")
        # If you know your approximate sampling rate (e.g., 10 Hz), use that
        estimated_sampling_rate_hz = 10  # adjust this based on your device
        df['elapsed_time_sec'] = df.index / estimated_sampling_rate_hz
    
    # Also compute elapsed time in minutes for plotting
    df['elapsed_time_min'] = df['elapsed_time_sec'] / 60
    
    return df


# ============================================================================
# PART 4: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_fNIRS_derived_metrics(df, title_prefix="fNIRS Derived Metrics", save_path = None):
    """
    Create comprehensive visualization of TOI, BVI, OEF, and Filling Index.
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    
    # Row 1: TOI across channels
    ax = axes[0, 0]
    for prefix in ['left_outer', 'right_outer', 'left_inner', 'right_inner']:
        col = f'{prefix}_TOI'
        if col in df.columns:
            ax.plot(df.index, df[col], label=prefix.replace('_', ' ').title(), alpha=0.7, linewidth=1)
    ax.set_ylabel('TOI (%)')
    ax.set_title(f'{title_prefix}: Tissue Oxygenation Index by Channel')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 1 right: TOI mean and asymmetry
    ax = axes[0, 1]
    if 'TOI_mean' in df.columns:
        ax.plot(df.index, df['TOI_mean'], 'b-', linewidth=2, label='Mean TOI')
        ax_twin = ax.twinx()
        if 'TOI_asymmetry' in df.columns:
            ax_twin.plot(df.index, df['TOI_asymmetry'], 'r--', linewidth=1.5, label='L-R Asymmetry')
            ax_twin.set_ylabel('Asymmetry', color='r')
            ax_twin.tick_params(axis='y', labelcolor='r')
    ax.set_ylabel('TOI (%)', color='b')
    ax.set_title('Mean TOI and Left-Right Asymmetry')
    ax.grid(True, alpha=0.3)
    
    # Row 2: BVI across channels
    ax = axes[1, 0]
    for prefix in ['left_outer', 'right_outer', 'left_inner', 'right_inner']:
        col = f'{prefix}_BVI'
        if col in df.columns:
            ax.plot(df.index, df[col], label=prefix.replace('_', ' ').title(), alpha=0.7, linewidth=1)
    ax.set_ylabel('BVI (a.u.)')
    ax.set_title('Blood Volume Index by Channel')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 2 right: BVI gradient
    ax = axes[1, 1]
    if 'BVI_gradient' in df.columns:
        ax.plot(df.index, df['BVI_gradient'], 'g-', linewidth=2, label='Outer-Inner Gradient')
    if 'BVI_mean' in df.columns:
        ax.plot(df.index, df['BVI_mean'] - df['BVI_mean'].mean(), 'gray', alpha=0.5, label='Centered Mean BVI')
    ax.set_ylabel('BVI Gradient')
    ax.set_title('Blood Volume Index Gradient (Outer - Inner)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Row 3: OEF Proxy
    ax = axes[2, 0]
    for prefix in ['left_outer', 'right_outer', 'left_inner', 'right_inner']:
        col = f'{prefix}_OEF_proxy'
        if col in df.columns:
            ax.plot(df.index, df[col], label=prefix.replace('_', ' ').title(), alpha=0.7, linewidth=1)
    ax.set_ylabel('OEF Proxy')
    ax.set_title('Oxygen Extraction Fraction Proxy by Channel')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 3 right: Rates of change
    ax = axes[2, 1]
    if 'd_TOI_mean_dt_smooth' in df.columns:
        ax.plot(df.index, df['d_TOI_mean_dt_smooth'], 'b-', linewidth=1.5, label='d(TOI)/dt')
    if 'd_BVI_mean_dt_smooth' in df.columns:
        ax.plot(df.index, df['d_BVI_mean_dt_smooth'], 'g-', linewidth=1.5, label='d(BVI)/dt')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('Rate of Change')
    ax.set_title('Rates of Change (Smoothed)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Row 4: Bladder Filling Index
    ax = axes[3, 0]
    if 'Bladder_Filling_Index' in df.columns:
        ax.plot(df.index, df['Bladder_Filling_Index'], 'gray', alpha=0.5, linewidth=0.8, label='Raw')
        ax.plot(df.index, df['Bladder_Filling_Index_Smooth'], 'b-', linewidth=2.5, label='Smoothed')
        ax.fill_between(df.index, 0, df['Bladder_Filling_Index_Smooth'], alpha=0.3)
    ax.set_ylabel('Filling Index')
    ax.set_xlabel('Sample Index')
    ax.set_title('Bladder Filling Index (0 = Empty, 1 = Full)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Row 4 right: Estimated Volume (if available)
    ax = axes[3, 1]
    if 'Estimated_Bladder_Volume_mL' in df.columns:
        ax.plot(df.index, df['Estimated_Bladder_Volume_mL'], 'b-', linewidth=2.5)
        ax.fill_between(df.index, 0, df['Estimated_Bladder_Volume_mL'], alpha=0.3)
        ax.set_ylabel('Estimated Volume (mL)')
        ax.set_title('Estimated Continuous Bladder Volume')
    else:
        # Show component contributions instead
        components = ['TOI_norm', 'BVI_norm', 'OEF_norm', 'Asymmetry_norm']
        colors = ['blue', 'green', 'red', 'orange']
        for comp, color in zip(components, colors):
            if comp in df.columns:
                ax.plot(df.index, df[comp], label=comp.replace('_norm', ''), color=color, alpha=0.7)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Filling Index Components')
        ax.legend(loc='best')
    ax.set_xlabel('Sample Index')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()  # Prevents hanging if running multiple plots
    return fig


def plot_correlation_with_volume(df, pre_vol, post_vol, save_path=None):
    """
    Show how metrics correlate with the known volume change.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Compute simple trend indicators
    metrics_to_check = ['TOI_mean', 'BVI_mean', 'OEF_proxy_mean', 
                        'TOI_asymmetry', 'BVI_gradient', 'wavelength_ratio_outer']
    
    volume_change = post_vol - pre_vol
    
    for ax, metric in zip(axes.flatten(), metrics_to_check):
        if metric in df.columns:
            # Overall trend (end - start)
            trend = df[metric].iloc[-1] - df[metric].iloc[0]
            trend_pct = (trend / df[metric].mean()) * 100 if df[metric].mean() != 0 else 0
            
            # Plot the time series
            ax.plot(df.index, df[metric], 'b-', linewidth=1.5)
            ax.axhline(y=df[metric].mean(), color='gray', linestyle='--', alpha=0.5)
            
            # Add trend line
            z = np.polyfit(range(len(df)), df[metric].values, 1)
            p = np.poly1d(z)
            ax.plot(df.index, p(range(len(df))), 'r--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(f"{metric}\nΔ = {trend:.3f} ({trend_pct:+.1f}%)")
            ax.set_xlabel('Sample Index')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Metric Trends During Session\nVolume Change: {volume_change:+.0f} mL', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()  # Prevents hanging if running multiple plots
    return fig


# ============================================================================
# PART 5: MAIN PROCESSING PIPELINE
# ============================================================================

def process_fNIRS_session(df, pre_wear_volume=None, post_wear_volume=None, 
                          session_name="Session", smooth_window=5):
    """
    Complete processing pipeline for a single fNIRS session.
    
    Parameters:
    -----------
    df : pandas DataFrame - raw lag-features dataframe
    pre_wear_volume : float - bladder volume at session start (mL)
    post_wear_volume : float - bladder volume at session end (mL)
    session_name : str - identifier for this session
    smooth_window : int - window size for smoothing
    
    Returns:
    --------
    df_processed : DataFrame with all derived metrics
    summary_stats : dict - key summary statistics for this session
    """
    
    print(f"\n{'='*60}")
    print(f"Processing: {session_name}")
    print(f"{'='*60}")

    df_processed = add_elapsed_time_feature(df, time_col='prediction_time_sec')
    
    # Step 1: Compute derived metrics
    df_processed, channelsXD = compute_fNIRS_derived_metrics(df)
    
    # Step 2: Compute rates of change
    df_processed = compute_rates_of_change(df_processed,channelsXD, smooth_window=smooth_window)
    
    # Step 3: Compute Bladder Filling Index
    df_processed, formula = compute_bladder_filling_index(
        df_processed, 
        pre_wear_volume=pre_wear_volume,
        post_wear_volume=post_wear_volume
    )
    
    print(f"\nFilling Index Formula: {formula}")
    
    # Generate summary statistics
    summary_stats = {
        'session': session_name,
        'n_samples': len(df_processed),
        'duration_sec': df_processed['elapsed_time_sec'].max() if 'elapsed_time_sec' in df_processed.columns else None,
        'pre_volume_mL': pre_wear_volume,
        'post_volume_mL': post_wear_volume,
        'volume_change_mL': post_wear_volume - pre_wear_volume if (pre_wear_volume and post_wear_volume) else None,
        'TOI_mean_start': df_processed['TOI_mean'].iloc[:100].mean() if len(df_processed) > 100 else df_processed['TOI_mean'].mean(),
        'TOI_mean_end': df_processed['TOI_mean'].iloc[-100:].mean() if len(df_processed) > 100 else df_processed['TOI_mean'].mean(),
        'TOI_trend': df_processed['TOI_mean'].iloc[-1] - df_processed['TOI_mean'].iloc[0],
        'BVI_trend': df_processed['BVI_mean'].iloc[-1] - df_processed['BVI_mean'].iloc[0],
        'Filling_Index_start': df_processed['Bladder_Filling_Index_Smooth'].iloc[0],
        'Filling_Index_end': df_processed['Bladder_Filling_Index_Smooth'].iloc[-1],
    }
    
    # Print summary
    print("\n" + "-"*40)
    print("SESSION SUMMARY")
    print("-"*40)
    for key, value in summary_stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    # Correlation analysis if volume available
    if pre_wear_volume is not None and post_wear_volume is not None:
        print("\n" + "-"*40)
        print("VOLUME RELATIONSHIP")
        print("-"*40)
        
        vol_change = post_wear_volume - pre_wear_volume
        toi_change = summary_stats['TOI_trend']
        
        print(f"  Volume change: {vol_change:+.1f} mL")
        print(f"  TOI change: {toi_change:+.3f} %")
        print(f"  Ratio (TOI Δ / Volume Δ): {toi_change/vol_change:.4f} %/mL")
        
        if 'Estimated_Bladder_Volume_mL' in df_processed.columns:
            est_start = df_processed['Estimated_Bladder_Volume_mL'].iloc[0]
            est_end = df_processed['Estimated_Bladder_Volume_mL'].iloc[-1]
            print(f"  Estimated start volume: {est_start:.1f} mL (actual: {pre_wear_volume:.1f})")
            print(f"  Estimated end volume: {est_end:.1f} mL (actual: {post_wear_volume:.1f})")
            print(f"  Estimation error: {(est_end - est_start) - vol_change:+.1f} mL")
    
    return df_processed, summary_stats


# ============================================================================
# PART 6: BATCH PROCESSING FOR MULTIPLE SESSIONS
# ============================================================================

def process_multiple_sessions(session_dfs, session_volumes, smooth_window=5):
    """
    Process multiple sessions and compile results.
    
    Parameters:
    -----------
    session_dfs : dict - {session_name: dataframe}
    session_volumes : dict - {session_name: (pre_vol, post_vol)}
    smooth_window : int - window size for smoothing
    
    Returns:
    --------
    all_results : dict - {session_name: (df_processed, summary_stats)}
    combined_summary : DataFrame - summary across all sessions
    """
    
    all_results = {}
    summaries = []
    
    for session_name, df in session_dfs.items():
        pre_vol, post_vol = session_volumes.get(session_name, (None, None))
        
        df_proc, summary = process_fNIRS_session(
            df, 
            pre_wear_volume=pre_vol,
            post_wear_volume=post_vol,
            session_name=session_name,
            smooth_window=smooth_window
        )
        
        all_results[session_name] = (df_proc, summary)
        summaries.append(summary)
    
    combined_summary = pd.DataFrame(summaries)
    
    # Overall correlation analysis
    if 'volume_change_mL' in combined_summary.columns and combined_summary['volume_change_mL'].notna().any():
        print("\n" + "="*60)
        print("CROSS-SESSION CORRELATION ANALYSIS")
        print("="*60)
        
        valid_data = combined_summary.dropna(subset=['volume_change_mL'])
        
        if len(valid_data) > 1:
            from scipy.stats import pearsonr
            
            for metric in ['TOI_trend', 'BVI_trend']:
                if metric in valid_data.columns:
                    r, p = pearsonr(valid_data['volume_change_mL'], valid_data[metric])
                    print(f"\n{metric} vs Volume Change:")
                    print(f"  Pearson r = {r:.3f} (p = {p:.3f})")
    
    return all_results, combined_summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract features from windowed NIRS data (output of temporal_overlap.py)'
    )
    parser.add_argument('input_file', help='Windowed CSV file (from temporal_overlap.py)')
    parser.add_argument('--output-dir', default='Feature_Extracted_y/', 
                       help='Output directory (default: Feature_Extracted)')
    parser.add_argument('--output-name', default=None,
                       help='Output filename (default: <input_name>_features.csv)')
    parser.add_argument('--output-name2', default=None,
                       help='Output filename (default: <input_name>_features2.csv)')
    
    args = parser.parse_args()
    
    # Set output filename
    if args.output_name is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{base_name}_features.csv")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, args.output_name)

    df = pd.read_csv(args.input_file)

        # Process the example data
    df_processed, summary = process_fNIRS_session(
        df,
        pre_wear_volume=None,   # Example: 100 mL at start
        post_wear_volume=None,  # Example: 250 mL at end (150 mL filling)
        session_name="Example_Session"
    )

    if output_file:
        df_processed.to_csv(output_file, index=False)
        print(f"✓ Saved features to: {output_file}")

    plot1_path = f"{base_name}_fNIRS_derived_metrics.png"
    plot2_path = f"{base_name}_corr_with_vol.png"
    
    # Create visualizations
    fig1 = plot_fNIRS_derived_metrics(
        df_processed, 
        title_prefix="Example Session",
        save_path=plot1_path
)
    fig2 = plot_correlation_with_volume(
        df_processed, 
        pre_vol=33.8688, 
        post_vol=124.60032,
        save_path=plot2_path
)
    
    #plt.show()
    
    # Save processed data
    # df_processed.to_csv('fNIRS_processed_with_metrics.csv', index=False)
    

if __name__ == "__main__":
    main()
    
