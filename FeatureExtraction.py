#!/usr/bin/env python3
"""
Feature Extraction for Windowed NIRS Data
Extracts features from temporally overlapped windows (output of temporal_overlap.py)
"""

import numpy as np
import pandas as pd
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def extract_signal_features(signal):
    """Extract time-domain and spectral features from a 1D signal"""
    features = {}
    signal = np.array(signal).flatten()
    
    # Handle degenerate cases
    if len(signal) == 0:
        return {k: np.nan for k in ['mean', 'std', 'skewness', 'kurtosis', 'range', 'rms', 'zero_crossings', 'dominant_freq', 'spectral_power']}
    
    diff = signal - np.mean(signal)
    std = np.std(signal)
    
    # Time-domain features
    features.update({
        'mean': np.mean(signal),
        'std': std,
        'skewness': np.mean(diff**3) / (std**3 + 1e-7),
        'kurtosis': np.mean(diff**4) / (std**4 + 1e-7),
        'range': np.max(signal) - np.min(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'zero_crossings': len(np.where(np.diff(np.sign(signal)))[0]) / len(signal)
    })
    
    # Frequency-domain features (if signal is long enough)
    if len(signal) > 10:
        fft = np.abs(np.fft.rfft(signal))
        fft_freqs = np.fft.rfftfreq(len(signal))
        # Skip DC component (index 0)
        if len(fft) > 1:
            features.update({
                'dominant_freq': fft_freqs[np.argmax(fft[1:]) + 1],
                'spectral_power': np.sum(fft**2)
            })
        else:
            features.update({
                'dominant_freq': 0,
                'spectral_power': 0
            })
    
    return features

def extract_hbo_hbr_features(df_window):
    """Extract features from HbO and HbR channels"""
    features = {}
    
    # Find HbO and HbR columns
    hbo_cols = [col for col in df_window.columns if 'HbO' in col]
    hbr_cols = [col for col in df_window.columns if 'HbR' in col]
    
    # Process each HbO/HbR pair
    for hbo_col in hbo_cols:
        # Get corresponding HbR column
        hbr_col = hbo_col.replace('HbO', 'HbR')
        location = hbo_col.replace('_HbO', '')
        
        if hbr_col in df_window.columns:
            hbo_signal = df_window[hbo_col].values
            hbr_signal = df_window[hbr_col].values
            
            # Individual signal features
            for name, value in extract_signal_features(hbo_signal).items():
                features[f'{location}_HbO_{name}'] = value
            
            for name, value in extract_signal_features(hbr_signal).items():
                features[f'{location}_HbR_{name}'] = value
            
            # Ratio features (tissue oxygenation index)
            ratio = (hbo_signal + 0.001) / (hbr_signal + 0.001)
            features[f'{location}_HbO_HbR_ratio_mean'] = np.mean(ratio)
            features[f'{location}_HbO_HbR_ratio_std'] = np.std(ratio)
            
            # Difference features
            diff = hbo_signal - hbr_signal
            features[f'{location}_HbO_minus_HbR_mean'] = np.mean(diff)
            features[f'{location}_HbO_minus_HbR_std'] = np.std(diff)
    
    return features

def wavelength_specific_features(df_window):
    """Extract features from raw optical channels by wavelength"""
    features = {}
    
    # Note: Column names have '_uA' suffix
    # 730nm channels: optics1_uA, optics2_uA, optics5_uA, optics6_uA
    nm730_cols = [col for col in df_window.columns if 'optics' in col and col not in ['optics9_uA', 'optics10_uA', 'optics11_uA', 'optics12_uA', 'optics13_uA', 'optics14_uA', 'optics15_uA', 'optics16_uA']]
    # More precise: only 1,2,5,6 are 730nm
    nm730_cols = ['optics1_uA', 'optics2_uA', 'optics5_uA', 'optics6_uA']
    nm730_cols = [col for col in nm730_cols if col in df_window.columns]
    
    if nm730_cols:
        nm730 = df_window[nm730_cols].mean(axis=1)
        for name, value in extract_signal_features(nm730).items():
            features[f'730nm_{name}'] = value
    
    # 850nm channels: optics3_uA, optics4_uA, optics7_uA, optics8_uA
    nm850_cols = ['optics3_uA', 'optics4_uA', 'optics7_uA', 'optics8_uA']
    nm850_cols = [col for col in nm850_cols if col in df_window.columns]
    
    if nm850_cols:
        nm850 = df_window[nm850_cols].mean(axis=1)
        for name, value in extract_signal_features(nm850).items():
            features[f'850nm_{name}'] = value
    
    # Red channels (if available)
    red_cols = ['optics9_uA', 'optics10_uA', 'optics13_uA', 'optics14_uA']
    red_cols = [col for col in red_cols if col in df_window.columns]
    
    if red_cols:
        red = df_window[red_cols].mean(axis=1)
        for name, value in extract_signal_features(red).items():
            features[f'red_{name}'] = value
    
    return features

def spatial_features(df_window):
    """Extract spatial relationship features (left/right, inner/outer)"""
    features = {}
    eps = 1e-7
    
    # Left/Right asymmetry for 730nm
    left_730 = ['optics1_uA', 'optics5_uA']
    right_730 = ['optics2_uA', 'optics6_uA']
    left_730 = [col for col in left_730 if col in df_window.columns]
    right_730 = [col for col in right_730 if col in df_window.columns]
    
    if left_730 and right_730:
        left_730_mean = df_window[left_730].mean(axis=1).mean()
        right_730_mean = df_window[right_730].mean(axis=1).mean()
        features['left_right_730_diff'] = left_730_mean - right_730_mean
        features['left_right_730_ratio'] = (left_730_mean + eps) / (right_730_mean + eps)
    
    # Left/Right asymmetry for 850nm
    left_850 = ['optics3_uA', 'optics7_uA']
    right_850 = ['optics4_uA', 'optics8_uA']
    left_850 = [col for col in left_850 if col in df_window.columns]
    right_850 = [col for col in right_850 if col in df_window.columns]
    
    if left_850 and right_850:
        left_850_mean = df_window[left_850].mean(axis=1).mean()
        right_850_mean = df_window[right_850].mean(axis=1).mean()
        features['left_right_850_diff'] = left_850_mean - right_850_mean
        features['left_right_850_ratio'] = (left_850_mean + eps) / (right_850_mean + eps)
    
    # Inner/Outer depth sensitivity
    outer_730 = ['optics1_uA', 'optics2_uA']
    inner_730 = ['optics5_uA', 'optics6_uA']
    outer_730 = [col for col in outer_730 if col in df_window.columns]
    inner_730 = [col for col in inner_730 if col in df_window.columns]
    
    if outer_730 and inner_730:
        outer_730_mean = df_window[outer_730].mean(axis=1).mean()
        inner_730_mean = df_window[inner_730].mean(axis=1).mean()
        features['depth_730_ratio'] = (inner_730_mean + eps) / (outer_730_mean + eps)
    
    outer_850 = ['optics3_uA', 'optics4_uA']
    inner_850 = ['optics7_uA', 'optics8_uA']
    outer_850 = [col for col in outer_850 if col in df_window.columns]
    inner_850 = [col for col in inner_850 if col in df_window.columns]
    
    if outer_850 and inner_850:
        outer_850_mean = df_window[outer_850].mean(axis=1).mean()
        inner_850_mean = df_window[inner_850].mean(axis=1).mean()
        features['depth_850_ratio'] = (inner_850_mean + eps) / (outer_850_mean + eps)
    
    return features

def wavelength_ratio_features(df_window):
    """Extract 850nm/730nm ratio features (sensitive to blood volume)"""
    features = {}
    eps = 1e-7
    
    # Outer channels
    outer_850 = ['optics3_uA', 'optics4_uA']
    outer_730 = ['optics1_uA', 'optics2_uA']
    outer_850 = [col for col in outer_850 if col in df_window.columns]
    outer_730 = [col for col in outer_730 if col in df_window.columns]
    
    if outer_850 and outer_730:
        ratio_outer = (df_window[outer_850].mean(axis=1) + eps) / (df_window[outer_730].mean(axis=1) + eps)
        features['wavelength_ratio_outer_mean'] = ratio_outer.mean()
        features['wavelength_ratio_outer_std'] = ratio_outer.std()
        features['wavelength_ratio_outer_slope'] = np.polyfit(range(len(ratio_outer)), ratio_outer, 1)[0]
    
    # Inner channels
    inner_850 = ['optics7_uA', 'optics8_uA']
    inner_730 = ['optics5_uA', 'optics6_uA']
    inner_850 = [col for col in inner_850 if col in df_window.columns]
    inner_730 = [col for col in inner_730 if col in df_window.columns]
    
    if inner_850 and inner_730:
        ratio_inner = (df_window[inner_850].mean(axis=1) + eps) / (df_window[inner_730].mean(axis=1) + eps)
        features['wavelength_ratio_inner_mean'] = ratio_inner.mean()
        features['wavelength_ratio_inner_std'] = ratio_inner.std()
        features['wavelength_ratio_inner_slope'] = np.polyfit(range(len(ratio_inner)), ratio_inner, 1)[0]
    
    return features

def signal_quality_metric(signal):
    """Combined quality metric for a single channel"""
    signal = np.array(signal).flatten()
    return {
        'snr': np.mean(signal) / (np.std(signal) + 1e-7),
        'artifacts': np.sum(np.abs(signal - np.mean(signal)) > 3 * np.std(signal)) / len(signal)
    }
def extract_hbo_hbr_features_row(row_data):
    """Extract features from HbO and HbR channels for a single time point"""
    features = {}
    
    # Find HbO and HbR columns
    hbo_cols = [col for col in row_data.index if 'HbO' in col]
    
    # Process each HbO/HbR pair
    for hbo_col in hbo_cols:
        # Get corresponding HbR column
        hbr_col = hbo_col.replace('HbO', 'HbR')
        location = hbo_col.replace('_HbO', '')
        
        if hbr_col in row_data.index:
            hbo_value = row_data[hbo_col]
            hbr_value = row_data[hbr_col]
            
            # For single points, store the raw values
            features[f'{location}_HbO_value'] = hbo_value
            features[f'{location}_HbR_value'] = hbr_value
            
            # Ratio features (tissue oxygenation index)
            ratio = (hbo_value + 0.001) / (hbr_value + 0.001)
            features[f'{location}_HbO_HbR_ratio'] = ratio
            
            # Difference features
            diff = hbo_value - hbr_value
            features[f'{location}_HbO_minus_HbR'] = diff
    
    return features

def wavelength_specific_features_row(row_data):
    """Extract features from raw optical channels by wavelength for a single time point"""
    features = {}
    
    # 730nm channels
    nm730_cols = ['optics1_uA', 'optics2_uA', 'optics5_uA', 'optics6_uA']
    nm730_cols = [col for col in nm730_cols if col in row_data.index]
    
    if nm730_cols:
        nm730_mean = np.mean([row_data[col] for col in nm730_cols])
        features['730nm_mean'] = nm730_mean
    
    # 850nm channels
    nm850_cols = ['optics3_uA', 'optics4_uA', 'optics7_uA', 'optics8_uA']
    nm850_cols = [col for col in nm850_cols if col in row_data.index]
    
    if nm850_cols:
        nm850_mean = np.mean([row_data[col] for col in nm850_cols])
        features['850nm_mean'] = nm850_mean
    
    # Red channels (if available)
    red_cols = ['optics9_uA', 'optics10_uA', 'optics13_uA', 'optics14_uA']
    red_cols = [col for col in red_cols if col in row_data.index]
    
    if red_cols:
        red_mean = np.mean([row_data[col] for col in red_cols])
        features['red_mean'] = red_mean
    
    return features

def spatial_features_row(row_data):
    """Extract spatial relationship features for a single time point"""
    features = {}
    eps = 1e-7
    
    # Left/Right asymmetry for 730nm
    left_730 = ['optics1_uA', 'optics5_uA']
    right_730 = ['optics2_uA', 'optics6_uA']
    left_730 = [col for col in left_730 if col in row_data.index]
    right_730 = [col for col in right_730 if col in row_data.index]
    
    if left_730 and right_730:
        left_730_mean = np.mean([row_data[col] for col in left_730])
        right_730_mean = np.mean([row_data[col] for col in right_730])
        features['left_right_730_diff'] = left_730_mean - right_730_mean
        features['left_right_730_ratio'] = (left_730_mean + eps) / (right_730_mean + eps)
    
    # Left/Right asymmetry for 850nm
    left_850 = ['optics3_uA', 'optics7_uA']
    right_850 = ['optics4_uA', 'optics8_uA']
    left_850 = [col for col in left_850 if col in row_data.index]
    right_850 = [col for col in right_850 if col in row_data.index]
    
    if left_850 and right_850:
        left_850_mean = np.mean([row_data[col] for col in left_850])
        right_850_mean = np.mean([row_data[col] for col in right_850])
        features['left_right_850_diff'] = left_850_mean - right_850_mean
        features['left_right_850_ratio'] = (left_850_mean + eps) / (right_850_mean + eps)
    
    # Inner/Outer depth sensitivity
    outer_730 = ['optics1_uA', 'optics2_uA']
    inner_730 = ['optics5_uA', 'optics6_uA']
    outer_730 = [col for col in outer_730 if col in row_data.index]
    inner_730 = [col for col in inner_730 if col in row_data.index]
    
    if outer_730 and inner_730:
        outer_730_mean = np.mean([row_data[col] for col in outer_730])
        inner_730_mean = np.mean([row_data[col] for col in inner_730])
        features['depth_730_ratio'] = (inner_730_mean + eps) / (outer_730_mean + eps)
    
    outer_850 = ['optics3_uA', 'optics4_uA']
    inner_850 = ['optics7_uA', 'optics8_uA']
    outer_850 = [col for col in outer_850 if col in row_data.index]
    inner_850 = [col for col in inner_850 if col in row_data.index]
    
    if outer_850 and inner_850:
        outer_850_mean = np.mean([row_data[col] for col in outer_850])
        inner_850_mean = np.mean([row_data[col] for col in inner_850])
        features['depth_850_ratio'] = (inner_850_mean + eps) / (outer_850_mean + eps)
    
    return features

def wavelength_ratio_features_row(row_data):
    """Extract 850nm/730nm ratio features for a single time point"""
    features = {}
    eps = 1e-7
    
    # Outer channels
    outer_850 = ['optics3_uA', 'optics4_uA']
    outer_730 = ['optics1_uA', 'optics2_uA']
    outer_850 = [col for col in outer_850 if col in row_data.index]
    outer_730 = [col for col in outer_730 if col in row_data.index]
    
    if outer_850 and outer_730:
        outer_850_mean = np.mean([row_data[col] for col in outer_850])
        outer_730_mean = np.mean([row_data[col] for col in outer_730])
        ratio_outer = (outer_850_mean + eps) / (outer_730_mean + eps)
        features['wavelength_ratio_outer'] = ratio_outer
    
    # Inner channels
    inner_850 = ['optics7_uA', 'optics8_uA']
    inner_730 = ['optics5_uA', 'optics6_uA']
    inner_850 = [col for col in inner_850 if col in row_data.index]
    inner_730 = [col for col in inner_730 if col in row_data.index]
    
    if inner_850 and inner_730:
        inner_850_mean = np.mean([row_data[col] for col in inner_850])
        inner_730_mean = np.mean([row_data[col] for col in inner_730])
        ratio_inner = (inner_850_mean + eps) / (inner_730_mean + eps)
        features['wavelength_ratio_inner'] = ratio_inner
    
    return features

def extract_window_statistics_for_each_row(df_window, window_id):
    """Extract window-level statistics and broadcast to each row"""
    features = {}
    
    # Find HbO and HbR columns
    hbo_cols = [col for col in df_window.columns if 'HbO' in col]
    
    for hbo_col in hbo_cols:
        hbr_col = hbo_col.replace('HbO', 'HbR')
        location = hbo_col.replace('_HbO', '')
        
        if hbr_col in df_window.columns:
            hbo_signal = df_window[hbo_col].values
            hbr_signal = df_window[hbr_col].values
            
            # Extract statistical features from the entire window
            hbo_stats = extract_signal_features(hbo_signal)
            hbr_stats = extract_signal_features(hbr_signal)
            
            # Add window-level statistics (same for all rows in this window)
            for name, value in hbo_stats.items():
                features[f'{location}_HbO_{name}'] = value
            
            for name, value in hbr_stats.items():
                features[f'{location}_HbR_{name}'] = value
            
            # Ratio and difference statistics over the window
            ratio = (hbo_signal + 0.001) / (hbr_signal + 0.001)
            diff = hbo_signal - hbr_signal
            
            features[f'{location}_HbO_HbR_ratio_mean'] = np.mean(ratio)
            features[f'{location}_HbO_HbR_ratio_std'] = np.std(ratio)
            features[f'{location}_HbO_minus_HbR_mean'] = np.mean(diff)
            features[f'{location}_HbO_minus_HbR_std'] = np.std(diff)
    
    # Also add instantaneous values for this specific row
    features.update(extract_instantaneous_values(df_window.iloc[-1]))  # or current row
    
    return features

def process_windowed_csv_with_full_features(input_file, output_file=None):
    """Extract window statistics AND row values for each row"""
    df = pd.read_csv(input_file)
    metadata_cols = ['window_id', 'prediction_time_sec']
    
    all_features = []
    
    for window_id in df['window_id'].unique():
        window_data = df[df['window_id'] == window_id]
        
        # Extract window-level statistics (once per window)
        window_stats = extract_window_statistics_for_each_row(window_data, window_id)
        
        # For each row in the window, combine window stats + row-specific values
        for idx, row in window_data.iterrows():
            features = window_stats.copy()  # Start with window stats
            
            # Add row-specific instantaneous features
            row_values = extract_instantaneous_values(row)
            features.update(row_values)
            
            # Add metadata
            features['window_id'] = row['window_id']
            features['prediction_time_sec'] = row['prediction_time_sec']
            features['row_within_window'] = idx
            
            all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_file, index=False)
    return features_df

def extract_instantaneous_values(row_data):
    """Extract just the raw values for a single row (no statistics)"""
    features = {}
    
    # HbO/HbR raw values
    hbo_cols = [col for col in row_data.index if 'HbO' in col]
    for hbo_col in hbo_cols:
        hbr_col = hbo_col.replace('HbO', 'HbR')
        location = hbo_col.replace('_HbO', '')
        
        if hbr_col in row_data.index:
            features[f'{location}_HbO_current'] = row_data[hbo_col]
            features[f'{location}_HbR_current'] = row_data[hbr_col]
            features[f'{location}_HbO_HbR_ratio_current'] = (row_data[hbo_col] + 0.001) / (row_data[hbr_col] + 0.001)
            features[f'{location}_HbO_minus_HbR_current'] = row_data[hbo_col] - row_data[hbr_col]
    
    # Add wavelength, spatial, and ratio features (instantaneous)
    features.update(wavelength_specific_features_row(row_data))
    features.update(spatial_features_row(row_data))
    features.update(wavelength_ratio_features_row(row_data))
    
    return features

def extract_all_features_from_row(row_data):
    """
    Extract ALL features from a single row (time point) of data
    """
    features = {}
    
    # 1. HbO/HbR features (most important for bladder volume)
    features.update(extract_hbo_hbr_features_row(row_data))
    
    # 2. Wavelength-specific features (raw optical channels)
    features.update(wavelength_specific_features_row(row_data))
    
    # 3. Spatial features (left/right asymmetry, depth ratios)
    features.update(spatial_features_row(row_data))
    
    # 4. Wavelength ratio features
    features.update(wavelength_ratio_features_row(row_data))
    
    return features

def extract_all_features_from_window(df_window):
    """
    Extract ALL features from a single window DataFrame
    """
    features = {}
    
    # 1. HbO/HbR features (most important for bladder volume)
    features.update(extract_hbo_hbr_features(df_window))
    
    # 2. Wavelength-specific features (raw optical channels)
    features.update(wavelength_specific_features(df_window))
    
    # 3. Spatial features (left/right asymmetry, depth ratios)
    features.update(spatial_features(df_window))
    
    # 4. Wavelength ratio features
    features.update(wavelength_ratio_features(df_window))
    
    # 5. Signal quality metrics (optional, can help with QC)
    # Uncomment if needed
    # for col in df_window.columns:
    #     if col.startswith('optics') and '_uA' in col:
    #         quality = signal_quality_metric(df_window[col])
    #         features[f'{col}_snr'] = quality['snr']
    #         features[f'{col}_artifact_ratio'] = quality['artifacts']
    
    return features

def process_windowed_csv_aggregated(input_file, output_file=None):
    """
    Original windowed version - aggregates each window into ONE row of statistical features
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION FOR WINDOWED DATA (AGGREGATED - ONE ROW PER WINDOW)")
    print("="*60)
    print(f"Input file: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Check if this is windowed data
    if 'window_id' not in df.columns:
        print("\nERROR: This doesn't look like windowed data (missing 'window_id' column)")
        return None
    
    # Get unique windows
    window_ids = sorted(df['window_id'].unique())
    print(f"\nFound {len(window_ids)} unique windows")
    
    # Extract features for each window
    print("\nExtracting features from each window...")
    all_features = []
    
    for i, window_id in enumerate(window_ids):
        # Get data for this window
        window_data = df[df['window_id'] == window_id].copy()
        
        # Remove metadata columns for feature extraction
        metadata_cols = ['window_id', 'prediction_time_sec']
        feature_data = window_data.drop(columns=[c for c in metadata_cols if c in window_data.columns])
        
        # Extract features using original window function
        features = extract_all_features_from_window(feature_data)
        
        # Add metadata back
        features['window_id'] = window_id
        features['prediction_time_sec'] = window_data['prediction_time_sec'].iloc[0]
        
        all_features.append(features)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(window_ids)} windows...")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"\n✓ Extracted {len(features_df.columns)} features from {len(window_ids)} windows")
    print(f"  Output shape: {features_df.shape} (one row per window)")
    
    # Save if output file specified
    if output_file:
        features_df.to_csv(output_file, index=False)
        print(f"✓ Saved features to: {output_file}")
    
    return features_df

def process_windowed_csv(input_file, output_file=None):
    """
    Process windowed CSV but extract features for EACH ROW (preserve temporal resolution)
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION FOR WINDOWED DATA (ROW-BY-ROW)")
    print("="*60)
    print(f"Input file: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Check if this is windowed data
    if 'window_id' not in df.columns:
        print("\nERROR: This doesn't look like windowed data (missing 'window_id' column)")
        return None
    
    # Get metadata columns
    metadata_cols = ['window_id', 'prediction_time_sec']
    
    print("\nExtracting features for each row (preserving all rows)...")
    
    # Process each row individually
    all_features = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # Extract features from this single row (using the row-by-row function)
        feature_data = row.drop(labels=[c for c in metadata_cols if c in row.index])
        features = extract_all_features_from_row(feature_data)  # Your existing row function
        
        # Add metadata back
        features['window_id'] = row['window_id']
        features['prediction_time_sec'] = row['prediction_time_sec'] if 'prediction_time_sec' in row else i
        features['original_row_idx'] = idx
        
        all_features.append(features)
        
        # Progress indicator
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{len(df)} rows...")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"\n✓ Extracted {len(features_df.columns)} features from {len(df)} rows")
    print(f"  Output shape: {features_df.shape} (preserved all rows!)")
    
    # Save if output file specified
    if output_file:
        features_df.to_csv(output_file, index=False)
        print(f"✓ Saved features to: {output_file}")
    
    return features_df

def process_windowed_csv2(input_file, output_file=None):
    """
    Process windowed CSV and extract BOTH window-level stats AND row-specific features for EACH ROW
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION FOR WINDOWED DATA (WINDOW STATS + ROW VALUES)")
    print("="*60)
    print(f"Input file: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Check if this is windowed data
    if 'window_id' not in df.columns:
        print("\nERROR: This doesn't look like windowed data (missing 'window_id' column)")
        return None
    
    # Get metadata columns
    metadata_cols = ['window_id', 'prediction_time_sec']
    
    print("\nExtracting window-level statistics and row-specific features...")
    
    # Dictionary to cache window statistics (avoid recomputing for same window)
    window_stats_cache = {}
    
    # Process each row individually
    all_features = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        window_id = row['window_id']
        
        # Check if we've already computed stats for this window
        if window_id not in window_stats_cache:
            # Get all rows for this window
            window_data = df[df['window_id'] == window_id]
            
            # Remove metadata columns for feature extraction
            feature_window = window_data.drop(columns=[c for c in metadata_cols if c in window_data.columns])
            
            # Extract window-level statistics using your original window function
            window_stats = extract_all_features_from_window(feature_window)
            
            # Cache it
            window_stats_cache[window_id] = window_stats
        
        # Start with window-level stats (same for all rows in this window)
        features = window_stats_cache[window_id].copy()
        
        # Add row-specific instantaneous features
        feature_data = row.drop(labels=[c for c in metadata_cols if c in row.index])
        row_features = extract_all_features_from_row(feature_data)
        
        # Rename row features to avoid confusion (add '_current' suffix)
        row_features_renamed = {f"{k}_current": v for k, v in row_features.items()}
        features.update(row_features_renamed)
        
        # Add metadata back
        features['window_id'] = window_id
        features['prediction_time_sec'] = row['prediction_time_sec'] if 'prediction_time_sec' in row else i
        features['original_row_idx'] = idx
        features['row_position_in_window'] = i - df[df['window_id'] == window_id].index[0] if len(df[df['window_id'] == window_id]) > 0 else 0
        
        all_features.append(features)
        
        # Progress indicator
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{len(df)} rows...")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"\n✓ Extracted {len(features_df.columns)} features from {len(df)} rows")
    print(f"  - Window-level statistical features: {len(window_stats_cache[list(window_stats_cache.keys())[0]]) if window_stats_cache else 0}")
    print(f"  - Row-specific instantaneous features: {len(features_df.columns) - len(window_stats_cache[list(window_stats_cache.keys())[0]]) - 4}")  # minus metadata
    print(f"  Output shape: {features_df.shape} (preserved all rows!)")
    
    # Save if output file specified
    if output_file:
        features_df.to_csv(output_file, index=False)
        print(f"✓ Saved features to: {output_file}")
    
    return features_df

def main():
    parser = argparse.ArgumentParser(
        description='Extract features from windowed NIRS data (output of temporal_overlap.py)'
    )
    parser.add_argument('input_file', help='Windowed CSV file (from temporal_overlap.py)')
    parser.add_argument('--output-dir', default='Feature_Extracted/', 
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
        output_file1 = os.path.join(args.output_dir, f"{base_name}_features.csv")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file1 = os.path.join(args.output_dir, args.output_name)

    if args.output_name2 is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        os.makedirs(args.output_dir, exist_ok=True)
        output_file2 = os.path.join(args.output_dir, f"{base_name}_features2.csv")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file2 = os.path.join(args.output_dir, args.output_name2)    
    
    # Process the file
    features_df = process_windowed_csv(args.input_file, output_file1) #This one does the per row features
    features_df2 = process_windowed_csv_aggregated(args.input_file, output_file2)
    
    if features_df is not None and features_df2 is not None:
        print("\n" + "="*60)
        print("FEATURE EXTRACTION COMPLETE")
        print("="*60)
        #print(f"Aggregated features (one row/window): {process_windowed_csv_aggregated.shape}")
        print(f"  Saved to: {output_file1}")
        #print(f"Row-wise features (all rows): {process_windowed_csv.shape}")
        print(f"  Saved to: {output_file2}")
        print("="*60)



if __name__ == "__main__":
    main()
