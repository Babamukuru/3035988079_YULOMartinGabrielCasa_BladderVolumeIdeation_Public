#!/usr/bin/env python3
"""
Bladder Volume NIRS Data Preprocessing Pipeline
Handles both 16-channel and 4-channel optical data formats
"""

import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import UnivariateSpline
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETER CONFIGURATION SECTION - ADJUST THESE AS NEEDED
# ============================================================================

class PreprocessingParams:
    """Centralized parameters for preprocessing pipeline"""
    
    # Channel filtering parameters
    MIN_INTENSITY_THRESHOLD = 0  # Minimum mean intensity to keep channel
    CHANNEL_REMOVAL_THRESHOLD = 0.9  # Remove if >50% of channels fail threshold
    
    # Wavelet filtering parameters
    WAVELET_FAMILY = 'db4'  # Daubechies 4 wavelet
    WAVELET_LEVEL = 5  # Decomposition level
    WAVELET_THRESHOLD_METHOD = 'soft'  # 'soft' or 'hard' thresholding
    
    # Spline correction parameters
    SPLINE_SMOOTHING_FACTOR = 0.01  # s parameter for UnivariateSpline
    SPLINE_PERCENTILE = 10  # Percentile for baseline estimation
    
    # Kalman filtering parameters
    KALMAN_PROCESS_NOISE = 0.001  # Q - process noise covariance
    KALMAN_MEASUREMENT_NOISE = 0.1  # R - measurement noise covariance
    KALMAN_INITIAL_ERROR = 1.0  # P0 - initial error covariance
    
    # Optical density conversion parameters
    PATH_LENGTH = 1.0  # Path length factor (adjust based on your setup)
    DPF = 6.0  # Differential pathlength factor (typical for tissue)
    
    # Bandpass filtering parameters (in Hz, assuming sample rate)
    SAMPLE_RATE = 10.0  # Hz - adjust based on your acquisition rate
    LOWPASS_CUTOFF = 0.67  # Hz - physiological signals typically <0.5 Hz
    HIGHPASS_CUTOFF = 0.001  # Hz - remove very slow drifts
    FILTER_ORDER = 2  # Butterworth filter order
    
    # Beer-Lambert parameters (extinction coefficients in mM^-1 cm^-1)
    # Wavelengths: 730nm and 850nm
    EXTINCTION_HBO_730 = 0.38  # HbO extinction at 730nm
    EXTINCTION_HBR_730 = 0.86  # HbR extinction at 730nm
    EXTINCTION_HBO_850 = 0.87  # HbO extinction at 850nm
    EXTINCTION_HBR_850 = 0.39  # HbR extinction at 850nm

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_and_validate_data(filepath):
    """
    Load CSV and identify data format (16 or 4 channels)
    Returns: dataframe, filename, channel_count, channel_names
    """
    # Extract filename without extension for output naming
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Identify timestamp column (assume first column if named 'timestamp')
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        timestamp_col = df.columns[0]
    
    # Identify optical channels (exclude timestamp and label columns)
    label_cols = [col for col in df.columns if 'label' in col.lower() or 'notes' in col.lower()]
    optical_cols = [col for col in df.columns if col not in [timestamp_col] + label_cols]
    
    channel_count = len(optical_cols)
    
    print(f"Loaded data from {filename}")
    print(f"  - {channel_count} optical channels detected")
    print(f"  - Data shape: {df.shape}")
    print(f"  - Time range: {df[timestamp_col].iloc[0]} to {df[timestamp_col].iloc[-1]}")
    
    # Validate channel count
    if channel_count not in [4, 16]:
        print(f"  WARNING: Expected 4 or 16 channels, found {channel_count}")
    
    return df, filename, timestamp_col, optical_cols, channel_count

# ============================================================================
# STEP 1: MEAN INTENSITY THRESHOLDING
# ============================================================================

def channel_intensity_filtering(df, optical_cols, params):
    """
    Calculate mean intensity per channel and exclude channels below threshold
    Returns: filtered dataframe, list of kept channels, removal report
    """
    optical_cols = list(optical_cols)
    mean_intensities = df[optical_cols].mean()
    
    # Identify channels to keep
    keep_mask = mean_intensities >= params.MIN_INTENSITY_THRESHOLD
    channels_to_keep = [optical_cols[i] for i in range(len(optical_cols)) if keep_mask.iloc[i]]
    channels_to_remove = [optical_cols[i] for i in range(len(optical_cols)) if not keep_mask.iloc[i]]
    
    # Check if too many channels are failing
    removal_ratio = len(channels_to_remove) / len(optical_cols)
    
    if removal_ratio > params.CHANNEL_REMOVAL_THRESHOLD:
        print(f"  WARNING: {removal_ratio*100:.1f}% of channels below threshold")
        print(f"  Consider lowering MIN_INTENSITY_THRESHOLD ({params.MIN_INTENSITY_THRESHOLD})")
    
    print(f"  Kept {len(channels_to_keep)}/{len(optical_cols)} channels")
    if len(channels_to_remove) > 0:
        print(f"  Removed channels (mean intensity): {channels_to_remove}")
        print(f"    Mean intensities: {mean_intensities[channels_to_remove].to_dict()}")
    
    # Create filtered dataframe
    df_filtered = df[list(channels_to_keep) + ['timestamp']].copy()
    
    return df_filtered, channels_to_keep, channels_to_remove

# ============================================================================
# STEP 2: WAVELET FILTERING
# ============================================================================

def wavelet_filter(data, params):
    """
    Apply wavelet thresholding for noise reduction
    """
    coeffs = pywt.wavedec(data, params.WAVELET_FAMILY, level=params.WAVELET_LEVEL)
    
    # Apply thresholding to detail coefficients
    coeffs_thresh = list(coeffs)
    for i in range(1, len(coeffs_thresh)):
        sigma = np.median(np.abs(coeffs_thresh[i])) / 0.6745  # MAD estimator
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        if params.WAVELET_THRESHOLD_METHOD == 'soft':
            coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='soft')
        else:  # hard thresholding
            coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='hard')
    
    # Reconstruct signal
    filtered_data = pywt.waverec(coeffs_thresh, params.WAVELET_FAMILY)
    
    # Ensure same length (sometimes reconstruction adds a sample)
    if len(filtered_data) > len(data):
        filtered_data = filtered_data[:len(data)]
    
    return filtered_data

# ============================================================================
# STEP 2: SPLINE CORRECTION FOR BASELINE DRIFT
# ============================================================================

def spline_baseline_correction(data, params):
    """
    Estimate baseline using spline fitting and subtract it
    """
    x = np.arange(len(data))
    
    # Find baseline points (e.g., local minima or percentile values)
    window_size = int(len(data) / 10)  # Adaptive window size
    baseline_points = []
    
    for i in range(0, len(data), window_size):
        segment = data[i:min(i+window_size, len(data))]
        if len(segment) > 0:
            baseline_val = np.percentile(segment, params.SPLINE_PERCENTILE)
            baseline_points.append(baseline_val)
    
    # Create spline through baseline points
    baseline_x = np.linspace(0, len(data)-1, len(baseline_points))
    spline = UnivariateSpline(baseline_x, baseline_points, s=params.SPLINE_SMOOTHING_FACTOR)
    baseline_estimate = spline(x)
    
    # Subtract baseline
    corrected_data = data - baseline_estimate
    
    return corrected_data

# ============================================================================
# STEP 2: KALMAN FILTER FOR MOTION ARTIFACT ATTENUATION
# ============================================================================

def kalman_filter(data, params):
    """
    Apply Kalman filter to reduce motion artifacts
    """
    n_samples = len(data)
    
    # Initialize Kalman filter
    x_est = data[0]  # Initial state estimate
    P_est = params.KALMAN_INITIAL_ERROR  # Initial error covariance
    
    filtered_data = np.zeros(n_samples)
    filtered_data[0] = x_est
    
    for k in range(1, n_samples):
        # Prediction step
        x_pred = x_est  # Assume constant state (no motion model)
        P_pred = P_est + params.KALMAN_PROCESS_NOISE
        
        # Update step
        K = P_pred / (P_pred + params.KALMAN_MEASUREMENT_NOISE)  # Kalman gain
        x_est = x_pred + K * (data[k] - x_pred)
        P_est = (1 - K) * P_pred
        
        filtered_data[k] = x_est
    
    return filtered_data

# ============================================================================
# STEP 2: COMBINED MOTION ARTIFACT ATTENUATION
# ============================================================================

def motion_artifact_attenuation(df, channel_cols, params):
    """
    Apply wavelet filtering, spline correction, and Kalman filtering
    """
    df_motion_corrected = df.copy()
    
    for col in channel_cols:
        data = df[col].values
        
        # Step 2a: Wavelet filtering
        wavelet_filtered = wavelet_filter(data, params)
        
        # Step 2b: Spline baseline correction
        spline_corrected = spline_baseline_correction(wavelet_filtered, params)
        
        # Step 2c: Kalman filtering
        kalman_filtered = kalman_filter(spline_corrected, params)
        
        df_motion_corrected[col] = kalman_filtered
    
    return df_motion_corrected

# ============================================================================
# STEP 3: CONVERT TO OPTICAL DENSITY
# ============================================================================

def convert_to_optical_density(df, channel_cols, params):
    """
    Convert intensity data to optical density (OD)
    OD = -log10(I / I0) where I0 is baseline intensity
    """
    df_od = df.copy()
    
    for col in channel_cols:
        # Use mean of first 10% of data as baseline intensity (I0)
        baseline_samples = int(len(df) * 0.1)
        I0 = np.mean(df[col].iloc[:baseline_samples])
        
        # Avoid log(0) or negative values
        I = np.maximum(df[col].values, 1e-10)
        I0 = max(I0, 1e-10)
        
        # Calculate OD
        od_values = -np.log10(I / I0)
        df_od[col] = od_values
    
    return df_od

# ============================================================================
# STEP 4: BANDPASS FILTERING
# ============================================================================

def butter_bandpass(lowcut, highcut, fs, order):
    """Design Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, params):
    """
    Apply bandpass filter to remove high-frequency noise and low-frequency drift
    """
    b, a = butter_bandpass(params.HIGHPASS_CUTOFF, params.LOWPASS_CUTOFF, 
                          params.SAMPLE_RATE, params.FILTER_ORDER)
    
    # Apply zero-phase filtering to avoid phase shift
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def apply_bandpass_filtering(df, channel_cols, params):
    """
    Apply bandpass filtering to all optical channels
    """
    df_filtered = df.copy()
    
    for col in channel_cols:
        df_filtered[col] = bandpass_filter(df[col].values, params)
    
    return df_filtered

# ============================================================================
# STEP 5: BEER-LAMBERT CONVERSION TO HbO/HbR
# ============================================================================

def beer_lambert_conversion(df_od_730, df_od_850, params):
    """
    Convert optical density to hemoglobin concentrations using modified Beer-Lambert law
    Returns: HbO and HbR concentrations
    """
    # Modified Beer-Lambert law:
    # OD = (ε_HbO * C_HbO + ε_HbR * C_HbR) * L * DPF
    # Where L is path length, DPF is differential pathlength factor
    
    # Solve linear system for each time point
    # [OD_730]   [ε_HbO_730  ε_HbR_730] [C_HbO]
    # [OD_850] = [ε_HbO_850  ε_HbR_850] [C_HbR] * (L * DPF)
    
    # Create extinction matrix
    epsilon = np.array([
        [params.EXTINCTION_HBO_730, params.EXTINCTION_HBR_730],
        [params.EXTINCTION_HBO_850, params.EXTINCTION_HBR_850]
    ])
    
    # Path length factor
    path_factor = params.PATH_LENGTH * params.DPF
    
    # Solve for concentrations
    epsilon_inv = np.linalg.inv(epsilon)
    
    n_samples = len(df_od_730)
    hbo = np.zeros(n_samples)
    hbr = np.zeros(n_samples)
    
    for i in range(n_samples):
        od_vector = np.array([df_od_730.iloc[i], df_od_850.iloc[i]])
        conc = np.dot(epsilon_inv, od_vector) / path_factor
        hbo[i] = conc[0]
        hbr[i] = conc[1]
    
    return hbo, hbr

def process_beer_lambert(df, channel_cols, params):
    """
    Process Beer-Lambert conversion for all channels
    For each optical channel pair (730nm and 850nm at same location)
    """
    # Determine channel pairs based on naming convention
    # Expecting channels like: '730nm left outer', '850nm left outer', etc.
    
    hbo_columns = []
    hbr_columns = []
    
    # Group channels by location and type (left/right, inner/outer)
    channel_pairs = [
        ('left_outer', 'optics1_uA', 'optics3_uA'),   # 730nm left outer + 850nm left outer
        ('right_outer', 'optics2_uA', 'optics4_uA'),  # 730nm right outer + 850nm right outer
        ('left_inner', 'optics5_uA', 'optics7_uA'),   # 730nm left inner + 850nm left inner
        ('right_inner', 'optics6_uA', 'optics8_uA'),  # 730nm right inner + 850nm right inner
    ]
    
    print("  Looking for channel pairs:")
    
    # Process each pair
    for location, channel_730, channel_850 in channel_pairs:
        # Check if both channels exist in the dataframe
        if channel_730 in df.columns and channel_850 in df.columns:
            print(f"    Found pair: {channel_730} (730nm) + {channel_850} (850nm) -> {location}")
            
            # Convert to HbO/HbR
            hbo, hbr = beer_lambert_conversion(
                df[channel_730], df[channel_850], params
            )
            
            # Store results
            hbo_col = f'{location}_HbO'
            hbr_col = f'{location}_HbR'
            df[hbo_col] = hbo
            df[hbr_col] = hbr
            hbo_columns.append(hbo_col)
            hbr_columns.append(hbr_col)
        else:
            print(f"    WARNING: Missing channels for {location}")
            if channel_730 not in df.columns:
                print(f"      Missing: {channel_730}")
            if channel_850 not in df.columns:
                print(f"      Missing: {channel_850}")
    
    # OPTIONAL: If you want to also process the 4-channel mode data
    # (for when you have 4-channel files)
    if len(df.columns) == 4:  # 4-channel mode detection
        print("  Detected 4-channel mode data")
        four_channel_pairs = [
            ('left', 'left_730', 'left_850'),
            ('right', 'right_730', 'right_850'),
        ]
        # Add logic here if needed
    
    return df, hbo_columns, hbr_columns

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Preprocess bladder volume NIRS data (16 or 4 channels)'
    )
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output-dir', default='Preprocessed_Data/', 
                       help='Output directory (default: current directory)')
    parser.add_argument('--sample-rate', type=float, 
                       help='Override sample rate (Hz)')
    
    args = parser.parse_args()
    
    # Initialize parameters
    params = PreprocessingParams()
    
    # Override sample rate if provided
    if args.sample_rate:
        params.SAMPLE_RATE = args.sample_rate
        print(f"Using sample rate: {params.SAMPLE_RATE} Hz")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("STEP 0: LOADING DATA")
    print("="*60)
    df, filename, timestamp_col, optical_cols, channel_count = load_and_validate_data(args.input_file)
    
    # Create output filename
    output_file = os.path.join(args.output_dir, f"{filename}.cleaned.csv")
    
    # STEP 1: Channel intensity filtering
    print("\n" + "="*60)
    print("STEP 1: CHANNEL INTENSITY FILTERING")
    print("="*60)
    df_filtered, kept_channels, removed_channels = channel_intensity_filtering(
        df, optical_cols, params
    )
    
    # STEP 2: Motion artifact attenuation
    print("\n" + "="*60)
    print("STEP 2: MOTION ARTIFACT ATTENUATION")
    print("  - Wavelet filtering")
    print("  - Spline baseline correction")
    print("  - Kalman filtering")
    print("="*60)
    df_motion_corrected = motion_artifact_attenuation(df_filtered, kept_channels, params)
    
    # STEP 3: Convert to Optical Density
    print("\n" + "="*60)
    print("STEP 3: CONVERT TO OPTICAL DENSITY")
    print("="*60)
    df_od = convert_to_optical_density(df_motion_corrected, kept_channels, params)
    
    # STEP 4: Bandpass filtering
    print("\n" + "="*60)
    print("STEP 4: BANDPASS FILTERING")
    print(f"  - Highpass cutoff: {params.HIGHPASS_CUTOFF} Hz")
    print(f"  - Lowpass cutoff: {params.LOWPASS_CUTOFF} Hz")
    print("="*60)
    df_bandpass = apply_bandpass_filtering(df_od, kept_channels, params)
    
    # STEP 5: Beer-Lambert conversion
    print("\n" + "="*60)
    print("STEP 5: BEER-LAMBERT CONVERSION (HbO/HbR)")
    print("="*60)
    df_final, hbo_cols, hbr_cols = process_beer_lambert(df_bandpass, kept_channels, params)
    
    # Ensure timestamp column is preserved
    df_final[timestamp_col] = df[timestamp_col].values
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    df_final.to_csv(output_file, index=False)
    print(f"Saved cleaned data to: {output_file}")
    print(f"  - {len(kept_channels)} optical channels")
    print(f"  - {len(hbo_cols)} HbO channels")
    print(f"  - {len(hbr_cols)} HbR channels")
    print(f"  - Total columns: {len(df_final.columns)}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Channels processed: {len(kept_channels)}")
    print(f"Channels removed: {len(removed_channels)}")
    
    if len(hbo_cols) > 0:
        print("\nHbO Statistics (first channel):")
        print(f"  Mean: {df_final[hbo_cols[0]].mean():.4f}")
        print(f"  Std: {df_final[hbo_cols[0]].std():.4f}")
        print(f"  Range: [{df_final[hbo_cols[0]].min():.4f}, {df_final[hbo_cols[0]].max():.4f}]")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
