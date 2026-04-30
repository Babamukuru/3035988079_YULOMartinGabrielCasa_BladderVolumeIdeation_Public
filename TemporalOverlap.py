#!/usr/bin/env python3
"""
Temporal Overlap for CONTINUOUS prediction (real-time bladder volume estimation)
Creates highly overlapping windows for dense time series predictions
"""

import numpy as np
import pandas as pd
import argparse
import os

class ContinuousWindowingParams:
    """Parameters for continuous prediction (high overlap)"""
    
    # For ~1 second prediction intervals
    WINDOW_LENGTH_SEC = 10.0     # Context window (10 seconds of data)
    PREDICTION_INTERVAL_SEC = 1.0  # How often to predict (every 1 second)
    
    SAMPLE_RATE = None  # Auto-detect from timestamps

def detect_time_unit_and_sample_rate(timestamps):
    """
    Detect the time unit and calculate sample rate from timestamp differences
    """
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    
    if len(diffs) == 0:
        return 1.0, 10.0  # Default
    
    median_diff = np.median(diffs)
    
    # Calculate what 1 second looks like in timestamp units
    # By looking at the time span and number of samples
    total_span = timestamps[-1] - timestamps[0]
    n_samples = len(timestamps)
    
    # Assume the recording has a reasonable duration (not 54 hours)
    # The actual duration in seconds is unknown, but we can calculate sample rate
    # from the fact that sample rate should be consistent
    
    # Method: The median diff in timestamp units corresponds to the sample interval
    # We need to find what 1 timestamp unit equals in seconds
    
    # Let's try to determine if the timestamps are in microseconds (1e-6)
    # Typical NIRS systems sample at 10-100 Hz, so sample interval is 0.01-0.1 seconds
    # If median_diff = 5, then 5 units = ~0.01-0.1 seconds, so 1 unit = 0.002-0.02 seconds
    
    # Check the magnitude of the timestamps
    if timestamps[0] > 1e12:
        # Very large numbers - likely microseconds or nanoseconds
        # Try microsecond assumption first (common in some systems)
        time_unit_seconds = 1e-6  # Assume microseconds
        sample_interval_seconds = median_diff * time_unit_seconds
        
        # Check if sample rate is reasonable (between 1 and 1000 Hz)
        sample_rate = 1 / sample_interval_seconds
        
        if 1 < sample_rate < 10000:
            print(f"  Detected time unit: Microseconds (1 unit = 0.000001 seconds)")
            print(f"  Sample interval: {sample_interval_seconds*1000:.2f} ms")
            print(f"  Sample rate: {sample_rate:.2f} Hz")
            return time_unit_seconds, sample_rate
    
    # If microsecond assumption fails, calculate based on typical sample rates
    # Assume the recording is reasonably short (minutes, not hours)
    # Use the median diff to estimate
    possible_units = [1e-9, 1e-6, 1e-3, 1]  # nano, micro, milli, seconds
    
    for unit in possible_units:
        sample_interval_seconds = median_diff * unit
        sample_rate = 1 / sample_interval_seconds
        if 1 < sample_rate < 10000:  # Reasonable sample rate range
            print(f"  Detected time unit: {unit*1e6:.0f} microseconds" if unit <= 1e-3 else f"  Detected time unit: {unit} seconds")
            print(f"  Sample interval: {sample_interval_seconds*1000:.2f} ms")
            print(f"  Sample rate: {sample_rate:.2f} Hz")
            return unit, sample_rate
    
    # Fallback
    print(f"  Warning: Could not detect time unit, using default 10 Hz")
    return 1.0, 10.0

def create_continuous_windows(data, window_sec, prediction_interval_sec, fs=None):
    """
    Create windows for continuous prediction
    """
    # Auto-detect sample rate from timestamps
    if fs is None and 'timestamp' in data.columns:
        timestamps = data['timestamp'].values
        
        # Detect time unit and calculate sample rate
        time_unit_seconds, fs = detect_time_unit_and_sample_rate(timestamps)
        
        # Convert timestamps to seconds for duration calculation
        timestamps_seconds = timestamps * time_unit_seconds
        
    elif fs is None:
        fs = 10.0  # Default fallback
        print(f"  No timestamp column, using default sample rate: {fs:.2f} Hz")
        timestamps_seconds = np.arange(len(data)) / fs
    
    else:
        if 'timestamp' in data.columns:
            # Use provided fs, but still need timestamps in seconds
            time_unit_seconds, _ = detect_time_unit_and_sample_rate(data['timestamp'].values)
            timestamps_seconds = data['timestamp'].values * time_unit_seconds
        else:
            timestamps_seconds = np.arange(len(data)) / fs
    
    # Calculate recording duration
    total_duration_sec = timestamps_seconds[-1] - timestamps_seconds[0]
    
    # Calculate window parameters in samples
    window_samples = int(window_sec * fs)
    step_samples = max(1, int(prediction_interval_sec * fs))
    
    overlap_ratio = 1 - (step_samples / window_samples) if window_samples > 0 else 0
    
    n_samples = len(data)
    
    print(f"\n  Data diagnostics:")
    print(f"    - Total samples: {n_samples}")
    print(f"    - Total duration: {total_duration_sec:.1f} seconds ({total_duration_sec/60:.1f} minutes)")
    print(f"    - Sample rate: {fs:.2f} Hz")
    print(f"    - Time per sample: {1000/fs:.2f} ms")
    
    print(f"\n  Window parameters for CONTINUOUS prediction:")
    print(f"    - Window length: {window_sec}s ({window_samples} samples)")
    print(f"    - Prediction every: {prediction_interval_sec}s ({step_samples} samples)")
    print(f"    - Overlap: {overlap_ratio*100:.1f}%")
    
    # Check if we can create at least one window
    if window_samples > n_samples:
        print(f"\n  ERROR: Window length ({window_samples} samples) exceeds total data ({n_samples} samples)")
        print(f"  Your recording is {total_duration_sec:.1f} seconds long")
        print(f"  Maximum window length: {total_duration_sec:.1f} seconds")
        print(f"\n  Try: --window-length {total_duration_sec/2:.1f}")
        return [], [], fs
    
    windows = []
    window_metadata = []
    
    start = 0
    window_id = 0
    
    expected_predictions = int((n_samples - window_samples) / step_samples) + 1
    print(f"\n  Creating {expected_predictions} windows...")
    
    while start + window_samples <= n_samples:
        end = start + window_samples
        
        # Extract window
        window_data = data.iloc[start:end].copy()
        
        # The prediction time is the END of the window (in seconds)
        prediction_time = timestamps_seconds[end-1]
        window_start_time = timestamps_seconds[start]
        
        # Add metadata as columns
        window_data['window_id'] = window_id
        window_data['prediction_time_sec'] = prediction_time
        
        windows.append(window_data)
        window_metadata.append({
            'window_id': window_id,
            'prediction_time_sec': prediction_time,
            'window_start_sec': window_start_time,
            'window_end_sec': prediction_time,
            'samples_in_window': len(window_data)
        })
        
        start += step_samples
        window_id += 1
        
        # Progress indicator
        if window_id % 100 == 0:
            print(f"    Created {window_id}/{expected_predictions} windows...")
    
    total_predictions = len(windows)
    
    print(f"\n  ✓ Created {total_predictions} prediction windows")
    print(f"    - Prediction every {prediction_interval_sec:.1f} second")
    print(f"    - Time range: {window_metadata[0]['prediction_time_sec']:.1f} to {window_metadata[-1]['prediction_time_sec']:.1f} seconds")
    
    return windows, window_metadata, fs

# Instead of saving 1000 separate files, save ONE file with a window_id column
def save_windows_as_single_file(windows, metadata, output_file):
    """
    Save all windows as a single CSV with window_id column
    """
    all_windows = []
    
    for window_id, window in enumerate(windows):
        # Add window identifier
        window_copy = window.copy()
        window_copy['window_id'] = window_id
        window_copy['prediction_time_sec'] = metadata[window_id]['prediction_time_sec']
        all_windows.append(window_copy)
    
    # Concatenate all windows
    combined_df = pd.concat(all_windows, ignore_index=True)
    
    # Save single file
    combined_df.to_csv(output_file, index=False)
    print(f"  Saved {len(windows)} windows to single file: {output_file}")
    print(f"  File size: {combined_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(
        description='Create overlapping windows for CONTINUOUS prediction'
    )
    parser.add_argument('input_file', help='Cleaned CSV file')
    parser.add_argument('--output-dir', default='TO_Data/', help='Output directory')
    parser.add_argument('--window-length', type=float, default=10.0, 
                       help='Window length in seconds (default: 10.0)')
    parser.add_argument('--prediction-interval', type=float, default=1.0,
                       help='Predict every N seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    filename = os.path.splitext(os.path.basename(args.input_file))[0]
    
    print("\n" + "="*60)
    print("CONTINUOUS PREDICTION PIPELINE (High Overlap)")
    print("="*60)
    print(f"Input: {args.input_file}")
    
    # Load data
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples")
    
    # Create windows
    windows, metadata, fs = create_continuous_windows(
        df, 
        args.window_length, 
        args.prediction_interval
    )
    
    if len(windows) == 0:
        print("\nERROR: No windows created!")
        return
    
    # Save windows
    output_dir = os.path.join(args.output_dir, f"{filename}_continuous_windows")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n  Saving {len(windows)} windows to: {output_dir}")
    for i, window in enumerate(windows):
        window_file = os.path.join(output_dir, f"pred_window_{i:05d}.csv")
        window.to_csv(window_file, index=False)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_file = os.path.join(args.output_dir, f"{filename}_prediction_times.csv")
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"\n  ✓ Saved metadata to: {metadata_file}")
    print(f"\n  Total training examples: {len(windows)}")
    print("="*60)

    output_file = os.path.join(args.output_dir, f"{filename}_windows.csv")
    all_windows = []
    for i, (window, meta) in enumerate(zip(windows, metadata)):
            # Add window metadata to each row
        window_copy = window.copy()
        window_copy['window_id'] = i
        window_copy['prediction_time_sec'] = meta['prediction_time_sec']
        all_windows.append(window_copy)
        
        # Concatenate all windows
    combined_df = pd.concat(all_windows, ignore_index=True)
        
        # Save single file
    combined_df.to_csv(output_file, index=False)
        
        # Calculate file size
    file_size_mb = os.path.getsize(output_file) / 1e6
    print(f"  ✓ Saved to: {output_file}")
    print(f"    File size: {file_size_mb:.1f} MB")
    print(f"    Total rows: {len(combined_df)}")
    print(f"    Total columns: {len(combined_df.columns)}")    

if __name__ == "__main__":
    main()