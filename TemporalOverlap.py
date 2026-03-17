"""
Temporal Overlap (Windowing) for NIRS Data
Creates overlapping windows from continuous time series
"""

import numpy as np
import pandas as pd

def create_overlapping_windows(data, window_sec, overlap_ratio=0.5, fs=10.2):
    """
    Create overlapping windows from continuous time series data
    
    Parameters:
    -----------
    data : DataFrame or array
        Time series data (samples × features)
    window_sec : float
        Window length in seconds [YOUR VALUE: ______]
    overlap_ratio : float
        Overlap between windows (0 to 1) [YOUR VALUE: ______]
    fs : float
        Sampling frequency in Hz [YOUR VALUE: ______]
        
    Returns:
    --------
    windows : list of DataFrames/arrays
        List of windowed data segments
    window_indices : list of tuples
        Start and end indices for each window
    """
    # Convert to numpy if DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        index = data.index
        columns = data.columns
    else:
        data_array = data
        index = None
        columns = None
    
    # Calculate window parameters
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap_ratio))
    
    if step_samples < 1:
        raise ValueError("Overlap too high: step size would be zero")
    
    n_samples = len(data_array)
    windows = []
    window_indices = []
    
    # Create windows
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        
        # Extract window
        if isinstance(data, pd.DataFrame):
            window = data.iloc[start:end]
        else:
            window = data_array[start:end]
        
        windows.append(window)
        window_indices.append((start, end))
        
        # Move to next window
        start += step_samples
    
    print(f"Created {len(windows)} windows of {window_sec}s with {overlap_ratio*100:.0f}% overlap")
    
    return windows, window_indices


def extract_window_features(windows, feature_func=None):
    """
    Extract features from each window
    """
    if feature_func is None:
        # Default: basic statistics
        feature_list = []
        
        for i, window in enumerate(windows):
            if isinstance(window, pd.DataFrame):
                stats = {
                    'window_id': i,
                    'window_start': window.index[0] if hasattr(window, 'index') else i,
                }
                
                for col in window.columns:
                    stats[f'{col}_mean'] = window[col].mean()
                    stats[f'{col}_std'] = window[col].std()
                    stats[f'{col}_min'] = window[col].min()
                    stats[f'{col}_max'] = window[col].max()
                
                feature_list.append(stats)
            else:
                stats = {
                    'window_id': i,
                    'mean': np.mean(window),
                    'std': np.std(window),
                    'min': np.min(window),
                    'max': np.max(window)
                }
                feature_list.append(stats)
        
        features_df = pd.DataFrame(feature_list)
    
    else:
        features = [feature_func(w) for w in windows]
        features_df = pd.DataFrame(features)
    
    return features_df

