"""
NIRS Preprocessing Pipeline with Kalman Filtering
Handles: quality check, motion correction, OD conversion, filtering, Beer-Lambert
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.linalg import solve_discrete_lyapunov
import pywt
from filterpy.kalman import KalmanFilter  # You'll need: pip install filterpy


class NIRSPreprocessor:
    """
    Flexible NIRS preprocessing class adaptable to different channel configurations
    
    Parameters (Complete List - Fill These In!)
    --------------------------------------------
    fs : float
        Sampling frequency in Hz [DEFAULT: 10.2, YOUR VALUE: ______]
    wavelengths : list
        Measurement wavelengths in nm [DEFAULT: [730, 850], YOUR VALUE: ______]
    source_detector_dist : float
        Source-detector separation in mm [DEFAULT: 30, YOUR VALUE: ______]
    intensity_threshold : float
        Minimum acceptable raw intensity [DEFAULT: 0.05, YOUR VALUE: ______]
    impedance_threshold : float
        Maximum acceptable impedance in kOhm [DEFAULT: 5, YOUR VALUE: ______]
    baseline_sec : float
        Seconds used for baseline calculation [DEFAULT: 30, YOUR VALUE: ______]
    lowcut : float
        Bandpass low cutoff in Hz [DEFAULT: 0.01, YOUR VALUE: ______]
    highcut : float
        Bandpass high cutoff in Hz [DEFAULT: 0.5, YOUR VALUE: ______]
    filter_order : int
        Butterworth filter order [DEFAULT: 3, YOUR VALUE: ______]
    wavelet : str
        Wavelet type for decomposition [DEFAULT: 'db4', YOUR VALUE: ______]
    wavelet_level : int
        Wavelet decomposition level [DEFAULT: 4, YOUR VALUE: ______]
    wavelet_threshold_factor : float
        Threshold multiplier for wavelet [DEFAULT: 2.5, YOUR VALUE: ______]
    artifact_std_threshold : float
        Std multiplier for artifact detection [DEFAULT: 3.0, YOUR VALUE: ______]
    spline_smoothing : float
        Spline smoothing factor (0=exact) [DEFAULT: 0, YOUR VALUE: ______]
    kalman_Q : float
        Kalman process noise covariance [DEFAULT: 1e-4, YOUR VALUE: ______]
    kalman_R : float
        Kalman measurement noise covariance [DEFAULT: 0.01, YOUR VALUE: ______]
    kalman_model_order : int
        Kalman AR model order [DEFAULT: 2, YOUR VALUE: ______]
    kalman_P_init : float
        Kalman initial state covariance [DEFAULT: 1.0, YOUR VALUE: ______]
    dpf_730 : float
        DPF at 730nm for abdominal tissue [DEFAULT: 5.0, YOUR VALUE: ______]
    dpf_850 : float
        DPF at 850nm for abdominal tissue [DEFAULT: 4.5, YOUR VALUE: ______]
    """
    
    def __init__(self, 
                 # System parameters
                 fs=10.2,
                 wavelengths=[730, 850],
                 source_detector_dist=30,
                 
                 # Quality check parameters
                 intensity_threshold=0.05,
                 impedance_threshold=5,
                 baseline_sec=30,
                 
                 # Filter parameters
                 lowcut=0.01,
                 highcut=0.5,
                 filter_order=3,
                 
                 # Wavelet parameters
                 wavelet='db4',
                 wavelet_level=4,
                 wavelet_threshold_factor=2.5,
                 
                 # Spline parameters
                 artifact_std_threshold=3.0,
                 spline_smoothing=0,
                 
                 # Kalman parameters
                 kalman_Q=1e-4,
                 kalman_R=0.01,
                 kalman_model_order=2,
                 kalman_P_init=1.0,
                 
                 # Beer-Lambert parameters
                 dpf_730=5.0,  # PLACEHOLDER - VERIFY FOR ABDOMEN!
                 dpf_850=4.5,  # PLACEHOLDER - VERIFY FOR ABDOMEN!
                 
                 # Additional settings
                 verbose=True):
        
        # Store all parameters
        self.fs = fs
        self.wavelengths = wavelengths
        self.dist = source_detector_dist
        self.intensity_threshold = intensity_threshold
        self.impedance_threshold = impedance_threshold
        self.baseline_sec = baseline_sec
        self.baseline_samples = int(fs * baseline_sec)
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_threshold_factor = wavelet_threshold_factor
        self.artifact_std_threshold = artifact_std_threshold
        self.spline_smoothing = spline_smoothing
        self.kalman_Q = kalman_Q
        self.kalman_R = kalman_R
        self.kalman_model_order = kalman_model_order
        self.kalman_P_init = kalman_P_init
        self.verbose = verbose
        
        # DPF values (wavelength specific)
        self.dpf = {730: dpf_730, 850: dpf_850}
        
        # Standard extinction coefficients [cm⁻¹/mM]
        # Source: [Your citation here]
        self.extinction = {
            730: {'HbO': 0.52, 'HbR': 1.28},
            850: {'HbO': 1.14, 'HbR': 0.79}
        }
        
        if self.verbose:
            self._print_parameters()
    
    def _print_parameters(self):
        """Print all current parameters for verification"""
        print("\n" + "="*60)
        print("NIRS PREPROCESSOR CONFIGURATION")
        print("="*60)
        print(f"\nSYSTEM PARAMETERS:")
        print(f"  fs: {self.fs} Hz")
        print(f"  wavelengths: {self.wavelengths} nm")
        print(f"  source-detector distance: {self.dist} mm")
        
        print(f"\nQUALITY CHECK:")
        print(f"  intensity_threshold: {self.intensity_threshold}")
        print(f"  impedance_threshold: {self.impedance_threshold} kOhm")
        print(f"  baseline_sec: {self.baseline_sec} s")
        
        print(f"\nFILTERING:")
        print(f"  bandpass: {self.lowcut}-{self.highcut} Hz")
        print(f"  filter_order: {self.filter_order}")
        
        print(f"\nWAVELET:")
        print(f"  type: {self.wavelet}")
        print(f"  level: {self.wavelet_level}")
        print(f"  threshold_factor: {self.wavelet_threshold_factor}")
        
        print(f"\nSPLINE:")
        print(f"  artifact_std_threshold: {self.artifact_std_threshold}")
        print(f"  spline_smoothing: {self.spline_smoothing}")
        
        print(f"\nKALMAN:")
        print(f"  Q (process noise): {self.kalman_Q}")
        print(f"  R (measurement noise): {self.kalman_R}")
        print(f"  model_order: {self.kalman_model_order}")
        print(f"  P_init: {self.kalman_P_init}")
        
        print(f"\nBEER-LAMBERT:")
        print(f"  DPF 730nm: {self.dpf[730]}")
        print(f"  DPF 850nm: {self.dpf[850]}")
        print(f"  Extinction coefficients: from literature")
        print("="*60 + "\n")
    
    def quality_check(self, raw_intensity_df, impedance_df=None):
        """
        Check channel quality based on raw intensity and impedance
        """
        # Calculate mean intensity during baseline period
        baseline_data = raw_intensity_df.iloc[:self.baseline_samples]
        mean_intensity = baseline_data.mean()
        
        # Find channels above threshold
        intensity_ok = mean_intensity[mean_intensity > self.intensity_threshold].index.tolist()
        
        # Cross-reference with impedance if available
        if impedance_df is not None:
            impedance_ok = impedance_df[impedance_df < self.impedance_threshold].index.tolist()
            good_channels = list(set(intensity_ok) & set(impedance_ok))
            
            if self.verbose:
                print(f"Quality check: {len(good_channels)}/{len(raw_intensity_df.columns)} channels passed")
                print(f"  - Passed intensity: {len(intensity_ok)}")
                print(f"  - Passed impedance: {len(impedance_ok)}")
        else:
            good_channels = intensity_ok
            if self.verbose:
                print(f"Quality check: {len(good_channels)}/{len(raw_intensity_df.columns)} channels passed (intensity only)")
        
        return good_channels
    
    def wavelet_filter(self, data):
        """
        Wavelet-based motion artifact filtering
        
        The wavelet transform decomposes the signal into different frequency bands.
        Motion artifacts appear as large coefficients that are outliers compared to
        the expected Gaussian distribution of clean physiological signals. By
        thresholding these outliers and reconstructing, we remove artifacts while
        preserving the underlying signal.
        """
        # Decompose signal
        coeffs = pywt.wavedec(data, self.wavelet, level=self.wavelet_level)
        
        # Calculate threshold using median absolute deviation (MAD)
        # MAD is robust to outliers, making it ideal for artifact detection
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = self.wavelet_threshold_factor * sigma
        
        # Apply soft thresholding to detail coefficients (skip approximation)
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs_thresh)):
            coeffs_thresh[i] = pywt.threshold(
                coeffs_thresh[i], 
                threshold, 
                mode='soft'  # Soft thresholding preserves continuity
            )
        
        # Reconstruct signal
        cleaned = pywt.waverec(coeffs_thresh, self.wavelet)
        
        # Trim to original length (wavelet reconstruction may add samples)
        return cleaned[:len(data)]
    
    def spline_correction(self, data):
        """
        Spline interpolation for baseline shift correction
        
        This method identifies segments where motion artifacts cause baseline shifts,
        then uses clean surrounding data to interpolate over the corrupted region.
        The cubic spline ensures smooth transitions between clean and corrected segments.
        """
        from scipy.interpolate import UnivariateSpline
        
        # Step 1: Detect artifacts using moving standard deviation
        window = int(self.fs * 2)  # 2-second window
        moving_std = pd.Series(data).rolling(window, center=True, min_periods=1).std().values
        
        # Identify artifact regions (where std exceeds threshold)
        artifact_mask = moving_std > (self.artifact_std_threshold * np.nanstd(data))
        artifact_indices = np.where(artifact_mask)[0]
        
        # If no artifacts, return original
        if len(artifact_indices) == 0:
            return data
        
        # Find clean segments (non-artifact)
        clean_indices = np.setdiff1d(np.arange(len(data)), artifact_indices)
        
        if len(clean_indices) < self.kalman_model_order + 2:
            if self.verbose:
                print("Warning: Insufficient clean data for spline correction")
            return data
        
        # Fit spline to clean data
        # s=self.spline_smoothing: 0 = exact interpolation through points
        spline = UnivariateSpline(
            clean_indices, 
            data[clean_indices], 
            k=3,  # cubic spline
            s=self.spline_smoothing
        )
        
        # Replace artifact segments with spline fit
        corrected = data.copy()
        corrected[artifact_indices] = spline(artifact_indices)
        
        if self.verbose:
            print(f"  Spline corrected {len(artifact_indices)} artifact samples ({100*len(artifact_indices)/len(data):.1f}%)")
        
        return corrected
    
    def kalman_filter(self, data):
        """
        Kalman filtering for adaptive motion artifact attenuation
        
        The Kalman filter operates in two steps:
        1. PREDICT: Use an AR model to predict the next state based on previous states
        2. UPDATE: Combine prediction with new measurement, weighted by uncertainties
        
        This creates an optimal estimate that adapts to signal characteristics while
        rejecting artifacts that don't match the expected dynamics.
        
        Mathematical basis:
        - State vector: x[k] = [signal, signal_derivative, signal_derivative2, ...]
        - Prediction: x[k|k-1] = A * x[k-1|k-1]
        - Update: x[k|k] = x[k|k-1] + K * (measurement - H * x[k|k-1])
        
        Where K (Kalman gain) balances trust in prediction vs measurement based on
        process noise (Q) and measurement noise (R).
        """
        n = len(data)
        filtered = np.zeros(n)
        
        # For short segments, return original
        if n < self.kalman_model_order + 5:
            return data
        
        # Step 1: Fit AR model to initial clean segment
        # We use the first 10 seconds of data to estimate signal dynamics
        init_samples = min(int(10 * self.fs), n // 4)
        
        # Build AR model using Yule-Walker equations
        from scipy.linalg import toeplitz
        from scipy.signal import lfilter
        
        # Estimate autocorrelation
        init_data = data[:init_samples]
        acf = np.correlate(init_data - init_data.mean(), 
                           init_data - init_data.mean(), 
                           mode='full')
        acf = acf[len(acf)//2:] / len(init_data)
        
        # Solve Yule-Walker for AR coefficients
        if len(acf) > self.kalman_model_order:
            R = toeplitz(acf[:self.kalman_model_order])
            r = acf[1:self.kalman_model_order+1]
            try:
                ar_coeffs = np.linalg.solve(R, r)
            except np.linalg.LinAlgError:
                # Fallback to simple model if matrix is singular
                ar_coeffs = np.zeros(self.kalman_model_order)
                ar_coeffs[0] = 0.9
        else:
            ar_coeffs = np.zeros(self.kalman_model_order)
            ar_coeffs[0] = 0.9
        
        # Step 2: Build state-space matrices
        # State vector includes current and previous values
        dim = self.kalman_model_order
        
        # State transition matrix (A)
        A = np.zeros((dim, dim))
        A[0, :] = ar_coeffs
        if dim > 1:
            A[1:, :-1] = np.eye(dim-1)
        
        # Measurement matrix (H) - we observe the first state
        H = np.zeros((1, dim))
        H[0, 0] = 1.0
        
        # Noise covariances
        Q = np.eye(dim) * self.kalman_Q  # Process noise
        R = np.eye(1) * self.kalman_R     # Measurement noise
        
        # Step 3: Initialize Kalman filter
        kf = KalmanFilter(dim_x=dim, dim_z=1)
        kf.F = A
        kf.H = H
        kf.Q = Q
        kf.R = R
        kf.P = np.eye(dim) * self.kalman_P_init  # Initial state covariance
        
        # Initialize state with first samples
        x_init = np.zeros(dim)
        for i in range(min(dim, n)):
            x_init[i] = data[i]
        kf.x = x_init
        
        # Step 4: Run Kalman filter
        for i in range(n):
            # Predict
            kf.predict()
            
            # Update (if we have a measurement)
            kf.update(data[i])
            
            # Store filtered value (first state component)
            filtered[i] = kf.x[0]
        
        if self.verbose:
            # Calculate noise reduction
            original_noise = np.std(data - filtered)
            print(f"  Kalman: noise reduced by {100*(1 - original_noise/np.std(data)):.1f}%")
        
        return filtered
    
    def hybrid_motion_correction(self, data):
        """
        Apply all three motion correction methods in sequence
        
        Order matters:
        1. Wavelet first: Removes sharp spikes and high-frequency artifacts
        2. Spline second: Corrects any remaining baseline shifts
        3. Kalman third: Smooths residual noise adaptively
        """
        if self.verbose:
            print(f"\nApplying motion correction to channel...")
        
        # Step 1: Wavelet filtering
        data_wavelet = self.wavelet_filter(data)
        
        # Step 2: Spline correction
        data_spline = self.spline_correction(data_wavelet)
        
        # Step 3: Kalman filtering
        data_kalman = self.kalman_filter(data_spline)
        
        return data_kalman
    
    def bandpass_filter(self, data):
        """
        Apply bandpass filter to OD data
        """
        nyquist = 0.5 * self.fs
        
        # Handle edge cases
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
            if self.verbose:
                print(f"Warning: High cutoff adjusted to {high*nyquist:.2f} Hz")
        
        if low <= 0:
            low = 0.001
            if self.verbose:
                print(f"Warning: Low cutoff adjusted to {low*nyquist:.2f} Hz")
        
        # Design and apply filter
        b, a = butter(self.filter_order, [low, high], btype='band')
        
        # Use filtfilt for zero-phase distortion
        if data.ndim == 1:
            filtered = filtfilt(b, a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = filtfilt(b, a, data[:, i])
        
        return filtered
    
    def od_conversion(self, raw_intensity):
        """
        Convert raw intensity to optical density
        
        OD = -log10(I / I0)
        where I0 is baseline intensity (mean of first baseline_samples)
        """
        # Calculate baseline intensity
        baseline_intensity = np.mean(raw_intensity[:self.baseline_samples], axis=0)
        
        # Avoid numerical issues
        baseline_intensity = np.maximum(baseline_intensity, 1e-10)
        raw_safe = np.maximum(raw_intensity, 1e-10)
        
        # Convert to OD
        if raw_safe.ndim == 1:
            od = -np.log10(raw_safe / np.mean(baseline_intensity))
        else:
            od = -np.log10(raw_safe / baseline_intensity[np.newaxis, :])
        
        return od
    
    def solve_two_wavelength(self, od_730, od_850):
        """
        Solve the two-wavelength system for HbO and HbR using Beer-Lambert law
        
        [ΔOD₇₃₀] = [ε_HbO₇₃₀  ε_HbR₇₃₀] · [ΔHbO] · d · DPF₇₃₀
        [ΔOD₈₅₀]   [ε_HbO₈₅₀  ε_HbR₈₅₀]   [ΔHbR]    · d · DPF₈₅₀
        
        Returns:
            dhbo: Change in oxygenated hemoglobin concentration (mM)
            dhbr: Change in deoxygenated hemoglobin concentration (mM)
        """
        # Build extinction coefficient matrix
        E = np.array([
            [self.extinction[730]['HbO'], self.extinction[730]['HbR']],
            [self.extinction[850]['HbO'], self.extinction[850]['HbR']]
        ])
        
        # Pathlength factors (convert mm to cm)
        d_cm = self.dist / 10  # mm to cm
        L730 = d_cm * self.dpf[730]
        L850 = d_cm * self.dpf[850]
        
        # Check if matrix is invertible
        if np.linalg.cond(E) > 1000:
            if self.verbose:
                print("Warning: Extinction matrix is ill-conditioned")
        
        # Initialize output
        n_samples = len(od_730)
        dhbo = np.zeros(n_samples)
        dhbr = np.zeros(n_samples)
        
        # Solve at each time point
        for i in range(n_samples):
            od_vec = np.array([od_730[i] / L730, od_850[i] / L850])
            
            try:
                # Solve linear system: E * [HbO; HbR] = OD
                conc = np.linalg.solve(E, od_vec)
                dhbo[i] = conc[0]
                dhbr[i] = conc[1]
            except np.linalg.LinAlgError:
                # If solving fails, use pseudoinverse
                conc = np.linalg.pinv(E) @ od_vec
                dhbo[i] = conc[0]
                dhbr[i] = conc[1]
        
        return dhbo, dhbr
    
    def preprocess_pipeline(self, raw_df, impedance_df=None, channel_wavelengths=None):
        """
        Run full preprocessing pipeline
        
        Parameters:
        -----------
        raw_df : DataFrame
            Raw intensity data (columns = channels, rows = timepoints)
        impedance_df : DataFrame, optional
            Impedance measurements per channel
        channel_wavelengths : dict, optional
            Mapping from channel names to wavelengths
            e.g., {'ch1': 730, 'ch2': 730, 'ch3': 850, 'ch4': 850}
            
        Returns:
        --------
        results : dict
            Contains processed HbO, HbR, and quality metrics
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STARTING PREPROCESSING PIPELINE")
            print("="*60)
        
        # Step 1: Quality check
        good_channels = self.quality_check(raw_df, impedance_df)
        
        if not good_channels:
            raise ValueError("No channels passed quality check")
        
        # Filter DataFrame to good channels
        data_good = raw_df[good_channels].copy()
        
        if self.verbose:
            print(f"\nStep 2: Motion artifact correction")
        
        # Step 2: Motion artifact attenuation (apply to each channel)
        data_clean = pd.DataFrame(index=data_good.index, columns=data_good.columns)
        for col in data_good.columns:
            if self.verbose:
                print(f"\nProcessing channel: {col}")
            data_clean[col] = self.hybrid_motion_correction(data_good[col].values)
        
        # Step 3: Convert to OD
        if self.verbose:
            print(f"\nStep 3: Converting to Optical Density")
        od_data = self.od_conversion(data_clean.values)
        od_df = pd.DataFrame(od_data, columns=data_clean.columns, index=data_clean.index)
        
        # Step 4: Bandpass filter
        if self.verbose:
            print(f"\nStep 4: Bandpass filtering ({self.lowcut}-{self.highcut} Hz)")
        od_filtered = self.bandpass_filter(od_df.values)
        od_filtered_df = pd.DataFrame(od_filtered, columns=od_df.columns, index=od_df.index)
        
        # Step 5: Beer-Lambert to get HbO/HbR
        if self.verbose:
            print(f"\nStep 5: Converting to HbO/HbR using Beer-Lambert law")
        
        # Group channels by wavelength
        if channel_wavelengths is None:
            # Assume channels are named with wavelength
            ch_730 = [c for c in good_channels if '730' in c or '760' in c]
            ch_850 = [c for c in good_channels if '850' in c or '830' in c]
        else:
            ch_730 = [c for c in good_channels if channel_wavelengths.get(c) == 730]
            ch_850 = [c for c in good_channels if channel_wavelengths.get(c) == 850]
        
        results = {
            'raw_cleaned': data_clean,
            'od_filtered': od_filtered_df,
            'good_channels': good_channels,
            'ch_730': ch_730,
            'ch_850': ch_850,
            'preprocessor': self
        }
        
        # If we have both wavelengths, compute HbO/HbR
        if ch_730 and ch_850:
            # For simplicity, average channels of same wavelength
            # You might want more sophisticated combination
            od_730_mean = od_filtered_df[ch_730].mean(axis=1).values
            od_850_mean = od_filtered_df[ch_850].mean(axis=1).values
            
            dhbo, dhbr = self.solve_two_wavelength(od_730_mean, od_850_mean)
            
            results['HbO'] = dhbo
            results['HbR'] = dhbr
            results['HbO_HbR_ratio'] = dhbo / (dhbo + dhbr + 1e-10)
            
            if self.verbose:
                print(f"  Computed HbO/HbR from {len(ch_730)} 730nm and {len(ch_850)} 850nm channels")
        
        if self.verbose:
            print("\n" + "="*60)
            print("PREPROCESSING COMPLETE")
            print("="*60 + "\n")
        
        return results


# Example usage
if __name__ == "__main__":
    # Create preprocessor with your parameters
    # FILL IN YOUR VALUES HERE
    pre = NIRSPreprocessor(
        # System
        fs=10.2,              # YOUR VALUE: ______ Hz
        wavelengths=[730, 850], # YOUR VALUE: ______ nm
        source_detector_dist=30, # YOUR VALUE: ______ mm
        
        # Quality
        intensity_threshold=0.05,  # YOUR VALUE: ______
        impedance_threshold=5,      # YOUR VALUE: ______ kOhm
        baseline_sec=30,            # YOUR VALUE: ______ s
        
        # Filtering
        lowcut=0.01,    # YOUR VALUE: ______ Hz
        highcut=0.5,    # YOUR VALUE: ______ Hz
        filter_order=3, # YOUR VALUE: ______
        
        # Wavelet
        wavelet='db4',              # YOUR VALUE: ______
        wavelet_level=4,            # YOUR VALUE: ______
        wavelet_threshold_factor=2.5, # YOUR VALUE: ______
        
        # Spline
        artifact_std_threshold=3.0, # YOUR VALUE: ______
        spline_smoothing=0,          # YOUR VALUE: ______
        
        # Kalman
        kalman_Q=1e-4,          # YOUR VALUE: ______
        kalman_R=0.01,          # YOUR VALUE: ______
        kalman_model_order=2,    # YOUR VALUE: ______
        kalman_P_init=1.0,       # YOUR VALUE: ______
        
        # Beer-Lambert
        dpf_730=5.0,   # YOUR VALUE: ______ (VERIFY!)
        dpf_850=4.5,   # YOUR VALUE: ______ (VERIFY!)
        
        verbose=True
    )
    
    # Load your data (adapt to your format)
    # raw_data = pd.read_csv('your_nirs_data.csv')
    # impedance = pd.read_csv('impedance.csv')
    
    # Run pipeline
    # results = pre.preprocess_pipeline(raw_data, impedance)
    
    print("\nScript ready. Uncomment and adapt the data loading lines above.")
