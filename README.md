# 3035988079_YULOMartinGabrielCasa_BladderVolumeIdeation_Public
Public version of the Github repository for the final year project of 3035988079 YULO Martin Gabriel Casa

Usage: The entire pipeline is built that users only need to run main.sh. It will sequentially run each script when needed on the appropriate data file. The pipeline automatically handles creation of data subfolders, proper output file-naming and creation, and saves any produced figures as png files.

Requirements: 
Code requires up-to-date python and R. List of necessary packages for each script are also listed at the top of the respective scripts, and are additionally listed below. For the code to also function properly, it requires a master metadata file that maps each raw sensor data with the appropriate before and after ultrasound-measured bladder volume based on their timestamp and file name. The file in question would also contain patient metadata and maps each patient with their respective raw sensor data csv files, raw bladder volume images, and timestamp and data collection period (E.g., if this was the first, second, third, or fourth wear and data measurement of the sensor).

Data is found in this google drive: https://drive.google.com/drive/folders/1aj9TIMCgy5Rh0zogOe_rCxWE9lY_HYnP?usp=sharing
Please download it and move the data into the same folder as the github repo (same folder as main.sh)

Workflow of main.sh:
Requires master csv: (for this project, it is "BladderVolFYP_MasterDataset_First8.csv" also found in the google drive.

///// #1 ///////
DataPreprocessing.py
This function loads raw intensity data from a 16-channel (or 4-channel) NIRS device, filter out noisy channels and motion artifacts, and convert the cleaned optical data into hemoglobin concentration changes (HbO and HbR) using the Modified Beer-Lambert Law. The output is a cleaned, time-series-ready CSV file.

Input csv is the base raw data files. It assumes (1) it is a CSV file with a timestamp column. (2) Optical data columns following a specific naming convention. The script is hardcoded to look for 16 channels (optics1_uA through optics16_uA), particularly using the first eight for Beer-Lambert conversion. (3) Channel pairing logic assumes specific numbering: optics1/3/5/7 are 730nm and optics2/4/6/8 are 850nm sources.

Functions within:
PreprocessingParams (Class):

A central configuration class. It defines all adjustable parameters for every step of the preprocessing pipeline, from wavelet filter settings to Beer-Lambert extinction coefficients. This prevents hardcoding values inside functions and makes experimentation easy.

load_and_validate_data(filepath):

Loads the CSV, identifies the timestamp column, determines the number of optical channels, and validates against expected formats (4 or 16 channels). It provides an initial data report.

channel_intensity_filtering(df, optical_cols, params):

The first quality control step. It calculates the mean intensity of each optical channel. Any channel with a mean intensity below MIN_INTENSITY_THRESHOLD is removed. It warns if too many channels are failing, as this could indicate a hardware or probe placement issue.

motion_artifact_attenuation(df, channel_cols, params):

The core noise removal step for each channel. It orchestrates three sub-functions:

(1) wavelet_filter: Decomposes the signal, thresholds detail coefficients to suppress noise, and reconstructs it.
(2) spline_baseline_correction: Estimates and removes slow baseline drift using percentile-based spline fitting.
(3) kalman_filter: Further attenuates rapid motion artifacts by assuming a constant signal state and adaptively filtering out sharp deviations.

convert_to_optical_density(df, channel_cols, params):

Converts raw intensity (I) to optical density (OD). OD is a linear measure of light attenuation, calculated as -log10(I / I0). The baseline intensity I0 is estimated from the first 10% of the data.

apply_bandpass_filtering(df, channel_cols, params):

Applies a zero-phase Butterworth bandpass filter to the OD data. This isolates the physiological signal of interest by removing low-frequency drift and high-frequency noise based on the configured HIGHPASS_CUTOFF and LOWPASS_CUTOFF frequencies.

process_beer_lambert(df, channel_cols, params):

The key physiological conversion step. It uses the beer_lambert_conversion function to solve a system of linear equations (the Modified Beer-Lambert Law) using known extinction coefficients for 730nm and 850nm light. This translates OD changes from two wavelengths into concentration changes of oxyhemoglobin (HbO) and deoxyhemoglobin (HbR) at four specific optode locations (e.g., left_outer_HbO).

main() is the pipeline's entry point. It parses command-line arguments, initializes parameters, creates an output directory, and sequentially calls all the processing steps listed above, finally saving the result as a cleaned CSV.

///// #2 ///////
TemporalOverlap.py
This script transforms the continuous, cleaned time-series data into a format suitable for training a model that makes high-frequency predictions. It creates a large number of overlapping windows from the input data, each representing a 'context window' leading up to a single prediction time point.

Input requirement (automatically vetted by main.sh): The cleaned CSV output from DataPreprocessing3.py. It must contain a timestamp column and all the processed signal columns.

Functions within:
detect_time_unit_and_sample_rate(timestamps):

Intelligently determines the sample rate (Hz) from the timestamp data. It checks the magnitude of timestamps to guess the unit (e.g., microseconds, seconds) and verifies if the calculated sample rate is physiologically plausible (1-10,000 Hz). This prevents manual specification of the sample rate in most cases.

create_continuous_windows(data, window_sec, prediction_interval_sec, fs):

The central logic of the script. For a given window_sec (e.g., 10 seconds) and prediction_interval_sec (e.g., 1 second), it slides a window across the data, creating a new window every prediction_interval_sec. The overlap is typically very high (e.g., 90% for a 1-second step on a 10-second window). Each window's 'target' time is the end of the window, simulating a real-time forecast.

save_windows_as_single_file(windows, metadata, output_file):

A more efficient storage method. Instead of saving potentially thousands of individual CSV files, it concatenates all window DataFrames into a single large CSV, preserving the window_id column to distinguish between them.

main() then loads the cleaned data, calls create_continuous_windows with user-defined or default parameters, and then saves the output. It saves both the individual window files and the combined single file for convenience.

///// #3 ///////
FeatureExtraction.py
This function extracts a rich feature set from the windowed data, transforming raw time-series segments (or single time points) into informative, fixed-length feature vectors that a machine learning model can understand. It provides multiple strategies to aggregate data across a window or preserve temporal resolution.

Input requirement (automatically vetted by main.sh): The output CSV from TemporalOverlap.py. It must have a window_id column to group the rows and contain the HbO/HbR and optical channel columns from the original preprocessing.

Each feature extraction function processes a 1D signal or a single row of data and returns a dictionary of features. These are combined by orchestrator functions.

Functions within:
Time-Domain & Spectral Signal Features (extract_signal_features): Computes statistics like mean, standard deviation, skewness, kurtosis, and root mean square, as well as spectral features like dominant frequency. This function works on a full signal array.

HbO/HbR Specific Features (extract_hbo_hbr_features, extract_hbo_hbr_features_row): For each optode location (e.g., left_outer), it extracts statistical features from the HbO and HbR signals (e.g., left_outer_HbO_mean) or instantaneous values. It also calculates their ratio (oxygenation index) and difference.

Wavelength-Specific Features (wavelength_specific_features, wavelength_specific_features_row): Aggregates all channels of a specific wavelength (e.g., all 730nm channels) and extracts signal features from the mean trend.

Spatial Relationship Features (spatial_features, spatial_features_row): Quantifies asymmetries and ratios between probe locations, such as left-to-right differences and inner-to-outer (depth) ratios, which can indicate tissue oxygenation gradients.

Wavelength Ratio Features (wavelength_ratio_features, wavelength_ratio_features_row): Calculates the ratio of 850nm to 730nm intensity, a feature sensitive to blood volume changes, for both inner and outer channel pairs.

Orchestration Strategies (The main function runs two):
process_windowed_csv_aggregated(input_file):

Purpose: Window-level aggregation. It iterates through each unique window_id, extracts all features from the entire window's signal using functions like extract_all_features_from_window, and produces one row per window. This is the classic static model input.

process_windowed_csv(input_file):

Purpose: Row-level feature extraction. To preserve the full temporal resolution, it iterates through every single row of the windowed file. For each row, it extracts features using extract_all_features_from_row, which only uses the instantaneous values at that time point. It produces a feature DataFrame with the same number of rows as the input. This is ideal for sequence models or when you need predictions at every sample.

main() parses arguments, sets output paths, and calls both process_windowed_csv and process_windowed_csv_aggregated in sequence, saving their respective outputs. This provides maximum flexibility for downstream model development.


///// #4 ///////
FeatureSelector.py

Functions within:
configure_plots():

Sets global matplotlib and seaborn styling for consistent, publication-quality plots with muted palettes and gridlines.

plot_wavelength_ratios(feature_df, save_path):

Visualizes the 850nm/730nm ratio features, which are proxies for blood volume changes. If a bladder_state column exists, it groups comparisons by state; otherwise, it generates simple boxplots of ratio distributions across different spatial locations (inner vs. outer optodes).

plot_spatial_asymmetry(feature_df, save_path):

Creates scatter plots comparing left-right asymmetry at 730nm vs. 850nm. This reveals spatial patterns in oxygenation—potentially critical for detecting asymmetric bladder distension or sensor misplacement. Falls back to histogram if only one asymmetry metric is available.

plot_feature_correlation(feature_df, top_n, save_path):

Generates a triangular correlation matrix heatmap for the top N features by variance. Prints feature pairs with |correlation| > 0.8 to console, flagging redundancy that could harm model interpretability.

plot_feature_importance(feature_df, target_col, save_path):

Ranks features using ANOVA F-statistic against a categorical target (e.g., bladder_state). Produces a horizontal bar chart of the top 20 features with significance asterisks, giving a univariate estimate of predictive power.

plot_time_series_features(feature_df, feature_col, window_col, save_path):

Plots how a single feature evolves over the session's prediction_time_sec. Overlays a Gaussian-smoothed trendline to reveal slow drifts associated with bladder filling cycles.

is_high_quality(features, snr_threshold, artifact_threshold):

A heuristic quality-check function that flags windows likely corrupted by motion artifacts. It uses HbO standard deviation and wavelength ratio reasonability as proxies for signal quality (since the feature extractor's explicit SNR metrics are commented out by default).

main() is the orchestrator that loads the feature CSV, creates the output directory, and sequentially generates all five plot types. It automatically selects the top three features by variance for individual time-series plots.

FeatureSelector3.py, which accepts the same type of input with same requirements, additionally generates the following graphs, and has the following functions:


///// #5 ///////
y_features.py
This script derives physiologically meaningful target variables from the cleaned NIRS features, creating a continuous "Bladder Filling Index" that can serve as a regression target for machine learning models. It synthesizes tissue oxygenation, blood volume, and spatial asymmetry into a normalized [0,1] score and optionally maps it to estimated milliliter volumes.

Functions within:
compute_fNIRS_derived_metrics(df):

Purpose: The foundational physiological computation. For each optode location, calculates three key metrics:
- TOI (Tissue Oxygenation Index): HbO / (HbO + HbR) × 100 — measures oxygen saturation percentage.
- BVI (Blood Volume Index): HbO + HbR — proxy for total hemoglobin concentration, reflecting localized blood volume shifts.
- OEF Proxy (Oxygen Extraction Fraction): HbR / (HbO + HbR) — inversely related to TOI, representing the fraction of deoxygenated hemoglobin.
It then computes aggregate means (TOI_mean, BVI_mean, OEF_proxy_mean) and dynamic asymmetries (left-right imbalances, inner-outer depth gradients).

Part 2: Temporal Dynamics
compute_rates_of_change(df, channels, metrics, smooth_window):

Captures the dynamics of bladder filling by computing first derivatives (rates of change) of the derived metrics. Applies Savitzky-Golay smoothing to suppress noise before computing gradients, producing columns like d_TOI_mean_dt_smooth. Also calculates percentage change rates for normalized comparisons.

Part 3: Bladder Filling Index
compute_bladder_filling_index(df, pre_wear_volume, post_wear_volume):

The core target engineering function. Normalizes five components to [0,1] and combines them with fixed weights:
- 25% TOI_norm (inverted: lower TOI → higher index, reflecting sympathetic activation during distension)
- 20% BVI_norm (blood volume shifts)
- 20% OEF_norm (oxygen extraction increases)
- 20% Asymmetry_norm (spatial heterogeneity with filling)
- 15% dTOI_norm (dynamic rate of change)
- Applies Savitzky-Golay smoothing to produce Bladder_Filling_Index_Smooth.

If pre/post volumes are provided, performs a linear mapping from index to Estimated_Bladder_Volume_mL using endpoint calibration.

add_elapsed_time_feature(df, time_col):

Adds elapsed_time_sec and elapsed_time_min columns, converting timestamps or row index to a continuous time variable. This is critical as elapsed time itself is often the single strongest predictor of bladder volume (per Fechner et al., 2023).

Part 4: Visualization
plot_fNIRS_derived_metrics(df, title_prefix, save_path):

Generates an 8-panel (4×2) comprehensive dashboard showing TOI per channel, mean TOI + asymmetry, BVI per channel, BVI gradients, OEF per channel, rates of change, the Bladder Filling Index, and either estimated volume or component contributions.

plot_correlation_with_volume(df, pre_vol, post_vol, save_path):

Creates a 2×3 grid plotting six key metrics over time with linear trendlines and annotations showing absolute and percentage changes. Designed to visually assess which metrics track with known volume changes between session start and end.

Part 5 & 6: Pipeline and Batch Processing
process_fNIRS_session(df, pre_wear_volume, post_wear_volume, session_name, smooth_window):

The single-session pipeline. Sequentially calls add_elapsed_time_feature, compute_fNIRS_derived_metrics, compute_rates_of_change, and compute_bladder_filling_index. Returns both the processed DataFrame and a summary statistics dictionary covering TOI trends, filling index values, and volume estimation errors.

process_multiple_sessions(session_dfs, session_volumes, smooth_window):

Extends the pipeline to multiple sessions (e.g., different subjects or conditions). Iterates through a dictionary of DataFrames, processes each, and performs cross-session Pearson correlation analysis between known volume changes and measured TOI/BVI trends.

main() parses command-line arguments, loads the feature CSV, runs process_fNIRS_session with optional volume calibration, saves the enriched DataFrame with all derived metrics and filling indices, and generates both diagnostic plots.

///// #6 ///////
MLmodel.py
To train and rigorously evaluate machine learning models (Random Forest and LSTM) for predicting continuous bladder volume from the derived NIRS features. The script implements leave-one-session-out cross-validation, feature selection, downsampling for efficiency, and comprehensive evaluation metrics including regression errors, classification accuracy, confusion matrices, and ROC curves. It saves trained models for deployment.

required input (automatically handled by main.sh:)
Four CSV files from y_features.py processing (sessions 1-4), each containing the derived physiological metrics, filling indices, and elapsed time features.
Ten positional command-line arguments: four CSV paths and six bladder volume measurements (mL) representing pre- and post-void volumes for each session plus a post-void residual (PVR).

Functions within:
Argument Parsing and Session Configuration
parse_args():
Defines and parses the fixed-order positional arguments: four CSV file paths, six volume measurements (bv1, bv3, bv4, bv5, bv6, bv_pvr), and optional flags for patient ID, output directory, downsampling factor, and maximum features.

parse_vol(val):
Safely parses volume arguments, returning None for "NA", empty, or invalid values. This allows flexible session inclusion—sessions with missing volumes are automatically skipped.

build_session_list(args):
Constructs a list of valid (session_name, csv_path, pre_volume, post_volume) tuples from the arguments. Session 1 uses PVR (if available) as pre-volume; subsequent sessions chain their volumes (session N's post-volume becomes session N+1's pre-volume). Sessions with missing data are silently excluded.

Data Loading and Feature Engineering
load_sessions(session_list):
Loads each CSV, validates it's non-empty, and returns parallel lists of session names, DataFrames, and (pre_vol, post_vol) tuples. Prints a summary of each loaded session.

get_features(df):
Dynamically identifies which feature columns from the master list (CORE_FEATURES, CHANNEL_FEATURES, TIME_FEATURES, WAVELENGTH_FEATURES) are actually present and non-null in the DataFrame. This provides robustness to variations in upstream processing outputs.

add_labels(df, pre, post):
Creates the regression target column label_volume_mL by linearly interpolating between the known pre-session and post-session volumes across all rows. This assumes a constant filling rate, providing continuous supervision for each time point.

Efficiency Optimizations
downsample_df(df, factor):
Reduces dataset size by keeping every Nth row (default factor=100). This is critical because the high-overlap windowing from TemporalOverlap.py creates massive temporal redundancy—adjacent rows are nearly identical. Downsampling preserves the volume trajectory while dramatically reducing training time and overfitting risk for tree-based models.

select_features(X, y, all_features, max_features):
Uses a lightweight Random Forest (50 trees, max depth 10) to rank all available features by importance. Selects the top max_features (default 40) for the final model. This removes correlated/redundant features that can cause overfitting and slows training. Returns both the selected feature list and a full importance DataFrame for inspection.

2 main models were done: A standard RandomForest Regressor, and a Keras Sequential LSTM following the Fechner paper.
Model Classes (Object-Oriented Wrappers)
RFModel: Wraps scikit-learn's RandomForestRegressor (200 trees, max depth 15, min samples leaf 5) with standardization. Provides consistent fit(), predict(), and save() interfaces matching the LSTM wrapper, enabling interchangeable use in the cross-validation loop.

LSTMModel:  Wraps a Keras Sequential LSTM network with architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1). Includes separate scalers for features and targets. The _sequences() method creates sliding windows of length 60 for temporal context. Falls back gracefully (returning NaN predictions) if insufficient data for sequence creation.

Cross-Validation and Evaluation
build_train_data(indices, session_dfs, session_volumes, downsample, selected_features):
Combines multiple sessions into a single training array. Applies downsampling, labels creation, and feature selection. Returns stacked feature matrix X, target vector y, and the list of feature names actually used.

run_cv(model_class, model_name, names, dfs, vols, downsample, selected_features, **kw):
Implements leave-one-session-out cross-validation. For each fold, trains on all sessions except one, then predicts on the held-out session. Applies consistent downsampling and feature selection to both train and test sets. Returns a list of DataFrames with predictions, ground truth, and metadata.

classify_volume(vol, thresholds): Discretizes continuous volume predictions into three clinical categories: "low" (<100 mL), "moderate" (100-300 mL), or "high" (>300 mL). This enables classification evaluation alongside regression metrics.

evaluate_model(predictions, model_name, pid, thresholds): Does the following evaluation metrics
Extracts endpoint predictions (first and last row of each session).
Regression metrics: MAE, RMSE, and R² for start, end, and combined volumes.
Classification metrics: Accuracy, weighted precision/recall/F1, and confusion matrix for low/moderate/high categories.
ROC analysis: Binary "high vs. not-high" ROC curve with AUC.
Visualization: Saves confusion matrix heatmap and ROC curve plots.
Trace R²: Compares the full prediction trajectory against a linear ramp between pre and post volumes, revealing whether the model learned anything beyond the baseline interpolation.

main():
Parses arguments and builds session list.
Creates an initial combined dataset from all sessions with downsampling.
Performs feature selection, saving importance rankings.
Rebuilds dataset with selected features.
Trains final Random Forest on all data, performs sanity check (training MAE), and saves model artifacts (model pickle, scaler pickle, feature list CSV).
Runs leave-one-out CV for Random Forest with full evaluation.
Optionally trains and evaluates LSTM (using less aggressive downsampling since LSTMs benefit from more data), saving the Keras model.

///// #7 ///////
ScikitModel.py
A refactored, more rigorous implementation of MLModel2.py using scikit-learn's Pipeline API to prevent data leakage during feature selection and preprocessing. This version ensures that all transformations (imputation, scaling, feature selection) are fit only on training data within each cross-validation fold, then applied to held-out test data—a critical best practice that MLModel2.py's manual approach potentially violated. It also adds comprehensive result saving (predicted vs. actual bar charts, volume tables, metrics files) for production-grade reporting.

Follows a modified version of a ScikitLearn pipeline to extract relevant features and train a Random Forest model to estimate bladder volume.

Comparison with MLModel2.py
MLModel2.py performed feature selection once on all data before cross-validation—a subtle form of data leakage where test fold information could influence which features are selected. ScikitModel.py embeds feature selection inside the Pipeline, ensuring it's re-fit on each training fold independently during cross-validation.

Same input requirements as MLModel2.py, again automatically handled by main.sh

Functions within:
Argument Parsing and Data Loading (Similar to MLModel2.py)
parse_args(), parse_vol(val), build_session_list(args):

Identical in function to MLModel2.py—parse command-line arguments, handle missing volumes, and build the session list from the fixed-order positional arguments.

load_and_prepare_data(session_list):

Purpose: Loads and prepares all session data into a unified format for scikit-learn. Key differences from MLModel2.py:

Automatic feature detection: Dynamically identifies all numeric feature columns by excluding known non-feature columns (window_id, prediction_time_sec, target-derived columns like Bladder_Filling_Index, TOI_norm, etc.).

Forward/backward fill: Handles NaN values with forward-fill then backward-fill before any rows are dropped, minimizing data loss.

Group array: Creates a groups array where each row is labeled by its session index, essential for LeaveOneGroupOut cross-validation.

Session info tracking: Returns a list of dictionaries with per-session metadata (name, row count, pre/post volumes) used throughout evaluation.

Pipeline Construction
build_pipeline(n_features, max_features=40): Constructs a scikit-learn Pipeline with four sequential steps:

SimpleImputer: Fills any remaining NaN values with column means (safety net after forward/backward fill).
StandardScaler: Standardizes features to zero mean, unit variance—critical for models sensitive to feature scales.
SelectFromModel: Embedded feature selection using a lightweight Random Forest (100 trees, max depth 10). This fits on training data only, then transforms test data to keep only the most important features. The max_features parameter caps the number retained.
RandomForestRegressor: The final prediction model (200 trees, max depth 15, min samples leaf 5).
Data Leakage Prevention: Because feature selection is inside the pipeline, when cross_val_predict or LeaveOneGroupOut is used, the selector is re-fit on each training fold—test data never influences which features are chosen.

Cross-Validation
run_loso_cv(pipeline, X, y, groups, session_info): Performs strict leave-one-session-out cross-validation. For each fold, trains the entire pipeline (including feature selection) on all but one session, then predicts on the held-out session. Prints per-fold MAE and R² for immediate feedback.

run_train_test_split(pipeline, X, y, groups, session_info, test_size=0.2):
Alternative simpler 80/20 train-test split that respects session boundaries (sessions are kept intact, not split across train/test). Currently used as the primary evaluation method in main().

Evaluation and Reporting (Enhanced vs. MLModel2.py)
classify_volume(vol, thresholds):
Same as MLModel2.py—discretizes continuous volume into low/moderate/high categories for classification metrics.

evaluate_results(y_true, y_pred, session_info, model_name, pid):
Comprehensive evaluation that writes all results to both console and a timestamped metrics file. Includes:

Regression: MAE, RMSE, R² on all valid predictions.
Per-session endpoint errors: Start and end volume errors for each session, saved as CSV.
Classification: Accuracy, weighted precision/recall/F1, confusion matrix (saved as PNG), and ROC curve for "high vs. not-high" classification (saved as PNG).

Also calls plot_predicted_vs_actual() and save_volume_table() for additional outputs.
plot_predicted_vs_actual(session_info, y_true, y_pred, output_dir, model_name, pid):

Creates a side-by-side bar chart comparing predicted vs. actual bladder volumes for pre (start) and post (end) values across all sessions. Blue bars = actual, red bars = predicted. This provides an intuitive visual assessment of model performance per session.

save_volume_table(session_info, y_true, y_pred, output_dir, model_name, pid):
Generates a detailed CSV table with per-session actual vs. predicted volumes, errors, and volume changes. Also prints a formatted table to console. This is essential for clinical reporting and error analysis.

Main Pipeline
main():

The production training pipeline:

Loads and prepares data from all valid sessions.
Builds the scikit-learn pipeline with embedded feature selection.
Performs train-test split evaluation (with session boundary respect) and runs full evaluate_results.
Trains the final pipeline on all data for deployment.
Reports which features were selected by the final model and saves the list.
Serializes the complete pipeline as a single .pkl file using joblib.dump, enabling simple deployment with joblib.load() and pipeline.predict(new_data).

These aforementioned two model scripts trains a model on a class-by-class basis (in this case, calibrated and trained on an individuals patients data. A benefit is that it is trained for the specific biophyisical differences and data variances for this patient, but a drawback is there is less data to use.

///// #8 ///////
FeatureSelectorBig.py

A modified version of FeatureSelector that performs feature selection and visualization based on all 4 csvs, aimed to illustrate the difference between sessions.

The workflow and functions were generally the same, although the input is quite different:


//// #9 /////
ScikitModelBig.py

A modified version of ScikitModel.py that is applied to all accumulated data, in order to have a model with more training data available, at the cost of more generalization that isn't calibrated to the individual participants body type and biophysical factors.

The workflow and functions were generally the same, although the input is quite different:
Main.sh automatically adds all csvs and mapped validated Ultrasound bladder before and after volumes, as well as pvr volumes, per patient and stores it into CONFIG_CSV, which is then accessed and read by ScikitModelBig to know the bladder volume values and the names and locations of the feature-extracted csv files.


////////////////

OTHER FILES not incorporated into main.sh:

fyp_csv_viewer.Rmd:
Viewing csv files manually via excel or textedit was too time and resource intensive and quite inefficient, so i wrote a simple R-script to help me fiddle with csv files and inspect their file sizes, row or column counts, colnames, etc. 

This is merely for troubleshooting and assistance, and is not required for the main pipeline.

BladderDicomViewer.py
A dictionary within the file entails all the relevant dicom files that are used. This script displays them all side-by-side for easier viewing and double-checking of all the images to check their quality and to measure the bladder volume.

Ultrasound_validation2.Rmd:
This script takes in input of Depth, Height and Width of bladder volume images, manually measured and filled in a spreadsheet. It then calculates the bladder volume based on formula D*H*W*0.7, and appends this information to the resultant csv file. Additionally, it can filter out images labelled "0" as NA or "1" as poor quality, though doing so reduces the number of viable data as files would not have respective ground truth.
