"""
Helper functions for the ECG Analysis Streamlit app.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import sys
import base64
from datetime import datetime
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import data_processing, peak_detection, feature_extraction

def load_file(uploaded_file):
    """
    Load and preprocess an uploaded ECG file.
    
    Parameters:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (time, ecg_signal, fs, filename)
    """
    try:
        # Get file name and extension
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Create a temporary directory to save the file
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Save uploaded file to temp directory
        tmp_path = os.path.join(tmp_dir, file_name)
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Process based on file extension
        if file_ext in ['.csv', '.txt']:
            # Use data_processing module to load CSV
            time, ecg_signal, fs = data_processing.load_ecg_data(tmp_path)
            
            if time is None or ecg_signal is None or fs is None:
                st.error(f"Failed to load {file_name}. Ensure it contains ECG data.")
                return None, None, None, file_name
                
        elif file_ext in ['.dat', '.hea', '.atr']:
            try:
                # Try to load WFDB record
                import wfdb
                record_path = os.path.splitext(tmp_path)[0]  # Remove extension
                record = wfdb.rdrecord(record_path)
                ecg_signal = record.p_signal
                fs = record.fs
                time = np.arange(len(ecg_signal)) / fs
            except Exception as e:
                st.error(f"Failed to load {file_name}: {e}")
                return None, None, None, file_name
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None, None, None, file_name
            
        # Return loaded data
        return time, ecg_signal, fs, file_name
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, None, uploaded_file.name

def generate_synthetic_ecg(duration=10, fs=250, heart_rate=60, noise_level=0.05):
    """
    Generate synthetic ECG signal for demo purposes.
    
    Parameters:
        duration (float): Signal duration in seconds
        fs (float): Sampling frequency in Hz
        heart_rate (float): Heart rate in BPM
        noise_level (float): Amount of noise to add
        
    Returns:
        tuple: (time, ecg_signal, fs)
    """
    # Create time array
    time = np.arange(0, duration, 1/fs)
    
    # Calculate beat interval in seconds
    beat_interval = 60 / heart_rate
    
    # Create synthetic ECG
    ecg_signal = np.zeros((len(time), 2))  # Two channels
    
    # Create heartbeats
    for i in range(int(duration / beat_interval) + 1):
        t_center = i * beat_interval
        
        # QRS complex
        qrs_mask = (time > t_center - 0.03) & (time < t_center + 0.03)
        ecg_signal[qrs_mask, 0] = 1.5 * np.sin((time[qrs_mask] - t_center) * 150 + np.pi/2)
        ecg_signal[qrs_mask, 1] = 1.2 * np.sin((time[qrs_mask] - t_center) * 150 + np.pi/2)
        
        # P wave
        p_center = t_center - 0.15
        p_mask = (time > p_center - 0.05) & (time < p_center + 0.05)
        ecg_signal[p_mask, 0] += 0.25 * np.sin((time[p_mask] - p_center) * 100 + np.pi/2)
        ecg_signal[p_mask, 1] += 0.2 * np.sin((time[p_mask] - p_center) * 100 + np.pi/2)
        
        # T wave
        t_center = t_center + 0.2
        t_mask = (time > t_center - 0.1) & (time < t_center + 0.1)
        ecg_signal[t_mask, 0] += 0.35 * np.sin((time[t_mask] - t_center) * 50 + np.pi/2)
        ecg_signal[t_mask, 1] += 0.3 * np.sin((time[t_mask] - t_center) * 50 + np.pi/2)
    
    # Add baseline wander
    baseline_wander = 0.3 * np.sin(2 * np.pi * 0.05 * time)
    ecg_signal[:, 0] += baseline_wander
    ecg_signal[:, 1] += baseline_wander * 0.8
    
    # Add powerline interference (50 Hz)
    powerline = 0.2 * np.sin(2 * np.pi * 50 * time)
    ecg_signal[:, 0] += powerline
    ecg_signal[:, 1] += powerline * 0.9
    
    # Add random noise
    ecg_signal[:, 0] += np.random.normal(0, noise_level, len(time))
    ecg_signal[:, 1] += np.random.normal(0, noise_level * 0.8, len(time))
    
    return time, ecg_signal, fs

def process_ecg(time, ecg_signal, fs, processing_params):
    """
    Process ECG signal with the given parameters.
    
    Parameters:
        time (array): Time array
        ecg_signal (array): ECG signal data
        fs (float): Sampling frequency
        processing_params (dict): Processing parameters
        
    Returns:
        tuple: (processed_signal, r_peaks, pqrst_peaks, intervals, heart_rate)
    """
    try:
        # Apply signal preprocessing
        processed_signal = data_processing.preprocess_ecg(
            ecg_signal, 
            fs=fs,
            lowcut=processing_params.get('lowcut', 5),
            highcut=processing_params.get('highcut', 30),
            powerline_freq=processing_params.get('powerline_freq', 50)
        )
        
        # Detect R peaks
        r_peaks = peak_detection.detect_r_peaks(
            processed_signal, 
            fs=fs,
            method=processing_params.get('r_peak_method', 'xqrs')
        )
        
        # Detect PQRST peaks
        pqrst_peaks = peak_detection.detect_pqrst_peaks(processed_signal, r_peaks, fs)
        
        # Calculate ECG intervals
        intervals = peak_detection.calculate_ecg_intervals(pqrst_peaks, fs)
        
        # Calculate heart rate
        heart_rate, _, _ = peak_detection.calculate_heart_rate(r_peaks, fs)
        
        return processed_signal, r_peaks, pqrst_peaks, intervals, heart_rate
        
    except Exception as e:
        st.error(f"Error processing ECG: {e}")
        return None, None, None, None, None

def extract_heartbeats(ecg_signal, r_peaks, fs, window_params):
    """
    Extract heartbeat segments from ECG signal.
    
    Parameters:
        ecg_signal (array): ECG signal data
        r_peaks (array): R-peak indices
        fs (float): Sampling frequency
        window_params (dict): Window parameters
        
    Returns:
        array: Extracted heartbeat segments
    """
    try:
        # Extract heartbeats
        heartbeats = data_processing.extract_heartbeat_segments(
            ecg_signal, 
            r_peaks, 
            fs=fs,
            before_r=window_params.get('before_r', 0.25),
            after_r=window_params.get('after_r', 0.45)
        )
        
        return heartbeats
        
    except Exception as e:
        st.error(f"Error extracting heartbeats: {e}")
        return None

# In extract_features function in helpers.py
def extract_features(heartbeats, fs, feature_params):
    """Extract features from heartbeat segments."""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process in batches
        num_beats = len(heartbeats)
        features_list = []
        
        for i, beat in enumerate(heartbeats):
            # Extract features for this heartbeat
            beat_features = feature_extraction.extract_heartbeat_features(beat, fs, feature_params['include_advanced'])
            features_list.append(beat_features)
            
            # Update progress
            progress = (i + 1) / num_beats
            progress_bar.progress(progress)
            status_text.text(f"Processing: {i+1}/{num_beats} heartbeats")
            
        # Create DataFrame from features
        features_df = pd.DataFrame(features_list)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return features_df
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def get_download_link(dataframe, filename, text="Download CSV"):
    """
    Generate a download link for a DataFrame.
    
    Parameters:
        dataframe (DataFrame): Data to download
        filename (str): Name of the file to download
        text (str): Text for the download link
        
    Returns:
        str: HTML link for downloading the data
    """
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_figure_download_link(fig, filename="figure.png", text="Download Figure"):
    """
    Generate a download link for a matplotlib figure.
    
    Parameters:
        fig (Figure): Matplotlib figure
        filename (str): Name of the file to download
        text (str): Text for the download link
        
    Returns:
        str: HTML link for downloading the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def save_results(output_dir, ecg_data, processed_signal, r_peaks, pqrst_peaks, intervals, features_df, metadata):
    """
    Save analysis results to files.
    
    Parameters:
        output_dir (str): Directory to save results
        ecg_data (tuple): (time, ecg_signal, fs)
        processed_signal (array): Processed ECG signal
        r_peaks (array): R-peak indices
        pqrst_peaks (dict): PQRST peak indices
        intervals (dict): ECG intervals
        features_df (DataFrame): Extracted features
        metadata (dict): Additional metadata
        
    Returns:
        str: Path to the results directory
    """
    try:
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_dir, f"ecg_analysis_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Unpack ECG data
        time, ecg_signal, fs = ecg_data
        
        # Save raw ECG data
        raw_df = pd.DataFrame({"Time": time})
        for i in range(ecg_signal.shape[1]):
            raw_df[f"Channel_{i+1}"] = ecg_signal[:, i]
        raw_df.to_csv(os.path.join(results_dir, "raw_ecg.csv"), index=False)
        
        # Save processed ECG data
        processed_df = pd.DataFrame({"Time": time})
        for i in range(processed_signal.shape[1]):
            processed_df[f"Channel_{i+1}"] = processed_signal[:, i]
        processed_df.to_csv(os.path.join(results_dir, "processed_ecg.csv"), index=False)
        
        # Save peaks
        peaks_df = pd.DataFrame({"R_peaks": r_peaks})
        for wave, indices in pqrst_peaks.items():
            if wave != "R":  # R peaks already added
                peaks_df[f"{wave}_peaks"] = pd.Series(indices)
        peaks_df.to_csv(os.path.join(results_dir, "peaks.csv"), index=False)
        
        # Save intervals
        intervals_dict = {}
        for interval_name, values in intervals.items():
            if not interval_name.endswith("_mean") and not interval_name.endswith("_std"):
                intervals_dict[interval_name] = values
        
        intervals_df = pd.DataFrame(intervals_dict)
        intervals_df.to_csv(os.path.join(results_dir, "intervals.csv"), index=False)
        
        # Save interval statistics
        stats_dict = {key: [value] for key, value in intervals.items() 
                    if key.endswith("_mean") or key.endswith("_std")}
        stats_df = pd.DataFrame(stats_dict)
        stats_df.to_csv(os.path.join(results_dir, "interval_stats.csv"), index=False)
        
        # Save features
        if features_df is not None:
            features_df.to_csv(os.path.join(results_dir, "features.csv"), index=False)
        
        # Save metadata
        metadata["timestamp"] = timestamp
        metadata["fs"] = fs
        metadata["num_channels"] = ecg_signal.shape[1]
        metadata["duration"] = len(time) / fs
        metadata["num_heartbeats"] = len(r_peaks)
        metadata["heart_rate"] = metadata.get("heart_rate", 0)
        
        with open(os.path.join(results_dir, "metadata.txt"), "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        return results_dir
        
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return None

def plot_ecg_with_peaks(time, ecg_signal, processed_signal, pqrst_peaks, fs, 
                        start_time=0, duration=5, channel=0):
    """
    Plot ECG signal with detected PQRST peaks.
    
    Parameters:
        time (array): Time array
        ecg_signal (array): Original ECG signal
        processed_signal (array): Processed ECG signal
        pqrst_peaks (dict): Dictionary of PQRST peak indices
        fs (float): Sampling frequency
        start_time (float): Start time for plotting in seconds
        duration (float): Duration for plotting in seconds
        channel (int): Channel to plot
        
    Returns:
        Figure: Matplotlib figure
    """
    # Convert time to samples
    start_sample = int(start_time * fs)
    end_sample = min(len(time), start_sample + int(duration * fs))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original and processed signals
    ax.plot(time[start_sample:end_sample], 
            ecg_signal[start_sample:end_sample, channel], 
            'b-', alpha=0.5, label='Raw ECG')
    ax.plot(time[start_sample:end_sample], 
            processed_signal[start_sample:end_sample, channel], 
            'g-', label='Processed ECG')
    
    # Colors and markers for each wave
    wave_styles = {
        'P': {'color': 'blue', 'marker': 'o', 'label': 'P-wave'},
        'Q': {'color': 'green', 'marker': 's', 'label': 'Q-wave'},
        'R': {'color': 'red', 'marker': '^', 'label': 'R-peak'},
        'S': {'color': 'purple', 'marker': 'd', 'label': 'S-wave'},
        'T': {'color': 'cyan', 'marker': '*', 'label': 'T-wave'}
    }
    
    # Plot peaks
    for wave, indices in pqrst_peaks.items():
        # Filter peaks within the plotting window
        indices_in_window = indices[(indices >= start_sample) & (indices < end_sample)]
        if len(indices_in_window) > 0:
            ax.plot(time[indices_in_window], 
                    processed_signal[indices_in_window, channel], 
                    marker=wave_styles[wave]['marker'], 
                    color=wave_styles[wave]['color'], 
                    linestyle='none', 
                    markersize=8, 
                    label=wave_styles[wave]['label'])
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'ECG Signal with PQRST Peaks (Channel {channel+1})')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Set axis limits
    ax.set_xlim(start_time, start_time + duration)
    
    # Add heart rate if available
    if 'R' in pqrst_peaks and len(pqrst_peaks['R']) > 1:
        heart_rate, _, _ = peak_detection.calculate_heart_rate(pqrst_peaks['R'], fs)
        ax.text(0.02, 0.95, f'Heart Rate: {heart_rate:.1f} BPM', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

def plot_heartbeats(heartbeats, fs, num_beats=5):
    """
    Plot individual heartbeat segments.
    
    Parameters:
        heartbeats (array): Array of heartbeat segments
        fs (float): Sampling frequency
        num_beats (int): Number of beats to plot
        
    Returns:
        Figure: Matplotlib figure
    """
    # Limit number of beats to plot
    num_beats = min(num_beats, len(heartbeats))
    
    # Create time array for each beat
    beat_samples = heartbeats.shape[1]
    beat_time = np.arange(beat_samples) / fs - 0.25  # Assuming 0.25s before R peak
    
    # Create figure
    fig, axes = plt.subplots(num_beats, 1, figsize=(10, 2 * num_beats), sharex=True)
    if num_beats == 1:
        axes = [axes]
    
    # Plot each beat
    for i in range(num_beats):
        axes[i].plot(beat_time, heartbeats[i])
        axes[i].axvline(x=0, color='r', linestyle='--', label='R Peak' if i == 0 else '')
        axes[i].set_title(f'Heartbeat {i+1}')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)
    
    # Set common x-label
    axes[-1].set_xlabel('Time (s)')
    
    # Add legend to first subplot only
    axes[0].legend()
    
    plt.tight_layout()
    return fig

def plot_average_heartbeat(heartbeats, fs):
    """
    Plot average heartbeat with standard deviation.
    
    Parameters:
        heartbeats (array): Array of heartbeat segments
        fs (float): Sampling frequency
        
    Returns:
        Figure: Matplotlib figure
    """
    # Calculate mean and std
    mean_beat = np.mean(heartbeats, axis=0)
    std_beat = np.std(heartbeats, axis=0)
    
    # Create time array
    beat_samples = heartbeats.shape[1]
    beat_time = np.arange(beat_samples) / fs - 0.25  # Assuming 0.25s before R peak
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean beat
    ax.plot(beat_time, mean_beat, 'b-', label='Mean Beat')
    
    # Plot standard deviation
    ax.fill_between(beat_time, mean_beat - std_beat, mean_beat + std_beat, 
                   color='b', alpha=0.2, label='±1 SD')
    
    # Mark R peak
    ax.axvline(x=0, color='r', linestyle='--', label='R Peak')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Average Heartbeat Pattern (n={len(heartbeats)})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_feature_distribution(features_df, feature_name):
    """
    Plot the distribution of a specific feature.
    
    Parameters:
        features_df (DataFrame): DataFrame of features
        feature_name (str): Name of the feature to plot
        
    Returns:
        Figure: Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if feature exists
    if feature_name not in features_df.columns:
        ax.text(0.5, 0.5, f"Feature '{feature_name}' not found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Plot histogram
    ax.hist(features_df[feature_name], bins=30, alpha=0.7, color='b')
    
    # Add vertical lines for statistics
    ax.axvline(x=features_df[feature_name].mean(), color='r', linestyle='--', 
              label=f'Mean: {features_df[feature_name].mean():.3f}')
    ax.axvline(x=features_df[feature_name].median(), color='g', linestyle='-', 
              label=f'Median: {features_df[feature_name].median():.3f}')
    
    # Set labels and title
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature_name}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

# In helpers.py
@st.cache_data
def cached_extract_features(heartbeats_array, fs, include_advanced, quick_mode):
    """Cached version of feature extraction to avoid recomputation."""
    # Convert to tuple for hashing (arrays aren't hashable)
    heartbeats_tuple = tuple(map(tuple, heartbeats_array))
    
    # Use the actual feature extraction function
    if quick_mode:
        features = feature_extraction.extract_features_in_batches(
            heartbeats_array, fs, include_advanced, batch_size=50)
    else:
        features = feature_extraction.extract_features_from_heartbeats(
            heartbeats_array, fs, include_advanced)
            
    return features