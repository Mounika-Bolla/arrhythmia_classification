"""
ECG Signal Processing Module

This module provides functions for preprocessing ECG signals, including:
- Baseline wander removal
- Powerline interference removal
- Bandpass filtering
- Signal normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter
import pandas as pd

def remove_baseline_wander(signal, fs=360):
    """
    Remove baseline wander using a high-pass filter.
    
    Parameters:
        signal (array): The ECG signal
        fs (float): Sampling frequency (Hz)
        
    Returns:
        array: Signal with baseline wander removed
    """
    # Check signal dimensions
    if signal.ndim > 1:  # Multi-channel
        filtered_signal = np.zeros_like(signal)
        for channel in range(signal.shape[1]):
            filtered_signal[:, channel] = remove_baseline_wander(signal[:, channel], fs)
        return filtered_signal
        
    # Make sure signal is long enough for filtering
    min_length = 10  # Minimum length for filtering
    if len(signal) < min_length:
        print(f"Warning: Signal too short for baseline filter (length={len(signal)}, min={min_length})")
        return signal
    
    try:
        cutoff = 0.5  # 0.5 Hz cutoff frequency
        nyquist = 0.5 * fs
        high_pass = cutoff / nyquist
        b, a = butter(1, high_pass, btype="high")
        
        # Use lfilter for very short signals
        if len(signal) < 20:
            return lfilter(b, a, signal)
        else:
            return filtfilt(b, a, signal)
    except Exception as e:
        print(f"Warning: Baseline filter failed: {e}")
        return signal

def remove_powerline_interference(signal, fs=250, freq=50):
    """
    Remove powerline interference using a notch filter.
    
    Parameters:
        signal (array): The ECG signal
        fs (float): Sampling frequency (Hz)
        freq (float): Powerline frequency (50Hz in Europe, 60Hz in US)
        
    Returns:
        array: Signal with powerline interference removed
    """
    # Check signal dimensions
    if signal.ndim > 1:  # Multi-channel
        filtered_signal = np.zeros_like(signal)
        for channel in range(signal.shape[1]):
            filtered_signal[:, channel] = remove_powerline_interference(signal[:, channel], fs, freq)
        return filtered_signal
    
    # Make sure signal is long enough for filtering
    min_length = 10  # Minimum length for filtering
    if len(signal) < min_length:
        print(f"Warning: Signal too short for notch filter (length={len(signal)}, min={min_length})")
        return signal
        
    try:
        q = 30.0  # Quality factor
        w0 = freq / (fs/2)
        b, a = iirnotch(w0, q)
        
        # Use lfilter for very short signals
        if len(signal) < 20:
            return lfilter(b, a, signal)
        else:
            return filtfilt(b, a, signal)
    except Exception as e:
        print(f"Warning: Notch filter failed: {e}")
        return signal

def bandpass_filter(signal, lowcut=5, highcut=15, fs=250, order=2):
    """
    Apply bandpass filter to ECG signal.
    
    Parameters:
        signal (array): The ECG signal
        lowcut (float): Lower cutoff frequency (Hz)
        highcut (float): Upper cutoff frequency (Hz)
        fs (float): Sampling frequency (Hz)
        order (int): Filter order
        
    Returns:
        array: Filtered signal
    """
    # Check signal dimensions
    if signal.ndim > 1:  # Multi-channel
        filtered_signal = np.zeros_like(signal)
        for channel in range(signal.shape[1]):
            filtered_signal[:, channel] = bandpass_filter(signal[:, channel], lowcut, highcut, fs, order)
        return filtered_signal
    
    # Check if signal is long enough for the filter order
    min_length = 3 * (order * 2 + 1)  # Rule of thumb
    if len(signal) < min_length:
        print(f"Warning: Signal too short for bandpass filter (length={len(signal)}, min={min_length})")
        return signal
        
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        
        # Use lfilter for very short signals
        if len(signal) < 6 * order:
            return lfilter(b, a, signal)
        else:
            return filtfilt(b, a, signal)
    except Exception as e:
        print(f"Warning: Bandpass filter failed: {e}")
        return signal

def normalize_signal(signal):
    """
    Normalize the signal to have values between -1 and 1.
    
    Parameters:
        signal (array): The ECG signal
        
    Returns:
        array: Normalized signal
    """
    # Check signal dimensions
    if signal.ndim > 1:  # Multi-channel
        normalized_signal = np.zeros_like(signal)
        for channel in range(signal.shape[1]):
            normalized_signal[:, channel] = normalize_signal(signal[:, channel])
        return normalized_signal
    
    # Handle the case where min and max are the same (constant signal)
    sig_min = np.min(signal)
    sig_max = np.max(signal)
    
    if sig_min == sig_max:
        return np.zeros_like(signal)
    
    return 2 * (signal - sig_min) / (sig_max - sig_min) - 1

def preprocess_ecg(signal, fs=250, lowcut=5, highcut=30, powerline_freq=50):
    """
    Complete ECG preprocessing pipeline.
    
    Parameters:
        signal (array): The ECG signal
        fs (float): Sampling frequency (Hz)
        lowcut (float): Lower cutoff frequency (Hz)
        highcut (float): Upper cutoff frequency (Hz)
        powerline_freq (float): Powerline frequency (Hz)
        
    Returns:
        array: Preprocessed signal
    """
    try:
        # Make a copy of the input signal
        signal_filtered = signal.copy()
        
        # Remove baseline wander
        signal_filtered = remove_baseline_wander(signal_filtered, fs)
        
        # Remove powerline interference
        signal_filtered = remove_powerline_interference(signal_filtered, fs, powerline_freq)
        
        # Apply bandpass filter
        signal_filtered = bandpass_filter(signal_filtered, lowcut, highcut, fs)
        
        # Normalize
        signal_filtered = normalize_signal(signal_filtered)
        
        return signal_filtered
        
    except Exception as e:
        print(f"Warning: Signal processing failed: {e}")
        print("Falling back to simple normalization")
        # Fall back to just normalization if all else fails
        return normalize_signal(signal)

def load_ecg_data(file_path):
    """
    Load ECG data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (time, ecg_signal, fs) - time array, ECG signal, and sampling frequency
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Check for time column
        if 'Time (s)' in df.columns:
            time = df['Time (s)'].values
            # Calculate sampling frequency from time array
            fs = int(1 / (time[1] - time[0]))
        else:
            # Create time array based on assumed frequency
            fs = 250  # Default 250 Hz
            time = np.arange(len(df)) / fs
        
        # Detect ECG signal columns
        ecg_columns = [col for col in df.columns if col.startswith('Channel') or 
                       col.startswith('ECG') or col.startswith('Lead')]
        
        if not ecg_columns:
            # If no specific ECG columns found, use all numeric columns except time
            ecg_columns = [col for col in df.columns if col != 'Time (s)' and 
                           np.issubdtype(df[col].dtype, np.number)]
        
        # Extract ECG signal
        ecg_signal = df[ecg_columns].values
        
        return time, ecg_signal, fs
        
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return None, None, None

def extract_heartbeat_segments(ecg_signal, r_peaks, fs=250, before_r=0.25, after_r=0.45):
    """
    Extract heartbeat segments centered around R-peaks.
    
    Parameters:
        ecg_signal (array): The ECG signal
        r_peaks (array): Array of R-peak indices
        fs (float): Sampling frequency (Hz)
        before_r (float): Time before R-peak in seconds
        after_r (float): Time after R-peak in seconds
        
    Returns:
        array: Array of heartbeat segments
    """
    # Convert times to samples
    before_samples = int(before_r * fs)
    after_samples = int(after_r * fs)
    
    # Initialize array to store heartbeats
    heartbeats = []
    
    # Extract heartbeats
    for r_peak in r_peaks:
        # Check if we can extract a complete segment
        if r_peak >= before_samples and r_peak + after_samples <= len(ecg_signal):
            # Extract segment
            segment = ecg_signal[r_peak - before_samples:r_peak + after_samples]
            heartbeats.append(segment)
    
    return np.array(heartbeats)