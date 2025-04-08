"""
ECG Peak Detection Module

This module provides functions for detecting PQRST peaks in ECG signals.
It implements various algorithms for R-peak detection and methods for
finding other peaks (P, Q, S, T) relative to the R-peaks.
"""

import numpy as np
from scipy.signal import find_peaks
try:
    from wfdb.processing import XQRS
    wfdb_available = True
except ImportError:
    wfdb_available = False

def detect_r_peaks_pan_tompkins(ecg_signal, fs=250):
    """
    Detect R-peaks using the Pan-Tompkins algorithm.
    
    Parameters:
        ecg_signal (array): The ECG signal
        fs (float): Sampling frequency (Hz)
        
    Returns:
        array: Array of R-peak indices
    """
    # If multi-channel, use the first channel
    if ecg_signal.ndim > 1:
        signal = ecg_signal[:, 0]
    else:
        signal = ecg_signal
    
    # Differentiation
    diff_signal = np.diff(signal)
    # Append a zero to match the original signal length
    diff_signal = np.append(diff_signal, 0)
    
    # Squaring
    squared_signal = diff_signal ** 2
    
    # Moving Average (Integration)
    window_size = int(0.15 * fs)  # 150 ms window
    if window_size < 1:
        window_size = 1
    
    integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode="same")
    
    # Adaptive thresholding
    threshold = 0.3 * np.max(integrated_signal)
    
    # Find peaks
    peaks, _ = find_peaks(integrated_signal, height=threshold, distance=int(0.2 * fs))
    
    # Adjust peak locations to actual R-peaks
    r_peaks = []
    for peak in peaks:
        # Search for maximum in a small window around the detected peak
        window_start = max(0, peak - int(0.05 * fs))
        window_end = min(len(signal), peak + int(0.05 * fs))
        r_peak = window_start + np.argmax(signal[window_start:window_end])
        r_peaks.append(r_peak)
    
    return np.array(r_peaks)

def detect_r_peaks_xqrs(ecg_signal, fs=250):
    """
    Detect R-peaks using the XQRS algorithm from WFDB.
    
    Parameters:
        ecg_signal (array): The ECG signal
        fs (float): Sampling frequency (Hz)
        
    Returns:
        array: Array of R-peak indices
    """
    # Check if WFDB is available
    if not wfdb_available:
        print("WFDB library not available. Using Pan-Tompkins algorithm instead.")
        return detect_r_peaks_pan_tompkins(ecg_signal, fs)
    
    # If multi-channel, use the first channel
    if ecg_signal.ndim > 1:
        signal = ecg_signal[:, 0]
    else:
        signal = ecg_signal
    
    try:
        # Initialize XQRS detector
        xqrs = XQRS(sig=signal, fs=fs)
        
        # Detect peaks
        xqrs.detect()
        
        # Get detected peak indices
        r_peaks = xqrs.qrs_inds
        
        return r_peaks
    except Exception as e:
        print(f"XQRS detection failed: {e}")
        print("Falling back to Pan-Tompkins algorithm.")
        return detect_r_peaks_pan_tompkins(ecg_signal, fs)

def detect_r_peaks(ecg_signal, fs=250, method='xqrs'):
    """
    Detect R-peaks using specified method.
    
    Parameters:
        ecg_signal (array): The ECG signal
        fs (float): Sampling frequency (Hz)
        method (str): Detection method ('xqrs' or 'pan_tompkins')
        
    Returns:
        array: Array of R-peak indices
    """
    if method.lower() == 'xqrs' and wfdb_available:
        return detect_r_peaks_xqrs(ecg_signal, fs)
    else:
        return detect_r_peaks_pan_tompkins(ecg_signal, fs)

def detect_pqrst_peaks(ecg_signal, r_peaks, fs=250):
    """
    Detect P, Q, S, and T peaks based on R-peaks.
    
    Parameters:
        ecg_signal (array): The ECG signal
        r_peaks (array): Array of R-peak indices
        fs (float): Sampling frequency (Hz)
        
    Returns:
        dict: Dictionary containing P, Q, R, S, and T peak indices
    """
    # If multi-channel, use the first channel
    if ecg_signal.ndim > 1:
        signal = ecg_signal[:, 0]
    else:
        signal = ecg_signal
    
    p_peaks, q_peaks, s_peaks, t_peaks = [], [], [], []
    
    for r_peak in r_peaks:
        # Q-wave: Look for minimum before R-peak
        q_start = max(0, r_peak - int(0.05 * fs))  # 50 ms before R
        q_end = r_peak
        if q_start < q_end:
            q_index = q_start + np.argmin(signal[q_start:q_end])
            q_peaks.append(q_index)
        
        # S-wave: Look for minimum after R-peak
        s_start = r_peak
        s_end = min(len(signal), r_peak + int(0.05 * fs))  # 50 ms after R
        if s_start < s_end:
            s_index = s_start + np.argmin(signal[s_start:s_end])
            s_peaks.append(s_index)
        
        # P-wave: Look for maximum before Q-wave
        if len(q_peaks) > 0:
            p_start = max(0, q_peaks[-1] - int(0.15 * fs))  # 150 ms before Q
            p_end = q_peaks[-1]
            if p_start < p_end:
                p_index = p_start + np.argmax(signal[p_start:p_end])
                p_peaks.append(p_index)
        
        # T-wave: Look for maximum after S-wave
        if len(s_peaks) > 0:
            t_start = s_peaks[-1]
            t_end = min(len(signal), s_peaks[-1] + int(0.3 * fs))  # 300 ms after S
            if t_start < t_end:
                t_index = t_start + np.argmax(signal[t_start:t_end])
                t_peaks.append(t_index)
    
    # Package results
    pqrst_peaks = {
        'P': np.array(p_peaks),
        'Q': np.array(q_peaks),
        'R': np.array(r_peaks),
        'S': np.array(s_peaks),
        'T': np.array(t_peaks)
    }
    
    return pqrst_peaks

def calculate_heart_rate(r_peaks, fs=250, window_size=3):
    """
    Calculate instantaneous and average heart rate from R-peaks.
    
    Parameters:
        r_peaks (array): Array of R-peak indices
        fs (float): Sampling frequency (Hz)
        window_size (float): Window size in seconds for averaging
        
    Returns:
        tuple: (average_hr, inst_hr, hr_times)
            - average_hr: Average heart rate in BPM
            - inst_hr: Instantaneous heart rate values in BPM
            - hr_times: Time points for instantaneous heart rate measurements
    """
    if len(r_peaks) < 2:
        return 0, np.array([]), np.array([])
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    
    # Convert to BPM
    inst_heart_rates = 60 / rr_intervals
    
    # Time points for heart rate (at the midpoint of each RR interval)
    hr_times = r_peaks[:-1] / fs + rr_intervals / 2
    
    # Calculate average heart rate
    average_hr = np.mean(inst_heart_rates)
    
    return average_hr, inst_heart_rates, hr_times

def calculate_ecg_intervals(pqrst_peaks, fs=250):
    """
    Calculate important ECG intervals from PQRST peaks.
    
    Parameters:
        pqrst_peaks (dict): Dictionary of P, Q, R, S, and T peak indices
        fs (float): Sampling frequency (Hz)
        
    Returns:
        dict: Dictionary of ECG intervals in milliseconds
    """
    intervals = {}
    
    # Convert from samples to milliseconds
    ms_factor = 1000 / fs
    
    # RR intervals (time between consecutive R peaks)
    if len(pqrst_peaks['R']) > 1:
        rr_intervals = np.diff(pqrst_peaks['R']) * ms_factor
        intervals['RR'] = rr_intervals
        intervals['RR_mean'] = np.mean(rr_intervals)
        intervals['RR_std'] = np.std(rr_intervals)
        
    # PR interval (from P to R)
    pr_intervals = []
    for i in range(min(len(pqrst_peaks['P']), len(pqrst_peaks['R']))):
        if i < len(pqrst_peaks['P']) and i < len(pqrst_peaks['R']):
            pr_intervals.append((pqrst_peaks['R'][i] - pqrst_peaks['P'][i]) * ms_factor)
    if pr_intervals:
        intervals['PR'] = np.array(pr_intervals)
        intervals['PR_mean'] = np.mean(pr_intervals)
        intervals['PR_std'] = np.std(pr_intervals)
    
    # QRS duration (from Q to S)
    qrs_durations = []
    for i in range(min(len(pqrst_peaks['Q']), len(pqrst_peaks['S']))):
        if i < len(pqrst_peaks['Q']) and i < len(pqrst_peaks['S']):
            qrs_durations.append((pqrst_peaks['S'][i] - pqrst_peaks['Q'][i]) * ms_factor)
    if qrs_durations:
        intervals['QRS'] = np.array(qrs_durations)
        intervals['QRS_mean'] = np.mean(qrs_durations)
        intervals['QRS_std'] = np.std(qrs_durations)
    
    # QT interval (from Q to T)
    qt_intervals = []
    for i in range(min(len(pqrst_peaks['Q']), len(pqrst_peaks['T']))):
        if i < len(pqrst_peaks['Q']) and i < len(pqrst_peaks['T']):
            qt_intervals.append((pqrst_peaks['T'][i] - pqrst_peaks['Q'][i]) * ms_factor)
    if qt_intervals:
        intervals['QT'] = np.array(qt_intervals)
        intervals['QT_mean'] = np.mean(qt_intervals)
        intervals['QT_std'] = np.std(qt_intervals)
    
    # ST segment (from S to T)
    st_segments = []
    for i in range(min(len(pqrst_peaks['S']), len(pqrst_peaks['T']))):
        if i < len(pqrst_peaks['S']) and i < len(pqrst_peaks['T']):
            st_segments.append((pqrst_peaks['T'][i] - pqrst_peaks['S'][i]) * ms_factor)
    if st_segments:
        intervals['ST'] = np.array(st_segments)
        intervals['ST_mean'] = np.mean(st_segments)
        intervals['ST_std'] = np.std(st_segments)
    
    return intervals