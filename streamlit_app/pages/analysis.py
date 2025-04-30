"""
ECG Analysis page for the Streamlit app with enhanced file support.
Can handle a wide variety of file formats including .xyz and other custom extensions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time as time_module
import io
import tempfile
import zipfile
import json
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import helper functions
from streamlit_app.utils.helpers import (
    load_file, generate_synthetic_ecg, process_ecg, extract_heartbeats, 
    extract_features, plot_ecg_with_peaks, plot_heartbeats, 
    plot_average_heartbeat, get_download_link, get_figure_download_link,
    save_results
)

# Import processing modules
from src import data_processing, peak_detection, feature_extraction

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

def process_uploaded_file(uploaded_file, temp_dir=None, companion_files=None):
    """
    Process uploaded files of various formats and convert to standardized ECG data.
    Enhanced to handle a wider variety of file types including multi-channel binary formats.
    
    Parameters:
        uploaded_file: The uploaded file object
        temp_dir: Temporary directory for file operations
        companion_files: Dictionary of other uploaded files by name
        
    Returns:
        tuple: (time, ecg_signal, fs, filename) or (None, None, None, None) on failure
    """
    try:
        import logging
        import io
        import sys
        
        # Redirect stdout and stderr to capture library messages
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        try:
            # Set up logging to capture messages without displaying to user
            logger = logging.getLogger("file_processor")
            logger.setLevel(logging.ERROR)
            
            # Ensure temp_dir exists
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            
            # Show a processing message
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Extract file name and extension
                file_name = uploaded_file.name
                base_name = os.path.splitext(file_name)[0]
                file_extension = os.path.splitext(file_name)[1].lower()[1:] if '.' in file_name else ''
                
                # Save file to temp directory 
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Check if this is a WFDB file (.dat or .hea)
                if file_extension in ['dat', 'hea']:
                    # Process as WFDB file
                    return process_wfdb_file(uploaded_file, temp_dir, companion_files)
                
                # Process other file types
                if file_extension in ['csv', 'txt']:
                    result = process_csv_txt_file(uploaded_file)
                elif file_extension == 'json':
                    result = process_json_file(uploaded_file)
                elif file_extension == 'xml':
                    result = process_xml_file(uploaded_file)
                elif file_extension == 'mat':
                    result = process_mat_file(uploaded_file, temp_dir)
                elif file_extension == 'zip':
                    result = process_zip_file(uploaded_file, temp_dir)
                else:
                    # Try as binary file
                    for channels_to_try in [12, 8, 3, 2, 1]:
                        result = process_binary_ecg_file(file_path, file_name, num_channels=channels_to_try)
                        if result[0] is not None:
                            break
                    
                    # If binary processing failed, try generic
                    if result[0] is None:
                        result = process_generic_file(uploaded_file, temp_dir)
                
                # Check if processing was successful
                if result[0] is not None:
                    return result
                else:
                    # If all attempts failed
                    st.error(f"Could not process {file_name}. Please try another format.")
                    return None, None, None, None
        
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing {uploaded_file.name}. Please check the file and try again.")
        return None, None, None, None

def process_wfdb_file(uploaded_file, temp_dir, companion_files=None):
    """
    Process WFDB format files (MIT-BIH). Handles .hea and .dat pairs.
    
    Parameters:
        uploaded_file: The uploaded file object
        temp_dir: Temporary directory for file operations
        companion_files: Dictionary of other uploaded files by name
        
    Returns:
        tuple: (time, ecg_signal, fs, filename) or (None, None, None, None) on failure
    """
    try:
        import wfdb
        import logging
        
        # Set up logging
        logger = logging.getLogger("wfdb_processor")
        logger.setLevel(logging.ERROR)
        
        # Extract file info
        file_name = uploaded_file.name
        base_name = os.path.splitext(file_name)[0]
        file_extension = os.path.splitext(file_name)[1].lower()
        record_name = base_name
        
        # Check for companion file
        companion_file_name = None
        if file_extension == '.dat':
            companion_file_name = base_name + '.hea'
        elif file_extension == '.hea':
            companion_file_name = base_name + '.dat'
        
        # Check if companion file exists in uploaded files
        companion_file_path = None
        if companion_files and companion_file_name in companion_files:
            companion_file = companion_files[companion_file_name]
            companion_file_path = os.path.join(temp_dir, companion_file_name)
            # Save companion file if not already saved
            if not os.path.exists(companion_file_path):
                with open(companion_file_path, 'wb') as f:
                    companion_file.seek(0)
                    f.write(companion_file.read())
        
        # Check if companion file exists in temp directory
        if not companion_file_path or not os.path.exists(companion_file_path):
            companion_file_path = os.path.join(temp_dir, companion_file_name)
            if not os.path.exists(companion_file_path):
                st.warning(f"Companion file {companion_file_name} not found. This may affect processing quality.")
                # For .hea files, we must have a .dat file
                if file_extension == '.hea':
                    st.error(f"Data file {companion_file_name} is required to process this record.")
                    return None, None, None, None
                # For .dat files, we can try to create a minimal header
                elif file_extension == '.dat':
                    # Create a minimal header file
                    with open(companion_file_path, 'w') as f:
                        # Get file size to estimate the number of samples
                        dat_path = os.path.join(temp_dir, file_name)
                        dat_size = os.path.getsize(dat_path)
                        num_samples = dat_size // 2  # Assuming 16-bit samples
                        
                        # Write a minimal header
                        f.write(f"{record_name} 1 250 {num_samples}\n")
                        f.write("ECG 16 1 0 0 0 0 0\n")
                        st.info(f"Created minimal header file for {file_name}.")
        
        try:
            # Try to read the record using wfdb
            record_path = os.path.join(temp_dir, record_name)
            record = wfdb.rdrecord(record_path)
            
            # Create time array using actual sampling frequency from file
            time = np.arange(record.sig_len) / record.fs
            
            # Get signal data
            ecg_signal = record.p_signal if hasattr(record, 'p_signal') else record.d_signal
            
            # Ensure ecg_signal is 2D with shape (samples, channels)
            if len(ecg_signal.shape) == 1:
                ecg_signal = ecg_signal.reshape(-1, 1)
            
            st.success(f"Successfully processed WFDB record {record_name} with {ecg_signal.shape[1]} channel(s) at {record.fs} Hz.")
            return time, ecg_signal, record.fs, record_name
            
        except Exception as e:
            logger.error(f"Standard WFDB reading failed: {str(e)}")
            
            # Try alternative approach for binary data
            try:
                # Read header file to get sampling frequency and format
                fs = 250  # Default
                num_channels = 1  # Default
                
                hea_path = os.path.join(temp_dir, record_name + '.hea')
                if os.path.exists(hea_path):
                    with open(hea_path, 'r') as f:
                        header_lines = f.readlines()
                        if header_lines and len(header_lines[0].split()) >= 3:
                            parts = header_lines[0].split()
                            # Format: record_name num_signals sampling_frequency ...
                            if len(parts) >= 2 and parts[1].isdigit():
                                num_channels = int(parts[1])
                            if len(parts) >= 3 and parts[2].replace('.', '', 1).isdigit():
                                fs = float(parts[2])
                
                # Read the binary data file
                dat_path = os.path.join(temp_dir, record_name + '.dat')
                with open(dat_path, 'rb') as f:
                    binary_data = f.read()
                
                # Try to convert binary data to signal
                # Assume 16-bit integers (most common for ECG)
                values = np.frombuffer(binary_data, dtype='int16')
                
                # Reshape to handle multiple channels
                if num_channels > 1 and len(values) % num_channels == 0:
                    ecg_signal = values.reshape(-1, num_channels)
                else:
                    # Fallback to single channel
                    ecg_signal = values.reshape(-1, 1)
                
                # Create time array
                time = np.arange(len(ecg_signal)) / fs
                
                st.success(f"Successfully processed WFDB record {record_name} using alternative method.")
                return time, ecg_signal, fs, record_name
                
            except Exception as inner_e:
                logger.error(f"Alternative WFDB processing failed: {str(inner_e)}")
                st.error(f"Could not process {record_name}. Please check the file format.")
                return None, None, None, None
    
    except Exception as e:
        logger.error(f"Error processing WFDB file: {str(e)}")
        st.error(f"Error processing {file_name}. Please check the file and try again.")
        return None, None, None, None

def process_wfdb_pair(uploaded_files, temp_dir):
    """
    Process WFDB files as pairs (.dat and .hea with same base name).
    
    Parameters:
        uploaded_files: List of uploaded file objects
        temp_dir: Temporary directory for file operations
        
    Returns:
        tuple: (time, ecg_signal, fs, filename) or (None, None, None, None) on failure
    """
    try:
        import wfdb
        
        # Group files by base name
        files_by_name = {}
        for file in uploaded_files:
            # Save file to temp directory
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, 'wb') as f:
                file.seek(0)  # Reset file pointer
                f.write(file.read())
            
            # Group by base name
            base_name = os.path.splitext(file.name)[0]
            extension = os.path.splitext(file.name)[1].lower()
            
            if base_name not in files_by_name:
                files_by_name[base_name] = {}
            
            files_by_name[base_name][extension] = file.name
        
        # Process each file group that has both .dat and .hea
        for base_name, files in files_by_name.items():
            if '.dat' in files and '.hea' in files:
                # We have a complete pair
                try:
                    # Try to read the record
                    record_path = os.path.join(temp_dir, base_name)
                    record = wfdb.rdrecord(record_path)
                    
                    # Create time array
                    time = np.arange(record.sig_len) / record.fs
                    
                    # Get signal data
                    ecg_signal = record.p_signal if hasattr(record, 'p_signal') else record.d_signal
                    
                    # Ensure ecg_signal is 2D with shape (samples, channels)
                    if len(ecg_signal.shape) == 1:
                        ecg_signal = ecg_signal.reshape(-1, 1)
                    
                    st.success(f"Successfully processed WFDB record {base_name} with {ecg_signal.shape[1]} channel(s) at {record.fs} Hz.")
                    return time, ecg_signal, record.fs, base_name
                
                except Exception as e:
                    # Try alternative approach
                    try:
                        # Read header to get parameters
                        fs = 250  # Default
                        num_channels = 1  # Default
                        
                        hea_path = os.path.join(temp_dir, base_name + '.hea')
                        with open(hea_path, 'r') as f:
                            header_lines = f.readlines()
                            if header_lines and len(header_lines[0].split()) >= 3:
                                parts = header_lines[0].split()
                                if len(parts) >= 2 and parts[1].isdigit():
                                    num_channels = int(parts[1])
                                if len(parts) >= 3 and parts[2].replace('.', '', 1).isdigit():
                                    fs = float(parts[2])
                        
                        # Read binary data
                        dat_path = os.path.join(temp_dir, base_name + '.dat')
                        with open(dat_path, 'rb') as f:
                            binary_data = f.read()
                        
                        # Convert to signal
                        values = np.frombuffer(binary_data, dtype='int16')
                        
                        # Reshape based on channels
                        if num_channels > 1 and len(values) % num_channels == 0:
                            ecg_signal = values.reshape(-1, num_channels)
                        else:
                            ecg_signal = values.reshape(-1, 1)
                        
                        # Create time array
                        time = np.arange(len(ecg_signal)) / fs
                        
                        st.success(f"Successfully processed WFDB record {base_name} using alternative method.")
                        return time, ecg_signal, fs, base_name
                    
                    except Exception as inner_e:
                        st.error(f"Could not process {base_name}. Error: {str(inner_e)}")
        
        # If we get here, no valid pairs were found
        st.warning("No valid WFDB record pairs (.dat and .hea) were found.")
        return None, None, None, None
    
    except Exception as e:
        st.error(f"Error processing WFDB files: {str(e)}")
        return None, None, None, None

def process_csv_txt_file(uploaded_file):
    """Process CSV or TXT files."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Read the file with the current encoding
                uploaded_file.seek(0)
                content = uploaded_file.read().decode(encoding)
                buffer = io.StringIO(content)
                
                # Attempt to read with auto-separation detection
                df = pd.read_csv(buffer, sep=None, engine='python')
                
                # If only one column, try whitespace separator
                if len(df.columns) == 1:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, delim_whitespace=True)
                
                # Convert columns to numeric
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(axis=1, how='all')
                
                # Extract time and signal data
                if 'time' in df.columns or 'Time' in df.columns:
                    time_col = 'time' if 'time' in df.columns else 'Time'
                    time = df[time_col].values
                    
                    # Identify signal columns
                    signal_cols = [col for col in df.columns if col.lower() not in ['time', 'annotation']]
                    
                    # Extract signal data
                    ecg_signal = df[signal_cols].values
                    
                    # Ensure ecg_signal is 2D
                    if len(ecg_signal.shape) == 1:
                        ecg_signal = ecg_signal.reshape(-1, 1)
                        
                    # Calculate sampling rate from time column
                    if len(time) > 1:
                        fs = 1.0 / (time[1] - time[0])
                    else:
                        fs = 250  # Default
                else:
                    # Create time array if not present
                    time = np.arange(len(df))
                    ecg_signal = df.values
                    
                    # Ensure ecg_signal is 2D
                    if len(ecg_signal.shape) == 1:
                        ecg_signal = ecg_signal.reshape(-1, 1)
                        
                    fs = 250  # Default
                
                st.success(f"Successfully processed {uploaded_file.name}.")
                return time, ecg_signal, fs, uploaded_file.name
            
            except Exception:
                continue
        
        # If all encodings failed
        st.error(f"Could not parse {uploaded_file.name}. Try a different format.")
        return None, None, None, None
    
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None, None, None, None

def process_generic_file(uploaded_file, temp_dir=None):
    """
    Process a file with an unknown format by trying multiple methods.
    Attempts to interpret the file as a simple text-based format with columns of data.
    
    Parameters:
        uploaded_file: The uploaded file object
        temp_dir: Temporary directory for file operations
        
    Returns:
        tuple: (time, ecg_signal, fs, filename) or (None, None, None, None) on failure
    """
    try:
        import logging
        import io
        import sys
        
        # Redirect stdout and stderr to capture library messages
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        try:
            # Set up logging to capture messages without displaying to user
            logger = logging.getLogger("generic_processor")
            logger.setLevel(logging.ERROR)
            
            # Ensure temp_dir exists
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            
            # Save the file to temp directory
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Try direct binary reading first for non-standard formats like .xyz
            try:
                # Try to read as binary and assume 16-bit integers
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                
                # Check if file size makes sense as int16
                if len(binary_data) % 2 == 0:  # Must be even number of bytes for 16-bit integers
                    values = np.frombuffer(binary_data, dtype='int16')
                    
                    # Check if the values seem reasonable for ECG data
                    if len(values) > 0 and np.std(values) > 0 and np.max(np.abs(values)) < 1e6:
                        ecg_signal = values.reshape(-1, 1)
                        fs = 250  # Default sampling frequency
                        time = np.arange(len(ecg_signal)) / fs
                        
                        return time, ecg_signal, fs, uploaded_file.name
            except Exception:
                pass  # Silently continue to next method if binary reading fails
            
            # Try to determine if it's text or binary
            try:
                with open(file_path, 'rb') as f:
                    sample = f.read(1024)
                    is_binary = bool(sample.translate(None, bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})))
                
                if is_binary:
                    # Binary file - try other binary formats
                    for dtype, size in [('float32', 4), ('int32', 4), ('float64', 8), ('int8', 1)]:
                        if len(binary_data) % size == 0:
                            try:
                                values = np.frombuffer(binary_data, dtype=dtype)
                                if np.std(values) > 0 and np.max(np.abs(values)) < 1e6:
                                    ecg_signal = values.reshape(-1, 1)
                                    fs = 250  # Default sampling frequency
                                    time = np.arange(len(ecg_signal)) / fs
                                    return time, ecg_signal, fs, uploaded_file.name
                            except:
                                continue
                else:
                    # Text file - try to parse with various delimiters
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    delimiters = [None, ',', '\t', ' ', ';', '|']
                    
                    for encoding in encodings:
                        for delimiter in delimiters:
                            try:
                                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                                    df = pd.read_csv(f, sep=delimiter, engine='python', header=None, 
                                                    skiprows=0, error_bad_lines=False)
                                
                                # If successful, process the dataframe
                                if not df.empty:
                                    # Try to convert all columns to numeric
                                    df = df.apply(pd.to_numeric, errors='coerce')
                                    df = df.dropna(axis=1, how='all')
                                    
                                    if not df.empty:
                                        # Create time array
                                        time = np.arange(len(df))
                                        
                                        # Use all columns as signal data
                                        ecg_signal = df.values
                                        
                                        # Ensure ecg_signal is 2D
                                        if len(ecg_signal.shape) == 1:
                                            ecg_signal = ecg_signal.reshape(-1, 1)
                                        
                                        # Default sampling rate
                                        fs = 250
                                        
                                        return time, ecg_signal, fs, uploaded_file.name
                            except:
                                continue
            except Exception:
                pass  # Silently continue to next method
            
            # Last resort - try to read as text and extract numbers
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Try to extract all numbers from the file using regex
                numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', content)
                if numbers and len(numbers) > 10:
                    values = [float(num) for num in numbers]
                    ecg_signal = np.array(values).reshape(-1, 1)
                    fs = 250  # Default sampling rate
                    time = np.arange(len(ecg_signal)) / fs
                    
                    return time, ecg_signal, fs, uploaded_file.name
            except Exception:
                pass  # Silently continue
            
            # If we get here, all methods failed
            return None, None, None, None
        
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
    except Exception as e:
        # Log detailed error but show generic message to user
        logging.error(f"Error in generic file processor: {str(e)}")
        return None, None, None, None

def process_binary_ecg_file(file_path, filename, num_channels=1):
    """
    Process binary ECG file with support for multiple channels.
    
    Parameters:
        file_path: Path to the file
        filename: Original filename for reporting
        num_channels: Number of channels in the data (default: 1)
        
    Returns:
        tuple: (time, ecg_signal, fs, filename) or (None, None, None, None) on failure
    """
    try:
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        
        # Try different data types
        for dtype, size in [('int16', 2), ('float32', 4), ('int32', 4), ('float64', 8), ('int8', 1)]:
            if len(binary_data) % (size * num_channels) == 0:
                try:
                    # Convert binary data to numpy array
                    values = np.frombuffer(binary_data, dtype=dtype)
                    total_samples = len(values)
                    
                    # Reshape to handle multiple channels
                    if num_channels > 1:
                        # Check if the data can be reshaped to the specified number of channels
                        if total_samples % num_channels == 0:
                            samples_per_channel = total_samples // num_channels
                            # Reshape to (samples, channels) format
                            # Two possible layouts: interleaved or sequential blocks
                            
                            # Try interleaved layout first (ch1, ch2, ch3, ch1, ch2, ch3, ...)
                            try:
                                ecg_signal = values.reshape(-1, num_channels)
                                
                                # Quick validation - channels should be somewhat correlated in ECG
                                if ecg_signal.shape[0] > 100:  # Need enough data to validate
                                    # Calculate correlations between first channel and others
                                    correlations = []
                                    for i in range(1, num_channels):
                                        corr = np.corrcoef(ecg_signal[:100, 0], ecg_signal[:100, i])[0, 1]
                                        correlations.append(abs(corr))
                                    
                                    avg_corr = np.mean(correlations) if correlations else 0
                                    
                                    # If average correlation is very low, try sequential layout
                                    if avg_corr < 0.1:
                                        raise ValueError("Low correlation suggests wrong layout")
                            except:
                                # Try sequential block layout (all ch1, all ch2, all ch3, ...)
                                ecg_signal = np.zeros((samples_per_channel, num_channels))
                                for i in range(num_channels):
                                    start_idx = i * samples_per_channel
                                    end_idx = (i + 1) * samples_per_channel
                                    ecg_signal[:, i] = values[start_idx:end_idx]
                            
                            # Create time array
                            fs = 250  # Default sampling frequency
                            time = np.arange(ecg_signal.shape[0]) / fs
                            
                            return time, ecg_signal, fs, filename
                        
                    # Single channel case
                    else:
                        ecg_signal = values.reshape(-1, 1)
                        fs = 250  # Default sampling frequency
                        time = np.arange(len(ecg_signal)) / fs
                        return time, ecg_signal, fs, filename
                except Exception as e:
                    continue
        
        return None, None, None, None
    
    except Exception as e:
        return None, None, None, None

def process_json_file(uploaded_file):
    """Process JSON format files."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Try to read the file with the current encoding
                uploaded_file.seek(0)
                content = uploaded_file.read().decode(encoding)
                
                # Parse JSON data
                json_data = json.loads(content)
                
                # Try to identify ECG data in the JSON structure
                ecg_data = None
                
                # Look for common patterns in ECG JSON files
                # Look for common patterns in ECG JSON files
                if isinstance(json_data, list):
                    # Check if it's a list of data points
                    ecg_data = json_data
                elif isinstance(json_data, dict):
                    # Look for keys that might contain ECG data
                    potential_keys = ['data', 'ecg', 'ecg_data', 'signal', 'values', 'samples', 
                                     'measurements', 'points', 'result', 'results', 'waveform']
                    
                    for key in potential_keys:
                        if key in json_data and isinstance(json_data[key], (list, np.ndarray)):
                            ecg_data = json_data[key]
                            break
                    
                    # If no known keys are found, try to find any array in the data
                    if ecg_data is None:
                        for key, value in json_data.items():
                            if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                                ecg_data = value
                                break
                
                if ecg_data is not None:
                    # Convert to numpy array
                    if isinstance(ecg_data[0], (dict, list)):
                        # Handle nested structures
                        df = pd.json_normalize(ecg_data)
                        df = df.apply(pd.to_numeric, errors='coerce')
                        df = df.dropna(axis=1, how='all')
                        ecg_signal = df.values
                    else:
                        # Simple list of values
                        ecg_signal = np.array(ecg_data, dtype=float)
                    
                    # Ensure ecg_signal is 2D with shape (samples, channels)
                    if len(ecg_signal.shape) == 1:
                        ecg_signal = ecg_signal.reshape(-1, 1)
                    
                    # Create time array with default sampling rate
                    fs = 250  # Default
                    time = np.arange(len(ecg_signal)) / fs
                    
                    return time, ecg_signal, fs, uploaded_file.name
                else:
                    st.warning("Could not identify ECG data in the JSON file.")
                    return None, None, None, None
            
            except Exception:
                # Try the next encoding
                continue
        
        # If we get here, none of the encodings worked
        raise Exception("Could not parse JSON file with any supported encoding")
    
    except Exception as e:
        st.error(f"Error processing JSON file: {str(e)}")
        return None, None, None, None

def process_xml_file(uploaded_file):
    """Process XML format files."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Try to read the file with the current encoding
                uploaded_file.seek(0)
                content = uploaded_file.read().decode(encoding)
                
                # Parse XML data
                tree = ET.fromstring(content)
                
                # Try different strategies to find ECG data in XML
                ecg_data = []
                
                # Strategy 1: Look for specific tags that might contain ECG values
                potential_tags = ['data', 'value', 'sample', 'ecg', 'point', 'measurement', 
                                 'amplitude', 'waveform', 'signal', 'result']
                
                for tag in potential_tags:
                    elements = tree.findall(f'.//{tag}')
                    if elements:
                        try:
                            values = []
                            for element in elements:
                                if element.text and element.text.strip():
                                    values.append(float(element.text.strip()))
                            
                            if len(values) > 10:  # Only consider if we found enough values
                                ecg_data = values
                                break
                        except ValueError:
                            # If conversion to float fails, this isn't numerical data
                            continue
                
                # Strategy 2: Try to extract values from attributes
                if not ecg_data:
                    for attr in ['value', 'data', 'sample', 'amplitude', 'measurement']:
                        values = []
                        for element in tree.iter():
                            if attr in element.attrib:
                                try:
                                    values.append(float(element.attrib[attr]))
                                except ValueError:
                                    continue
                        
                        if len(values) > 10:  # Only consider if we found enough values
                            ecg_data = values
                            break
                
                # If strategies failed, try to extract all numbers from the XML text
                if not ecg_data:
                    xml_string = ET.tostring(tree, encoding='unicode')
                    numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', xml_string)
                    
                    if len(numbers) > 10:
                        ecg_data = [float(num) for num in numbers]
                
                if ecg_data:
                    # Convert to numpy array
                    ecg_signal = np.array(ecg_data).reshape(-1, 1)
                    
                    # Create time array with default sampling rate
                    fs = 250  # Default
                    time = np.arange(len(ecg_signal)) / fs
                    
                    return time, ecg_signal, fs, uploaded_file.name
                else:
                    continue  # Try next encoding if no data found
            
            except Exception:
                # Try the next encoding
                continue
        
        # If we get here, none of the encodings worked
        st.warning("Could not identify ECG data in the XML file.")
        return None, None, None, None
    
    except Exception as e:
        st.error(f"Error processing XML file: {str(e)}")
        return None, None, None, None

def process_mat_file(uploaded_file, temp_dir):
    """Process MATLAB .mat files."""
    try:
        # Ensure temp_dir exists
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        # Save the uploaded .mat file
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the .mat file
        mat_data = loadmat(file_path)
        
        # Attempt to find the ECG data in the .mat file
        ecg_data = None
        for key, value in mat_data.items():
            if key.startswith('__'): continue  # Skip metadata
            if isinstance(value, np.ndarray) and value.size > 10:
                if len(value.shape) <= 2:  # Only consider 1D or 2D arrays
                    # Check if values are reasonable for ECG data
                    if np.std(value) > 0 and np.max(np.abs(value)) < 1e6:
                        ecg_data = value
                        break
        
        if ecg_data is not None:
            # Check if we need to transpose the data
            if len(ecg_data.shape) == 2 and ecg_data.shape[0] > ecg_data.shape[1]:
                ecg_signal = ecg_data
            else:
                ecg_signal = ecg_data.T if len(ecg_data.shape) == 2 else ecg_data.reshape(-1, 1)
            
            # Create time array with default sampling rate
            fs = 250  # Default sampling frequency
            time = np.arange(len(ecg_signal)) / fs
            
            return time, ecg_signal, fs, uploaded_file.name
        else:
            st.warning("Could not extract ECG data from the MAT file.")
            return None, None, None, None
    
    except Exception as e:
        st.error(f"Error processing MAT file: {str(e)}")
        return None, None, None, None

def process_zip_file(uploaded_file, temp_dir):
    """Process ZIP archives containing ECG data files."""
    try:
        # Ensure temp_dir exists
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        # Save the zip file
        zip_path = os.path.join(temp_dir, uploaded_file.name)
        with open(zip_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Look for ECG data files in the extracted contents
        ecg_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                # Include common ECG file extensions plus support for nonstandard extensions
                if file.endswith(('.csv', '.txt', '.dat', '.json', '.xml', '.mat', 
                                 '.ecg', '.edf', '.scp', '.hl7', '.dicom', '.xyz')):
                    ecg_files.append((os.path.join(root, file), file))
        
        if not ecg_files:
            # If no known extensions found, try to process any non-system file
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    # Skip system files and already checked extensions
                    if not file.startswith('.') and not file.endswith(('.zip', '.hea')):
                        ecg_files.append((os.path.join(root, file), file))
        
        if not ecg_files:
            st.warning("No potential ECG data files found in the ZIP archive.")
            return None, None, None, None
        
        # Try to process each file until one succeeds
        for file_path, file_name in ecg_files:
            st.info(f"Attempting to process {file_name} from the ZIP archive.")
            
            # Get file extension
            extension = os.path.splitext(file_path)[1].lower()[1:]
            
            # Process based on extension or use generic processor
            if extension in ['csv', 'txt']:
                with open(file_path, 'r', errors='ignore') as f:
                    try:
                        # Try to read as CSV
                        df = pd.read_csv(f, sep=None, engine='python', error_bad_lines=False)
                        
                        # Process dataframe 
                        df = df.apply(pd.to_numeric, errors='coerce')
                        df = df.dropna(axis=1, how='all')
                        
                        if not df.empty:
                            # Use the first column as time if it's monotonically increasing
                            if df.shape[1] > 1 and df.iloc[:, 0].is_monotonic_increasing:
                                time = df.iloc[:, 0].values
                                signal_cols = df.iloc[:, 1:]
                            else:
                                # Create a time array
                                time = np.arange(len(df))
                                signal_cols = df
                            
                            # Extract signal data
                            ecg_signal = signal_cols.values
                            
                            # Ensure ecg_signal is 2D with shape (samples, channels)
                            if len(ecg_signal.shape) == 1:
                                ecg_signal = ecg_signal.reshape(-1, 1)
                                
                            # Estimate or set default sampling rate
                            if len(time) > 1 and np.issubdtype(time.dtype, np.number) and time[1] > time[0]:
                                fs = 1.0 / (time[1] - time[0])
                            else:
                                fs = 250  # Default
                            
                            return time, ecg_signal, fs, file_name
                    except Exception as e:
                        st.warning(f"Error reading {file_name} as CSV: {str(e)}")
            
            # Continue processing other file types...
            # (Code for other file types would go here)
            
        # If we get here, no valid files were found
        st.error("No files in the ZIP archive could be processed as ECG data.")
        return None, None, None, None
        
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return None, None, None, None

def convert_to_csv(time, ecg_signal, filename):
    """
    Convert processed ECG data to CSV format.
    
    Parameters:
        time: Time array
        ecg_signal: ECG signal data
        filename: Original filename
        
    Returns:
        tuple: (csv_data, csv_filename)
    """
    try:
        # Create DataFrame
        df = pd.DataFrame({"time": time})
        
        # Add channels
        for i in range(ecg_signal.shape[1]):
            df[f"channel_{i+1}"] = ecg_signal[:, i]
        
        # Convert to CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        # Create filename
        base_name = os.path.splitext(filename)[0]
        csv_filename = f"{base_name}_converted.csv"
        
        return csv_data, csv_filename
    
    except Exception as e:
        st.error(f"Error converting to CSV: {str(e)}")
        return None, None

def show():
    """Show the ECG Analysis page."""
    st.title("ECG Signal Analysis")
    
    # Initialize session state variables
    if 'ecg_data' not in st.session_state:
        st.session_state.ecg_data = None
    if 'processed_signal' not in st.session_state:
        st.session_state.processed_signal = None
    if 'r_peaks' not in st.session_state:
        st.session_state.r_peaks = None
    if 'pqrst_peaks' not in st.session_state:
        st.session_state.pqrst_peaks = None
    if 'intervals' not in st.session_state:
        st.session_state.intervals = None
    if 'heart_rate' not in st.session_state:
        st.session_state.heart_rate = None
    if 'heartbeats' not in st.session_state:
        st.session_state.heartbeats = None
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None
    
    # Sidebar for data input and parameters
    with st.sidebar:
        st.header("Data Input")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Generate Synthetic ECG"]
        )
        
        if data_source == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload ECG data file(s):",
                type=None,  # Allow any file type
                accept_multiple_files=True,
                help="Upload any ECG data files - all formats will be attempted"
            )
    
            if uploaded_files:
                # Create action buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Load Data", key="load_data_button"):
                        with st.spinner("Loading data..."):
                            # Create temporary directory
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Check for WFDB files
                                has_wfdb_files = any(f.name.lower().endswith(('.dat', '.hea')) for f in uploaded_files)
                                
                                if has_wfdb_files:
                                    # Process as WFDB pair
                                    time, ecg_signal, fs, filename = process_wfdb_pair(uploaded_files, temp_dir)
                                else:
                                    # Create dictionary of uploaded files by name
                                    companion_files = {f.name: f for f in uploaded_files}
                                    
                                    # Try to process each file
                                    time, ecg_signal, fs, filename = None, None, None, None
                                    for file in uploaded_files:
                                        result = process_uploaded_file(file, temp_dir, companion_files)
                                        if result[0] is not None:
                                            time, ecg_signal, fs, filename = result
                                            break
                                
                                if time is not None and ecg_signal is not None and fs is not None:
                                    st.session_state.ecg_data = (time, ecg_signal, fs)
                                    st.session_state.filename = filename
                                    st.success(f"Successfully loaded {filename} with {ecg_signal.shape[1]} channel(s) at {fs} Hz")
                                    
                                    # Reset processing results
                                    st.session_state.processed_signal = None
                                    st.session_state.r_peaks = None
                                    st.session_state.pqrst_peaks = None
                                    st.session_state.intervals = None
                                    st.session_state.heart_rate = None
                                    st.session_state.heartbeats = None
                                    st.session_state.features_df = None
                                else:
                                    st.error("Failed to load ECG data. Please check the file format.")
                
                with col2:
                    if st.button("Convert to CSV", key="convert_to_csv_button"):
                        with st.spinner("Converting files to CSV..."):
                            # Create temporary directory
                            with tempfile.TemporaryDirectory() as temp_dir:
                                converted_files = []
                                
                                # Check for WFDB files
                                has_wfdb_files = any(f.name.lower().endswith(('.dat', '.hea')) for f in uploaded_files)
                                
                                if has_wfdb_files:
                                    # Process as WFDB pair
                                    time, ecg_signal, fs, filename = process_wfdb_pair(uploaded_files, temp_dir)
                                    
                                    if time is not None and ecg_signal is not None:
                                        # Convert to CSV
                                        csv_data, csv_filename = convert_to_csv(time, ecg_signal, filename)
                                        if csv_data is not None:
                                            converted_files.append((csv_filename, csv_data))
                                            st.success(f"Successfully converted {filename} to CSV")
                                
                                # Create dictionary of uploaded files by name
                                companion_files = {f.name: f for f in uploaded_files}
                                
                                # Process remaining files individually
                                for file in uploaded_files:
                                    # Skip WFDB files if we already processed them as a pair
                                    if has_wfdb_files and file.name.lower().endswith(('.dat', '.hea')):
                                        continue
                                    
                                    # Process the file
                                    time, ecg_signal, fs, filename = process_uploaded_file(file, temp_dir, companion_files)
                                    
                                    if time is not None and ecg_signal is not None:
                                        # Convert to CSV
                                        csv_data, csv_filename = convert_to_csv(time, ecg_signal, filename)
                                        if csv_data is not None:
                                            converted_files.append((csv_filename, csv_data))
                                            st.success(f"Successfully converted {filename} to CSV")
                                
                                # Create download buttons
                                if converted_files:
                                    st.subheader("Download Converted CSV Files")
                                    for csv_filename, csv_data in converted_files:
                                        st.download_button(
                                            label=f"Download {csv_filename}",
                                            data=csv_data,
                                            file_name=csv_filename,
                                            mime="text/csv",
                                            key=f"download_{csv_filename}"
                                        )
                                    
                                    # Also load the first converted file for analysis
                                    if len(converted_files) > 0:
                                        first_filename, first_csv_data = converted_files[0]
                                        
                                        # Load CSV data
                                        df = pd.read_csv(io.StringIO(first_csv_data.decode('utf-8')))
                                        time = df['time'].values
                                        
                                        # Extract signal columns
                                        signal_cols = [col for col in df.columns if col != 'time']
                                        ecg_signal = df[signal_cols].values
                                        
                                        # Calculate sampling rate
                                        if len(time) > 1:
                                            fs = 1.0 / (time[1] - time[0])
                                        else:
                                            fs = 250
                                        
                                        # Store in session state
                                        st.session_state.ecg_data = (time, ecg_signal, fs)
                                        st.session_state.filename = first_filename
                                        
                                        # Reset processing results
                                        st.session_state.processed_signal = None
                                        st.session_state.r_peaks = None
                                        st.session_state.pqrst_peaks = None
                                        st.session_state.intervals = None
                                        st.session_state.heart_rate = None
                                        st.session_state.heartbeats = None
                                        st.session_state.features_df = None
                                        
                                        st.success(f"Loaded {first_filename} for analysis")
                                else:
                                    st.error("No files were successfully converted to CSV")
        
        else:  # Generate Synthetic ECG
            st.subheader("Synthetic ECG Parameters")
            
            duration = st.slider(
                "Duration (seconds):",
                min_value=5,
                max_value=60,
                value=10,
                step=5
            )
            
            fs = st.slider(
                "Sampling Frequency (Hz):",
                min_value=100,
                max_value=1000,
                value=250,
                step=50
            )
            
            heart_rate = st.slider(
                "Heart Rate (BPM):",
                min_value=40,
                max_value=200,
                value=70,
                step=5
            )
            
            noise_level = st.slider(
                "Noise Level:",
                min_value=0.0,
                max_value=0.5,
                value=0.05,
                step=0.05
            )
            
            if st.button("Generate ECG", key="generate_ecg_button"):
                with st.spinner("Generating synthetic ECG..."):
                    time, ecg_signal, fs = generate_synthetic_ecg(
                        duration=duration,
                        fs=fs,
                        heart_rate=heart_rate,
                        noise_level=noise_level
                    )
                    
                    st.session_state.ecg_data = (time, ecg_signal, fs)
                    st.session_state.filename = "synthetic_ecg.csv"
                    st.success("Successfully generated synthetic ECG")
                    
                    # Reset processing results
                    st.session_state.processed_signal = None
                    st.session_state.r_peaks = None
                    st.session_state.pqrst_peaks = None
                    st.session_state.intervals = None
                    st.session_state.heart_rate = None
                    st.session_state.heartbeats = None
                    st.session_state.features_df = None
        
        # Processing parameters (only show if data is loaded)
        if st.session_state.ecg_data is not None:
            st.header("Processing Parameters")
            
            st.subheader("Filtering")
            lowcut = st.slider(
                "Low Cutoff Frequency (Hz):",
                min_value=0.5,
                max_value=10.0,
                value=5.0,
                step=0.5
            )
            
            highcut = st.slider(
                "High Cutoff Frequency (Hz):",
                min_value=15.0,
                max_value=100.0,
                value=30.0,
                step=5.0
            )
            
            powerline_freq = st.radio(
                "Powerline Frequency (Hz):",
                [50, 60],
                index=0
            )
            
            st.subheader("R-Peak Detection")
            r_peak_method = st.radio(
                "Detection Method:",
                ["pan_tompkins", "xqrs"],
                index=1,
                format_func=lambda x: "Pan-Tompkins" if x == "pan_tompkins" else "XQRS"
            )
            
            # Create a dictionary of processing parameters
            processing_params = {
                'lowcut': lowcut,
                'highcut': highcut,
                'powerline_freq': powerline_freq,
                'r_peak_method': r_peak_method
            }
            
            # Process button
            if st.button("Process ECG", key="process_ecg_button"):
                with st.spinner("Processing ECG signal..."):
                    # Extract data from session state
                    time, ecg_signal, fs = st.session_state.ecg_data
                    
                    # Process ECG signal
                    processed_signal, r_peaks, pqrst_peaks, intervals, heart_rate = process_ecg(
                        time, ecg_signal, fs, processing_params
                    )
                    
                    if processed_signal is not None:
                        # Store results in session state
                        st.session_state.processed_signal = processed_signal
                        st.session_state.r_peaks = r_peaks
                        st.session_state.pqrst_peaks = pqrst_peaks
                        st.session_state.intervals = intervals
                        st.session_state.heart_rate = heart_rate
                        
                        # Reset heartbeats and features
                        st.session_state.heartbeats = None
                        st.session_state.features_df = None
                        
                        st.success("ECG processing completed")
    
    # Main content area
    if st.session_state.ecg_data is None:
        st.info("Please load ECG data from the sidebar.")
    
    else:
        # Show basic information
        time, ecg_signal, fs = st.session_state.ecg_data
        
        st.header("ECG Data Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sampling Rate", f"{fs} Hz")
        with col2:
            st.metric("Duration", f"{len(time)/fs:.2f} sec")
        with col3:
            st.metric("Channels", f"{ecg_signal.shape[1]}")
        
        # Show raw ECG data
        st.subheader("Raw ECG Signal")
        
        # Allow selection of view duration
        view_duration = st.slider(
            "View Duration (seconds):",
            min_value=1.0,
            max_value=min(30.0, len(time)/fs),
            value=5.0,
            step=1.0
        )
        
        # Allow selection of start time
        max_start_time = float(max(0, len(time)/fs - view_duration))
        start_time = st.slider(
            "Start Time (seconds):",
            min_value=0.0,
            max_value=max_start_time,
            value=0.0,
            step=1.0
        )
        
        # Allow selection of channel
        channel = st.selectbox(
            "Channel:",
            range(ecg_signal.shape[1]),
            format_func=lambda x: f"Channel {x+1}"
        )
        
        # Plot raw ECG
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Convert time to seconds for plotting
        start_sample = int(start_time * fs)
        end_sample = min(len(time), start_sample + int(view_duration * fs))
        
        # Plot ECG signal
        ax.plot(time[start_sample:end_sample], ecg_signal[start_sample:end_sample, channel])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Raw ECG Signal - Channel {channel+1}")
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Download link for raw ECG
        download_df = pd.DataFrame({"Time": time})
        for i in range(ecg_signal.shape[1]):
            download_df[f"Channel_{i+1}"] = ecg_signal[:, i]
        
        st.markdown(get_download_link(download_df, "raw_ecg.csv", "Download Raw ECG Data"), unsafe_allow_html=True)
        
        # Display processed ECG if available
        if st.session_state.processed_signal is not None:
            st.header("ECG Analysis Results")
            
            # Show heart rate
            if st.session_state.heart_rate is not None:
                st.metric("Heart Rate", f"{st.session_state.heart_rate:.1f} BPM")
            
            # Show processed ECG with detected peaks
            st.subheader("ECG Signal with Detected Peaks")
            
            # Plot processed ECG with peaks
            fig = plot_ecg_with_peaks(
                time, 
                ecg_signal, 
                st.session_state.processed_signal, 
                st.session_state.pqrst_peaks, 
                fs,
                start_time=start_time,
                duration=view_duration,
                channel=channel
            )
            
            st.pyplot(fig)
            
            # Download links
            st.markdown(get_figure_download_link(fig, "ecg_with_peaks.png", "Download Figure"), unsafe_allow_html=True)
            
            # Show ECG intervals if available
            if st.session_state.intervals is not None:
                st.subheader("ECG Intervals")
                
                # Create a DataFrame of interval statistics
                interval_stats = {}
                for name, value in st.session_state.intervals.items():
                    if name.endswith("_mean") or name.endswith("_std"):
                        interval_stats[name] = value
                
                interval_df = pd.DataFrame([interval_stats])
                
                # Rename columns for better display
                rename_dict = {
                    "RR_mean": "RR Interval (ms)",
                    "PR_mean": "PR Interval (ms)",
                    "QRS_mean": "QRS Duration (ms)",
                    "QT_mean": "QT Interval (ms)",
                    "ST_mean": "ST Segment (ms)",
                    "RR_std": "RR Std Dev (ms)",
                    "PR_std": "PR Std Dev (ms)",
                    "QRS_std": "QRS Std Dev (ms)",
                    "QT_std": "QT Std Dev (ms)",
                    "ST_std": "ST Std Dev (ms)"
                }
                
                interval_df = interval_df.rename(columns=rename_dict)
                
                # Display as table
                st.dataframe(interval_df)
                
                # Download link
                st.markdown(get_download_link(interval_df, "ecg_intervals.csv", "Download Interval Data"), unsafe_allow_html=True)
            
            # Heartbeat segmentation section
            st.header("Heartbeat Segmentation")
            
            # Parameters for heartbeat segmentation
            col1, col2 = st.columns(2)
            with col1:
                before_r = st.slider(
                    "Time before R-peak (s):",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.25,
                    step=0.05
                )
            
            with col2:
                after_r = st.slider(
                    "Time after R-peak (s):",
                    min_value=0.1,
                    max_value=0.8,
                    value=0.45,
                    step=0.05
                )
            
            # Window parameters
            window_params = {
                'before_r': before_r,
                'after_r': after_r
            }
            
            # Extract heartbeats button
            if st.button("Extract Heartbeats", key="extract_heartbeats_button"):
                with st.spinner("Extracting heartbeats..."):
                    # Extract heartbeats
                    heartbeats = extract_heartbeats(
                        st.session_state.processed_signal[:, channel],  # Use selected channel
                        st.session_state.r_peaks,
                        fs,
                        window_params
                    )
                    
                    if heartbeats is not None and len(heartbeats) > 0:
                        st.session_state.heartbeats = heartbeats
                        st.success(f"Extracted {len(heartbeats)} heartbeats")
                    else:
                        st.error("Failed to extract heartbeats")
            
            # Display heartbeats if available
            if st.session_state.heartbeats is not None:
                st.subheader("Individual Heartbeats")
                
                # Number of heartbeats to display
                num_beats = st.slider(
                    "Number of beats to display:",
                    min_value=1,
                    max_value=min(10, len(st.session_state.heartbeats)),
                    value=5,
                    step=1
                )
                
                # Plot heartbeats
                fig = plot_heartbeats(st.session_state.heartbeats, fs, num_beats=num_beats)
                st.pyplot(fig)
                
                # Plot average heartbeat
                st.subheader("Average Heartbeat")
                fig = plot_average_heartbeat(st.session_state.heartbeats, fs)
                st.pyplot(fig)
                
                # Feature extraction section
                st.header("Feature Extraction")
                
                # Parameters for feature extraction
                include_advanced = st.checkbox("Include Advanced Features", value=True)
                
                # Feature parameters
                feature_params = {
                    'include_advanced': include_advanced
                }
                
                # Extract features button
                if st.button("Extract Features", key="extract_features_button"):
                    with st.spinner("Extracting features..."):
                        # Extract features
                        features_df = extract_features(
                            st.session_state.heartbeats,
                            fs,
                            feature_params
                        )
                        
                        if features_df is not None:
                            st.session_state.features_df = features_df
                            st.success(f"Extracted {features_df.shape[1]} features from {features_df.shape[0]} heartbeats")
                        else:
                            st.error("Failed to extract features")
                
                # Display features if available
                if st.session_state.features_df is not None:
                    st.subheader("Extracted Features")
                    
                    # Display feature summary
                    st.dataframe(st.session_state.features_df.describe())
                    
                    # Download link for features
                    st.markdown(get_download_link(st.session_state.features_df, "heartbeat_features.csv", "Download Features"), unsafe_allow_html=True)
                    
                    # Feature visualization
                    st.subheader("Feature Visualization")
                    
                    # Select feature to visualize
                    feature_options = st.session_state.features_df.select_dtypes(include=[np.number]).columns.tolist()
                    selected_feature = st.selectbox("Select Feature:", feature_options)
                    
                    if selected_feature:
                        # Plot feature distribution
                        fig = plot_feature_distribution(st.session_state.features_df, selected_feature)
                        st.pyplot(fig)
            
            # Save results section
            st.header("Save Analysis Results")
            
            # Output directory
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
            
            # Save button
            if st.button("Save All Results", key="save_results_button"):
                with st.spinner("Saving results..."):
                    # Prepare metadata
                    metadata = {
                        'filename': st.session_state.filename,
                        'heart_rate': st.session_state.heart_rate,
                        'processing_params': processing_params
                    }
                    
                    # Save results
                    results_dir = save_results(
                        output_dir,
                        st.session_state.ecg_data,
                        st.session_state.processed_signal,
                        st.session_state.r_peaks,
                        st.session_state.pqrst_peaks,
                        st.session_state.intervals,
                        st.session_state.features_df,
                        metadata
                    )
                    
                    if results_dir:
                        st.success(f"Results saved to {results_dir}")
                    else:
                        st.error("Failed to save results")