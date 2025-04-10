"""
ECG Analysis page for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time as time_module

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

def show():
    """Show the ECG Analysis page."""
    st.title("ECG Signal Analysis")
    
    # Initialize session state variables if they don't exist
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
    
    # Create sidebar for data input and parameters
    with st.sidebar:
        st.header("Data Input")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Generate Synthetic ECG"]
        )
        
        if data_source == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload ECG data file(s):",
                type=['csv', 'txt', 'dat', 'hea'],
                accept_multiple_files=True,
                help="Supported formats: CSV, TXT, or WFDB files (.dat and .hea together)"
            )
    
            if uploaded_files:
                if st.button("Load Data"):
                    with st.spinner("Loading data..."):
                        time, ecg_signal, fs, filename = load_file(uploaded_files)
                    
                    if time is not None and ecg_signal is not None and fs is not None:
                        st.session_state.ecg_data = (time, ecg_signal, fs)
                        st.session_state.filename = filename
                        st.success(f"Successfully loaded {filename}")
                        
                        # Reset processing results
                        st.session_state.processed_signal = None
                        st.session_state.r_peaks = None
                        st.session_state.pqrst_peaks = None
                        st.session_state.intervals = None
                        st.session_state.heart_rate = None
                        st.session_state.heartbeats = None
                        st.session_state.features_df = None
                
            if uploaded_files is not None:
                if st.button("Load Data"):
                    with st.spinner("Loading data..."):
                        time, ecg_signal, fs, filename = load_file(uploaded_files)
                        
                        if time is not None and ecg_signal is not None and fs is not None:
                            st.session_state.ecg_data = (time, ecg_signal, fs)
                            st.session_state.filename = filename
                            st.success(f"Successfully loaded {filename}")
                            
                            # Reset processing results
                            st.session_state.processed_signal = None
                            st.session_state.r_peaks = None
                            st.session_state.pqrst_peaks = None
                            st.session_state.intervals = None
                            st.session_state.heart_rate = None
                            st.session_state.heartbeats = None
                            st.session_state.features_df = None
        
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
            
            if st.button("Generate ECG"):
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
            if st.button("Process ECG"):
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
        max_start_time = max(0, len(time)/fs - view_duration)
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
            if st.button("Extract Heartbeats"):
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
                if st.button("Extract Features"):
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
            if st.button("Save All Results"):
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