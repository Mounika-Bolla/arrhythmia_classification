"""
ECG Arrhythmia Classification page for the Streamlit app.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time as time_module
import joblib
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import helper functions
from streamlit_app.utils.helpers import (
    load_file, generate_synthetic_ecg, process_ecg, extract_heartbeats, 
    plot_ecg_with_peaks, get_download_link, get_figure_download_link
)

# Import processing modules
from src import data_processing, peak_detection, feature_extraction, arrhythmia_classifier

def show():
    """Show the ECG Arrhythmia Classification page."""
    st.title("ECG Arrhythmia Classification")
    
    # Sidebar for options
    with st.sidebar:
        st.header("Classification Options")
        
        # Model selection
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'saved')
        
        # Make sure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Find available models
        try:
            model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        except:
            model_dirs = []
        
        if not model_dirs:
            st.warning("No classification models found. Please train models using the arrhythmia classification notebook.")
            model_options = ["None"]
            selected_model = "None"
        else:
            model_options = ["ensemble"] + [d for d in model_dirs if d != "ensemble"]
            selected_model = st.selectbox(
                "Select Classification Model:",
                model_options,
                index=0 if "ensemble" in model_options else 0,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        # Classification threshold
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum probability required for classification"
        )
        
        # Beat segmentation parameters
        st.subheader("Heartbeat Segmentation")
        before_r = st.slider(
            "Pre-R Peak Window (s):",
            min_value=0.1,
            max_value=0.5,
            value=0.25,
            step=0.05
        )
        
        after_r = st.slider(
            "Post-R Peak Window (s):",
            min_value=0.1,
            max_value=0.8,
            value=0.45,
            step=0.05
        )
    
    # Main area
    if 'ecg_data' not in st.session_state or st.session_state.ecg_data is None:
        st.info("No ECG data loaded. Please go to the Analysis page to load ECG data.")
    elif 'processed_signal' not in st.session_state or st.session_state.processed_signal is None:
        st.info("ECG data not processed. Please go to the Analysis page to process the ECG signal.")
    elif 'r_peaks' not in st.session_state or st.session_state.r_peaks is None:
        st.info("R-peaks not detected. Please go to the Analysis page to process the ECG signal.")
    elif selected_model == "None":
        st.warning("No classification model selected or available. Please train models using the arrhythmia classification notebook.")
    else:
        # Get data from session state
        time, ecg_signal, fs = st.session_state.ecg_data
        processed_signal = st.session_state.processed_signal
        r_peaks = st.session_state.r_peaks
        
        # Try to load the selected model
        try:
            model_path = os.path.join(model_dir, selected_model)
            pipeline, label_names = arrhythmia_classifier.load_classifier(model_path)
            
            st.success(f"Loaded classification model: {selected_model}")
            
            # Extract heartbeats from the signal
            with st.spinner("Extracting heartbeats..."):
                channel_to_use = 0  # Default to first channel
                
                # Check if signal has multiple channels and offer selection
                if processed_signal.shape[1] > 1:
                    channel_to_use = st.selectbox(
                        "Select channel for classification:",
                        range(processed_signal.shape[1]),
                        format_func=lambda x: f"Channel {x+1}"
                    )
                
                heartbeats = data_processing.extract_heartbeat_segments(
                    processed_signal[:, channel_to_use],
                    r_peaks,
                    fs,
                    before_r=before_r,
                    after_r=after_r
                )
                
                if heartbeats is None or len(heartbeats) == 0:
                    st.error("Failed to extract heartbeats. Check if R-peaks were detected correctly.")
                    return
            
            # Classify heartbeats
            with st.spinner("Classifying heartbeats..."):
                try:
                    class_ids, class_names, probabilities = arrhythmia_classifier.classify_multiple_heartbeats(
                        heartbeats, pipeline, feature_extraction, label_names
                    )
                except ValueError as e:
                    if "X has 51 features, but StandardScaler is expecting 10 features as input" in str(e):
                        st.warning(f"Feature mismatch detected: {str(e)}")
                        
                        # Extract features manually and use only first 10 features
                        features_list = []
                        for beat in heartbeats:
                            features = feature_extraction.extract_heartbeat_features(beat, include_advanced=False)
                            features_df = pd.DataFrame([features])
                            features_df = features_df.fillna(0)
                            # Take only first 10 columns
                            features_df = features_df.iloc[:, :10]
                            features_list.append(features_df.values[0])
                        
                        # Create array with only 10 features
                        features_array = np.array(features_list)
                        
                        # Apply pipeline steps manually
                        X_scaled = pipeline.named_steps['scaler'].transform(features_array)
                        class_ids = pipeline.named_steps['classifier'].predict(X_scaled)
                        
                        # Get probabilities if available
                        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                            probabilities = pipeline.named_steps['classifier'].predict_proba(X_scaled)
                        else:
                            probabilities = None
                        
                        # Get class names
                        class_names = []
                        for class_id in class_ids:
                            if class_id in label_names:
                                class_names.append(label_names[class_id])
                            else:
                                class_names.append(f"Unknown ({class_id})")
                    else:
                        raise e
            
            # Count arrhythmia types
            class_counts = pd.Series(class_names).value_counts()
            
            # Display classification summary
            st.header("Classification Summary")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Beats", len(heartbeats))
            
            with col2:
                normal_beats = class_counts.get("Normal", 0)
                st.metric("Normal Beats", normal_beats)
            
            with col3:
                abnormal_beats = len(heartbeats) - normal_beats
                st.metric("Abnormal Beats", abnormal_beats)
            
            # Display class distribution
            st.subheader("Beat Classification Distribution")
            
            # Create a DataFrame for the distribution
            distribution_df = pd.DataFrame({
                'Beat Type': class_counts.index,
                'Count': class_counts.values,
                'Percentage': class_counts.values / len(heartbeats) * 100
            })
            
            # Display as a table
            st.dataframe(distribution_df)
            
            # Plot class distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(distribution_df['Beat Type'], distribution_df['Count'], color='skyblue')
            ax.set_xlabel('Beat Type')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Classified Beat Types')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Plot signal with classifications
            st.header("ECG Signal with Classifications")
            
            # Select segment to display
            duration = st.slider(
                "Segment Duration (seconds):",
                min_value=5,
                max_value=30,
                value=10,
                step=5
            )
            
            max_start_time = float(max(0.1, len(time)/fs - duration))
            start_time = st.slider(
                "Start Time (seconds):",
                min_value=0.0,
                max_value=max_start_time,
                value=0.0,
                step=5.0
            )
            
            # Calculate sample range
            start_sample = int(start_time * fs)
            end_sample = min(len(time), start_sample + int(duration * fs))
            
            # Plot the signal with beat classifications
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot the ECG signal
            ax.plot(time[start_sample:end_sample], processed_signal[start_sample:end_sample, channel_to_use], 'b-')
            
            # Find R-peaks in the selected window
            r_peaks_in_window = r_peaks[(r_peaks >= start_sample) & (r_peaks < end_sample)]
            
            # Create color map for different classes
            class_colors = {
                'Normal': 'green',
                'Atrial Premature': 'orange',
                'Ventricular Premature': 'red',
                'Fusion': 'purple',
                'Paced': 'brown',
                'Unclassifiable': 'gray'
            }
            
            # Mark R-peaks with classification
            for r_peak in r_peaks_in_window:
                # Find the corresponding beat in the classification results
                beat_idx = np.where(r_peaks == r_peak)[0][0]
                
                if beat_idx < len(class_names):
                    beat_type = class_names[beat_idx]
                    confidence = np.max(probabilities[beat_idx]) if probabilities is not None else 1.0
                    
                    # Only label if confidence is above threshold
                    if confidence >= confidence_threshold:
                        color = class_colors.get(beat_type, 'black')
                        
                        # Mark R peak
                        ax.plot(time[r_peak], processed_signal[r_peak, channel_to_use], 'o', color=color, markersize=8)
                        
                        # Add beat type as text
                        ax.annotate(beat_type, (time[r_peak], processed_signal[r_peak, channel_to_use]),
                                   xytext=(0, 20), textcoords='offset points',
                                   color=color, ha='center',
                                   bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('ECG Signal with Beat Classifications')
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Download link for classification results
            # Ensure all arrays are the same length
            min_length = min(len(r_peaks), len(class_names))
            if probabilities is not None:
                min_length = min(min_length, len(probabilities))
            
            results_df = pd.DataFrame({
                'Time (s)': time[r_peaks[:min_length]],
                'R_Peak_Index': r_peaks[:min_length],
                'Beat_Type': class_names[:min_length],
                'Confidence': [np.max(prob) if prob is not None else None for prob in probabilities[:min_length]] if probabilities is not None else [None] * min_length
            })
            
            st.markdown(get_download_link(results_df, "classification_results.csv", "Download Classification Results"), unsafe_allow_html=True)
            st.markdown(get_figure_download_link(fig, "ecg_classification.png", "Download Figure"), unsafe_allow_html=True)
            
            # Generate detailed beat report
            st.header("Abnormal Beat Analysis")
            
            # Filter abnormal beats
            abnormal_indices = [i for i, name in enumerate(class_names) if name != "Normal" and i < min_length]
            
            if abnormal_indices:
                # Select which abnormal beats to display
                max_beats = min(10, len(abnormal_indices))
                num_beats = st.slider(
                    "Number of abnormal beats to display:",
                    min_value=1,
                    max_value=max_beats,
                    value=min(5, max_beats)
                )
                
                # Select beats to display (prioritize different types)
                selected_indices = []
                displayed_types = set()
                
                # First, try to get one of each type
                for i in abnormal_indices:
                    if class_names[i] not in displayed_types:
                        selected_indices.append(i)
                        displayed_types.add(class_names[i])
                    
                    if len(selected_indices) >= num_beats:
                        break
                
                # If we still need more, add beats regardless of type
                if len(selected_indices) < num_beats:
                    for i in abnormal_indices:
                        if i not in selected_indices:
                            selected_indices.append(i)
                        
                        if len(selected_indices) >= num_beats:
                            break
                
                # Display selected abnormal beats
                for i, idx in enumerate(selected_indices[:num_beats]):
                    st.subheader(f"Abnormal Beat {i+1}: {class_names[idx]}")
                    
                    # Create columns for beat and info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Plot the beat
                        fig, ax = plt.subplots(figsize=(8, 4))
                        beat_time = np.linspace(-before_r, after_r, len(heartbeats[idx]))
                        ax.plot(beat_time, heartbeats[idx])
                        ax.axvline(x=0, color='r', linestyle='--', label='R Peak')
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Amplitude')
                        ax.set_title(f'Beat Type: {class_names[idx]}')
                        ax.grid(True)
                        st.pyplot(fig)
                    
                    with col2:
                        # Show beat information
                        st.write(f"R-Peak Index: {r_peaks[idx]}")
                        st.write(f"Time: {time[r_peaks[idx]]:.2f} seconds")
                        
                        if probabilities is not None and idx < len(probabilities):
                            st.write(f"Confidence: {np.max(probabilities[idx]):.2f}")
                            
                            # Show probability breakdown
                            prob_df = pd.DataFrame({
                                'Beat Type': [label_names[i] for i in range(len(probabilities[idx]))],
                                'Probability': probabilities[idx]
                            }).sort_values('Probability', ascending=False)
                            
                            st.dataframe(prob_df)
            else:
                st.info("No abnormal beats detected in this ECG signal.")
            
            # Generate rhythm report
            st.header("Rhythm Analysis")
            
            # Calculate RR intervals
            rr_intervals = np.diff(r_peaks) / fs
            
            # Identify irregular rhythms
            if len(rr_intervals) > 0:
                mean_rr = np.mean(rr_intervals)
                std_rr = np.std(rr_intervals)
                
                # Check for arrhythmias based on RR intervals
                rhythm_irregular = std_rr > 0.1  # Simple threshold for demonstration
                
                # Plot RR intervals
                fig, ax = plt.subplots(figsize=(12, 6))
                beat_numbers = np.arange(1, len(rr_intervals) + 1)
                ax.plot(beat_numbers, rr_intervals, 'o-')
                ax.axhline(y=mean_rr, color='r', linestyle='--', label=f'Mean RR: {mean_rr:.2f}s')
                ax.fill_between(beat_numbers, mean_rr - std_rr, mean_rr + std_rr, color='r', alpha=0.2, label=f'±1 SD: {std_rr:.2f}s')
                ax.set_xlabel('Beat Number')
                ax.set_ylabel('RR Interval (s)')
                ax.set_title('RR Interval Variability')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                # Heart rate variability metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_hr = 60 / mean_rr
                    st.metric("Average Heart Rate", f"{avg_hr:.1f} BPM")
                
                with col2:
                    st.metric("RR Interval SD", f"{std_rr:.3f}s")
                
                with col3:
                    # Calculate RMSSD (Root Mean Square of Successive Differences)
                    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
                    st.metric("RMSSD", f"{rmssd:.3f}s")
                
                # Assessment of rhythm
                if rhythm_irregular:
                    st.warning("Irregular rhythm detected (high RR interval variability)")
                else:
                    st.success("Regular rhythm (normal RR interval variability)")
                
                # Display histogram of RR intervals
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(rr_intervals, bins=20, alpha=0.7, color='skyblue')
                ax.axvline(x=mean_rr, color='r', linestyle='--', label=f'Mean: {mean_rr:.2f}s')
                ax.axvline(x=mean_rr + std_rr, color='g', linestyle='--', label=f'+1 SD: {mean_rr + std_rr:.2f}s')
                ax.axvline(x=mean_rr - std_rr, color='g', linestyle='--', label=f'-1 SD: {mean_rr - std_rr:.2f}s')
                ax.set_xlabel('RR Interval (s)')
                ax.set_ylabel('Frequency')
                ax.set_title('RR Interval Distribution')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("Not enough beats detected for rhythm analysis.")
            
            # Save results
            if st.button("Save Classification Results", key="save_classification_button"):
                # Create output directory
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                         'results', 'classifications')
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate timestamp
                timestamp = time_module.strftime("%Y%m%d_%H%M%S")
                
                # Save classification results
                results_df.to_csv(os.path.join(output_dir, f"classification_{timestamp}.csv"), index=False)
                
                # Save summary
                summary_df = pd.DataFrame({
                    'Beat Type': class_counts.index,
                    'Count': class_counts.values,
                    'Percentage': class_counts.values / len(heartbeats) * 100
                })
                summary_df.to_csv(os.path.join(output_dir, f"summary_{timestamp}.csv"), index=False)
                
                st.success(f"Classification results saved to results/classifications directory")
        
        except Exception as e:
            st.error(f"Error in classification: {str(e)}")
            st.exception(e)