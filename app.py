import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# In app.py, change the import to:
from streamlit_app.pages.home import show as show_home
from streamlit_app.pages.analysis import show as show_analysis
from streamlit_app.pages.visualization import show as show_visualization
from streamlit_app.pages.classification import show as show_classification

# Configure page settings
st.set_page_config(
    page_title="ECG Analysis Tool",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "streamlit_app", "assets", "styles.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Try to load CSS, but don't crash if the file doesn't exist yet
try:
    load_css()
except:
    pass

# Display logo in sidebar
try:
    logo_path = os.path.join(os.path.dirname(__file__), "streamlit_app", "assets", "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
except:
    pass

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "ECG Analysis", "Arrhythmia Classification", "Visualization"]
)

# Display the selected page
if page == "Home":
    show_home()
elif page == "ECG Analysis":
    show_analysis()
elif page == "Arrhythmia Classification":
    show_classification()
elif page == "Visualization":
    show_visualization()

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application performs ECG signal analysis, including "
    "PQRST peak detection, feature extraction, and arrhythmia classification."
)
st.sidebar.text("© 2025 ECG Research Team")