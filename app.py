"""
Simplified ECG Analysis app that loads pages directly.
Place this file in your project root directory.
"""

import streamlit as st
import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Configure page settings
st.set_page_config(
    page_title="ECG Analysis Tool",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import page modules
# This avoids the circular import issue by importing directly
sys.path.append(os.path.join(project_root, "streamlit_app", "pages"))
import home
import analysis
import visualization

# Custom CSS
def load_css():
    css_file = os.path.join(project_root, "streamlit_app", "assets", "styles.css")
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
    logo_path = os.path.join(project_root, "streamlit_app", "assets", "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
except:
    pass

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "ECG Analysis", "Visualization"]
)

# Display the selected page
if page == "Home":
    home.show()
elif page == "ECG Analysis":
    analysis.show()
elif page == "Visualization":
    visualization.show()

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application performs ECG signal analysis, including "
    "PQRST peak detection and feature extraction."
)
st.sidebar.text("© 2025 ECG Research Team")