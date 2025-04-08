# Make the page modules importable
# Instead of relative imports, use absolute imports
import sys
import os

# Add the parent directory to Python's module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules
import streamlit_app.pages.home
import streamlit_app.pages.analysis
import streamlit_app.pages.visualization

# For easier access, create aliases
home = streamlit_app.pages.home
analysis = streamlit_app.pages.analysis
visualization = streamlit_app.pages.visualization