# ECG Analysis Tool

A Python-based application for analyzing ECG signals, detecting PQRST peaks, and extracting features for research purposes.

## Features

- **ECG Signal Processing**: Filtering, baseline removal, and normalization
- **PQRST Peak Detection**: Automated detection of P, Q, R, S, and T wave peaks
- **Feature Extraction**: Extract statistical and morphological features from ECG signals
- **Interactive Visualization**: Visualize ECG signals and detected peaks
- **User-friendly Interface**: Easy-to-use Streamlit web application

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ecg-analysis.git
   cd ecg-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   cd streamlit_app
   streamlit run app.py
   ```

## Usage

1. Navigate to the application in your web browser (typically at http://localhost:8501)
2. Upload an ECG signal file (CSV, DAT, etc.)
3. Process the signal and detect PQRST peaks
4. Visualize the results and extract features

## Deployment on Azure

This application is designed to be easily deployed to Azure App Service. See the documentation for detailed deployment instructions.

## Project Structure

- `data/`: Directory for ECG data files
- `models/`: Saved models and model configurations
- `src/`: Core processing modules
  - `data_processing.py`: Signal processing functions
  - `peak_detection.py`: PQRST peak detection algorithms
  - `feature_extraction.py`: Feature extraction from ECG signals
- `streamlit_app/`: Streamlit web application
  - `app.py`: Main application entry point
  - `pages/`: Application pages
  - `assets/`: UI assets (CSS, images)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [WFDB Python Package](https://github.com/MIT-LCP/wfdb-python)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [Streamlit](https://streamlit.io/)