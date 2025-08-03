
import numpy as np
import tensorflow as tf
import json
from sklearn.preprocessing import StandardScaler

class ECGRiskPredictor:
    def __init__(self, model_path, metadata_path):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata['class_names']
        self.window_size = self.metadata['window_size']
        
    def preprocess_beats(self, beats):
        """Preprocess ECG beats for prediction"""
        # Normalize each beat
        normalized_beats = np.zeros_like(beats, dtype=np.float32)
        for i, beat in enumerate(beats):
            mean_val = np.mean(beat)
            std_val = np.std(beat)
            if std_val > 1e-8:
                normalized_beats[i] = (beat - mean_val) / std_val
            else:
                normalized_beats[i] = beat - mean_val
        
        # Reshape for model input
        return normalized_beats.reshape(normalized_beats.shape[0], -1, 1)
    
    def predict_risk(self, ecg_beats):
        """Predict risk from ECG beats"""
        # Preprocess
        processed_beats = self.preprocess_beats(ecg_beats)
        
        # Predict
        predictions = self.model.predict(processed_beats)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate risk distribution
        from collections import Counter
        class_counts = Counter(predicted_classes)
        total_beats = len(predicted_classes)
        
        risk_distribution = {
            self.class_names[i]: (class_counts.get(i, 0) / total_beats) * 100
            for i in range(len(self.class_names))
        }
        
        return {
            "risk_distribution": risk_distribution,
            "predictions": predicted_classes.tolist(),
            "probabilities": predictions.tolist(),
            "total_beats": total_beats
        }

# Usage:
# predictor = ECGRiskPredictor('ecg_model_production.h5', 'model_metadata.json')
# result = predictor.predict_risk(ecg_beats_array)
