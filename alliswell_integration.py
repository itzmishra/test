"""
Integration module for ALLiswell.py models with Streamlit app
=============================================================
This module provides a wrapper to use ALLiswell models in the Streamlit app.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from feature_extraction import OptimizedFeatureExtractor
except ImportError:
    OptimizedFeatureExtractor = None


class ALLiswellModelWrapper:
    """
    Wrapper class to use ALLiswell models in Streamlit app.
    Provides interface compatible with TwoStageMLPipeline.
    """
    
    def __init__(self, model_path='best_engine_model.pkl', 
                 scaler_path='best_engine_scaler.pkl',
                 encoder_path='label_encoder.pkl'):
        """
        Initialize wrapper with saved models.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            encoder_path: Path to saved label encoder
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_extractor = OptimizedFeatureExtractor() if OptimizedFeatureExtractor else None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load model, scaler, and label encoder from files."""
        try:
            # Load model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Loaded model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Loaded scaler from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            # Load label encoder
            if os.path.exists(self.encoder_path):
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✅ Loaded label encoder from {self.encoder_path}")
            else:
                # Create default encoder if not found
                self.label_encoder = LabelEncoder()
                print(f"⚠️  Label encoder not found. Using default.")
        
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict(self, audio_path_or_array, sr=None):
        """
        Predict engine fault from audio.
        
        Args:
            audio_path_or_array: Path to audio file or audio array
            sr: Sample rate (if audio_path_or_array is array)
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not available")
        
        # Extract features
        if isinstance(audio_path_or_array, (str, Path)):
            feature_data = self.feature_extractor.extract_all_features(audio_path_or_array, sr=sr)
        else:
            # If array provided, need sample rate
            if sr is None:
                sr = 48000  # Default
            feature_data = self.feature_extractor.extract_all_features(audio_path_or_array, sr=sr)
        
        # Convert feature data to array
        # Following the exact feature order as in model_training.csv (36 features)
        # Order: MFCC (13), Spectral (3), ZCR (1), RMS (1), DWT (8), Envelope (2), Bispectrum (8)
        if isinstance(feature_data, dict):
            # Use the 'features' key if available (contains pre-combined feature vector from extract_all_features)
            # This matches the exact order from feature_extraction.py: MFCC(13) + Spectral(3) + ZCR(1) + RMS(1) + DWT(8) + Envelope(2) + Bispectrum(8) = 36
            if 'features' in feature_data:
                feature_vector = np.array(feature_data['features'])
                # Ensure correct length (36 features matching model_training.csv)
                if len(feature_vector) != 36:
                    if len(feature_vector) < 36:
                        feature_vector = np.pad(feature_vector, (0, 36 - len(feature_vector)), 'constant', constant_values=0.0)
                    else:
                        feature_vector = feature_vector[:36]
            else:
                # Build feature vector in exact order matching CSV columns
                # This matches the order from feature_extraction.py extract_all_features method
                feature_vector = []
                
                # 1. MFCC (13 features) - MFCC_1 to MFCC_13
                mfcc = feature_data.get('mfcc', np.zeros(13))
                if isinstance(mfcc, np.ndarray):
                    mfcc_flat = mfcc.flatten()
                    feature_vector.extend(mfcc_flat[:13].tolist() if len(mfcc_flat) >= 13 else list(mfcc_flat) + [0.0] * (13 - len(mfcc_flat)))
                elif isinstance(mfcc, (list, tuple)):
                    mfcc_list = list(mfcc)[:13]
                    feature_vector.extend(mfcc_list + [0.0] * (13 - len(mfcc_list)))
                else:
                    feature_vector.extend([0.0] * 13)
                
                # 2. Spectral features (3): Centroid, Bandwidth, Rolloff
                spectral = feature_data.get('spectral', [0.0, 0.0, 0.0])
                if isinstance(spectral, (list, tuple, np.ndarray)):
                    spectral_list = list(spectral)[:3]
                    feature_vector.extend(spectral_list + [0.0] * (3 - len(spectral_list)))
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
                
                # 3. Zero Crossing Rate (1) - extract from signal if needed
                # For now, we'll need to extract it separately or use 0
                feature_vector.append(0.0)  # Will be extracted properly in feature_extraction
                
                # 4. RMS Energy (1) - extract from signal if needed
                feature_vector.append(0.0)  # Will be extracted properly in feature_extraction
                
                # 5. DWT features (8): D1_Mean, D1_Std, D2_Mean, D2_Std, D3_Mean, D3_Std, A3_Mean, A3_Std
                dwt = feature_data.get('dwt', np.zeros(8))
                if isinstance(dwt, np.ndarray):
                    dwt_flat = dwt.flatten()
                    feature_vector.extend(dwt_flat[:8].tolist() if len(dwt_flat) >= 8 else list(dwt_flat) + [0.0] * (8 - len(dwt_flat)))
                elif isinstance(dwt, (list, tuple)):
                    dwt_list = list(dwt)[:8]
                    feature_vector.extend(dwt_list + [0.0] * (8 - len(dwt_list)))
                else:
                    feature_vector.extend([0.0] * 8)
                
                # 6. Envelope features (2): RMS_Envelope_Mean, Hilbert_Envelope_Mean
                envelope = feature_data.get('envelope', [0.0, 0.0])
                if isinstance(envelope, (list, tuple, np.ndarray)):
                    env_list = list(envelope)[:2]
                    feature_vector.extend(env_list + [0.0] * (2 - len(env_list)))
                else:
                    feature_vector.extend([0.0, 0.0])
                
                # 7. Bispectrum features (8)
                bispectrum = feature_data.get('bispectrum', np.zeros(8))
                if isinstance(bispectrum, np.ndarray):
                    bispec_flat = bispectrum.flatten()
                    feature_vector.extend(bispec_flat[:8].tolist() if len(bispec_flat) >= 8 else list(bispec_flat) + [0.0] * (8 - len(bispec_flat)))
                elif isinstance(bispectrum, (list, tuple)):
                    bispec_list = list(bispectrum)[:8]
                    feature_vector.extend(bispec_list + [0.0] * (8 - len(bispec_list)))
                else:
                    feature_vector.extend([0.0] * 8)
                
                feature_vector = np.array(feature_vector)
        else:
            # If it's already an array, use it directly
            feature_vector = np.array(feature_data)
        
        # Ensure it has exactly 36 features (matching model_training.csv structure)
        if len(feature_vector) < 36:
            feature_vector = np.pad(feature_vector, (0, 36 - len(feature_vector)), 'constant', constant_values=0.0)
        elif len(feature_vector) > 36:
            feature_vector = feature_vector[:36]
        
        # Reshape for single sample
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Decode label
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
            # Create probability dictionary
            prob_dict = {}
            if probabilities is not None:
                for i, class_name in enumerate(self.label_encoder.classes_):
                    prob_dict[class_name] = probabilities[i]
        else:
            prediction_label = str(prediction)
            prob_dict = {f"Class_{i}": prob for i, prob in enumerate(probabilities)} if probabilities is not None else {}
        
        confidence = probabilities[prediction] if probabilities is not None else 1.0
        
        return {
            'prediction': prediction_label,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }


def create_streamlit_compatible_pipeline(model_path='best_engine_model.pkl',
                                        scaler_path='best_engine_scaler.pkl',
                                        encoder_path='label_encoder.pkl'):
    """
    Create a Streamlit-compatible pipeline wrapper.
    
    Returns:
        Wrapper object compatible with TwoStageMLPipeline interface
    """
    wrapper = ALLiswellModelWrapper(model_path, scaler_path, encoder_path)
    
    # Create adapter class
    class StreamlitAdapter:
        """Adapter to make ALLiswell wrapper work like TwoStageMLPipeline"""
        
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.feature_extractor = wrapper.feature_extractor
        
        def predict(self, audio_path_or_array, sr=None):
            """Predict using ALLiswell model, format like TwoStageMLPipeline"""
            result = self.wrapper.predict(audio_path_or_array, sr)
            
            # Extract features for visualization
            if isinstance(audio_path_or_array, (str, Path)):
                feature_data = self.feature_extractor.extract_all_features(audio_path_or_array, sr=sr)
            else:
                if sr is None:
                    sr = 48000
                feature_data = self.feature_extractor.extract_all_features(audio_path_or_array, sr=sr)
            
            # Map prediction to fault types
            fault_pred = result['prediction']
            fault_confidence = result['confidence']
            
            # Create vehicle result (not available in single-stage model)
            vehicle_result = {
                'prediction': 'Other/Unknown',
                'confidence': 0.5,
                'probabilities': {
                    'Ford_EcoSport': 0.33,
                    'Ford_Figo': 0.33,
                    'Other/Unknown': 0.34
                },
                'warning': '⚠️ Single-stage model: Vehicle identification not available'
            }
            
            # Create fault result
            fault_result = {
                'prediction': fault_pred,
                'confidence': fault_confidence,
                'probabilities': result['probabilities'],
                'warning': None if fault_confidence > 0.6 else '⚠️ Low confidence prediction'
            }
            
            return {
                'vehicle': vehicle_result,
                'fault': fault_result,
                'feature_data': feature_data
            }
    
    return StreamlitAdapter(wrapper)
