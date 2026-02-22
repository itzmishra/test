"""
ALLiswell.py - Model Testing and Comparison Script
==================================================
This script tests multiple ML models (Random Forest, SVM, XGBoost, Logistic Regression)
on engine fault detection data following the research paper methodology.

Based on: "Engine Fault Detection by Sound Analysis and Machine Learning"
Features: MFCC, DWT, Spectral features
Models: Random Forest, SVM, XGBoost, Logistic Regression
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. XGBoost model will be skipped.")
    print("Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')

# Try to import feature extractor
try:
    from feature_extraction import OptimizedFeatureExtractor
except ImportError:
    print("Warning: Could not import OptimizedFeatureExtractor. Using basic feature extraction.")
    OptimizedFeatureExtractor = None


class EngineFaultModelTester:
    """
    Model testing class for engine fault detection.
    Tests multiple models and compares their performance.
    """
    
    def __init__(self, csv_file='model_training.csv', test_size=0.2, random_state=42):
        """
        Initialize the model tester.
        
        Args:
            csv_file: Path to the master training CSV file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.csv_file = csv_file
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_extractor = OptimizedFeatureExtractor() if OptimizedFeatureExtractor else None
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models with default parameters."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Add XGBoost only if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        
    def load_data(self):
        """
        Load and preprocess data from CSV file.
        
        Returns:
            X: Feature matrix
            y: Label vector
            feature_names: List of feature names
        """
        print("=" * 60)
        print("Loading Data from CSV")
        print("=" * 60)
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        # Load CSV
        df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(df)} samples from {self.csv_file}")
        
        # Check for label column
        if 'Label' in df.columns:
            y = df['Label'].values
        elif 'Label_ID' in df.columns:
            y = df['Label_ID'].values
        else:
            raise ValueError("No 'Label' or 'Label_ID' column found in CSV")
        
        # Encode labels if they are strings
        if isinstance(y[0], str):
            y = self.label_encoder.fit_transform(y)
            print(f"   Encoded labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Get feature columns (exclude metadata columns)
        exclude_cols = ['File_name', 'Label', 'Label_ID']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("   Found missing values. Filling with median...")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
        
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")
        print(f"   Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            label_name = self.label_encoder.inverse_transform([u])[0] if hasattr(self.label_encoder, 'classes_') else u
            print(f"      Class {u} ({label_name}): {c} samples")
        
        return X, y, feature_cols
    
    def train_and_evaluate(self, X, y, feature_names):
        """
        Train and evaluate all models.
        
        Args:
            X: Feature matrix
            y: Label vector
            feature_names: List of feature names
            
        Returns:
            results: Dictionary with model results
        """
        print("\n" + "=" * 60)
        print("Training and Evaluating Models")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nData Split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'f1_macro': f1_macro,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                # Print results
                print(f"{model_name} Results:")
                print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
                print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
                print(f"   F1-Score (Weighted): {f1:.4f} ({f1*100:.2f}%)")
                print(f"   F1-Score (Macro): {f1_macro:.4f} ({f1_macro*100:.2f}%)")
                print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def compare_models(self, results):
        """
        Compare all models and identify the best one.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            best_model_name: Name of the best model
        """
        print("\n" + "=" * 60)
        print("Model Comparison Summary")
        print("=" * 60)
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score (Weighted)': result['f1_score'],
                'F1-Score (Macro)': result['f1_macro'],
                'CV Accuracy': result['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model (based on accuracy, with F1 as tiebreaker)
        best_model_name = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        return best_model_name
    
    def save_best_model(self, results, best_model_name, output_dir='.'):
        """
        Save the best model and scaler.
        
        Args:
            results: Dictionary with model results
            best_model_name: Name of the best model
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        best_result = results[best_model_name]
        
        # Save model
        model_file = output_path / f'best_engine_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(best_result['model'], f)
        
        # Save scaler
        scaler_file = output_path / f'best_engine_scaler.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(best_result['scaler'], f)
        
        # Save label encoder
        encoder_file = output_path / f'label_encoder.pkl'
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\nSaved best model to:")
        print(f"   Model: {model_file}")
        print(f"   Scaler: {scaler_file}")
        print(f"   Encoder: {encoder_file}")
    
    def predict_from_audio(self, audio_path, model_name=None):
        """
        Predict engine fault from audio file.
        
        Args:
            audio_path: Path to audio file
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            prediction: Predicted label
            confidence: Prediction confidence
            probabilities: Class probabilities
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not available. Cannot extract features from audio.")
        
        # Extract features
        feature_data = self.feature_extractor.extract_all_features(audio_path)
        
        # Convert to array (assuming feature_data is a dict)
        if isinstance(feature_data, dict):
            # Extract feature values in the same order as training
            # This is a simplified version - you may need to adjust based on your feature extractor
            feature_vector = np.array([v for v in feature_data.values() if isinstance(v, (int, float, np.number))])
        else:
            feature_vector = np.array(feature_data)
        
        # Reshape for single sample
        feature_vector = feature_vector.reshape(1, -1)
        
        # Scale features
        if model_name and model_name in self.scalers:
            scaler = self.scalers[model_name]
        else:
            # Use first available scaler
            scaler = list(self.scalers.values())[0] if self.scalers else None
        
        if scaler:
            feature_vector = scaler.transform(feature_vector)
        
        # Predict
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            # Use first available model
            model = list(self.models.values())[0]
        
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode label
        if hasattr(self.label_encoder, 'classes_'):
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = prediction
        
        confidence = probabilities[prediction] if probabilities is not None else 1.0
        
        return prediction_label, confidence, probabilities


def main():
    """Main function to run model testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ML models for engine fault detection')
    parser.add_argument('--csv', type=str, default='model_training.csv',
                       help='Path to training CSV file')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to audio file for prediction (optional)')
    parser.add_argument('--save', action='store_true',
                       help='Save the best model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = EngineFaultModelTester(
        csv_file=args.csv,
        test_size=args.test_size
    )
    
    # Load data
    X, y, feature_names = tester.load_data()
    
    # Train and evaluate
    results = tester.train_and_evaluate(X, y, feature_names)
    
    # Compare models
    best_model = tester.compare_models(results)
    
    # Save best model if requested
    if args.save:
        tester.save_best_model(results, best_model)
    
        # Predict from audio if provided
        if args.audio:
            print(f"\nPredicting from audio: {args.audio}")
            try:
                prediction, confidence, probabilities = tester.predict_from_audio(args.audio)
                print(f"   Prediction: {prediction}")
                print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                if probabilities is not None:
                    print(f"   Probabilities: {dict(zip(tester.label_encoder.classes_, probabilities))}")
            except Exception as e:
                print(f"Error predicting from audio: {e}")
    
    print("\n" + "=" * 60)
    print("Model Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
