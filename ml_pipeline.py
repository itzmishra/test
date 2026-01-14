"""
Two-Stage ML Pipeline for Engine Fault Detection
================================================
Stage 1: Vehicle Identification (Ford EcoSport, Ford Figo, Other/Unknown)
Stage 2: Fault Classification (Healthy, Misfire, Air Intake Irregularity)

Both stages use shared feature extraction pipeline.
Designed for limited datasets and class imbalance handling.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import warnings
from feature_extraction import OptimizedFeatureExtractor

warnings.filterwarnings("ignore")


class TwoStageMLPipeline:
    """
    Two-stage ML pipeline:
    1. Vehicle identification classifier
    2. Fault classification (vehicle-agnostic)
    """
    
    def __init__(self, feature_extractor=None):
        """
        Initialize pipeline with feature extractor.
        
        Args:
            feature_extractor: OptimizedFeatureExtractor instance (creates new if None)
        """
        self.feature_extractor = feature_extractor or OptimizedFeatureExtractor()
        
        # Stage 1: Vehicle identification models
        self.vehicle_model = None
        self.vehicle_scaler = None
        self.vehicle_classes = ['Ford_EcoSport', 'Ford_Figo', 'Other/Unknown']
        
        # Stage 2: Fault classification models
        self.fault_model = None
        self.fault_scaler = None
        self.fault_classes = ['Healthy', 'Misfire', 'Air_Intake_Irregularity']
        
        # Confidence thresholds
        self.vehicle_confidence_threshold = 0.5  # Below this = "Other/Unknown"
        self.fault_confidence_threshold = 0.4  # Below this = low confidence warning
    
    def infer_vehicle_from_filename(self, filename):
        """
        Infer vehicle type from filename.
        This is a helper function for data labeling.
        
        Args:
            filename: Audio filename
            
        Returns:
            Vehicle label (0: EcoSport, 1: Figo, 2: Other)
        """
        filename_lower = filename.lower()
        if 'ecosport' in filename_lower or 'eco' in filename_lower:
            return 0  # Ford EcoSport
        elif 'figo' in filename_lower:
            return 1  # Ford Figo
        else:
            return 2  # Other/Unknown
    
    def infer_fault_from_filename(self, filename, label=None):
        """
        Infer fault type from filename or explicit label.
        
        Args:
            filename: Audio filename
            label: Explicit label (0: Healthy, 1: Misfire, 2: Air Intake)
            
        Returns:
            Fault label (0: Healthy, 1: Misfire, 2: Air Intake)
        """
        if label is not None:
            return int(label)
        
        filename_lower = filename.lower()
        if 'misfire' in filename_lower or 'unh' in filename_lower or 'unhealthy' in filename_lower:
            return 1  # Misfire
        elif 'intake' in filename_lower or 'air' in filename_lower:
            return 2  # Air Intake Irregularity
        else:
            return 0  # Healthy
    
    def prepare_training_data(self, csv_path, feature_extraction_fn=None):
        """
        Prepare training data from CSV file.
        
        Args:
            csv_path: Path to CSV file with features
            feature_extraction_fn: Optional function to extract features from audio files
            
        Returns:
            Tuple of (vehicle_X, vehicle_y, fault_X, fault_y) and feature names
        """
        df = pd.read_csv(csv_path)
        
        # Extract feature columns (exclude metadata columns)
        feature_cols = [col for col in df.columns if col not in ['File name', 'filename', 'label', 'Label', 'vehicle', 'Vehicle']]
        
        # Get features
        X = df[feature_cols].values.astype(np.float32)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Extract labels
        vehicle_labels = []
        fault_labels = []
        
        # Infer labels from filenames if not present
        if 'File name' in df.columns or 'filename' in df.columns:
            filename_col = 'File name' if 'File name' in df.columns else 'filename'
            for filename in df[filename_col]:
                vehicle_labels.append(self.infer_vehicle_from_filename(str(filename)))
                fault_labels.append(self.infer_fault_from_filename(str(filename)))
        else:
            # Use explicit labels if available
            if 'label' in df.columns:
                for label in df['label']:
                    fault_labels.append(int(label))
                    vehicle_labels.append(2)  # Default to Other/Unknown
            else:
                raise ValueError("No filename or label column found in CSV")
        
        vehicle_y = np.array(vehicle_labels)
        fault_y = np.array(fault_labels)
        
        return X, vehicle_y, fault_y, feature_cols
    
    def train_vehicle_classifier(self, X, y, test_size=0.2, use_class_weights=True, n_estimators=100):
        """
        Train Stage 1: Vehicle identification classifier.
        
        Args:
            X: Feature matrix
            y: Vehicle labels (0: EcoSport, 1: Figo, 2: Other)
            test_size: Test split ratio
            use_class_weights: Whether to use class weights for imbalance
            n_estimators: Number of trees for Random Forest
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.vehicle_scaler = StandardScaler()
        X_train_scaled = self.vehicle_scaler.fit_transform(X_train)
        X_test_scaled = self.vehicle_scaler.transform(X_test)
        
        # Compute class weights if needed
        class_weights = None
        if use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
        
        # Train Random Forest
        self.vehicle_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        self.vehicle_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.vehicle_model.predict(X_test_scaled)
        y_proba = self.vehicle_model.predict_proba(X_test_scaled)
        
        print("\n=== Stage 1: Vehicle Identification Results ===")
        print(f"Test Accuracy: {np.mean(y_pred == y_test):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.vehicle_classes))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        joblib.dump(self.vehicle_model, 'vehicle_model.pkl')
        joblib.dump(self.vehicle_scaler, 'vehicle_scaler.pkl')
        print("\n✅ Vehicle classifier saved as 'vehicle_model.pkl' and 'vehicle_scaler.pkl'")
        
        return {
            'accuracy': np.mean(y_pred == y_test),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro')
        }
    
    def train_fault_classifier(self, X, y, test_size=0.2, use_class_weights=True, n_estimators=100):
        """
        Train Stage 2: Fault classification (vehicle-agnostic).
        
        Args:
            X: Feature matrix
            y: Fault labels (0: Healthy, 1: Misfire, 2: Air Intake)
            test_size: Test split ratio
            use_class_weights: Whether to use class weights for imbalance
            n_estimators: Number of trees for Random Forest
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.fault_scaler = StandardScaler()
        X_train_scaled = self.fault_scaler.fit_transform(X_train)
        X_test_scaled = self.fault_scaler.transform(X_test)
        
        # Compute class weights if needed
        class_weights = None
        if use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
        
        # Train Random Forest
        self.fault_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        self.fault_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.fault_model.predict(X_test_scaled)
        y_proba = self.fault_model.predict_proba(X_test_scaled)
        
        print("\n=== Stage 2: Fault Classification Results ===")
        print(f"Test Accuracy: {np.mean(y_pred == y_test):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.fault_classes))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        joblib.dump(self.fault_model, 'fault_model.pkl')
        joblib.dump(self.fault_scaler, 'fault_scaler.pkl')
        print("\n✅ Fault classifier saved as 'fault_model.pkl' and 'fault_scaler.pkl'")
        
        return {
            'accuracy': np.mean(y_pred == y_test),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro')
        }
    
    def load_models(self, vehicle_model_path='vehicle_model.pkl', vehicle_scaler_path='vehicle_scaler.pkl',
                    fault_model_path='fault_model.pkl', fault_scaler_path='fault_scaler.pkl'):
        """
        Load trained models and scalers.
        
        Args:
            vehicle_model_path: Path to vehicle model
            vehicle_scaler_path: Path to vehicle scaler
            fault_model_path: Path to fault model
            fault_scaler_path: Path to fault scaler
        """
        if os.path.exists(vehicle_model_path) and os.path.exists(vehicle_scaler_path):
            self.vehicle_model = joblib.load(vehicle_model_path)
            self.vehicle_scaler = joblib.load(vehicle_scaler_path)
        else:
            print(f"⚠️ Warning: Vehicle model files not found at {vehicle_model_path}")
        
        if os.path.exists(fault_model_path) and os.path.exists(fault_scaler_path):
            self.fault_model = joblib.load(fault_model_path)
            self.fault_scaler = joblib.load(fault_scaler_path)
        else:
            print(f"⚠️ Warning: Fault model files not found at {fault_model_path}")
    
    def predict(self, audio_path_or_array, sr=None):
        """
        Two-stage prediction: vehicle identification + fault classification.
        
        Args:
            audio_path_or_array: Audio file path or numpy array
            sr: Sample rate (if audio_path_or_array is array)
            
        Returns:
            Dictionary with vehicle and fault predictions, confidence scores, and warnings
        """
        if self.vehicle_model is None or self.fault_model is None:
            raise ValueError("Models not loaded. Call load_models() first or train models.")
        
        # Extract features
        feature_data = self.feature_extractor.extract_all_features(audio_path_or_array, sr)
        features = feature_data['features'].reshape(1, -1)
        
        # Stage 1: Vehicle identification
        features_vehicle_scaled = self.vehicle_scaler.transform(features)
        vehicle_proba = self.vehicle_model.predict_proba(features_vehicle_scaled)[0]
        vehicle_pred = self.vehicle_model.predict(features_vehicle_scaled)[0]
        vehicle_confidence = vehicle_proba[vehicle_pred]
        
        # Apply confidence threshold for "Other/Unknown"
        if vehicle_confidence < self.vehicle_confidence_threshold:
            vehicle_label = "Other/Unknown"
            vehicle_confidence = 1.0 - vehicle_confidence  # Invert for "unknown" confidence
            vehicle_warning = "⚠️ Vehicle type confidence is low. This may be an unseen vehicle type."
        else:
            vehicle_label = self.vehicle_classes[vehicle_pred]
            vehicle_warning = None
        
        # Stage 2: Fault classification (always execute, independent of vehicle)
        features_fault_scaled = self.fault_scaler.transform(features)
        fault_proba = self.fault_model.predict_proba(features_fault_scaled)[0]
        fault_pred = self.fault_model.predict(features_fault_scaled)[0]
        fault_confidence = fault_proba[fault_pred]
        
        # Check for low confidence
        fault_warning = None
        if fault_confidence < self.fault_confidence_threshold:
            fault_warning = "⚠️ Fault detection confidence is low. Please verify with a mechanic."
        
        # Create probability dictionaries
        vehicle_proba_dict = {
            self.vehicle_classes[i]: float(vehicle_proba[i])
            for i in range(len(self.vehicle_classes))
        }
        
        fault_proba_dict = {
            self.fault_classes[i]: float(fault_proba[i])
            for i in range(len(self.fault_classes))
        }
        
        return {
            'vehicle': {
                'prediction': vehicle_label,
                'confidence': float(vehicle_confidence),
                'probabilities': vehicle_proba_dict,
                'warning': vehicle_warning
            },
            'fault': {
                'prediction': self.fault_classes[fault_pred],
                'confidence': float(fault_confidence),
                'probabilities': fault_proba_dict,
                'warning': fault_warning
            },
            'feature_data': feature_data  # Include feature data for visualization
        }


def train_two_stage_pipeline(csv_path, save_models=True):
    """
    Train the complete two-stage pipeline.
    
    Args:
        csv_path: Path to training CSV file
        save_models: Whether to save trained models
        
    Returns:
        Trained pipeline instance
    """
    pipeline = TwoStageMLPipeline()
    
    print("Loading and preparing training data...")
    X, vehicle_y, fault_y, feature_names = pipeline.prepare_training_data(csv_path)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"\nVehicle distribution: {np.bincount(vehicle_y)}")
    print(f"Fault distribution: {np.bincount(fault_y)}")
    
    # Train Stage 1: Vehicle identification
    print("\n" + "="*60)
    print("Training Stage 1: Vehicle Identification")
    print("="*60)
    vehicle_metrics = pipeline.train_vehicle_classifier(X, vehicle_y)
    
    # Train Stage 2: Fault classification
    print("\n" + "="*60)
    print("Training Stage 2: Fault Classification")
    print("="*60)
    fault_metrics = pipeline.train_fault_classifier(X, fault_y)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nVehicle Classifier - F1 Macro: {vehicle_metrics['f1_macro']:.4f}")
    print(f"Fault Classifier - F1 Macro: {fault_metrics['f1_macro']:.4f}")
    
    return pipeline


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "MASTER.csv"  # Default CSV file
    
    if os.path.exists(csv_path):
        pipeline = train_two_stage_pipeline(csv_path)
        print("\n✅ Models trained and saved successfully!")
    else:
        print(f"❌ Error: CSV file '{csv_path}' not found.")
        print("Please provide a valid CSV file path as an argument.")



