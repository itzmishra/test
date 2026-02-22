"""
Model Accuracy Testing Script
==============================
This script evaluates the trained model's performance on both training and testing data.
It provides detailed metrics to assess model performance and detect overfitting.

Usage:
    python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, 
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")


class ModelAccuracyTester:
    """
    Class to test model accuracy on training and testing data.
    """
    
    def __init__(self, csv_file, model_path, scaler_path=None, encoder_path=None):
        """
        Initialize the accuracy tester.
        
        Args:
            csv_file: Path to training CSV file
            model_path: Path to saved model (.pkl file)
            scaler_path: Path to saved scaler (.pkl file)
            encoder_path: Path to saved label encoder (.pkl file)
        """
        self.csv_file = csv_file
        self.model_path = model_path
        self.scaler_path = scaler_path or model_path.replace('model.pkl', 'scaler.pkl').replace('_model.pkl', '_scaler.pkl')
        self.encoder_path = encoder_path or model_path.replace('model.pkl', 'encoder.pkl').replace('_model.pkl', '_encoder.pkl')
        
        # Load components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        self._load_components()
    
    def _load_components(self):
        """Load model, scaler, and label encoder."""
        print("=" * 60)
        print("Loading Model Components")
        print("=" * 60)
        
        # Load model
        if not os.path.exists(self.model_path):
            print(f"\nERROR: Model file not found: {self.model_path}")
            print("\nTo fix this, you need to train a model first:")
            print("   1. Create master CSV: python create_master_csv.py")
            print("   2. Train models: python ALLiswell.py --csv model_training.csv --save")
            print("\n   This will create:")
            print("   - best_engine_model.pkl")
            print("   - best_engine_scaler.pkl")
            print("   - label_encoder.pkl")
            print("\n   Then run this script again with:")
            print("   python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl")
            raise FileNotFoundError(f"Model file not found: {self.model_path}\n\nPlease train a model first using: python ALLiswell.py --csv model_training.csv --save")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Loaded model from: {self.model_path}")
        
        # Load scaler
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from: {self.scaler_path}")
        else:
            print(f"Scaler not found at {self.scaler_path}. Will create new one.")
            self.scaler = StandardScaler()
        
        # Load label encoder
        if os.path.exists(self.encoder_path):
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"Loaded label encoder from: {self.encoder_path}")
        else:
            print(f"Label encoder not found. Will create new one.")
            self.label_encoder = LabelEncoder()
    
    def load_data(self, test_size=0.2, random_state=42):
        """
        Load and split data.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        print("\n" + "=" * 60)
        print("Loading and Preparing Data")
        print("=" * 60)
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        # Load CSV
        df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(df)} samples from {self.csv_file}")
        
        # Get labels
        if 'Label' in df.columns:
            y = df['Label'].values
        elif 'Label_ID' in df.columns:
            y = df['Label_ID'].values
        else:
            raise ValueError("No 'Label' or 'Label_ID' column found in CSV")
        
        # Encode labels if needed
        if isinstance(y[0], str):
            y = self.label_encoder.fit_transform(y)
            print(f"   Encoded labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Get features
        exclude_cols = ['File_name', 'Label', 'Label_ID']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("   Found missing values. Filling with median...")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")
        
        # Scale features
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Scaler already fitted, just transform
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate model on both training and testing sets.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "=" * 60)
        print("Evaluating Model Performance")
        print("=" * 60)
        
        results = {}
        
        # Evaluate on training set
        print("\nTraining Set Performance:")
        print("-" * 60)
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train) if hasattr(self.model, 'predict_proba') else None
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
        
        results['training'] = {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1_score': train_f1,
            'f1_macro': train_f1_macro,
            'y_true': y_train,
            'y_pred': y_train_pred,
            'y_proba': y_train_proba,
            'confusion_matrix': confusion_matrix(y_train, y_train_pred)
        }
        
        print(f"   Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   Precision: {train_precision:.4f} ({train_precision*100:.2f}%)")
        print(f"   Recall:    {train_recall:.4f} ({train_recall*100:.2f}%)")
        print(f"   F1-Score:  {train_f1:.4f} ({train_f1*100:.2f}%)")
        print(f"   F1-Macro:  {train_f1_macro:.4f} ({train_f1_macro*100:.2f}%)")
        
        # Evaluate on testing set
        print("\nTesting Set Performance:")
        print("-" * 60)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        results['testing'] = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'f1_macro': test_f1_macro,
            'y_true': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"   Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
        print(f"   F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
        print(f"   F1-Macro:  {test_f1_macro:.4f} ({test_f1_macro*100:.2f}%)")
        
        # Compare training vs testing
        print("\nTraining vs Testing Comparison:")
        print("-" * 60)
        accuracy_diff = train_accuracy - test_accuracy
        f1_diff = train_f1 - test_f1
        
        print(f"   Accuracy Difference:  {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
        print(f"   F1-Score Difference:  {f1_diff:.4f} ({f1_diff*100:.2f}%)")
        
        if accuracy_diff > 0.15:
            print("   WARNING: Large gap detected! Model may be overfitting.")
        elif accuracy_diff > 0.05:
            print("   CAUTION: Moderate gap detected. Monitor for overfitting.")
        else:
            print("   Good: Small gap indicates good generalization.")
        
        results['comparison'] = {
            'accuracy_diff': accuracy_diff,
            'f1_diff': f1_diff,
            'overfitting_risk': 'high' if accuracy_diff > 0.15 else ('moderate' if accuracy_diff > 0.05 else 'low')
        }
        
        return results
    
    def print_detailed_report(self, results, label_encoder=None):
        """
        Print detailed classification reports.
        
        Args:
            results: Dictionary with evaluation results
            label_encoder: Label encoder for class names
        """
        print("\n" + "=" * 60)
        print("Detailed Classification Reports")
        print("=" * 60)
        
        # Get class names
        if label_encoder and hasattr(label_encoder, 'classes_'):
            class_names = label_encoder.classes_
        else:
            n_classes = len(np.unique(results['training']['y_true']))
            class_names = [f"Class_{i}" for i in range(n_classes)]
        
        # Calculate per-class metrics for training
        print("\n" + "=" * 60)
        print("PER-CLASS ACCURACY ANALYSIS - TRAINING SET")
        print("=" * 60)
        train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(
            results['training']['y_true'],
            results['training']['y_pred'],
            labels=range(len(class_names)),
            zero_division=0
        )
        
        # Calculate per-class accuracy (correct predictions / total samples for that class)
        train_class_accuracy = {}
        for class_idx, class_name in enumerate(class_names):
            mask = results['training']['y_true'] == class_idx
            if mask.sum() > 0:
                correct = (results['training']['y_pred'][mask] == class_idx).sum()
                total = mask.sum()
                acc = correct / total
                train_class_accuracy[class_name] = acc
            else:
                train_class_accuracy[class_name] = 0.0
        
        print(f"\n{'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        for idx, class_name in enumerate(class_names):
            acc = train_class_accuracy[class_name]
            prec = train_precision[idx]
            rec = train_recall[idx]
            f1 = train_f1[idx]
            supp = train_support[idx]
            print(f"{class_name:<20} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {supp:<10}")
        
        # Calculate per-class metrics for testing
        print("\n" + "=" * 60)
        print("PER-CLASS ACCURACY ANALYSIS - TESTING SET")
        print("=" * 60)
        test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(
            results['testing']['y_true'],
            results['testing']['y_pred'],
            labels=range(len(class_names)),
            zero_division=0
        )
        
        # Calculate per-class accuracy
        test_class_accuracy = {}
        for class_idx, class_name in enumerate(class_names):
            mask = results['testing']['y_true'] == class_idx
            if mask.sum() > 0:
                correct = (results['testing']['y_pred'][mask] == class_idx).sum()
                total = mask.sum()
                acc = correct / total
                test_class_accuracy[class_name] = acc
            else:
                test_class_accuracy[class_name] = 0.0
        
        print(f"\n{'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        for idx, class_name in enumerate(class_names):
            acc = test_class_accuracy[class_name]
            prec = test_precision[idx]
            rec = test_recall[idx]
            f1 = test_f1[idx]
            supp = test_support[idx]
            print(f"{class_name:<20} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {supp:<10}")
        
        # Store per-class results
        results['training']['per_class'] = {
            'accuracy': train_class_accuracy,
            'precision': dict(zip(class_names, train_precision)),
            'recall': dict(zip(class_names, train_recall)),
            'f1': dict(zip(class_names, train_f1)),
            'support': dict(zip(class_names, train_support))
        }
        
        results['testing']['per_class'] = {
            'accuracy': test_class_accuracy,
            'precision': dict(zip(class_names, test_precision)),
            'recall': dict(zip(class_names, test_recall)),
            'f1': dict(zip(class_names, test_f1)),
            'support': dict(zip(class_names, test_support))
        }
        
        # Summary comparison
        print("\n" + "=" * 60)
        print("PER-CLASS ACCURACY SUMMARY - TESTING SET")
        print("=" * 60)
        print(f"\n{'Class':<20} {'Accuracy':<15} {'Status':<20}")
        print("-" * 60)
        for class_name in class_names:
            acc = test_class_accuracy[class_name]
            if acc >= 0.90:
                status = "Excellent"
            elif acc >= 0.80:
                status = "Good"
            elif acc >= 0.70:
                status = "Fair"
            elif acc >= 0.60:
                status = "Poor"
            else:
                status = "Very Poor"
            print(f"{class_name:<20} {acc*100:>6.2f}%        {status:<20}")
        
        # Training report
        print("\nTraining Set - Full Classification Report:")
        print("-" * 60)
        train_report = classification_report(
            results['training']['y_true'],
            results['training']['y_pred'],
            target_names=class_names,
            zero_division=0
        )
        print(train_report)
        
        # Testing report
        print("\nTesting Set - Full Classification Report:")
        print("-" * 60)
        test_report = classification_report(
            results['testing']['y_true'],
            results['testing']['y_pred'],
            target_names=class_names,
            zero_division=0
        )
        print(test_report)
        
        # Confusion matrices
        print("\nConfusion Matrices:")
        print("-" * 60)
        
        print("\nTraining Set Confusion Matrix:")
        print("Rows = True Labels, Columns = Predicted Labels")
        cm_train_df = pd.DataFrame(
            results['training']['confusion_matrix'],
            index=[f"True_{name}" for name in class_names],
            columns=[f"Pred_{name}" for name in class_names]
        )
        print(cm_train_df)
        
        print("\nTesting Set Confusion Matrix:")
        print("Rows = True Labels, Columns = Predicted Labels")
        cm_test_df = pd.DataFrame(
            results['testing']['confusion_matrix'],
            index=[f"True_{name}" for name in class_names],
            columns=[f"Pred_{name}" for name in class_names]
        )
        print(cm_test_df)
    
    def plot_results(self, results, label_encoder=None, save_path=None):
        """
        Plot comparison charts.
        
        Args:
            results: Dictionary with evaluation results
            label_encoder: Label encoder for class names
            save_path: Path to save plots
        """
        if not PLOTTING_AVAILABLE:
            print("\nPlotting not available. Install matplotlib and seaborn for visualizations.")
            return
        
        print("\n" + "=" * 60)
        print("Generating Visualizations")
        print("=" * 60)
        
        # Get class names
        if label_encoder and hasattr(label_encoder, 'classes_'):
            class_names = label_encoder.classes_
        else:
            n_classes = len(np.unique(results['training']['y_true']))
            class_names = [f"Class_{i}" for i in range(n_classes)]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance: Training vs Testing', fontsize=16, fontweight='bold')
        
        # 1. Metrics comparison bar chart
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        train_values = [
            results['training']['accuracy'],
            results['training']['precision'],
            results['training']['recall'],
            results['training']['f1_score']
        ]
        test_values = [
            results['testing']['accuracy'],
            results['testing']['precision'],
            results['testing']['recall'],
            results['testing']['f1_score']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
        ax1.bar(x + width/2, test_values, width, label='Testing', alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Training confusion matrix
        ax2 = axes[0, 1]
        sns.heatmap(results['training']['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax2, xticklabels=class_names, yticklabels=class_names)
        ax2.set_title('Training Set Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. Testing confusion matrix
        ax3 = axes[1, 0]
        sns.heatmap(results['testing']['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Greens', ax=ax3, xticklabels=class_names, yticklabels=class_names)
        ax3.set_title('Testing Set Confusion Matrix')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')
        
        # 4. Per-class accuracy comparison
        ax4 = axes[1, 1]
        if 'per_class' in results['testing']:
            class_names = list(results['testing']['per_class']['accuracy'].keys())
            test_accs = [results['testing']['per_class']['accuracy'][name] for name in class_names]
            train_accs = [results['training']['per_class']['accuracy'][name] for name in class_names]
            
            x = np.arange(len(class_names))
            width = 0.35
            ax4.bar(x - width/2, train_accs, width, label='Training', alpha=0.8)
            ax4.bar(x + width/2, test_accs, width, label='Testing', alpha=0.8)
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Per-Class Accuracy Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(class_names, rotation=45, ha='right')
            ax4.legend()
            ax4.set_ylim([0, 1.1])
            ax4.grid(axis='y', alpha=0.3)
        else:
            comparison_data = {
                'Training': results['training']['accuracy'],
                'Testing': results['testing']['accuracy'],
                'Difference': results['comparison']['accuracy_diff']
            }
            colors = ['blue', 'green', 'red' if results['comparison']['accuracy_diff'] > 0.1 else 'orange']
            ax4.bar(comparison_data.keys(), comparison_data.values(), color=colors, alpha=0.7)
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Training vs Testing Accuracy')
            ax4.set_ylim([0, 1.1])
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to: model_accuracy_comparison.png")
        
        plt.close()
    
    def save_results(self, results, output_file='accuracy_test_results.txt'):
        """
        Save results to a text file.
        
        Args:
            results: Dictionary with evaluation results
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Model Accuracy Test Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Training Set Performance:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:  {results['training']['accuracy']:.4f} ({results['training']['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {results['training']['precision']:.4f} ({results['training']['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {results['training']['recall']:.4f} ({results['training']['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {results['training']['f1_score']:.4f} ({results['training']['f1_score']*100:.2f}%)\n")
            f.write(f"F1-Macro:  {results['training']['f1_macro']:.4f} ({results['training']['f1_macro']*100:.2f}%)\n\n")
            
            # Per-class training results
            if 'per_class' in results['training']:
                f.write("Training Set - Per-Class Accuracy:\n")
                f.write("-" * 60 + "\n")
                for class_name, acc in results['training']['per_class']['accuracy'].items():
                    prec = results['training']['per_class']['precision'][class_name]
                    rec = results['training']['per_class']['recall'][class_name]
                    f1 = results['training']['per_class']['f1'][class_name]
                    supp = results['training']['per_class']['support'][class_name]
                    f.write(f"{class_name}:\n")
                    f.write(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
                    f.write(f"  Precision: {prec:.4f} ({prec*100:.2f}%)\n")
                    f.write(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)\n")
                    f.write(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
                    f.write(f"  Support:   {supp}\n\n")
            
            f.write("\nTesting Set Performance:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:  {results['testing']['accuracy']:.4f} ({results['testing']['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {results['testing']['precision']:.4f} ({results['testing']['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {results['testing']['recall']:.4f} ({results['testing']['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {results['testing']['f1_score']:.4f} ({results['testing']['f1_score']*100:.2f}%)\n")
            f.write(f"F1-Macro:  {results['testing']['f1_macro']:.4f} ({results['testing']['f1_macro']*100:.2f}%)\n\n")
            
            # Per-class testing results
            if 'per_class' in results['testing']:
                f.write("Testing Set - Per-Class Accuracy:\n")
                f.write("-" * 60 + "\n")
                for class_name, acc in results['testing']['per_class']['accuracy'].items():
                    prec = results['testing']['per_class']['precision'][class_name]
                    rec = results['testing']['per_class']['recall'][class_name]
                    f1 = results['testing']['per_class']['f1'][class_name]
                    supp = results['testing']['per_class']['support'][class_name]
                    f.write(f"{class_name}:\n")
                    f.write(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
                    f.write(f"  Precision: {prec:.4f} ({prec*100:.2f}%)\n")
                    f.write(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)\n")
                    f.write(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
                    f.write(f"  Support:   {supp}\n\n")
            
            f.write("\nComparison:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy Difference: {results['comparison']['accuracy_diff']:.4f} ({results['comparison']['accuracy_diff']*100:.2f}%)\n")
            f.write(f"F1-Score Difference: {results['comparison']['f1_diff']:.4f} ({results['comparison']['f1_diff']*100:.2f}%)\n")
            f.write(f"Overfitting Risk: {results['comparison']['overfitting_risk'].upper()}\n")
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Test model accuracy on training and testing data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First, train a model:
  python ALLiswell.py --csv model_training.csv --save
  
  # Then test the model:
  python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl
  
  # With visualizations:
  python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl --plot
  
  # Save results to file:
  python test_model_accuracy.py --csv model_training.csv --model best_engine_model.pkl --save-results
        """
    )
    parser.add_argument('--csv', type=str, default='model_training.csv',
                       help='Path to training CSV file (default: model_training.csv)')
    parser.add_argument('--model', type=str, default='best_engine_model.pkl',
                       help='Path to saved model file (default: best_engine_model.pkl)')
    parser.add_argument('--scaler', type=str, default=None,
                       help='Path to saved scaler file (auto-detected if not provided)')
    parser.add_argument('--encoder', type=str, default=None,
                       help='Path to saved label encoder file (auto-detected if not provided)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to text file')
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"\nERROR: CSV file not found: {args.csv}")
        print("\nTo fix this:")
        print("   1. Run: python create_master_csv.py")
        print("   2. This will create model_training.csv")
        print("   3. Then run this script again")
        sys.exit(1)
    
    # Initialize tester
    try:
        tester = ModelAccuracyTester(
            csv_file=args.csv,
            model_path=args.model,
            scaler_path=args.scaler,
            encoder_path=args.encoder
        )
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = tester.load_data(
        test_size=args.test_size
    )
    
    # Evaluate model
    results = tester.evaluate_model(X_train, X_test, y_train, y_test)
    
    # Print detailed reports
    tester.print_detailed_report(results, tester.label_encoder)
    
    # Generate plots
    if args.plot:
        tester.plot_results(results, tester.label_encoder)
    
    # Save results
    if args.save_results:
        tester.save_results(results)
    
    print("\n" + "=" * 60)
    print("Accuracy Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
