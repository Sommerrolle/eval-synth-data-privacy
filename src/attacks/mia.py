#!/usr/bin/env python3
"""
Distance-based Membership Inference Attack (MIA) for synthetic health data evaluation.

This script implements a membership inference attack where:
1. Given a real patient record r and synthetic dataset S, find closest synthetic record s
2. Calculate distance d(r,s) 
3. Classify r as member of training set if d(r,s) < threshold
4. Evaluate attack success using holdout test data with known membership labels
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

# Import existing modules (adjust paths as needed)
from duckdb_manager import DuckDBManager
from feature_preprocessing import FeaturePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mia_attack.log'),
        logging.StreamHandler()
    ]
)

class MembershipInferenceAttack:
    """
    Distance-based Membership Inference Attack implementation.
    """
    
    def __init__(self, results_dir: str = 'results/mia_attacks'):
        """Initialize the MIA evaluator."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
    def prepare_attack_data(self, 
                           training_data: pd.DataFrame,
                           holdout_data: pd.DataFrame, 
                           synthetic_data: pd.DataFrame,
                           feature_columns: List[str],
                           sample_size: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Prepare data for membership inference attack.
        
        Args:
            training_data: Data used to train the synthetic model (members)
            holdout_data: Data NOT used to train the synthetic model (non-members)
            synthetic_data: Synthetic dataset generated from training_data
            feature_columns: Columns to use for distance calculation
            sample_size: Maximum number of records to use from each dataset
            
        Returns:
            Tuple of (processed_training, processed_holdout, processed_synthetic, membership_labels)
        """
        logging.info("Preparing data for membership inference attack...")
        
        # Validate feature columns exist in all datasets
        missing_features = []
        for dataset_name, dataset in [("training", training_data), ("holdout", holdout_data), ("synthetic", synthetic_data)]:
            missing = [col for col in feature_columns if col not in dataset.columns]
            if missing:
                missing_features.append(f"{dataset_name}: {missing}")
        
        if missing_features:
            raise ValueError(f"Missing feature columns in datasets: {missing_features}")
        
        # Sample data to manageable size
        if len(training_data) > sample_size:
            training_sample = training_data.sample(n=sample_size, random_state=42)
        else:
            training_sample = training_data.copy()
            
        if len(holdout_data) > sample_size:
            holdout_sample = holdout_data.sample(n=sample_size, random_state=42)
        else:
            holdout_sample = holdout_data.copy()
            
        if len(synthetic_data) > sample_size * 2:  # Allow larger synthetic dataset
            synthetic_sample = synthetic_data.sample(n=sample_size * 2, random_state=42)
        else:
            synthetic_sample = synthetic_data.copy()
        
        # Select feature columns
        training_features = training_sample[feature_columns].copy()
        holdout_features = holdout_sample[feature_columns].copy()
        synthetic_features = synthetic_sample[feature_columns].copy()
        
        # Create membership labels (1 = member, 0 = non-member)
        membership_labels = np.concatenate([
            np.ones(len(training_features)),   # Training data = members
            np.zeros(len(holdout_features))    # Holdout data = non-members
        ])
        
        logging.info(f"Attack data prepared:")
        logging.info(f"  Members (training): {len(training_features):,}")
        logging.info(f"  Non-members (holdout): {len(holdout_features):,}")
        logging.info(f"  Synthetic auxiliary: {len(synthetic_features):,}")
        logging.info(f"  Feature dimensions: {len(feature_columns)}")
        
        return training_features, holdout_features, synthetic_features, membership_labels
    
    def preprocess_features(self, 
                           training_features: pd.DataFrame,
                           holdout_features: pd.DataFrame,
                           synthetic_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess features for distance calculation using existing FeaturePreprocessor.
        
        Args:
            training_features: Training data features
            holdout_features: Holdout data features  
            synthetic_features: Synthetic data features
            
        Returns:
            Tuple of encoded feature arrays (training, holdout, synthetic)
        """
        logging.info("Preprocessing features for distance calculation...")
        
        # Combine all data for consistent preprocessing
        combined_target = pd.concat([training_features, holdout_features], ignore_index=True)
        
        # Use FeaturePreprocessor to handle healthcare-specific encoding
        processed_target, processed_synthetic, numeric_cols, string_cols = FeaturePreprocessor.preprocess_dataframes(
            combined_target, synthetic_features
        )
        
        # Create transformer pipeline
        transformers = []
        if numeric_cols:
            transformers.append((
                make_pipeline(
                    SimpleImputer(strategy='mean'),
                    StandardScaler()
                ), 
                numeric_cols
            ))
        
        if not transformers:
            raise ValueError("No valid numeric columns for distance calculation")
        
        # Fit transformer on all data
        transformer = make_column_transformer(*transformers, remainder='drop')
        
        # Transform all datasets
        target_encoded = transformer.fit_transform(processed_target)
        synthetic_encoded = transformer.transform(processed_synthetic)
        
        # Split target data back into training and holdout
        n_training = len(training_features)
        training_encoded = target_encoded[:n_training]
        holdout_encoded = target_encoded[n_training:]
        
        logging.info(f"Feature preprocessing complete:")
        logging.info(f"  Encoded dimensions: {training_encoded.shape[1]}")
        logging.info(f"  Training shape: {training_encoded.shape}")
        logging.info(f"  Holdout shape: {holdout_encoded.shape}")
        logging.info(f"  Synthetic shape: {synthetic_encoded.shape}")
        
        return training_encoded, holdout_encoded, synthetic_encoded
    
    def calculate_distances_to_synthetic(self, 
                                       target_data: np.ndarray,
                                       synthetic_data: np.ndarray,
                                       metric: str = 'euclidean') -> np.ndarray:
        """
        Calculate minimum distances from target records to synthetic dataset.
        
        Args:
            target_data: Target records to evaluate
            synthetic_data: Synthetic dataset to search
            metric: Distance metric to use
            
        Returns:
            Array of minimum distances for each target record
        """
        logging.info(f"Calculating distances using {metric} metric...")
        
        # Use k-NN to find closest synthetic record for each target record
        knn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1)
        knn.fit(synthetic_data)
        
        # Get distances to nearest synthetic neighbors
        distances, indices = knn.kneighbors(target_data)
        
        # Return minimum distances (first column since k=1)
        min_distances = distances.flatten()
        
        logging.info(f"Distance calculation complete:")
        logging.info(f"  Min distance: {min_distances.min():.6f}")
        logging.info(f"  Max distance: {min_distances.max():.6f}")
        logging.info(f"  Mean distance: {min_distances.mean():.6f}")
        logging.info(f"  Std distance: {min_distances.std():.6f}")
        
        return min_distances
    
    def optimize_threshold(self, 
                          distances: np.ndarray, 
                          true_labels: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, Dict]:
        """
        Find optimal threshold for membership classification.
        
        Args:
            distances: Minimum distances to synthetic data
            true_labels: True membership labels (1=member, 0=non-member)
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, optimization_results)
        """
        logging.info(f"Optimizing threshold using {metric} metric...")
        
        # Try different threshold percentiles
        percentiles = np.linspace(1, 99, 99)
        thresholds = np.percentile(distances, percentiles)
        
        best_score = 0
        best_threshold = 0
        threshold_results = []
        
        for threshold in thresholds:
            # Predict membership (1 if distance < threshold, 0 otherwise)
            predictions = (distances < threshold).astype(int)
            
            # Calculate metrics
            try:
                accuracy = accuracy_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions, zero_division=0)
                recall = recall_score(true_labels, predictions, zero_division=0)
                f1 = f1_score(true_labels, predictions, zero_division=0)
                
                # Select score based on chosen metric
                if metric == 'accuracy':
                    score = accuracy
                elif metric == 'precision':
                    score = precision
                elif metric == 'recall':
                    score = recall
                else:  # default to f1
                    score = f1
                
                threshold_results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                logging.warning(f"Error calculating metrics for threshold {threshold}: {e}")
                continue
        
        optimization_results = {
            'best_threshold': best_threshold,
            'best_score': best_score,
            'optimization_metric': metric,
            'all_results': threshold_results
        }
        
        logging.info(f"Threshold optimization complete:")
        logging.info(f"  Best threshold: {best_threshold:.6f}")
        logging.info(f"  Best {metric}: {best_score:.4f}")
        
        return best_threshold, optimization_results
    
    def evaluate_attack(self, 
                       distances: np.ndarray,
                       true_labels: np.ndarray,
                       threshold: float) -> Dict:
        """
        Evaluate membership inference attack performance.
        
        Args:
            distances: Minimum distances to synthetic data
            true_labels: True membership labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = (distances < threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Calculate AUC-ROC (using negative distance as score since lower distance = higher membership probability)
        try:
            auc_roc = roc_auc_score(true_labels, -distances)
        except Exception as e:
            logging.warning(f"Could not calculate AUC-ROC: {e}")
            auc_roc = 0.5
        
        # Calculate confusion matrix components
        true_positives = np.sum((predictions == 1) & (true_labels == 1))
        false_positives = np.sum((predictions == 1) & (true_labels == 0))
        true_negatives = np.sum((predictions == 0) & (true_labels == 0))
        false_negatives = np.sum((predictions == 0) & (true_labels == 1))
        
        # Calculate distance statistics by membership
        member_distances = distances[true_labels == 1]
        non_member_distances = distances[true_labels == 0]
        
        results = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': {
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'false_negatives': int(false_negatives)
            },
            'distance_statistics': {
                'member_distances': {
                    'mean': float(member_distances.mean()),
                    'std': float(member_distances.std()),
                    'min': float(member_distances.min()),
                    'max': float(member_distances.max()),
                    'median': float(np.median(member_distances))
                },
                'non_member_distances': {
                    'mean': float(non_member_distances.mean()),
                    'std': float(non_member_distances.std()),
                    'min': float(non_member_distances.min()),
                    'max': float(non_member_distances.max()),
                    'median': float(np.median(non_member_distances))
                }
            }
        }
        
        logging.info(f"Attack evaluation complete:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1-Score: {f1:.4f}")
        logging.info(f"  AUC-ROC: {auc_roc:.4f}")
        
        return results
    
    def plot_distance_distributions(self, 
                                   distances: np.ndarray,
                                   true_labels: np.ndarray,
                                   threshold: float,
                                   save_path: str = None) -> str:
        """
        Plot distance distributions for members vs non-members.
        
        Args:
            distances: Minimum distances to synthetic data
            true_labels: True membership labels
            threshold: Classification threshold
            save_path: Path to save the plot
            
        Returns:
            Path where plot was saved
        """
        member_distances = distances[true_labels == 1]
        non_member_distances = distances[true_labels == 0]
        
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        plt.subplot(2, 2, 1)
        plt.hist(member_distances, bins=50, alpha=0.7, label='Members', color='red')
        plt.hist(non_member_distances, bins=50, alpha=0.7, label='Non-members', color='blue')
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Distance to Closest Synthetic Record')
        plt.ylabel('Frequency')
        plt.title('Distance Distributions')
        plt.legend()
        plt.yscale('log')
        
        # Plot box plots
        plt.subplot(2, 2, 2)
        plt.boxplot([member_distances, non_member_distances], 
                   labels=['Members', 'Non-members'])
        plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.ylabel('Distance to Closest Synthetic Record')
        plt.title('Distance Distributions (Box Plot)')
        plt.legend()
        
        # Plot ROC curve
        plt.subplot(2, 2, 3)
        try:
            fpr, tpr, _ = roc_curve(true_labels, -distances)
            auc_roc = roc_auc_score(true_labels, -distances)
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_roc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        except Exception as e:
            plt.text(0.5, 0.5, f'ROC curve error:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('ROC Curve (Error)')
        
        # Plot distance vs membership
        plt.subplot(2, 2, 4)
        member_indices = np.where(true_labels == 1)[0]
        non_member_indices = np.where(true_labels == 0)[0]
        
        plt.scatter(member_indices[:1000], member_distances[:1000], 
                   alpha=0.6, s=1, color='red', label='Members')
        plt.scatter(non_member_indices[:1000], non_member_distances[:1000], 
                   alpha=0.6, s=1, color='blue', label='Non-members')
        plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Record Index')
        plt.ylabel('Distance to Closest Synthetic Record')
        plt.title('Distance by Record (Sample)')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, f'mia_distances_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Distance distribution plot saved to: {save_path}")
        return save_path
    
    def run_membership_inference_attack(self,
                                      training_data: pd.DataFrame,
                                      holdout_data: pd.DataFrame,
                                      synthetic_data: pd.DataFrame,
                                      feature_columns: List[str],
                                      sample_size: int = 10000,
                                      distance_metric: str = 'euclidean',
                                      optimization_metric: str = 'f1') -> Dict:
        """
        Run complete membership inference attack pipeline.
        
        Args:
            training_data: Data used to train synthetic model (members)
            holdout_data: Data not used to train synthetic model (non-members)
            synthetic_data: Synthetic dataset
            feature_columns: Features to use for distance calculation
            sample_size: Maximum records per dataset
            distance_metric: Distance metric for k-NN
            optimization_metric: Metric to optimize threshold selection
            
        Returns:
            Dictionary with complete attack results
        """
        logging.info("Starting membership inference attack...")
        
        # Prepare data
        training_features, holdout_features, synthetic_features, membership_labels = self.prepare_attack_data(
            training_data, holdout_data, synthetic_data, feature_columns, sample_size
        )
        
        # Preprocess features
        training_encoded, holdout_encoded, synthetic_encoded = self.preprocess_features(
            training_features, holdout_features, synthetic_features
        )
        
        # Combine target data (training + holdout)
        target_encoded = np.vstack([training_encoded, holdout_encoded])
        
        # Calculate distances to synthetic data
        distances = self.calculate_distances_to_synthetic(
            target_encoded, synthetic_encoded, distance_metric
        )
        
        # Optimize threshold
        optimal_threshold, optimization_results = self.optimize_threshold(
            distances, membership_labels, optimization_metric
        )
        
        # Evaluate attack
        attack_results = self.evaluate_attack(distances, membership_labels, optimal_threshold)
        
        # Create visualization
        plot_path = self.plot_distance_distributions(distances, membership_labels, optimal_threshold)
        
        # Compile complete results
        complete_results = {
            'attack_config': {
                'distance_metric': distance_metric,
                'optimization_metric': optimization_metric,
                'sample_size': sample_size,
                'feature_columns': feature_columns,
                'n_features': len(feature_columns)
            },
            'dataset_info': {
                'n_training_records': len(training_features),
                'n_holdout_records': len(holdout_features),
                'n_synthetic_records': len(synthetic_features),
                'total_target_records': len(target_encoded)
            },
            'threshold_optimization': optimization_results,
            'attack_performance': attack_results,
            'visualization_path': plot_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return complete_results
    
    def save_results(self, results: Dict, original_db: str, synthetic_db: str, table_name: str) -> str:
        """Save MIA results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mia_results_{original_db}_{synthetic_db}_{table_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def json_serialize(obj):
            if isinstance(obj, dict):
                return {key: json_serialize(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [json_serialize(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = json_serialize(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"MIA results saved to: {filepath}")
        return filepath


def main():
    """Main function to demonstrate MIA usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Membership Inference Attack on synthetic health data')
    parser.add_argument('--original_db', required=True, help='Original database (containing training + holdout)')
    parser.add_argument('--synthetic_db', required=True, help='Synthetic database')
    parser.add_argument('--table', required=True, help='Table name to analyze')
    parser.add_argument('--training_table', help='Training table name (if different from --table)')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size per dataset')
    parser.add_argument('--results_dir', default='results/mia_attacks', help='Results directory')
    
    args = parser.parse_args()
    
    # Initialize components
    db_manager = DuckDBManager()
    mia_evaluator = MembershipInferenceAttack(results_dir=args.results_dir)
    
    # Feature columns for health data (adjust as needed)
    if 'inpatient' in args.table.lower():
        feature_columns = [
            "insurants_year_of_birth",
            "insurants_gender", 
            "insurance_data_regional_code",
            "inpatient_cases_date_of_admission",
            "inpatient_cases_department_admission",
            "inpatient_diagnosis_diagnosis",
            "inpatient_procedures_procedure_code"
        ]
    elif 'outpatient' in args.table.lower():
        feature_columns = [
            "insurants_year_of_birth",
            "insurants_gender",
            "insurance_data_regional_code", 
            "outpatient_cases_practice_code",
            "outpatient_cases_year",
            "outpatient_cases_quarter",
            "outpatient_diagnosis_diagnosis",
            "outpatient_procedures_procedure_code"
        ]
    elif 'drugs' in args.table.lower():
        feature_columns = [
            "insurants_year_of_birth",
            "insurants_gender",
            "insurance_data_regional_code",
            "drugs_date_of_prescription",
            "drugs_pharma_central_number",
            "drugs_atc"
        ]
    else:
        raise ValueError(f"Unknown table type: {args.table}")
    
    try:
        # Load datasets
        orig_path = db_manager.get_database_path(args.original_db)
        synth_path = db_manager.get_database_path(args.synthetic_db)
        
        # For this example, we'll simulate training/holdout split
        # In practice, you should have separate tables or split info
        original_data = db_manager.load_table_data(orig_path, args.table)
        synthetic_data = db_manager.load_table_data(synth_path, args.table)
        
        # Split original data into training (80%) and holdout (20%)
        train_size = int(0.8 * len(original_data))
        training_data = original_data.iloc[:train_size]
        holdout_data = original_data.iloc[train_size:]
        
        print(f"Dataset sizes:")
        print(f"  Training (members): {len(training_data):,}")
        print(f"  Holdout (non-members): {len(holdout_data):,}")
        print(f"  Synthetic: {len(synthetic_data):,}")
        
        # Run MIA
        results = mia_evaluator.run_membership_inference_attack(
            training_data=training_data,
            holdout_data=holdout_data,
            synthetic_data=synthetic_data,
            feature_columns=feature_columns,
            sample_size=args.sample_size
        )
        
        # Display results
        attack_perf = results['attack_performance']
        print(f"\nMembership Inference Attack Results:")
        print(f"  Accuracy: {attack_perf['accuracy']:.4f}")
        print(f"  Precision: {attack_perf['precision']:.4f}")
        print(f"  Recall: {attack_perf['recall']:.4f}")
        print(f"  F1-Score: {attack_perf['f1_score']:.4f}")
        print(f"  AUC-ROC: {attack_perf['auc_roc']:.4f}")
        print(f"  Optimal Threshold: {attack_perf['threshold']:.6f}")
        
        # Save results
        output_file = mia_evaluator.save_results(
            results, args.original_db.replace('.duckdb', ''), 
            args.synthetic_db.replace('.duckdb', ''), args.table
        )
        
        print(f"\nResults saved to: {output_file}")
        print(f"Visualization saved to: {results['visualization_path']}")
        
    except Exception as e:
        logging.error(f"MIA evaluation failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()