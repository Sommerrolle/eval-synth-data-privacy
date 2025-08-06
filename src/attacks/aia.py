"""
Attribute Inference Attack (AIA) Module

This module implements Attribute Inference Attacks for evaluating the privacy protection
of synthetic health data. It provides comprehensive attack strategies to assess how
well synthetic data protects against attribute inference attacks using partial knowledge
of quasi-identifiers.

The module includes:
- AttributeInferenceAttack: Main class for implementing AIA attacks
- prepare_data_for_attack: Data preparation for attack scenarios
- sample_overlapping_data: Sampling strategies for attack evaluation
- partial_knowledge_attack: K-nearest neighbors based attack
- machine_learning_attack: Machine learning based attack using Random Forest
- evaluate_aia_vulnerability: Comprehensive vulnerability assessment
- save_results: Utility function to save attack results

Key Features:
- Multiple attack strategies (k-NN, Random Forest)
- Partial knowledge simulation (configurable knowledge ratios)
- Overlapping data sampling for realistic attack scenarios
- Comprehensive vulnerability assessment and risk scoring
- Support for multiple sensitive attributes
- Detailed attack result analysis and reporting
- Integration with feature preprocessing pipeline

Attack Strategies:
- Partial Knowledge Attack: Uses k-nearest neighbors with limited quasi-identifier knowledge
- Machine Learning Attack: Uses Random Forest classifier for attribute inference
- Vulnerability Assessment: Comprehensive evaluation of attack success rates

Usage:
    from attacks.aia import AttributeInferenceAttack
    
    aia = AttributeInferenceAttack()
    results = aia.evaluate_aia_vulnerability(
        original_df, synthetic_df, 
        quasi_identifiers, sensitive_attributes
    )

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
from typing import List, Dict, Tuple
import json
from datetime import datetime

from feature_preprocessing import FeaturePreprocessor

class AttributeInferenceAttack:
    """
    Implement Attribute Inference Attack (AIA) for privacy evaluation of synthetic health data.
    
    This implementation follows the approach where we use partial quasi-identifiers 
    to infer sensitive attributes using k-nearest neighbors in synthetic data.
    """
    
    def __init__(self, results_dir: str = 'results/aia_attacks'):
        """Initialize the AIA evaluator."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def prepare_data_for_attack(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                              quasi_identifiers: List[str], sensitive_attributes: List[str]) -> Tuple:
        """
        Prepare data for the attribute inference attack.
        
        Args:
            original_df: Original dataset
            synthetic_df: Synthetic dataset  
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns
            
        Returns:
            Tuple of processed datasets and column information
        """
        # Ensure we have common columns
        common_cols = set(original_df.columns) & set(synthetic_df.columns)
        required_cols = set(quasi_identifiers + sensitive_attributes)
        
        if not required_cols.issubset(common_cols):
            missing = required_cols - common_cols
            raise ValueError(f"Missing columns in datasets: {missing}")
        
        # Select relevant columns
        cols_to_use = quasi_identifiers + sensitive_attributes
        orig_subset = original_df[cols_to_use].copy()
        synth_subset = synthetic_df[cols_to_use].copy()
        
        # Remove rows with missing sensitive attributes (we need ground truth)
        # for sensitive_attr in sensitive_attributes:
        #     orig_subset = orig_subset.dropna(subset=[sensitive_attr])
        #     synth_subset = synth_subset.dropna(subset=[sensitive_attr])
        #     # Also remove rows where sensitive attribute is "UNKNOWN" or similar placeholder values
        #     # 00000000 für pharma central number
        #     # 0000 department of admission 
        #     unknown_values = ['UNKNOWN', 'UUU', 'NULL', 'NaN', '', 'NA', 'MISSING', '00000000', '0000']
        #     orig_subset = orig_subset[~orig_subset[sensitive_attr].astype(str).str.upper().isin(unknown_values)]
        #     synth_subset = synth_subset[~synth_subset[sensitive_attr].astype(str).str.upper().isin(unknown_values)]
        
        return orig_subset, synth_subset, quasi_identifiers, sensitive_attributes


    def sample_overlapping_data(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                            sensitive_attribute: str, synthetic_sample_size: int = 5000, 
                            strategy: str = 'random') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Sample synthetic data and filter original data to only include overlapping sensitive attribute values.
        Ensures exact sample size by filtering out unknown values before sampling.
        
        Args:
            original_df: Original dataset
            synthetic_df: Synthetic dataset
            sensitive_attribute: Column name of the sensitive attribute to match on
            synthetic_sample_size: Size of synthetic sample to create (exact size)
            strategy: Sampling strategy - 'random', 'stratified', or 'common'
        
        Returns:
            Tuple of (filtered_original_df, synthetic_sample_df, overlap_stats)
        """
        
        logging.info(f"Sampling overlapping data for attribute: {sensitive_attribute}")
        
        # Validate inputs
        if sensitive_attribute not in original_df.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in original data")
        if sensitive_attribute not in synthetic_df.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in synthetic data")
        
        # Step 1: Pre-filter synthetic data to remove unknown/placeholder values
        unknown_values = {'UNKNOWN', 'UUU', 'NULL', 'NaN', '', 'NA', 'MISSING', '00000000', '0000'}
        
        # Filter synthetic data to exclude unknown values and missing data
        synthetic_clean = synthetic_df[
            ~synthetic_df[sensitive_attribute].astype(str).isin(unknown_values) &
            synthetic_df[sensitive_attribute].notna()
        ].copy()
        
        logging.info(f"Synthetic data: {len(synthetic_df):,} --> {len(synthetic_clean):,} records after removing unknown values")
        
        if len(synthetic_clean) < synthetic_sample_size:
            logging.warning(f"Not enough clean synthetic data ({len(synthetic_clean)}) to reach desired sample size ({synthetic_sample_size})")
            effective_sample_size = len(synthetic_clean)
        else:
            effective_sample_size = synthetic_sample_size
        
        # Step 2: Sample synthetic data using specified strategy
        if strategy == 'random':
            synthetic_sample = synthetic_clean.sample(
                n=effective_sample_size, 
                random_state=42
            )
            
        elif strategy == 'stratified':
            # Maintain diagnosis distribution proportionally
            synthetic_sample = synthetic_clean.groupby(sensitive_attribute, group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), max(1, effective_sample_size * len(x) // len(synthetic_clean))),
                    random_state=42
                )
            ).reset_index(drop=True)
            
            # Ensure we get exactly the desired sample size (or as close as possible)
            if len(synthetic_sample) > effective_sample_size:
                synthetic_sample = synthetic_sample.sample(n=effective_sample_size, random_state=42)
            elif len(synthetic_sample) < effective_sample_size:
                # If stratified sampling didn't get enough samples, fill with random samples
                remaining_needed = effective_sample_size - len(synthetic_sample)
                remaining_pool = synthetic_clean.drop(synthetic_sample.index)
                if len(remaining_pool) >= remaining_needed:
                    additional_samples = remaining_pool.sample(n=remaining_needed, random_state=42)
                    synthetic_sample = pd.concat([synthetic_sample, additional_samples]).reset_index(drop=True)
                
        elif strategy == 'common':
            # Focus on most common values (realistic adversary knowledge)
            top_values = synthetic_clean[sensitive_attribute].value_counts().head(500).index
            synthetic_filtered = synthetic_clean[synthetic_clean[sensitive_attribute].isin(top_values)]
            
            if len(synthetic_filtered) < effective_sample_size:
                logging.warning(f"Not enough records with common values ({len(synthetic_filtered)}) for desired sample size")
                synthetic_sample = synthetic_filtered.copy()
            else:
                synthetic_sample = synthetic_filtered.sample(n=effective_sample_size, random_state=42)
                
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'random', 'stratified', or 'common'")
        
        # Step 3: Get overlapping values from the clean synthetic sample
        synthetic_values = set(synthetic_sample[sensitive_attribute].astype(str).unique())
        
        # Step 4: Filter original data to overlapping values only (also exclude unknown values)
        original_clean = original_df[
            ~original_df[sensitive_attribute].astype(str).isin(unknown_values) &
            original_df[sensitive_attribute].notna()
        ].copy()
        
        original_filtered = original_clean[
            original_clean[sensitive_attribute].astype(str).isin(synthetic_values)
        ].copy()
        
        # Step 5: Calculate overlap statistics
        orig_total_values = len(original_clean[sensitive_attribute].unique())
        orig_filtered_values = len(original_filtered[sensitive_attribute].unique())
        synth_values_count = len(synthetic_values)
        
        overlap_stats = {
            'original_total_records': len(original_df),
            'original_clean_records': len(original_clean),
            'original_filtered_records': len(original_filtered),
            'synthetic_total_records': len(synthetic_df),
            'synthetic_clean_records': len(synthetic_clean),
            'synthetic_sample_records': len(synthetic_sample),
            'requested_sample_size': synthetic_sample_size,
            'achieved_sample_size': len(synthetic_sample),
            'original_total_values': orig_total_values,
            'original_filtered_values': orig_filtered_values,
            'synthetic_values': synth_values_count,
            'overlap_ratio': orig_filtered_values / orig_total_values if orig_total_values > 0 else 0,
            'filtering_ratio': len(original_filtered) / len(original_clean) if len(original_clean) > 0 else 0,
            'strategy_used': strategy,
            'sensitive_attribute': sensitive_attribute
        }
        
        # Log statistics
        logging.info(f"Overlap Sampling Results:")
        logging.info(f"  Strategy: {strategy}")
        logging.info(f"  Requested sample size: {synthetic_sample_size:,}")
        logging.info(f"  Achieved sample size: {len(synthetic_sample):,}")
        logging.info(f"  Original (clean): {len(original_clean):,} --> {len(original_filtered):,} records "
                    f"({overlap_stats['filtering_ratio']:.1%} retained)")
        logging.info(f"  Synthetic (clean): {len(synthetic_clean):,} --> {len(synthetic_sample):,} records")
        logging.info(f"  Value overlap: {orig_filtered_values}/{orig_total_values} "
                    f"({overlap_stats['overlap_ratio']:.1%})")
        
        return original_filtered, synthetic_sample, overlap_stats
    

    def add_overlap_statistics_to_results(self, results: Dict, overlap_stats: Dict, 
                                        sensitive_attribute: str = None) -> Dict:
        """
        Add overlap statistics to existing AIA results dictionary.
        
        Args:
            results: Existing AIA results dictionary
            overlap_stats: Overlap statistics returned from sample_overlapping_data()
            sensitive_attribute: Name of the sensitive attribute (optional, will be extracted from stats if not provided)
            
        Returns:
            Updated results dictionary with overlap statistics
        """
        # Extract sensitive attribute name from stats if not provided
        if sensitive_attribute is None:
            sensitive_attribute = overlap_stats.get('sensitive_attribute', 'unknown_attribute')
        
        # Initialize overlap_statistics section if it doesn't exist
        if 'overlap_statistics' not in results:
            results['overlap_statistics'] = {}
        
        # Add the overlap stats for this sensitive attribute
        results['overlap_statistics'][sensitive_attribute] = overlap_stats.copy()
        
        # Add a timestamp for when these stats were added
        # results['overlap_statistics'][sensitive_attribute]['stats_added_timestamp'] = datetime.now().isoformat()
        
        # logging.info(f"Added overlap statistics for {sensitive_attribute} to results")
        # logging.info(f"  Overlap ratio: {overlap_stats.get('overlap_ratio', 'N/A'):.1%}")
        # logging.info(f"  Filtering ratio: {overlap_stats.get('filtering_ratio', 'N/A'):.1%}")
        # logging.info(f"  Synthetic sample size: {overlap_stats.get('achieved_sample_size', 'N/A'):,}")
        
        return results


    def encode_features(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                       quasi_identifiers: List[str]) -> Tuple:
        """
        Encode features for distance calculation with healthcare-specific handling.
        Uses FeaturePreprocessor methods for consistent timestamp conversion.
        
        Args:
            original_df: Original dataset
            synthetic_df: Synthetic dataset
            quasi_identifiers: Quasi-identifier columns to encode
            
        Returns:
            Tuple of encoded datasets and fitted transformer
        """
        # Work with subsets directly (no copies needed)
        orig_subset = original_df[quasi_identifiers]
        synth_subset = synthetic_df[quasi_identifiers]
        
        # Combine datasets for consistent encoding
        combined_df = pd.concat([orig_subset, synth_subset], ignore_index=True)
        
        # Identify column types
        numeric_cols = []
        categorical_cols = []
        timestamp_cols = []
        
        for col in quasi_identifiers:
            col_lower = col.lower()
            # Use same logic as FeaturePreprocessor for timestamp detection
            if any(term in col_lower for term in ["date", "time", "from", "to"]):
                timestamp_cols.append(col)
            elif combined_df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Convert timestamps to Unix epoch using FeaturePreprocessor (ONCE)
        if timestamp_cols:
            logging.info(f"Converting {len(timestamp_cols)} timestamp columns using FeaturePreprocessor")
            
            # Convert timestamps in combined dataframe
            combined_df, updated_numeric_cols = FeaturePreprocessor.convert_timestamps_to_epoch(
                combined_df, timestamp_cols, numeric_cols.copy()
            )
            
            # Update numeric_cols to include converted timestamp columns
            numeric_cols = updated_numeric_cols
        
        # Handle categorical columns with label encoding (modify combined_df in place)
        if categorical_cols:
            for col in categorical_cols:
                # Use LabelEncoder to convert categories to integers
                le = LabelEncoder()
                combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        
        # Create transformer pipeline
        transformers = []
        
        # Handle numeric columns (including converted timestamps)
        if numeric_cols:
            transformers.append((
                make_pipeline(
                    SimpleImputer(strategy='mean'),
                    StandardScaler()
                ), 
                numeric_cols
            ))
        
        # Handle categorical columns (already label encoded)
        if categorical_cols:
            transformers.append((
                make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    StandardScaler()  # Scale encoded categories for distance calculations
                ), 
                categorical_cols
            ))
        
        if not transformers:
            raise ValueError("No valid columns for transformation")
        
        # Create and fit transformer on combined data
        transformer = make_column_transformer(*transformers, remainder='drop')
        transformer.fit(combined_df)
        
        # Split the combined dataframe back to original and synthetic
        n_orig = len(orig_subset)
        combined_orig = combined_df.iloc[:n_orig]
        combined_synth = combined_df.iloc[n_orig:]
        
        # Transform both datasets
        orig_encoded = transformer.transform(combined_orig)
        synth_encoded = transformer.transform(combined_synth)
        
        logging.info(f"Feature encoding complete:")
        logging.info(f"  - Numeric columns (incl. timestamps): {len(numeric_cols)}")
        logging.info(f"  - Categorical columns: {len(categorical_cols)}")
        logging.info(f"  - Total dimensions: {orig_encoded.shape[1]}")
        
        return orig_encoded, synth_encoded, transformer
    
    def partial_knowledge_attack(self, original_encoded: np.ndarray, synthetic_encoded: np.ndarray,
                                original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                sensitive_attr: str, knowledge_ratio: float = 0.7,
                                k_neighbors: int = 5) -> Dict:
        """
        Perform partial knowledge AIA using k-nearest neighbors.
        
        Args:
            original_encoded: Encoded original quasi-identifiers
            synthetic_encoded: Encoded synthetic quasi-identifiers  
            original_df: Original dataframe with sensitive attributes
            synthetic_df: Synthetic dataframe with sensitive attributes
            sensitive_attr: Target sensitive attribute
            knowledge_ratio: Ratio of quasi-identifiers known to attacker (0.0-1.0)
            k_neighbors: Number of neighbors to consider
            
        Returns:
            Dictionary with attack results
        """
        n_features = original_encoded.shape[1]
        n_known_features = max(1, int(n_features * knowledge_ratio))
        
        # Randomly select which features the attacker knows
        np.random.seed(42)  # For reproducibility
        known_feature_indices = np.random.choice(n_features, n_known_features, replace=False)
        
        # Create partial knowledge versions
        orig_partial = original_encoded[:, known_feature_indices]
        synth_partial = synthetic_encoded[:, known_feature_indices]
        
        # Fit k-NN on synthetic data (this is the attacker's auxiliary knowledge)
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        knn.fit(synth_partial)
        
        # For each original record, find nearest synthetic neighbors
        distances, neighbor_indices = knn.kneighbors(orig_partial)
        
        # Perform inference attack
        attack_results = []
        successful_attacks = 0
        total_attacks = len(original_df)
        
        for i, orig_idx in enumerate(range(len(original_df))):
            true_value = original_df.iloc[orig_idx][sensitive_attr]
            
            # Get sensitive attribute values from k nearest synthetic neighbors
            neighbor_values = []
            for neighbor_idx in neighbor_indices[i]:
                if neighbor_idx < len(synthetic_df):
                    neighbor_val = synthetic_df.iloc[neighbor_idx][sensitive_attr]
                    if pd.notna(neighbor_val):
                        neighbor_values.append(neighbor_val)
            
            if not neighbor_values:
                continue
            
            # Infer value (most common among neighbors)
            if pd.api.types.is_numeric_dtype(original_df[sensitive_attr]):
                # For numeric attributes, use mean
                inferred_value = np.mean(neighbor_values)
                # Consider attack successful if within reasonable range
                success = abs(inferred_value - true_value) <= abs(true_value * 0.1)  # 10% tolerance
            else:
                # For categorical attributes, use mode
                inferred_value = max(set(neighbor_values), key=neighbor_values.count)
                success = (inferred_value == true_value)
            
            attack_results.append({
                'true_value': true_value,
                'inferred_value': inferred_value,
                'success': success,
                'avg_distance': np.mean(distances[i])
            })
            
            if success:
                successful_attacks += 1
        
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0
        
        return {
            'success_rate': success_rate,
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'knowledge_ratio': knowledge_ratio,
            'k_neighbors': k_neighbors,
            'n_known_features': n_known_features,
            'n_total_features': n_features,
            'avg_distance_to_neighbors': np.mean([r['avg_distance'] for r in attack_results]),
            # 'detailed_results': attack_results[:100]  # Store first 100 for analysis
        }
    
    def machine_learning_attack(self, original_encoded: np.ndarray, synthetic_encoded: np.ndarray,
                               original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                               sensitive_attr: str, knowledge_ratio: float = 0.7) -> Dict:
        """
        Perform ML-based AIA using synthetic data as training set.
        
        Args:
            original_encoded: Encoded original quasi-identifiers
            synthetic_encoded: Encoded synthetic quasi-identifiers
            original_df: Original dataframe with sensitive attributes
            synthetic_df: Synthetic dataframe with sensitive attributes
            sensitive_attr: Target sensitive attribute
            knowledge_ratio: Ratio of quasi-identifiers known to attacker
            
        Returns:
            Dictionary with attack results
        """
        n_features = original_encoded.shape[1]
        n_known_features = max(1, int(n_features * knowledge_ratio))
        
        # Select known features
        np.random.seed(42)
        known_feature_indices = np.random.choice(n_features, n_known_features, replace=False)
        
        # Prepare training data from synthetic dataset
        X_train = synthetic_encoded[:, known_feature_indices]
        y_train = synthetic_df[sensitive_attr]
        
        # Remove missing values from training data
        valid_idx = pd.notna(y_train)
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        if len(X_train) == 0:
            return {'error': 'No valid training data'}
        
        # Prepare test data from original dataset
        X_test = original_encoded[:, known_feature_indices]
        y_test = original_df[sensitive_attr]
        
        # Remove missing values from test set
        valid_test_idx = pd.notna(y_test)
        X_test = X_test[valid_test_idx]
        y_test = y_test[valid_test_idx]
        
        if len(X_test) == 0:
            return {'error': 'No valid test data'}
        
        # Handle unseen labels by creating a combined label encoder
        logging.info(f"Training labels: {len(np.unique(y_train))} unique values")
        logging.info(f"Test labels: {len(np.unique(y_test))} unique values")
        
        # Combine all labels (training + test) for consistent encoding
        all_labels = pd.concat([y_train, y_test]).astype(str)
        
        # Create label encoder on all possible labels
        le = LabelEncoder()
        le.fit(all_labels)
        
        # Encode training and test labels
        y_train_encoded = le.transform(y_train.astype(str))
        y_test_encoded = le.transform(y_test.astype(str))
        
        # Check if this is a numeric attribute that should be binned
        if pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train_encoded)) > 50:
            # For numeric attributes with many values, bin them for classification
            from sklearn.preprocessing import KBinsDiscretizer
            
            # Fit discretizer on training data only
            discretizer = KBinsDiscretizer(n_bins=min(20, len(np.unique(y_train_encoded))), 
                                         encode='ordinal', strategy='uniform')
            
            # Transform training labels
            y_train_binned = discretizer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            # Transform test labels, handling out-of-range values
            try:
                y_test_binned = discretizer.transform(y_test.values.reshape(-1, 1)).ravel()
            except ValueError:
                # If test values are out of training range, clip them
                y_test_clipped = np.clip(y_test.values, 
                                       y_train.min(), y_train.max())
                y_test_binned = discretizer.transform(y_test_clipped.reshape(-1, 1)).ravel()
            
            y_train_final = y_train_binned.astype(int)
            y_test_final = y_test_binned.astype(int)
            
        else:
            # For categorical attributes, use label encoded values
            y_train_final = y_train_encoded
            y_test_final = y_test_encoded
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=128, random_state=42, n_jobs=-1)
        
        try:
            clf.fit(X_train, y_train_final)
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}
        
        # Predict and evaluate
        try:
            y_pred = clf.predict(X_test)
            success_rate = np.mean(y_pred == y_test_final)
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
        
        # Cross-validation on synthetic data to assess model quality
        try:
            cv_scores = cross_val_score(clf, X_train, y_train_final, cv=min(5, len(np.unique(y_train_final))))
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception as e:
            logging.warning(f"Cross-validation failed: {str(e)}")
            cv_mean = cv_std = 0.0
        
        # Calculate additional metrics for better insights
        unique_train_labels = len(np.unique(y_train_final))
        unique_test_labels = len(np.unique(y_test_final))
        
        return {
            'success_rate': float(success_rate),
            'cv_score_mean': cv_mean,
            'cv_score_std': cv_std,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'knowledge_ratio': knowledge_ratio,
            'n_known_features': n_known_features,
            'n_total_features': n_features,
            'unique_train_labels': unique_train_labels,
            'unique_test_labels': unique_test_labels,
            'labels_overlap': len(set(y_train.astype(str)) & set(y_test.astype(str))),
            'total_unique_labels': len(np.unique(all_labels))
        }
    
    def evaluate_aia_vulnerability(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                  quasi_identifiers: List[str], sensitive_attributes: List[str],
                                  sample_size: int = 10000) -> Dict:
        """
        Comprehensive AIA evaluation.
        
        Args:
            original_df: Original dataset
            synthetic_df: Synthetic dataset
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns
            sample_size: Maximum number of records to use
            
        Returns:
            Dictionary with comprehensive AIA results
        """
        logging.info("Starting Attribute Inference Attack evaluation")
        
        # Sample data if needed
        if len(original_df) > sample_size:
            original_df = original_df.sample(n=sample_size, random_state=42)
        if len(synthetic_df) > sample_size:
            synthetic_df = synthetic_df.sample(n=sample_size, random_state=42)
        
        # Prepare data
        orig_df, synth_df, qi_cols, sens_attrs = self.prepare_data_for_attack(
            original_df, synthetic_df, quasi_identifiers, sensitive_attributes
        )
        
        # Encode features
        orig_encoded, synth_encoded, transformer = self.encode_features(
            orig_df, synth_df, qi_cols
        )
        
        results = {
            'dataset_info': {
                'original_size': len(orig_df),
                'synthetic_size': len(synth_df),
                'n_quasi_identifiers': len(qi_cols),
                'n_sensitive_attributes': len(sens_attrs)
            },
            'overlap_statistics': {},
            'attacks': {}
        }
        
        # Test different knowledge ratios
        # knowledge_ratios = [0.3, 0.5, 0.7, 0.9]
        # k_values = [3, 5, 10]
        knowledge_ratios = [1.0]
        k_values = [1, 10, 100]
        
        for sensitive_attr in sens_attrs:
            if sensitive_attr not in orig_df.columns or sensitive_attr not in synth_df.columns:
                continue
                
            results['attacks'][sensitive_attr] = {
                'knn_attacks': [],
                'ml_attacks': []
            }
            
            # K-NN based attacks
            for knowledge_ratio in knowledge_ratios:
                for k in k_values:
                    try:
                        attack_result = self.partial_knowledge_attack(
                            orig_encoded, synth_encoded, orig_df, synth_df,
                            sensitive_attr, knowledge_ratio, k
                        )
                        attack_result['attack_type'] = f'knn_k{k}_knowledge{knowledge_ratio}'
                        results['attacks'][sensitive_attr]['knn_attacks'].append(attack_result)
                        
                        logging.info(f"KNN attack on {sensitive_attr} (k={k}, knowledge={knowledge_ratio}): "
                                   f"Success rate = {attack_result['success_rate']:.3f}")
                    except Exception as e:
                        logging.error(f"KNN attack failed for {sensitive_attr}: {str(e)}")
            
            # ML-based attacks
            for knowledge_ratio in knowledge_ratios:
                try:
                    ml_result = self.machine_learning_attack(
                        orig_encoded, synth_encoded, orig_df, synth_df,
                        sensitive_attr, knowledge_ratio
                    )
                    ml_result['attack_type'] = f'ml_knowledge{knowledge_ratio}'
                    results['attacks'][sensitive_attr]['ml_attacks'].append(ml_result)
                    
                    logging.info(f"ML attack on {sensitive_attr} (knowledge={knowledge_ratio}): "
                               f"Success rate = {ml_result['success_rate']:.3f}")
                except Exception as e:
                    logging.error(f"ML attack failed for {sensitive_attr}: {str(e)}")
        
        # Calculate summary statistics
        results['summary'] = self._calculate_attack_summary(results['attacks'])
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    
    def _calculate_attack_summary(self, attack_results: Dict) -> Dict:
        """Calculate summary statistics for all attacks."""
        summary = {
            'max_success_rates': {},
            'avg_success_rates': {},
            'vulnerability_assessment': {}
        }
        
        for sensitive_attr, attacks in attack_results.items():
            all_success_rates = []
            
            # Collect all success rates
            for attack_type in ['knn_attacks', 'ml_attacks']:
                for attack in attacks.get(attack_type, []):
                    if 'success_rate' in attack:
                        all_success_rates.append(attack['success_rate'])
            
            if all_success_rates:
                summary['max_success_rates'][sensitive_attr] = max(all_success_rates)
                summary['avg_success_rates'][sensitive_attr] = np.mean(all_success_rates)
                
                # Vulnerability assessment
                max_rate = max(all_success_rates)
                if max_rate > 0.8:
                    risk_level = "HIGH"
                elif max_rate > 0.6:
                    risk_level = "MEDIUM"
                elif max_rate > 0.4:
                    risk_level = "LOW"
                else:
                    risk_level = "MINIMAL"
                
                summary['vulnerability_assessment'][sensitive_attr] = {
                    'risk_level': risk_level,
                    'max_success_rate': max_rate,
                    'interpretation': self._interpret_risk_level(risk_level, max_rate)
                }
        
        return summary
    
    def _interpret_risk_level(self, risk_level: str, success_rate: float) -> str:
        """Provide interpretation of risk level."""
        interpretations = {
            "HIGH": f"Critical privacy risk (success rate: {success_rate:.1%}). "
                   "Synthetic data may allow easy inference of sensitive attributes.",
            "MEDIUM": f"Moderate privacy risk (success rate: {success_rate:.1%}). "
                     "Some inference possible but not trivial.",
            "LOW": f"Low privacy risk (success rate: {success_rate:.1%}). "
                   "Limited inference capability.",
            "MINIMAL": f"Minimal privacy risk (success rate: {success_rate:.1%}). "
                      "Inference attacks largely unsuccessful."
        }
        return interpretations.get(risk_level, "Unknown risk level")
    
    def save_results(self, results: Dict, db_name: str, table_name: str):
        """Save AIA results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aia_results_{db_name}_{table_name}_{timestamp}.json"
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
        
        logging.info(f"AIA results saved to: {filepath}")
        return filepath

