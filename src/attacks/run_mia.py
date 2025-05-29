#!/usr/bin/env python3
"""
Focused MIA test for membership inference attack evaluation
"""

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import logging
from mia import MembershipInferenceAttack
from duckdb_manager import DuckDBManager

def test_membership_inference():
    """
    Simplified test focusing on membership inference attack evaluation
    """
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    db_manager = DuckDBManager()
    mia_evaluator = MembershipInferenceAttack()
    
    # Configuration for focused test
    ORIGINAL_DB = "claims_data.duckdb"  # Your original database
    SYNTHETIC_DB = "cle_test.duckdb"  # Test with one synthetic database first
    TABLE_NAME = "clean_join_2017_inpatient"  # Focus on comprehensive inpatient table
    
    # Your specified quasi-identifiers and features for MIA
    feature_columns = [
        "insurants_year_of_birth",
        "insurants_gender", 
        "insurance_data_regional_code",
        "inpatient_cases_date_of_admission",
        "inpatient_cases_department_admission",
        "inpatient_diagnosis_diagnosis",
        "inpatient_procedures_procedure_code"
    ]
    
    print("="*80)
    print("FOCUSED MIA TEST: Membership Inference Attack")
    print("="*80)
    print(f"Original Database: {ORIGINAL_DB}")
    print(f"Synthetic Database: {SYNTHETIC_DB}")
    print(f"Table: {TABLE_NAME}")
    print(f"Feature Columns: {len(feature_columns)}")
    print("="*80)
    
    # Load data
    try:
        orig_path = db_manager.get_database_path(ORIGINAL_DB)
        synth_path = db_manager.get_database_path(SYNTHETIC_DB)
        
        original_data = db_manager.load_table_data(orig_path, TABLE_NAME)
        synthetic_data = db_manager.load_table_data(synth_path, TABLE_NAME)
        
        print(f"Original data shape: {original_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        
        # Check if required columns exist  
        missing_orig = [col for col in feature_columns 
                       if col not in original_data.columns]
        missing_synth = [col for col in feature_columns 
                        if col not in synthetic_data.columns]
        
        if missing_orig or missing_synth:
            print(f"ERROR: Missing columns in original: {missing_orig}")
            print(f"ERROR: Missing columns in synthetic: {missing_synth}")
            return
        
        # Split original data into training (members) and holdout (non-members)
        # Simulate the scenario where synthetic model was trained on part of the data
        
        # Use a deterministic split based on patient ID to ensure consistency
        # Sort by pid first to make split deterministic
        original_sorted = original_data.sort_values('pid').reset_index(drop=True)
        
        # Get unique patient IDs
        unique_pids = original_sorted['pid'].unique()
        n_patients = len(unique_pids)
        
        # Split patients 80/20 for training/holdout
        train_patients = int(0.8 * n_patients)
        training_pids = set(unique_pids[:train_patients])
        holdout_pids = set(unique_pids[train_patients:])
        
        # Create training and holdout datasets based on patient splits
        training_data = original_sorted[original_sorted['pid'].isin(training_pids)].copy()
        holdout_data = original_sorted[original_sorted['pid'].isin(holdout_pids)].copy()
        
        print(f"\nDataset split (by patients):")
        print(f"  Total patients: {n_patients:,}")
        print(f"  Training patients: {len(training_pids):,}")
        print(f"  Holdout patients: {len(holdout_pids):,}")
        print(f"  Training records (members): {len(training_data):,}")
        print(f"  Holdout records (non-members): {len(holdout_data):,}")
        print(f"  Synthetic records: {len(synthetic_data):,}")
        
        # Verify no patient overlap between training and holdout
        overlap_patients = training_pids & holdout_pids
        if overlap_patients:
            print(f"WARNING: Patient overlap detected: {len(overlap_patients)} patients")
        else:
            print("✓ No patient overlap between training and holdout sets")
        
        # Run MIA evaluation
        print(f"\nRunning MIA evaluation...")
        results = mia_evaluator.run_membership_inference_attack(
            training_data=training_data,
            holdout_data=holdout_data, 
            synthetic_data=synthetic_data,
            feature_columns=feature_columns,
            sample_size=50000,  # Reasonable sample for testing
            distance_metric='euclidean',
            optimization_metric='f1'
        )
        
        # Display results
        print(f"\n" + "="*60)
        print("MIA EVALUATION RESULTS")
        print("="*60)
        
        attack_perf = results['attack_performance']
        dataset_info = results['dataset_info']
        config = results['attack_config']
        
        print(f"\nDataset Information:")
        print(f"  Training records used: {dataset_info['n_training_records']:,}")
        print(f"  Holdout records used: {dataset_info['n_holdout_records']:,}")
        print(f"  Synthetic records used: {dataset_info['n_synthetic_records']:,}")
        print(f"  Total target records: {dataset_info['total_target_records']:,}")
        
        print(f"\nAttack Configuration:")
        print(f"  Distance metric: {config['distance_metric']}")
        print(f"  Optimization metric: {config['optimization_metric']}")
        print(f"  Features used: {config['n_features']}")
        
        print(f"\nAttack Performance:")
        print(f"  Accuracy: {attack_perf['accuracy']:.4f}")
        print(f"  Precision: {attack_perf['precision']:.4f}")
        print(f"  Recall: {attack_perf['recall']:.4f}")
        print(f"  F1-Score: {attack_perf['f1_score']:.4f}")
        print(f"  AUC-ROC: {attack_perf['auc_roc']:.4f}")
        print(f"  Optimal Threshold: {attack_perf['threshold']:.6f}")
        
        print(f"\nConfusion Matrix:")
        cm = attack_perf['confusion_matrix']
        print(f"  True Positives: {cm['true_positives']:,}")
        print(f"  False Positives: {cm['false_positives']:,}")
        print(f"  True Negatives: {cm['true_negatives']:,}")
        print(f"  False Negatives: {cm['false_negatives']:,}")
        
        print(f"\nDistance Statistics:")
        member_stats = attack_perf['distance_statistics']['member_distances']
        non_member_stats = attack_perf['distance_statistics']['non_member_distances']
        
        print(f"  Member distances (training data):")
        print(f"    Mean: {member_stats['mean']:.6f}")
        print(f"    Median: {member_stats['median']:.6f}")
        print(f"    Std: {member_stats['std']:.6f}")
        print(f"    Range: [{member_stats['min']:.6f}, {member_stats['max']:.6f}]")
        
        print(f"  Non-member distances (holdout data):")
        print(f"    Mean: {non_member_stats['mean']:.6f}")
        print(f"    Median: {non_member_stats['median']:.6f}")
        print(f"    Std: {non_member_stats['std']:.6f}")
        print(f"    Range: [{non_member_stats['min']:.6f}, {non_member_stats['max']:.6f}]")
        
        # Privacy assessment
        print(f"\n" + "="*60)
        print("PRIVACY ASSESSMENT")
        print("="*60)
        
        auc_roc = attack_perf['auc_roc']
        accuracy = attack_perf['accuracy']
        
        if auc_roc > 0.8 or accuracy > 0.8:
            risk_level = "HIGH PRIVACY RISK"
            interpretation = "The synthetic data shows strong membership inference vulnerability. " \
                           "Attackers can reliably determine if specific records were used for training."
        elif auc_roc > 0.7 or accuracy > 0.7:
            risk_level = "MODERATE PRIVACY RISK"
            interpretation = "The synthetic data shows moderate membership inference vulnerability. " \
                           "Some inference attacks may succeed."
        elif auc_roc > 0.6 or accuracy > 0.6:
            risk_level = "LOW PRIVACY RISK"
            interpretation = "The synthetic data shows limited membership inference vulnerability. " \
                           "Inference attacks have limited success."
        else:
            risk_level = "MINIMAL PRIVACY RISK"
            interpretation = "The synthetic data appears robust against membership inference attacks. " \
                           "Attack success is close to random guessing."
        
        print(f"Risk Level: {risk_level}")
        print(f"AUC-ROC: {auc_roc:.4f} (0.5 = random, 1.0 = perfect attack)")
        print(f"Accuracy: {accuracy:.4f} (0.5 = random, 1.0 = perfect attack)")
        print(f"\nInterpretation: {interpretation}")
        
        if member_stats['mean'] < non_member_stats['mean']:
            print(f"\n✓ Expected pattern: Members have lower average distance to synthetic data")
            print(f"  Member avg distance: {member_stats['mean']:.6f}")
            print(f"  Non-member avg distance: {non_member_stats['mean']:.6f}")
            print(f"  Difference: {non_member_stats['mean'] - member_stats['mean']:.6f}")
        else:
            print(f"\n⚠ Unexpected pattern: Non-members have lower average distance to synthetic data")
            print(f"  This may indicate issues with the synthetic data generation process")
        
        # Save results
        output_file = mia_evaluator.save_results(
            results, 
            ORIGINAL_DB.replace('.duckdb', ''), 
            SYNTHETIC_DB.replace('.duckdb', ''), 
            TABLE_NAME
        )
        
        print(f"\nTest completed successfully!")
        print(f"Results saved to: {output_file}")
        print(f"Visualization saved to: {results['visualization_path']}")
        
        # Additional recommendations
        print(f"\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if auc_roc > 0.7:
            print("• Consider using stronger privacy-preserving techniques")
            print("• Evaluate differential privacy parameters")
            print("• Consider reducing the synthetic dataset size")
            print("• Review data preprocessing and feature selection")
        elif auc_roc > 0.6:
            print("• Monitor privacy metrics when deploying synthetic data")
            print("• Consider additional privacy safeguards for sensitive use cases")
        else:
            print("• Current privacy protection appears adequate")
            print("• Continue monitoring with additional attack methods")
        
        print("• Test with different distance metrics and sample sizes")
        print("• Consider testing other synthetic datasets from different vendors")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        logging.error(f"MIA test failed: {str(e)}", exc_info=True)


def test_multiple_synthetic_datasets():
    """
    Test MIA against multiple synthetic datasets for comparison
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    db_manager = DuckDBManager()
    mia_evaluator = MembershipInferenceAttack()
    
    # Configuration
    ORIGINAL_DB = "claims_data.duckdb"
    SYNTHETIC_DBS = [
        "ai4medicine_arf.duckdb",
        "ai4medicine_gan.duckdb", 
        "cprd_bn.duckdb",
        "limebit_bn.duckdb",
        "limebit_mtgan.duckdb"
    ]
    TABLE_NAME = "all_inpatient"
    
    feature_columns = [
        "insurants_year_of_birth",
        "insurants_gender", 
        "insurance_data_regional_code",
        "inpatient_cases_date_of_admission",
        "inpatient_cases_department_admission",
        "inpatient_diagnosis_diagnosis",
        "inpatient_procedures_procedure_code"
    ]
    
    print("="*80)
    print("COMPARATIVE MIA TEST: Multiple Synthetic Datasets")
    print("="*80)
    
    # Load original data once
    orig_path = db_manager.get_database_path(ORIGINAL_DB)
    original_data = db_manager.load_table_data(orig_path, TABLE_NAME)
    
    # Create training/holdout split
    original_sorted = original_data.sort_values('pid').reset_index(drop=True)
    unique_pids = original_sorted['pid'].unique()
    n_patients = len(unique_pids)
    train_patients = int(0.8 * n_patients)
    training_pids = set(unique_pids[:train_patients])
    holdout_pids = set(unique_pids[train_patients:])
    
    training_data = original_sorted[original_sorted['pid'].isin(training_pids)].copy()
    holdout_data = original_sorted[original_sorted['pid'].isin(holdout_pids)].copy()
    
    results_summary = []
    
    for synthetic_db in SYNTHETIC_DBS:
        print(f"\n{'='*60}")
        print(f"Testing: {synthetic_db}")
        print(f"{'='*60}")
        
        try:
            # Load synthetic data
            synth_path = db_manager.get_database_path(synthetic_db)
            synthetic_data = db_manager.load_table_data(synth_path, TABLE_NAME)
            
            print(f"Synthetic data shape: {synthetic_data.shape}")
            
            # Check if required columns exist
            missing_synth = [col for col in feature_columns 
                            if col not in synthetic_data.columns]
            
            if missing_synth:
                print(f"ERROR: Missing columns in {synthetic_db}: {missing_synth}")
                continue
            
            # Run MIA evaluation
            results = mia_evaluator.run_membership_inference_attack(
                training_data=training_data,
                holdout_data=holdout_data,
                synthetic_data=synthetic_data,
                feature_columns=feature_columns,
                sample_size=25000,  # Smaller sample for multiple tests
                distance_metric='euclidean',
                optimization_metric='f1'
            )
            
            # Extract key metrics
            attack_perf = results['attack_performance']
            summary = {
                'synthetic_db': synthetic_db,
                'accuracy': attack_perf['accuracy'],
                'precision': attack_perf['precision'],
                'recall': attack_perf['recall'],
                'f1_score': attack_perf['f1_score'],
                'auc_roc': attack_perf['auc_roc'],
                'threshold': attack_perf['threshold']
            }
            results_summary.append(summary)
            
            print(f"Results: AUC-ROC={attack_perf['auc_roc']:.4f}, "
                  f"Accuracy={attack_perf['accuracy']:.4f}, "
                  f"F1={attack_perf['f1_score']:.4f}")
            
            # Save individual results
            output_file = mia_evaluator.save_results(
                results, 
                ORIGINAL_DB.replace('.duckdb', ''), 
                synthetic_db.replace('.duckdb', ''), 
                TABLE_NAME
            )
            print(f"Results saved to: {output_file}")
            
        except Exception as e:
            print(f"Error testing {synthetic_db}: {str(e)}")
            continue
    
    # Print comparative summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    if results_summary:
        # Sort by AUC-ROC (lower is better for privacy)
        results_summary.sort(key=lambda x: x['auc_roc'])
        
        print(f"{'Database':<20} {'AUC-ROC':<8} {'Accuracy':<8} {'F1-Score':<8} {'Privacy':<15}")
        print("-" * 80)
        
        for result in results_summary:
            auc_roc = result['auc_roc']
            if auc_roc > 0.8:
                privacy_level = "HIGH RISK"
            elif auc_roc > 0.7:
                privacy_level = "MODERATE RISK"
            elif auc_roc > 0.6:
                privacy_level = "LOW RISK"
            else:
                privacy_level = "MINIMAL RISK"
            
            db_name = result['synthetic_db'].replace('.duckdb', '')
            print(f"{db_name:<20} {auc_roc:<8.4f} {result['accuracy']:<8.4f} "
                  f"{result['f1_score']:<8.4f} {privacy_level:<15}")
        
        print("\nBest privacy protection (lowest AUC-ROC):")
        best = results_summary[0]
        print(f"  {best['synthetic_db']}: AUC-ROC = {best['auc_roc']:.4f}")
        
        print("\nWorst privacy protection (highest AUC-ROC):")
        worst = results_summary[-1]
        print(f"  {worst['synthetic_db']}: AUC-ROC = {worst['auc_roc']:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run comparative test across multiple synthetic datasets
        test_multiple_synthetic_datasets()
    else:
        # Run focused test on single synthetic dataset
        test_membership_inference()