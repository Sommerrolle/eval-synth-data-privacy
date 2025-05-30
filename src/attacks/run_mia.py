#!/usr/bin/env python3
"""
Focused MIA test for membership inference attack evaluation
"""

import os.path, sys
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from mia import MembershipInferenceAttack
from duckdb_manager.duckdb_manager import DuckDBManager

# MIA sample configurations
MIA_SAMPLE_CONFIGS = {
    'target_sample_per_set': 40000,  # 40k each for training & holdout (80k total target data)
    'synthetic_multiplier': 1.5,     # Synthetic dataset will be 1.5x the total target data
    'dataset_specific_limits': {
        'limebit_mtgan': 30000,      # Smaller due to dataset limitations
        'limebit_bn': 25000,         # Smaller due to dataset limitations
        'cprd_bn': 20000             # Smallest due to dataset limitations
    }
}

def test_membership_inference():
    """
    Simplified test focusing on membership inference attack evaluation
    """
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    db_manager = DuckDBManager()
    mia_evaluator = MembershipInferenceAttack()
    
    # Configuration for focused test - using separate databases
    TRAINING_DB = "claims_data.duckdb"      # Training database (members)
    HOLDOUT_DB = "cle_test.duckdb"         # Holdout database (non-members)  
    SYNTHETIC_DB = "limebit_mtgan.duckdb"     # Synthetic database
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
    
    # Determine dataset-specific limit based on synthetic DB name
    dataset_specific_limit = None
    for db_name, limit in MIA_SAMPLE_CONFIGS['dataset_specific_limits'].items():
        if db_name in SYNTHETIC_DB:
            dataset_specific_limit = limit
            break
    
    print("="*80)
    print("FOCUSED MIA TEST: Membership Inference Attack")
    print("="*80)
    print(f"Training Database (members): {TRAINING_DB}")
    print(f"Holdout Database (non-members): {HOLDOUT_DB}")
    print(f"Synthetic Database: {SYNTHETIC_DB}")
    print(f"Table: {TABLE_NAME}")
    print(f"Feature Columns: {len(feature_columns)}")
    print(f"Target Sample Per Set: {MIA_SAMPLE_CONFIGS['target_sample_per_set']:,}")
    print(f"Synthetic Multiplier: {MIA_SAMPLE_CONFIGS['synthetic_multiplier']}")
    if dataset_specific_limit:
        print(f"Dataset-Specific Limit: {dataset_specific_limit:,}")
    print("="*80)
    
    # Load data from separate databases
    try:
        training_path = db_manager.get_database_path(TRAINING_DB)
        holdout_path = db_manager.get_database_path(HOLDOUT_DB)
        synth_path = db_manager.get_database_path(SYNTHETIC_DB)
        
        training_data = db_manager.load_table_data(training_path, TABLE_NAME)
        holdout_data = db_manager.load_table_data(holdout_path, TABLE_NAME)
        synthetic_data = db_manager.load_table_data(synth_path, TABLE_NAME)
        
        print(f"Training data shape: {training_data.shape}")
        print(f"Holdout data shape: {holdout_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        
        # Check if required columns exist in all databases
        missing_training = [col for col in feature_columns 
                           if col not in training_data.columns]
        missing_holdout = [col for col in feature_columns 
                          if col not in holdout_data.columns]
        missing_synth = [col for col in feature_columns 
                        if col not in synthetic_data.columns]
        
        if missing_training or missing_holdout or missing_synth:
            print(f"ERROR: Missing columns in training: {missing_training}")
            print(f"ERROR: Missing columns in holdout: {missing_holdout}")
            print(f"ERROR: Missing columns in synthetic: {missing_synth}")
            return
        
        print(f"\nDataset information:")
        print(f"  Training records (members): {len(training_data):,}")
        print(f"  Holdout records (non-members): {len(holdout_data):,}")
        print(f"  Synthetic records: {len(synthetic_data):,}")
        
        # Optional: Check for patient overlap between training and holdout
        if 'pid' in training_data.columns and 'pid' in holdout_data.columns:
            training_pids = set(training_data['pid'].unique())
            holdout_pids = set(holdout_data['pid'].unique())
            overlap_patients = training_pids & holdout_pids
            
            if overlap_patients:
                print(f"WARNING: Patient overlap detected: {len(overlap_patients)} patients")
                print("  This could affect the validity of the membership inference attack!")
            else:
                print("✓ No patient overlap between training and holdout sets")
        
        # Run MIA evaluation
        print(f"\nRunning MIA evaluation...")
        results = mia_evaluator.run_membership_inference_attack(
            training_data=training_data,
            holdout_data=holdout_data, 
            synthetic_data=synthetic_data,
            feature_columns=feature_columns,
            target_sample_per_set=MIA_SAMPLE_CONFIGS['target_sample_per_set'],
            synthetic_multiplier=MIA_SAMPLE_CONFIGS['synthetic_multiplier'],
            dataset_specific_limit=dataset_specific_limit
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
        print(f"  Target sample per set: {config['target_sample_per_set']:,}")
        print(f"  Synthetic multiplier: {config['synthetic_multiplier']}")
        print(f"  Dataset specific limit: {config['dataset_specific_limit']}")
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
            f"{TRAINING_DB.replace('.duckdb', '')}_vs_{HOLDOUT_DB.replace('.duckdb', '')}", 
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
    
    # Configuration - using separate training and holdout databases
    TRAINING_DB = "claims_data.duckdb"      # Training database (members)
    HOLDOUT_DB = "cle_test.duckdb"         # Holdout database (non-members)
    SYNTHETIC_DBS = [
        "ai4medicine_arf.duckdb",
        "ai4medicine_gan.duckdb", 
        "cprd_bn.duckdb",
        "limebit_bn.duckdb",
        "limebit_mtgan.duckdb"
    ]
    TABLE_NAME = "clean_join_2017_inpatient"  # Updated to use specific year table
    
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
    print(f"Training Database (members): {TRAINING_DB}")
    print(f"Holdout Database (non-members): {HOLDOUT_DB}")
    print(f"Table: {TABLE_NAME}")
    print("="*80)
    
    # Load training and holdout data once
    training_path = db_manager.get_database_path(TRAINING_DB)
    holdout_path = db_manager.get_database_path(HOLDOUT_DB)
    
    training_data = db_manager.load_table_data(training_path, TABLE_NAME)
    holdout_data = db_manager.load_table_data(holdout_path, TABLE_NAME)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Holdout data shape: {holdout_data.shape}")
    
    # Check for patient overlap
    if 'pid' in training_data.columns and 'pid' in holdout_data.columns:
        training_pids = set(training_data['pid'].unique())
        holdout_pids = set(holdout_data['pid'].unique())
        overlap_patients = training_pids & holdout_pids
        
        if overlap_patients:
            print(f"WARNING: Patient overlap detected: {len(overlap_patients)} patients")
        else:
            print("✓ No patient overlap between training and holdout sets")
    
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
            
            # Determine dataset-specific limit
            dataset_specific_limit = None
            for db_name, limit in MIA_SAMPLE_CONFIGS['dataset_specific_limits'].items():
                if db_name in synthetic_db:
                    dataset_specific_limit = limit
                    print(f"Using dataset-specific limit: {limit:,}")
                    break
            
            # Run MIA evaluation
            results = mia_evaluator.run_membership_inference_attack(
                training_data=training_data,
                holdout_data=holdout_data,
                synthetic_data=synthetic_data,
                feature_columns=feature_columns,
                target_sample_per_set=MIA_SAMPLE_CONFIGS['target_sample_per_set'],
                synthetic_multiplier=MIA_SAMPLE_CONFIGS['synthetic_multiplier'],
                dataset_specific_limit=dataset_specific_limit
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
                f"{TRAINING_DB.replace('.duckdb', '')}_vs_{HOLDOUT_DB.replace('.duckdb', '')}", 
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