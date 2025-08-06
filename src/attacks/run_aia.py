"""
Attribute Inference Attack (AIA) Runner Module

This module provides a focused testing framework for running Attribute Inference Attacks
on health claims data, specifically designed to evaluate the privacy protection of
synthetic data against attribute inference attacks.

The module includes:
- test_inpatient_aia: Main function for testing inpatient diagnosis and procedure inference
- AIA_SAMPLE_CONFIGS: Configuration for different sample sizes based on dataset
- SAMPLE_SIZE: Default sample size for AIA testing

Key Features:
- Focused testing on specific sensitive attributes (diagnosis codes, procedure codes)
- Configurable sample sizes for different synthetic datasets
- Comprehensive logging and result reporting
- Integration with DuckDB manager for data loading
- Support for multiple synthetic data sources
- Detailed analysis of attack success rates

Attack Focus:
- Inpatient diagnosis code inference
- Inpatient procedure code inference
- Multiple knowledge ratio testing
- Comprehensive vulnerability assessment

Usage:
    python run_aia.py
    # Modify the configuration variables to test different datasets and attributes

Configuration:
- ORIGINAL_DB: Original database name
- SYNTHETIC_DB: Synthetic database name to test
- TABLE_NAME: Table to analyze
- quasi_identifiers: List of quasi-identifier columns
- sensitive_attributes: List of sensitive attributes to test

Author: [Your Name]
Date: [Date]
"""
#!/usr/bin/env python3
"""
Focused AIA test for inpatient diagnosis and procedure code inference
"""

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import logging
from aia import AttributeInferenceAttack
from duckdb_manager.duckdb_manager import DuckDBManager

# numbers for 2017 inpatient table
AIA_SAMPLE_CONFIGS = {
    'base_sample': 50000,      # Minimum for all datasets
    'bonus_samples': {         # Additional samples for larger datasets
        'claims_data': 50000,   # Total: 100k
        'cle_test': 70000,      # Total: 120k  
        'ai4medicine_arf': 70000, # Total: 120k
        'limebit_mtgan': 10000,  # Total: 60k
        'limebit_bn': 5000,      # Total: 55k
        'cprd_bn': 0             # Total: 50k (limited by available data)
    }
}

SAMPLE_SIZE = 50000

def test_inpatient_aia():
    """
    Main function to test Attribute Inference Attacks on inpatient data.
    
    This function:
    1. Configures logging and initializes AIA evaluator and database manager
    2. Sets up test parameters for inpatient diagnosis and procedure inference
    3. Loads original and synthetic datasets from DuckDB databases
    4. Validates that all required columns exist in both datasets
    5. Analyzes sensitive attribute distributions before filtering
    6. Runs comprehensive AIA evaluation with multiple attack strategies
    7. Provides detailed analysis of attack success rates and vulnerability assessment
    8. Saves results to JSON file with comprehensive reporting
    
    Test Configuration:
    - Original Database: claims_data.duckdb
    - Synthetic Database: cprd_bn.duckdb (configurable)
    - Table: clean_join_2017_inpatient
    - Sensitive Attributes: inpatient_diagnosis_diagnosis, inpatient_procedures_procedure_code
    - Quasi-Identifiers: 5 demographic and admission-related fields
    
    Attack Strategies:
    - k-NN based attacks with varying knowledge ratios
    - Machine learning attacks using Random Forest
    - Multiple sampling strategies for realistic attack scenarios
    
    Returns:
        None: Results are printed to console and saved to file
    """
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    db_manager = DuckDBManager()
    aia_evaluator = AttributeInferenceAttack()
    
    # Configuration for focused test
    ORIGINAL_DB = "claims_data.duckdb"  # Your original database
    SYNTHETIC_DB = "cprd_bn.duckdb"  # Test with one synthetic database first
    TABLE_NAME = "clean_join_2017_inpatient"  # Focus on inpatient table from 2018
    # TABLE_NAME = "all_inpatient"  # alle mal testen
    
    # Your specified quasi-identifiers
    quasi_identifiers = [
        "insurants_year_of_birth",
        "insurants_gender", 
        "insurance_data_regional_code",
        "inpatient_cases_date_of_admission",
        "inpatient_cases_department_admission"
    ]
    
    # Both sensitive attributes for comprehensive test
    sensitive_attributes = [
        "inpatient_diagnosis_diagnosis",
        "inpatient_procedures_procedure_code"
    ]
    
    print("="*80)
    print("FOCUSED AIA TEST: Inpatient Diagnosis & Procedure Code Inference")
    print("="*80)
    print(f"Original Database: {ORIGINAL_DB}")
    print(f"Synthetic Database: {SYNTHETIC_DB}")
    print(f"Table: {TABLE_NAME}")
    print(f"Target Attributes: {', '.join(sensitive_attributes)}")
    print(f"Quasi-Identifiers: {len(quasi_identifiers)}")
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
        missing_orig = [col for col in quasi_identifiers + sensitive_attributes 
                       if col not in original_data.columns]
        missing_synth = [col for col in quasi_identifiers + sensitive_attributes 
                        if col not in synthetic_data.columns]
        
        if missing_orig or missing_synth:
            print(f"ERROR: Missing columns in original: {missing_orig}")
            print(f"ERROR: Missing columns in synthetic: {missing_synth}")
            return
        
        # Check both sensitive attributes distribution before filtering
        print(f"\nSensitive attribute analysis:")
        for attr in sensitive_attributes:
            print(f"{attr}:")
            print(f"  Original unique values: {original_data[attr].nunique()}")
            print(f"  Synthetic unique values: {synthetic_data[attr].nunique()}")
            
            # Show most common values
            orig_top = original_data[attr].value_counts().head(5)
            print(f"  Top 5 original values:")
            for value, count in orig_top.items():
                print(f"    {value}: {count}")
            print()

        # Initialize combined results dictionary
        combined_results = {
            'dataset_info': {
                'original_size': len(original_data),
                'synthetic_size': len(synthetic_data),
                'n_quasi_identifiers': len(quasi_identifiers),
                'n_sensitive_attributes': len(sensitive_attributes)
            },
            'overlap_statistics': {},
            'attacks': {},
            'summary': {
                'max_success_rates': {},
                'avg_success_rates': {},
                'vulnerability_assessment': {}
            }
        }
        
        # Process each sensitive attribute separately with independent sampling
        print(f"\nRunning separate AIA evaluation for each sensitive attribute...")
        print("="*80)
        
        for i, attr in enumerate(sensitive_attributes, 1):
            print(f"\nProcessing attribute {i}/{len(sensitive_attributes)}: {attr}")
            print("-" * 60)
            
            # Independent sampling for this attribute
            original_filtered, synthetic_sample, overlap_stats = aia_evaluator.sample_overlapping_data(
                original_data, 
                synthetic_data,
                sensitive_attribute=attr,
                synthetic_sample_size=SAMPLE_SIZE,  # Exact size of clean samples
                strategy='common'
            )

            print(f"Overlap sampling for '{attr}':")
            print(f"  Requested: {overlap_stats['requested_sample_size']:,}")
            print(f"  Achieved: {overlap_stats['achieved_sample_size']:,}")
            print(f"  Clean synthetic records available: {overlap_stats['synthetic_clean_records']:,}")
            print(f"  Original records after filtering: {len(original_filtered):,}")
            print(f"  Value overlap ratio: {overlap_stats['overlap_ratio']:.1%}")
            
            # Run AIA evaluation for this single attribute
            print(f"  Running AIA attacks for {attr}...")
            attr_results = aia_evaluator.evaluate_aia_vulnerability(
                original_filtered, 
                synthetic_sample,
                quasi_identifiers, 
                [attr],  # Only this single attribute
                sample_size=SAMPLE_SIZE  # Sample size for testing
            )
            
            # Add overlap statistics for this attribute
            attr_results = aia_evaluator.add_overlap_statistics_to_results(
                attr_results, 
                overlap_stats, 
                sensitive_attribute=attr
            )
            
            # Merge results into combined results
            combined_results['overlap_statistics'][attr] = overlap_stats
            
            if attr in attr_results.get('attacks', {}):
                combined_results['attacks'][attr] = attr_results['attacks'][attr]
            
            if 'summary' in attr_results:
                if attr in attr_results['summary'].get('max_success_rates', {}):
                    combined_results['summary']['max_success_rates'][attr] = attr_results['summary']['max_success_rates'][attr]
                if attr in attr_results['summary'].get('avg_success_rates', {}):
                    combined_results['summary']['avg_success_rates'][attr] = attr_results['summary']['avg_success_rates'][attr]
                if attr in attr_results['summary'].get('vulnerability_assessment', {}):
                    combined_results['summary']['vulnerability_assessment'][attr] = attr_results['summary']['vulnerability_assessment'][attr]
            
            print(f"  ✓ Completed AIA evaluation for {attr}")
        
        # Use the combined results
        results = combined_results
        results['timestamp'] = attr_results.get('timestamp', '')  # Add timestamp from last evaluation

        # Display results
        print(f"\n" + "="*80)
        print("AIA EVALUATION RESULTS")
        print("="*80)
        
        if 'summary' in results:
            print("VULNERABILITY ASSESSMENT:")
            print("-" * 50)
            for attr, assessment in results['summary']['vulnerability_assessment'].items():
                print(f"\nAttribute: {attr}")
                print(f"  Risk Level: {assessment['risk_level']}")
                print(f"  Max Success Rate: {assessment['max_success_rate']:.1%}")
                print(f"  Average Success Rate: {results['summary']['avg_success_rates'][attr]:.1%}")
                print(f"  Interpretation: {assessment['interpretation']}")
        
        # Show detailed attack results for both attributes
        if 'attacks' in results:
            print(f"\n" + "="*80)
            print("DETAILED ATTACK RESULTS")
            print("="*80)
            
            for attr in sensitive_attributes:
                if attr in results['attacks']:
                    attacks = results['attacks'][attr]
                    
                    print(f"\n{attr.upper()}:")
                    print("-" * 60)
                    
                    # k-NN attacks
                    knn_attacks = attacks.get('knn_attacks', [])
                    if knn_attacks:
                        print("k-NN Attack Results:")
                        for attack in knn_attacks:
                            print(f"  {attack['attack_type']}: {attack['success_rate']:.1%} success "
                                  f"({attack['successful_attacks']}/{attack['total_attacks']} attacks)")
                        
                        # Find best k-NN attack
                        best_knn = max(knn_attacks, key=lambda x: x['success_rate'])
                        print(f"  → Best k-NN: {best_knn['attack_type']} with {best_knn['success_rate']:.1%} success")
                    
                    # ML attacks  
                    ml_attacks = attacks.get('ml_attacks', [])
                    if ml_attacks:
                        print("\nML Attack Results:")
                        for attack in ml_attacks:
                            if 'error' not in attack:
                                print(f"  {attack['attack_type']}: {attack['success_rate']:.1%} success "
                                      f"(CV score: {attack.get('cv_score_mean', 0):.3f})")
                                print(f"    Training samples: {attack.get('training_samples', 'N/A')}")
                                print(f"    Test samples: {attack.get('test_samples', 'N/A')}")
                                print(f"    Label overlap: {attack.get('labels_overlap', 'N/A')}/{attack.get('total_unique_labels', 'N/A')}")
                            else:
                                print(f"  {attack['attack_type']}: ERROR - {attack['error']}")
        
        # Show overlap statistics summary
        if 'overlap_statistics' in results:
            print(f"\n" + "="*80)
            print("OVERLAP STATISTICS SUMMARY")
            print("="*80)
            
            for attr, overlap_stats in results['overlap_statistics'].items():
                print(f"\n{attr}:")
                print(f"  Strategy: {overlap_stats.get('strategy_used', 'N/A')}")
                print(f"  Original records (filtered): {overlap_stats.get('original_filtered_records', 0):,}")
                print(f"  Synthetic sample size: {overlap_stats.get('achieved_sample_size', 0):,}")
                print(f"  Value overlap ratio: {overlap_stats.get('overlap_ratio', 0):.1%}")
                print(f"  Filtering ratio: {overlap_stats.get('filtering_ratio', 0):.1%}")
        
        # Summary comparison
        print(f"\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        if 'summary' in results and len(sensitive_attributes) > 1:
            max_rates = results['summary']['max_success_rates']
            avg_rates = results['summary']['avg_success_rates']
            
            print("Maximum Success Rates:")
            for attr in sensitive_attributes:
                if attr in max_rates:
                    print(f"  {attr}: {max_rates[attr]:.1%}")
            
            print("\nAverage Success Rates:")
            for attr in sensitive_attributes:
                if attr in avg_rates:
                    print(f"  {attr}: {avg_rates[attr]:.1%}")
            
            # Determine which attribute is more vulnerable
            if len(max_rates) == 2:
                attrs = list(max_rates.keys())
                rates = list(max_rates.values())
                more_vulnerable = attrs[0] if rates[0] > rates[1] else attrs[1]
                difference = abs(rates[0] - rates[1])
                
                print(f"\nVulnerability Comparison:")
                print(f"  More vulnerable attribute: {more_vulnerable}")
                print(f"  Success rate difference: {difference:.1%}")
                
                if difference > 0.1:  # 10% difference
                    print(f"  → Significant difference in vulnerability")
                else:
                    print(f"  → Similar vulnerability levels")
        
        # Save results
        output_file = aia_evaluator.save_results(
            results, 
            f"{ORIGINAL_DB.replace('.duckdb', '')}_vs_{SYNTHETIC_DB.replace('.duckdb', '')}", 
            TABLE_NAME
        )
        
        print(f"\n" + "="*80)
        print(f"Test completed successfully!")
        print(f"Results saved to: {output_file}")
        print("="*80)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        logging.error(f"AIA test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Run the focused test
    test_inpatient_aia()