#!/usr/bin/env python3
"""
Focused AIA test for inpatient diagnosis inference
"""

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import logging
from aia import AttributeInferenceAttack
from duckdb_manager.duckdb_manager import DuckDBManager

def test_inpatient_aia():
    """
    Simplified test focusing on inpatient diagnosis inference
    """
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    db_manager = DuckDBManager()
    aia_evaluator = AttributeInferenceAttack()
    
    # Configuration for focused test
    ORIGINAL_DB = "claims_data.duckdb"  # Your original database
    SYNTHETIC_DB = "cle_test.duckdb"  # Test with one synthetic database first
    TABLE_NAME = "clean_join_2018_inpatient"  # Focus on inpatient table from 2017
    # TABLE_NAME = "all_inpatient"  # alle mal testen
    
    # Your specified quasi-identifiers
    quasi_identifiers = [
        "insurants_year_of_birth",
        "insurants_gender", 
        "insurance_data_regional_code",
        "inpatient_cases_date_of_admission",
        "inpatient_cases_department_admission"
    ]
    
    # Focus on diagnosis only for initial test
    # sensitive_attributes = [
    #     "inpatient_diagnosis_diagnosis"  # Start with just this one
    # ]
    sensitive_attributes = [
        "inpatient_procedures_procedure_code"
    ]
    
    print("="*80)
    print("FOCUSED AIA TEST: Inpatient Diagnosis Inference")
    print("="*80)
    print(f"Original Database: {ORIGINAL_DB}")
    print(f"Synthetic Database: {SYNTHETIC_DB}")
    print(f"Table: {TABLE_NAME}")
    print(f"Target Attribute: {sensitive_attributes[0]}")
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
        
        # Check diagnosis code distribution before filtering
        print(f"\nDiagnosis code analysis:")
        print(f"Original unique diagnoses: {original_data['inpatient_procedures_procedure_code'].nunique()}")
        print(f"Synthetic unique diagnoses: {synthetic_data['inpatient_procedures_procedure_code'].nunique()}")
        
        # Show most common diagnosis codes
        orig_top_diagnoses = original_data['inpatient_procedures_procedure_code'].value_counts().head(10)
        print(f"\nTop 10 original diagnoses:")
        print(orig_top_diagnoses)

        # Sample data with overlapping data function
        original_filtered, synthetic_sample, stats = aia_evaluator.sample_overlapping_data(
            original_data, 
            synthetic_data,
            sensitive_attribute='inpatient_procedures_procedure_code',
            synthetic_sample_size=100000,  # Exact size of clean samples
            strategy='common'
        )

        print(f"Requested: {stats['requested_sample_size']:,}")
        print(f"Achieved: {stats['achieved_sample_size']:,}")
        print(f"Clean synthetic records available: {stats['synthetic_clean_records']:,}")
        
        # Run AIA evaluation with smaller sample for testing
        print(f"\nRunning AIA evaluation...")
        results = aia_evaluator.evaluate_aia_vulnerability(
            original_filtered, 
            synthetic_sample,
            quasi_identifiers, 
            sensitive_attributes,
            sample_size=100000  # Small sample for testing
        )
        
        # Add overlap statistics to the results
        results = aia_evaluator.add_overlap_statistics_to_results(
            results, 
            stats, 
            sensitive_attribute='inpatient_procedures_procedure_code'
        )

        results = aia_evaluator.add_overlap_statistics_to_results(
            results, 
            stats, 
            sensitive_attribute='inpatient_diagnosis_diagnosise'
        )

        # Display results
        print(f"\n" + "="*60)
        print("AIA EVALUATION RESULTS")
        print("="*60)
        
        if 'summary' in results:
            for attr, assessment in results['summary']['vulnerability_assessment'].items():
                print(f"\nAttribute: {attr}")
                print(f"Risk Level: {assessment['risk_level']}")
                print(f"Max Success Rate: {assessment['max_success_rate']:.1%}")
                print(f"Interpretation: {assessment['interpretation']}")
        
        # Show detailed attack results for diagnosis
        if 'attacks' in results:
            diagnosis_attacks = results['attacks'].get('inpatient_procedures_procedure_code', {})
            
            print(f"\nDetailed Attack Results for Diagnosis:")
            print("-" * 50)
            
            # k-NN attacks
            knn_attacks = diagnosis_attacks.get('knn_attacks', [])
            if knn_attacks:
                print("k-NN Attack Results:")
                for attack in knn_attacks[:3]:  # Show first 3
                    print(f"  {attack['attack_type']}: {attack['success_rate']:.1%} success")
            
            # ML attacks  
            ml_attacks = diagnosis_attacks.get('ml_attacks', [])
            if ml_attacks:
                print("ML Attack Results:")
                for attack in ml_attacks[:3]:  # Show first 3
                    print(f"  {attack['attack_type']}: {attack['success_rate']:.1%} success")
        
        # Save results
        output_file = aia_evaluator.save_results(
            results, 
            f"{ORIGINAL_DB.replace('.duckdb', '')}_vs_{SYNTHETIC_DB.replace('.duckdb', '')}", 
            TABLE_NAME
        )
        
        print(f"\nTest completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        logging.error(f"AIA test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Run the focused test
    test_inpatient_aia()