#!/usr/bin/env python3
"""
Combined Privacy Metrics Calculator for Health Claims Data

This script calculates and combines multiple privacy metrics for health claims data:
1. Traditional metrics: k-anonymity, l-diversity, t-closeness
2. Distance-based metrics: DCR (Distance to Closest Record), NNDR (Nearest Neighbor Distance Ratio)

It works with the comprehensive tables created by the data_pipeline.py script:
- all_inpatient
- all_outpatient
- all_drugs
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import your existing modules
from duckdb_manager import DuckDBManager
from calculate_privacy_metrics import PrivacyMetricsCalculator
from distance_metrics import DistanceMetricsCalculator

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"combined_privacy_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

class CombinedPrivacyMetricsCalculator:
    """Calculate and combine all privacy metrics for health claims data."""
    
    def __init__(self, results_dir='results/combined_privacy_metrics'):
        """Initialize the calculator."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.db_manager = DuckDBManager()
        self.privacy_calculator = PrivacyMetricsCalculator(
            results_dir=os.path.join(results_dir, 'traditional_metrics')
        )
        self.distance_calculator = DistanceMetricsCalculator(
            results_dir=os.path.join(results_dir, 'distance_metrics')
        )
    
    def calculate_all_metrics(self, db1_path, db2_path, table_name, sample_size=10000):
        """
        Calculate all privacy metrics for a table.
        
        Args:
            db1_path: Path to the original database
            db2_path: Path to the synthetic database
            table_name: Name of the table to analyze
            sample_size: Maximum number of records to use for distance metrics
            
        Returns:
            Dictionary containing all privacy metrics
        """
        logging.info(f"Calculating all privacy metrics for table: {table_name}")
        
        # Define quasi-identifiers and sensitive attributes based on table type
        quasi_identifiers, sensitive_attributes = self._get_columns_for_table(table_name)
        
        # Step 1: Calculate k-anonymity, l-diversity, and t-closeness
        logging.info("Step 1: Calculating k-anonymity, l-diversity, and t-closeness")
        traditional_metrics = self._calculate_traditional_metrics(
            db1_path, db2_path, table_name, quasi_identifiers, sensitive_attributes
        )
        
        # Step 2: Calculate distance-based metrics
        logging.info("Step 2: Calculating distance-based privacy metrics")
        distance_metrics = self._calculate_distance_metrics(
            db1_path, db2_path, table_name, sample_size
        )
        
        # Combine all results
        results = {
            "table_name": table_name,
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": sensitive_attributes,
            "traditional_metrics": traditional_metrics,
            "distance_metrics": distance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save combined results
        self._save_results(results, db1_path, db2_path, table_name)
        
        return results
    
    def _get_columns_for_table(self, table_name):
        """
        Get appropriate quasi-identifiers and sensitive attributes for the table.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Tuple of (quasi_identifiers, sensitive_attributes)
        """
        if "inpatient" in table_name.lower():
            quasi_identifiers = [
                "insurants_year_of_birth",
                "insurants_gender",
                "inpatient_cases_date_of_admission",
                "inpatient_cases_department_admission"
            ]
            sensitive_attributes = [
                "inpatient_diagnosis_diagnosis",
                "inpatient_procedures_procedure_code",
                "outpatient_diagnosis_diagnosis",
                "outpatient_procedures_procedure_code"
            ]
        elif "outpatient" in table_name.lower():
            quasi_identifiers = [
                "insurants_year_of_birth",
                "insurants_gender",
                "outpatient_cases_from",
                "outpatient_cases_practice_code",
                "outpatient_cases_year",
                "outpatient_cases_quarter"
            ]
            sensitive_attributes = [
                "inpatient_diagnosis_diagnosis",
                "inpatient_procedures_procedure_code",
                "outpatient_diagnosis_diagnosis",
                "outpatient_procedures_procedure_code"
            ]
        elif "drugs" in table_name.lower():
            quasi_identifiers = [
                "insurants_year_of_birth",
                "insurants_gender",
                "drugs_date_of_prescription",
                "drugs_date_of_dispense",
                "drugs_specialty_of_prescriber",
            ]
            sensitive_attributes = [
                "drugs_pharma_central_number",
                "drugs_atc",
                "drugs_amount_due"
            ]
        else:
            logging.warning(f"Unknown table type: {table_name}. Using default columns.")
            # Default columns - may need adjustment
            quasi_identifiers = ["insurants_year_of_birth", "insurants_gender"]
            sensitive_attributes = []
        
        return quasi_identifiers, sensitive_attributes
    
    def _calculate_traditional_metrics(self, db1_path, db2_path, table_name, 
                                     quasi_identifiers, sensitive_attributes):
        """
        Calculate k-anonymity, l-diversity, and t-closeness.
        
        Args:
            db1_path: Path to the original database
            db2_path: Path to the synthetic database
            table_name: Name of the table to analyze
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns
            
        Returns:
            Dictionary containing the traditional privacy metrics
        """
        try:
            # Pass the parameters to the privacy calculator from calculate_privacy_metrics.py
            # Set the appropriate QI attributes based on table type
            if "inpatient" in table_name.lower():
                self.privacy_calculator.qi_inpatient = quasi_identifiers
                self.privacy_calculator.qi_outpatient = []
            elif "outpatient" in table_name.lower():
                self.privacy_calculator.qi_inpatient = []
                self.privacy_calculator.qi_outpatient = quasi_identifiers
            else:
                # For drugs table, we'll use inpatient QI for convenience
                self.privacy_calculator.qi_inpatient = quasi_identifiers
                self.privacy_calculator.qi_outpatient = []
            
            self.privacy_calculator.sensitive_attributes = sensitive_attributes
            
            # Compare tables - need to make sure the table exists in both DBs
            comparison = self.privacy_calculator.compare_tables(
                db1_path=db1_path,
                db2_path=db2_path,
                table_name=table_name
            )
            
            # Return the results
            return {
                "dataset1_results": comparison.get("dataset1_results", {}),
                "dataset2_results": comparison.get("dataset2_results", {})
            }
            
        except Exception as e:
            logging.error(f"Error calculating traditional privacy metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_distance_metrics(self, db1_path, db2_path, table_name, sample_size):
        """
        Calculate distance-based privacy metrics (DCR and NNDR).
        
        Args:
            db1_path: Path to the original database
            db2_path: Path to the synthetic database
            table_name: Name of the table to analyze
            sample_size: Maximum number of records to use
            
        Returns:
            Dictionary containing the distance-based privacy metrics
        """
        try:
            return self.distance_calculator.calculate_metrics(
                db1_path=db1_path,
                db2_path=db2_path,
                table_name=table_name,
                sample_size=sample_size
            )
        except Exception as e:
            logging.error(f"Error calculating distance metrics: {str(e)}")
            return {"error": str(e)}
    
    def _save_results(self, results, db1_path, db2_path, table_name):
        """
        Save all privacy metrics results to a JSON file.
        
        Args:
            results: Dictionary containing all privacy metrics
            db1_path: Path to the original database
            db2_path: Path to the synthetic database
            table_name: Name of the table analyzed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db1_stem = Path(db1_path).stem if str(db1_path).endswith('.duckdb') else str(db1_path)
        db2_stem = Path(db2_path).stem if str(db2_path).endswith('.duckdb') else str(db2_path)
        
        filename = f"combined_privacy_{db1_stem}_{db2_stem}_{table_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Combined results saved to: {filepath}")
        return filepath

def main():
    """Main function to run the combined privacy metrics calculation."""
    parser = argparse.ArgumentParser(description='Calculate all privacy metrics for health claims data')
    parser.add_argument('--original', required=True, help='Path to original DuckDB database')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic DuckDB database')
    parser.add_argument('--tables', nargs='+', default=['all_inpatient', 'all_outpatient', 'all_drugs'], 
                      help='Tables to analyze (default: all_inpatient all_outpatient all_drugs)')
    parser.add_argument('--sample_size', type=int, default=10000, 
                      help='Maximum number of records to use for distance metrics')
    parser.add_argument('--results_dir', default='results/combined_privacy_metrics', 
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 80)
    print("Combined Privacy Metrics Calculator for Health Claims Data")
    print("=" * 80)
    print(f"Original database: {args.original}")
    print(f"Synthetic database: {args.synthetic}")
    print(f"Tables to analyze: {', '.join(args.tables)}")
    print(f"Sample size for distance metrics: {args.sample_size}")
    print("-" * 80)
    
    # Create the calculator
    calculator = CombinedPrivacyMetricsCalculator(results_dir=args.results_dir)
    
    # Calculate metrics for each table
    all_results = {}
    for table in args.tables:
        logging.info(f"Processing table: {table}")
        print(f"\n{'='*80}\nProcessing table: {table}\n{'='*80}")
        
        try:
            results = calculator.calculate_all_metrics(
                db1_path=args.original,
                db2_path=args.synthetic,
                table_name=table,
                sample_size=args.sample_size
            )
            
            all_results[table] = results
            
            # Print summary
            print(f"\nSummary of privacy metrics for table '{table}':")
            print("-" * 70)
            
            # K-anonymity
            original_k = results["traditional_metrics"]["dataset1_results"].get("k_anonymity", "N/A")
            synthetic_k = results["traditional_metrics"]["dataset2_results"].get("k_anonymity", "N/A")
            print(f"K-anonymity: Original={original_k}, Synthetic={synthetic_k}")
            
            # Privacy score
            original_privacy = results["traditional_metrics"]["dataset1_results"].get("privacy_score", "N/A")
            synthetic_privacy = results["traditional_metrics"]["dataset2_results"].get("privacy_score", "N/A")
            if original_privacy != "N/A" and synthetic_privacy != "N/A":
                print(f"Privacy score: Original={original_privacy:.2f}, Synthetic={synthetic_privacy:.2f}")
            
            # Distance metrics
            if "dcr" in results["distance_metrics"]:
                dcr_syn_to_orig = results["distance_metrics"]["dcr"].get("synthetic_to_original_p5", "N/A")
                dcr_within_orig = results["distance_metrics"]["dcr"].get("within_original_p5", "N/A")
                if dcr_syn_to_orig != "N/A" and dcr_within_orig != "N/A":
                    print(f"DCR (5th percentile): Synthetic→Original={dcr_syn_to_orig:.6f}, Within Original={dcr_within_orig:.6f}")
                    
                    if isinstance(dcr_syn_to_orig, (int, float)) and isinstance(dcr_within_orig, (int, float)):
                        if dcr_syn_to_orig < dcr_within_orig:
                            print("PRIVACY RISK: Synthetic records may be too close to specific original records")
                        else:
                            print("DCR check passed: Synthetic records maintain safe distance from original records")
            
            print("-" * 70)
            
        except Exception as e:
            logging.error(f"Error processing table {table}: {str(e)}")
            print(f"Error processing table {table}: {str(e)}")
    
    print("\nAll calculations completed")
    print(f"Results saved to: {args.results_dir}")

if __name__ == "__main__":
    main()