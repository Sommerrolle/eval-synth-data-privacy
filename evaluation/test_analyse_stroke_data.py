import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, List
from calculate_privacy_metrics import PrivacyMetricsCalculator, NpEncoder

def analyze_stroke_dataset(filepath: str, selected_metrics: List[str]) -> Dict:
    """
    Analyze the stroke prediction dataset using the PrivacyMetricsCalculator.
    
    Args:
        filepath (str): Path to the stroke dataset CSV file
        selected_metrics (List[str]): List of privacy metrics to calculate
        
    Returns:
        Dict: Results of privacy analysis
    """
    try:
        # Read the dataset
        df = pd.read_csv(filepath)
        
        # Print data info
        print("\nDataset Info:")
        print(df.info())
        
        print("\nColumn Data Types:")
        for column in df.columns:
            print(f"{column}: {df[column].dtype}")
            
        print("\nSample of data:")
        print(df.head())
        
        # Define quasi-identifiers and sensitive attributes
        quasi_identifiers = [
            'gender',
            'age',
            'hypertension',
            'heart_disease',
            'ever_married',
            'work_type',
            'Residence_type',
            'smoking_status'
        ]
        
        sensitive_attributes = ['stroke']
        
        # Initialize calculator with appropriate attributes
        calculator = PrivacyMetricsCalculator(
            results_dir='stroke_results',
            qi_inpatient=[],  # Not needed for stroke data
            qi_outpatient=[], # Not needed for stroke data
            sensitive_attributes=sensitive_attributes
        )
        
        # Override selected metrics
        calculator.selected_metrics = selected_metrics
        
        results = {}
        
        # Calculate k-anonymity if selected
        if 'k-anonymity' in selected_metrics:
            results.update(calculator.calculate_k_anonymity(df, quasi_identifiers))
            
        # Calculate l-diversity if selected
        if 'l-diversity' in selected_metrics:
            results['l_diversity'] = calculator.calculate_l_diversity(
                df, quasi_identifiers, sensitive_attributes
            )
            
        # Calculate t-closeness if selected
        if 't-closeness' in selected_metrics:
            results['t_closeness'] = calculator.calculate_t_closeness(
                df, quasi_identifiers, sensitive_attributes
            )
            
        # Prepare final results
        final_results = {
            "dataset_name": "stroke_prediction",
            "file_path": filepath,
            "total_records": len(df),
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": sensitive_attributes,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        save_results(final_results)
        
        return final_results
        
    except Exception as e:
        print(f"Error analyzing stroke dataset: {str(e)}")
        return {"error": str(e)}

def save_results(results: Dict):
    """
    Save the analysis results to a JSON file.
    
    Args:
        results (Dict): Analysis results to save
    """
    # Create results directory if it doesn't exist
    os.makedirs('stroke_results', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"privacy_metrics_stroke_{timestamp}.json"
    filepath = os.path.join('stroke_results', filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, fp=f, cls=NpEncoder, indent=2)
    print(f"\nResults saved to: {filepath}")

def main():
    # Get the metrics selection from user
    print("\nAvailable privacy metrics:")
    print("1. k-anonymity")
    print("2. l-diversity")
    print("3. t-closeness")
    
    while True:
        print("\nEnter numbers for desired metrics (space-separated):")
        selections = input("> ").strip().split()
        
        # Map selections to metric names
        metrics_map = {
            '1': 'k-anonymity',
            '2': 'l-diversity',
            '3': 't-closeness'
        }
        
        if all(s in metrics_map for s in selections):
            selected_metrics = [metrics_map[s] for s in selections]
            break
        print("Invalid selection. Please try again.")
    
    # Get filepath from user
    while True:
        filepath = input("\nEnter path to stroke dataset CSV file: ").strip()
        if os.path.exists(filepath):
            break
        print("File not found. Please enter a valid path.")
    
    # Run analysis
    print("\nAnalyzing Stroke Prediction Dataset...")
    results = analyze_stroke_dataset(filepath, selected_metrics)
    
    # Print summary of results
    if 'error' not in results:
        print("\nAnalysis Summary:")
        print(f"Total records analyzed: {results['total_records']}")
        print(f"Quasi-identifiers used: {', '.join(results['quasi_identifiers'])}")
        print(f"Sensitive attributes analyzed: {', '.join(results['sensitive_attributes'])}")
        
        if 'k-anonymity' in selected_metrics:
            print(f"\nK-anonymity value: {results['results'].get('k_anonymity')}")
            print(f"Privacy score: {results['results'].get('privacy_score'):.2f}%")
            
        print("\nDetailed results have been saved to the stroke_results directory.")

if __name__ == "__main__":
    main()