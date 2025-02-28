import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
import argparse
import os
import json
from datetime import datetime
from pathlib import Path

# Import your duckdb_manager
from duckdb_manager import DuckDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('privacy_metrics.log'),
        logging.StreamHandler()
    ]
)

class PrivacyMetricsCalculator:
    """Calculate privacy metrics between original and synthetic datasets."""
    
    def __init__(self, results_dir='results/privacy_metrics'):
        """Initialize the PrivacyMetricsCalculator."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.db_manager = DuckDBManager()
        
    def calculate_metrics(self, db1_path, db2_path, table_name, sample_size=10000):
        """
        Calculate privacy metrics (DCR and NNDR) between original and synthetic datasets.
        
        Args:
            db1_path: Path to the original DuckDB database
            db2_path: Path to the synthetic DuckDB database
            table_name: Name of the table to compare
            sample_size: Maximum number of records to use for the comparison
        
        Returns:
            Dictionary containing the privacy metrics
        """
        logging.info(f"Calculating privacy metrics for table: {table_name}")
        logging.info(f"Original DB: {db1_path}, Synthetic DB: {db2_path}")
        
        # Get data from DuckDB databases using your manager
        try:
            original_data = self.db_manager.load_table_data(db1_path, table_name)
            synthetic_data = self.db_manager.load_table_data(db2_path, table_name)
            
            logging.info(f"Original data shape: {original_data.shape}")
            logging.info(f"Synthetic data shape: {synthetic_data.shape}")
            
            if original_data.empty or synthetic_data.empty:
                raise ValueError("One or both datasets are empty")
        except Exception as e:
            logging.error(f"Error retrieving data: {str(e)}")
            return {"error": str(e)}
        
        # Ensure we have the same columns in both datasets
        common_cols = list(set(original_data.columns).intersection(set(synthetic_data.columns)))
        if len(common_cols) == 0:
            logging.error("No common columns found between datasets")
            return {"error": "No common columns found between datasets"}
        
        original_data = original_data[common_cols]
        synthetic_data = synthetic_data[common_cols]
        
        # Sample the data to make computation feasible
        no_of_records = min(original_data.shape[0], synthetic_data.shape[0], sample_size)
        logging.info(f"Using {no_of_records} records for privacy metric calculations")
        
        # Sample from original and synthetic data
        original_sample = original_data.sample(n=no_of_records, random_state=42).reset_index(drop=True)
        synthetic_sample = synthetic_data.sample(n=no_of_records, random_state=42).reset_index(drop=True)
        
        # Identify column types
        string_cols = original_sample.select_dtypes(exclude=np.number).columns
        numeric_cols = original_sample.select_dtypes(include=np.number).columns
        
        logging.info(f"Numeric columns: {len(numeric_cols)}")
        logging.info(f"String columns: {len(string_cols)}")
        
        # Handle case where no string columns are present
        transformers = []
        if len(numeric_cols) > 0:
            transformers.append((SimpleImputer(missing_values=np.nan, strategy="mean"), numeric_cols))
        if len(string_cols) > 0:
            transformers.append((OneHotEncoder(handle_unknown='ignore'), string_cols))
            
        if not transformers:
            logging.error("No valid columns for transformation")
            return {"error": "No valid columns for transformation"}
            
        # Create transformer for feature encoding
        transformer = make_column_transformer(
            *transformers,
            remainder="passthrough",
        )
        
        # Fit and transform the data
        combined_df = pd.concat([original_sample, synthetic_sample], axis=0)
        logging.info(f"Fitting transformer on combined data of shape {combined_df.shape}")
        
        try:
            transformer.fit(combined_df)
            
            original_encoded = transformer.transform(original_sample)
            synthetic_encoded = transformer.transform(synthetic_sample)
            
            # Check for sparse matrices and convert to dense if needed
            if hasattr(original_encoded, 'toarray'):
                original_encoded = original_encoded.toarray()
            if hasattr(synthetic_encoded, 'toarray'):
                synthetic_encoded = synthetic_encoded.toarray()
                
            logging.info(f"Transformed data shapes - Original: {original_encoded.shape}, Synthetic: {synthetic_encoded.shape}")
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            return {"error": f"Transformation error: {str(e)}"}
        
        # Calculate distances between original and synthetic data
        logging.info("Calculating nearest neighbors...")
        try:
            # Calculate DCR from synthetic to original
            nn_synthetic_to_original = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="l2", n_jobs=-1)
            nn_synthetic_to_original.fit(original_encoded)
            distances_syn_to_orig, _ = nn_synthetic_to_original.kneighbors(synthetic_encoded)
            
            # Calculate DCR from original to synthetic
            nn_original_to_synthetic = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="l2", n_jobs=-1)
            nn_original_to_synthetic.fit(synthetic_encoded)
            distances_orig_to_syn, _ = nn_original_to_synthetic.kneighbors(original_encoded)
            
            # Calculate within-dataset distances for original data
            nn_original_within = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="l2", n_jobs=-1)
            nn_original_within.fit(original_encoded)
            distances_orig_within, _ = nn_original_within.kneighbors(original_encoded)
            
            # Calculate within-dataset distances for synthetic data
            nn_synthetic_within = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="l2", n_jobs=-1)
            nn_synthetic_within.fit(synthetic_encoded)
            distances_syn_within, _ = nn_synthetic_within.kneighbors(synthetic_encoded)
            
        except Exception as e:
            logging.error(f"Error during nearest neighbor calculation: {str(e)}")
            return {"error": f"Nearest neighbor calculation error: {str(e)}"}
        
        # Calculate DCR (Distance to Closest Record)
        # For DCR, we use the first column which is the distance to the nearest neighbor
        dcr_syn_to_orig = distances_syn_to_orig[:, 0]
        dcr_orig_to_syn = distances_orig_to_syn[:, 0]
        dcr_orig_within = distances_orig_within[:, 1]  # Use second column as first is distance to self (zero)
        dcr_syn_within = distances_syn_within[:, 1]    # Use second column as first is distance to self (zero)
        
        # Calculate NNDR (Nearest Neighbor Distance Ratio)
        # For NNDR, we use the ratio of distances to the first and second nearest neighbors
        nndr_syn_to_orig = distances_syn_to_orig[:, 0] / np.maximum(distances_syn_to_orig[:, 1], 1e-8)
        nndr_orig_to_syn = distances_orig_to_syn[:, 0] / np.maximum(distances_orig_to_syn[:, 1], 1e-8)
        
        # Calculate within-dataset NNDR - use 2nd and 3rd neighbors since 1st is self
        if distances_orig_within.shape[1] > 2:
            nndr_orig_within = distances_orig_within[:, 1] / np.maximum(distances_orig_within[:, 2], 1e-8)
            nndr_syn_within = distances_syn_within[:, 1] / np.maximum(distances_syn_within[:, 2], 1e-8)
        else:
            # If we only have 2 neighbors, use a placeholder value
            nndr_orig_within = np.ones(len(distances_orig_within)) * 0.5
            nndr_syn_within = np.ones(len(distances_syn_within)) * 0.5
        
        # Get the 5th percentile metrics
        dcr_syn_to_orig_p5 = np.percentile(dcr_syn_to_orig, 5)
        dcr_orig_to_syn_p5 = np.percentile(dcr_orig_to_syn, 5)
        dcr_orig_within_p5 = np.percentile(dcr_orig_within, 5)
        dcr_syn_within_p5 = np.percentile(dcr_syn_within, 5)
        
        nndr_syn_to_orig_p5 = np.percentile(nndr_syn_to_orig, 5)
        nndr_orig_to_syn_p5 = np.percentile(nndr_orig_to_syn, 5)
        nndr_orig_within_p5 = np.percentile(nndr_orig_within, 5)
        nndr_syn_within_p5 = np.percentile(nndr_syn_within, 5)
        
        # Log detailed results
        logging.info(f"DCR 5th percentile (synthetic to original): {dcr_syn_to_orig_p5:.4f}")
        logging.info(f"DCR 5th percentile (original to synthetic): {dcr_orig_to_syn_p5:.4f}")
        logging.info(f"DCR 5th percentile (within original): {dcr_orig_within_p5:.4f}")
        logging.info(f"DCR 5th percentile (within synthetic): {dcr_syn_within_p5:.4f}")
        
        logging.info(f"NNDR 5th percentile (synthetic to original): {nndr_syn_to_orig_p5:.4f}")
        logging.info(f"NNDR 5th percentile (original to synthetic): {nndr_orig_to_syn_p5:.4f}")
        logging.info(f"NNDR 5th percentile (within original): {nndr_orig_within_p5:.4f}")
        logging.info(f"NNDR 5th percentile (within synthetic): {nndr_syn_within_p5:.4f}")
        
        # Prepare results dictionary
        results = {
            "dcr": {
                "synthetic_to_original_p5": float(dcr_syn_to_orig_p5),
                "original_to_synthetic_p5": float(dcr_orig_to_syn_p5),
                "within_original_p5": float(dcr_orig_within_p5),
                "within_synthetic_p5": float(dcr_syn_within_p5)
            },
            "nndr": {
                "synthetic_to_original_p5": float(nndr_syn_to_orig_p5),
                "original_to_synthetic_p5": float(nndr_orig_to_syn_p5),
                "within_original_p5": float(nndr_orig_within_p5),
                "within_synthetic_p5": float(nndr_syn_within_p5)
            }
        }
        
        # Interpret privacy risk
        if dcr_syn_to_orig_p5 < dcr_orig_within_p5:
            logging.warning("Privacy risk: DCR of synthetic to original is lower than within-original distance")
        if nndr_syn_to_orig_p5 < nndr_orig_within_p5:
            logging.warning("Privacy risk: NNDR of synthetic to original is lower than within-original NNDR")
        
        return results
    
    def save_results(self, results, db1_name, db2_name, table_name):
        """
        Save privacy metrics results to a JSON file.
        
        Args:
            results: Dictionary containing the privacy metrics
            db1_name: Name of the first database
            db2_name: Name of the second database
            table_name: Name of the table compared
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db1_stem = Path(db1_name).stem if db1_name.endswith('.duckdb') else db1_name
        db2_stem = Path(db2_name).stem if db2_name.endswith('.duckdb') else db2_name
        
        filename = f"privacy_metrics_{db1_stem}_{db2_stem}_{table_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to: {filepath}")
        return filepath
    
    def check_type_inconsistencies(self, db_path: str, table_name: str) -> Dict[str, Dict]:
        """
        Check each column in the table for type inconsistencies.
        
        Args:
            db_path: Path to the DuckDB database
            table_name: Name of the table to check
            
        Returns:
            Dictionary with column names as keys and type inconsistency information as values
        """
        logging.info(f"Checking type inconsistencies in '{table_name}'")
        
        try:
            # Get column information from the database
            columns_info = self.db_manager.get_table_column_info(db_path, table_name)
            
            if not columns_info:
                logging.error(f"Failed to get column information for table '{table_name}'")
                return {}
            
            inconsistencies = {}
            
            for col_info in columns_info:
                col_name = col_info['name']
                declared_type = col_info['type']
                
                # Skip checking pid column
                if col_name.lower() == 'pid':
                    continue
                    
                logging.info(f"Checking column '{col_name}' with declared type '{declared_type}'")
                
                # Check for NULL values
                null_count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL"
                null_count_result = self.db_manager.execute_query(db_path, null_count_query)
                null_count = null_count_result[0][0] if null_count_result else 0
                
                # Determine type-specific checks based on declared column type
                if 'INT' in declared_type.upper() or 'BIGINT' in declared_type.upper():
                    # Check for non-integer values in integer columns (this should trigger an error in DuckDB)
                    try:
                        # Check if we can cast all non-NULL values to INTEGER without errors
                        non_integer_check = f"""
                            SELECT COUNT(*) FROM {table_name} 
                            WHERE {col_name} IS NOT NULL AND 
                            (TRY_CAST({col_name} AS INTEGER) IS NULL)
                        """
                        non_integer_result = self.db_manager.execute_query(db_path, non_integer_check)
                        non_integer_count = non_integer_result[0][0] if non_integer_result else 0
                        
                        # Get some examples of problematic values
                        examples_query = f"""
                            SELECT DISTINCT {col_name} FROM {table_name}
                            WHERE {col_name} IS NOT NULL AND 
                            (TRY_CAST({col_name} AS INTEGER) IS NULL)
                            LIMIT 5
                        """
                        examples = self.db_manager.execute_query(db_path, examples_query)
                        example_values = [ex[0] for ex in examples] if examples else []
                        
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'non_matching_type_count': non_integer_count,
                            'example_values': example_values,
                            'has_inconsistencies': non_integer_count > 0
                        }
                    except Exception as e:
                        logging.warning(f"Error checking integer type consistency for {col_name}: {str(e)}")
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'error': str(e),
                            'has_inconsistencies': False  # Can't determine
                        }
                    
                elif 'FLOAT' in declared_type.upper() or 'DOUBLE' in declared_type.upper():
                    # Check for non-numeric values in float/double columns
                    try:
                        non_numeric_check = f"""
                            SELECT COUNT(*) FROM {table_name} 
                            WHERE {col_name} IS NOT NULL AND 
                            (TRY_CAST({col_name} AS DOUBLE) IS NULL)
                        """
                        non_numeric_result = self.db_manager.execute_query(db_path, non_numeric_check)
                        non_numeric_count = non_numeric_result[0][0] if non_numeric_result else 0
                        
                        # Get examples
                        examples_query = f"""
                            SELECT DISTINCT {col_name} FROM {table_name}
                            WHERE {col_name} IS NOT NULL AND 
                            (TRY_CAST({col_name} AS DOUBLE) IS NULL)
                            LIMIT 5
                        """
                        examples = self.db_manager.execute_query(db_path, examples_query)
                        example_values = [ex[0] for ex in examples] if examples else []
                        
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'non_matching_type_count': non_numeric_count,
                            'example_values': example_values,
                            'has_inconsistencies': non_numeric_count > 0
                        }
                    except Exception as e:
                        logging.warning(f"Error checking float/double type consistency for {col_name}: {str(e)}")
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'error': str(e),
                            'has_inconsistencies': False  # Can't determine
                        }
                        
                elif 'TIMESTAMP' in declared_type.upper() or 'DATE' in declared_type.upper():
                    # Check for non-date values in date/timestamp columns
                    try:
                        non_date_check = f"""
                            SELECT COUNT(*) FROM {table_name} 
                            WHERE {col_name} IS NOT NULL AND 
                            (TRY_CAST({col_name} AS TIMESTAMP) IS NULL)
                        """
                        non_date_result = self.db_manager.execute_query(db_path, non_date_check)
                        non_date_count = non_date_result[0][0] if non_date_result else 0
                        
                        # Get examples
                        examples_query = f"""
                            SELECT DISTINCT {col_name} FROM {table_name}
                            WHERE {col_name} IS NOT NULL AND 
                            (TRY_CAST({col_name} AS TIMESTAMP) IS NULL)
                            LIMIT 5
                        """
                        examples = self.db_manager.execute_query(db_path, examples_query)
                        example_values = [ex[0] for ex in examples] if examples else []
                        
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'non_matching_type_count': non_date_count,
                            'example_values': example_values,
                            'has_inconsistencies': non_date_count > 0
                        }
                    except Exception as e:
                        logging.warning(f"Error checking date/timestamp type consistency for {col_name}: {str(e)}")
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'error': str(e),
                            'has_inconsistencies': False  # Can't determine
                        }
                        
                elif 'VARCHAR' in declared_type.upper() or 'TEXT' in declared_type.upper() or 'CHAR' in declared_type.upper():
                    # For text types, check for unexpected numbers or dates masquerading as strings
                    # Check what percentage can be cast to numeric types
                    try:
                        # Count total non-null values
                        total_non_null_query = f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NOT NULL"
                        total_non_null_result = self.db_manager.execute_query(db_path, total_non_null_query)
                        total_non_null = total_non_null_result[0][0] if total_non_null_result else 0
                        
                        # Count values that can be cast to INTEGER
                        cast_int_query = f"""
                            SELECT COUNT(*) FROM {table_name} 
                            WHERE {col_name} IS NOT NULL AND 
                            TRY_CAST({col_name} AS INTEGER) IS NOT NULL
                        """
                        cast_int_result = self.db_manager.execute_query(db_path, cast_int_query)
                        int_castable_count = cast_int_result[0][0] if cast_int_result else 0
                        
                        # Count values that can be cast to TIMESTAMP
                        cast_date_query = f"""
                            SELECT COUNT(*) FROM {table_name} 
                            WHERE {col_name} IS NOT NULL AND 
                            TRY_CAST({col_name} AS TIMESTAMP) IS NOT NULL
                        """
                        cast_date_result = self.db_manager.execute_query(db_path, cast_date_query)
                        date_castable_count = cast_date_result[0][0] if cast_date_result else 0
                        
                        # Calculate percentages
                        int_percentage = (int_castable_count / total_non_null * 100) if total_non_null > 0 else 0
                        date_percentage = (date_castable_count / total_non_null * 100) if total_non_null > 0 else 0
                        
                        # If more than 90% of values can be cast to a specific type, it might indicate
                        # that the column should actually be of that type
                        has_inconsistencies = int_percentage > 90 or date_percentage > 90
                        
                        # Get examples of the most common values
                        examples_query = f"""
                            SELECT {col_name}, COUNT(*) as count
                            FROM {table_name}
                            WHERE {col_name} IS NOT NULL
                            GROUP BY {col_name}
                            ORDER BY count DESC
                            LIMIT 5
                        """
                        examples = self.db_manager.execute_query(db_path, examples_query)
                        example_values = [ex[0] for ex in examples] if examples else []
                        
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'int_castable_percentage': round(int_percentage, 2),
                            'date_castable_percentage': round(date_percentage, 2),
                            'example_values': example_values,
                            'has_inconsistencies': has_inconsistencies,
                            'suggested_type': 'INTEGER' if int_percentage > 90 else 'TIMESTAMP' if date_percentage > 90 else declared_type
                        }
                    except Exception as e:
                        logging.warning(f"Error checking string type consistency for {col_name}: {str(e)}")
                        inconsistencies[col_name] = {
                            'declared_type': declared_type,
                            'null_count': null_count,
                            'error': str(e),
                            'has_inconsistencies': False  # Can't determine
                        }
                else:
                    # For other types, just report NULL counts
                    inconsistencies[col_name] = {
                        'declared_type': declared_type,
                        'null_count': null_count,
                        'has_inconsistencies': False,  # No way to check other types easily
                        'notes': f"Type checking not implemented for {declared_type}"
                    }
            
            # Count columns with inconsistencies
            problem_columns = sum(1 for col_data in inconsistencies.values() if col_data.get('has_inconsistencies', False))
            logging.info(f"Found {problem_columns} columns with potential type inconsistencies")
            
            return inconsistencies
        
        except Exception as e:
            logging.error(f"Error checking type inconsistencies in table {table_name}: {str(e)}")
            return {}

    def generate_type_inconsistency_report(self, db_path: str, table_name: str, output_path: Optional[str] = None) -> str:
        """
        Generate a detailed report of type inconsistencies in the specified table.
        
        Args:
            db_path: Path to the DuckDB database
            table_name: Name of the table to check
            output_path: Optional path to save the report (defaults to results_dir/table_name_type_report.txt)
            
        Returns:
            Path to the saved report
        """
        if output_path is None:
            output_path = Path(self.results_dir) / f"{table_name}_type_report.txt"
        
        inconsistencies = self.check_type_inconsistencies(db_path, table_name)
        
        with open(output_path, 'w') as f:
            f.write(f"Type Inconsistency Report for Table: {table_name}\n")
            f.write(f"Database: {db_path}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Count and write summary
            problem_columns = sum(1 for col_data in inconsistencies.values() if col_data.get('has_inconsistencies', False))
            total_columns = len(inconsistencies)
            
            f.write(f"Summary: {problem_columns} out of {total_columns} columns have type inconsistencies\n\n")
            f.write("Columns with Inconsistencies:\n")
            f.write("-"*80 + "\n")
            
            # Write detailed information for each column with inconsistencies
            for col_name, data in inconsistencies.items():
                if data.get('has_inconsistencies', False):
                    f.write(f"Column: {col_name}\n")
                    f.write(f"  Declared Type: {data['declared_type']}\n")
                    f.write(f"  NULL Count: {data['null_count']}\n")
                    
                    if 'non_matching_type_count' in data:
                        f.write(f"  Non-matching Type Count: {data['non_matching_type_count']}\n")
                    
                    if 'int_castable_percentage' in data:
                        f.write(f"  Integer Castable: {data['int_castable_percentage']}%\n")
                    
                    if 'date_castable_percentage' in data:
                        f.write(f"  Date/Time Castable: {data['date_castable_percentage']}%\n")
                    
                    if 'suggested_type' in data and data['suggested_type'] != data['declared_type']:
                        f.write(f"  Suggested Type: {data['suggested_type']}\n")
                    
                    if 'example_values' in data and data['example_values']:
                        f.write("  Example Values:\n")
                        for val in data['example_values']:
                            f.write(f"    - {val}\n")
                    
                    f.write("\n")
            
            # Write information about columns without inconsistencies
            f.write("\nColumns without Detected Inconsistencies:\n")
            f.write("-"*80 + "\n")
            for col_name, data in inconsistencies.items():
                if not data.get('has_inconsistencies', False):
                    f.write(f"Column: {col_name}\n")
                    f.write(f"  Declared Type: {data['declared_type']}\n")
                    f.write(f"  NULL Count: {data['null_count']}\n")
                    if 'error' in data:
                        f.write(f"  Error during check: {data['error']}\n")
                    if 'notes' in data:
                        f.write(f"  Notes: {data['notes']}\n")
                    f.write("\n")
        
        logging.info(f"Type inconsistency report saved to {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Calculate privacy metrics between original and synthetic datasets')
    parser.add_argument('--original', required=True, help='Path to original DuckDB database')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic DuckDB database')
    parser.add_argument('--table', required=True, help='Table name to compare')
    parser.add_argument('--sample_size', type=int, default=10000, help='Maximum number of records to use')
    parser.add_argument('--results_dir', default='results/privacy_metrics', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Ensure database files exist
    if not os.path.exists(args.original):
        logging.error(f"Original database file not found: {args.original}")
        return
    if not os.path.exists(args.synthetic):
        logging.error(f"Synthetic database file not found: {args.synthetic}")
        return
    
    # Create calculator and calculate metrics
    calculator = PrivacyMetricsCalculator(results_dir=args.results_dir)
    
    results = calculator.calculate_metrics(
        args.original, 
        args.synthetic, 
        args.table,
        args.sample_size
    )
    
    if "error" in results:
        logging.error(f"Error calculating metrics: {results['error']}")
        return
    
    # Save results
    db1_name = os.path.basename(args.original)
    db2_name = os.path.basename(args.synthetic)
    output_file = calculator.save_results(results, db1_name, db2_name, args.table)
    
    # Print results summary
    print("\nPrivacy Metrics Results:")
    print("-" * 70)
    print("Distance to Closest Record (DCR) - 5th percentile:")
    print(f"  Synthetic → Original: {results['dcr']['synthetic_to_original_p5']:.4f}")
    print(f"  Original → Synthetic: {results['dcr']['original_to_synthetic_p5']:.4f}")
    print(f"  Within Original:      {results['dcr']['within_original_p5']:.4f}")
    print(f"  Within Synthetic:     {results['dcr']['within_synthetic_p5']:.4f}")
    print("-" * 70)
    print("Nearest Neighbor Distance Ratio (NNDR) - 5th percentile:")
    print(f"  Synthetic → Original: {results['nndr']['synthetic_to_original_p5']:.4f}")
    print(f"  Original → Synthetic: {results['nndr']['original_to_synthetic_p5']:.4f}")
    print(f"  Within Original:      {results['nndr']['within_original_p5']:.4f}")
    print(f"  Within Synthetic:     {results['nndr']['within_synthetic_p5']:.4f}")
    print("-" * 70)
    print(f"Results saved to: {output_file}")
    
    # Privacy assessment
    print("\nPrivacy Assessment:")
    print("-" * 70)
    if results['dcr']['synthetic_to_original_p5'] < results['dcr']['within_original_p5']:
        print("⚠️  PRIVACY RISK: DCR of synthetic to original is lower than within-original distance")
        print("    This suggests synthetic records may be too similar to specific original records")
    else:
        print("✅ DCR check passed: Synthetic records maintain safe distance from original records")
        
    if results['nndr']['synthetic_to_original_p5'] < results['nndr']['within_original_p5']:
        print("⚠️  PRIVACY RISK: NNDR of synthetic to original is lower than within-original NNDR")
        print("    This suggests synthetic records may uniquely identify specific original records")
    else:
        print("✅ NNDR check passed: Synthetic records don't uniquely identify specific original records")
    print("-" * 70)

if __name__ == "__main__":
    main()