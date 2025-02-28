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
        
    def detect_mixed_type_columns(self, df):
        """
        Detect columns with mixed data types and categorize columns as numeric, timestamp, or string.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (mixed_columns, numeric_cols, timestamp_cols, string_cols)
        """
        mixed_columns = []
        numeric_cols = []
        timestamp_cols = []
        string_cols = []
        
        for col in df.columns:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            # Get non-null values and their types
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
                
            # Check types
            types = set(type(x).__name__ for x in non_null_values)
            
            if len(types) > 1:
                mixed_columns.append(col)
                logging.warning(f"Column '{col}' has mixed types: {types}")
                
                # Check if it's primarily numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_cols.append(col)
                    else:
                        # Try to convert to datetime
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            if not df[col].isna().all():
                                timestamp_cols.append(col)
                            else:
                                # Default to string
                                df[col] = df[col].astype(str)
                                string_cols.append(col)
                        except:
                            df[col] = df[col].astype(str)
                            string_cols.append(col)
                except:
                    # Not numeric, try datetime
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if not df[col].isna().all():
                            timestamp_cols.append(col)
                        else:
                            df[col] = df[col].astype(str)
                            string_cols.append(col)
                    except:
                        df[col] = df[col].astype(str)
                        string_cols.append(col)
            else:
                # Single type columns
                type_name = list(types)[0]
                if type_name in ('int', 'float', 'int64', 'float64', 'int32', 'float32'):
                    numeric_cols.append(col)
                elif type_name in ('Timestamp', 'datetime64[ns]', 'datetime'):
                    timestamp_cols.append(col)
                else:
                    # Treat as string
                    df[col] = df[col].astype(str)
                    string_cols.append(col)
        
        return mixed_columns, numeric_cols, timestamp_cols, string_cols
    
    def standardize_column_types(self, df1, df2):
        """
        Ensure both dataframes have the same column types.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Tuple of (df1, df2, numeric_cols, string_cols) with standardized types
        """
        # Get common columns
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        df1 = df1[common_cols].copy()
        df2 = df2[common_cols].copy()
        
        logging.info(f"Standardizing types for {len(common_cols)} common columns")
        
        # Detect mixed columns in each DataFrame
        mixed1, numeric1, timestamp1, string1 = self.detect_mixed_type_columns(df1)
        mixed2, numeric2, timestamp2, string2 = self.detect_mixed_type_columns(df2)
        
        # Get all mixed columns from both DataFrames
        all_mixed = set(mixed1 + mixed2)
        logging.info(f"Found {len(all_mixed)} columns with mixed types")
        if all_mixed:
            logging.info(f"Mixed type columns: {all_mixed}")
        
        # Standardize numeric columns (intersection of numeric columns in both DataFrames)
        numeric_cols = list(set(numeric1).intersection(set(numeric2)))
        for col in numeric_cols:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
        
        # For timestamp columns, convert to string to avoid encoding issues
        timestamp_cols = list(set(timestamp1).union(set(timestamp2)))
        for col in timestamp_cols:
            if col in df1.columns:
                try:
                    df1[col] = pd.to_datetime(df1[col], errors='coerce')
                    df1[col] = df1[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    df1[col] = df1[col].astype(str)
            
            if col in df2.columns:
                try:
                    df2[col] = pd.to_datetime(df2[col], errors='coerce')
                    df2[col] = df2[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    df2[col] = df2[col].astype(str)
            
            # Add these standardized timestamp columns to string columns
            if col not in string1:
                string1.append(col)
            if col not in string2:
                string2.append(col)
        
        # All non-numeric columns are treated as strings
        string_cols = list(set(df1.columns).difference(set(numeric_cols)))
        for col in string_cols:
            df1[col] = df1[col].astype(str)
            df2[col] = df2[col].astype(str)
        
        return df1, df2, numeric_cols, string_cols
        
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
            
            if original_data.empty or synthetic_data.empty:
                raise ValueError("One or both datasets are empty")
                
            logging.info(f"Original data shape: {original_data.shape}")
            logging.info(f"Synthetic data shape: {synthetic_data.shape}")
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
        
        # Standardize column types
        original_sample, synthetic_sample, numeric_cols, string_cols = self.standardize_column_types(
            original_sample, synthetic_sample
        )
        
        logging.info(f"After standardization - Numeric columns: {len(numeric_cols)}")
        logging.info(f"After standardization - String columns: {len(string_cols)}")
        
        # Create transformer for feature encoding
        transformers = []
        if len(numeric_cols) > 0:
            transformers.append((SimpleImputer(missing_values=np.nan, strategy="mean"), numeric_cols))
        if len(string_cols) > 0:
            transformers.append((OneHotEncoder(handle_unknown='ignore'), string_cols))
            
        if not transformers:
            logging.error("No valid columns for transformation")
            return {"error": "No valid columns for transformation"}
            
        transformer = make_column_transformer(
            *transformers,
            remainder="drop",
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

    # [keep other methods unchanged]

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