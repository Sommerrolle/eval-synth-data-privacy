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
from pca_analysis import analyze_principal_components

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/distance_metrics.log'),
        logging.StreamHandler()
    ]
)

class DistanceMetricsCalculator:
    """Calculate privacy metrics between original and synthetic datasets."""
    
    def __init__(self, results_dir='results/privacy_metrics'):
        """Initialize the PrivacyMetricsCalculator."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.db_manager = DuckDBManager()
    
    
    def preprocess_dataframes(self, df1, df2):
        """
        Preprocess two dataframes to standardize column types and handle special columns.
        
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
        
        logging.info(f"Preprocessing {len(common_cols)} common columns")
        
        # 1. First, identify special column types for preprocessing
        diagnosis_cols = [col for col in common_cols if "diagnosis_diagnosis" in col.lower()]
        procedure_cols = [col for col in common_cols if "procedure_code" in col.lower()]
        
        # Identify potential timestamp columns (we'll convert these later)
        timestamp_cols = []
        for col in common_cols:
            col_lower = col.lower()
            if any(term in col_lower for term in ["date", "time", "from", "to"]):
                timestamp_cols.append(col)
        
        # 2. Convert medical codes (both diagnosis and procedure) to numerical values
        all_medical_code_cols = []
        if diagnosis_cols:
            logging.info(f"Converting {len(diagnosis_cols)} diagnosis columns to numerical values")
            all_medical_code_cols.extend(diagnosis_cols)

        if procedure_cols:
            logging.info(f"Converting {len(procedure_cols)} procedure columns to numerical values")
            all_medical_code_cols.extend(procedure_cols)

        if all_medical_code_cols:
            logging.info(f"Using simplified encoding for {len(all_medical_code_cols)} medical code columns")
            df1 = self.encode_medical_codes(df1, all_medical_code_cols)
            df2 = self.encode_medical_codes(df2, all_medical_code_cols)
            
            # Update common_cols after medical code conversion (original cols removed, new numeric cols added)
            common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        
        # # 3. Verify and convert timestamp columns
        # timestamp_cols = []
        # for col in potential_timestamp_cols:
        #     if col in common_cols:  # Check if column still exists after medical code processing
        #         try:
        #             # Test if column can be converted to datetime
        #             pd.to_datetime(df1[col].head(100), errors='raise')
        #             pd.to_datetime(df2[col].head(100), errors='raise')
        #             timestamp_cols.append(col)
        #         except:
        #             pass  # Not a timestamp column
        
        # Initialize numeric_cols list
        numeric_cols = []
        
        # 4. Convert timestamp columns to Unix timestamps
        if timestamp_cols:
            logging.info(f"Converting {len(timestamp_cols)} timestamp columns to Unix timestamps")
            df1, numeric_cols = self.convert_timestamps_to_epoch(df1, timestamp_cols, numeric_cols)
            df2, numeric_cols = self.convert_timestamps_to_epoch(df2, timestamp_cols, numeric_cols)
        
        # 5. Now categorize remaining columns as numeric or string
        remaining_cols = [col for col in common_cols if col not in timestamp_cols]
        remaining_numeric, string_cols = self.categorize_columns(df1, df2, remaining_cols)

        # Remove medical code cols from string cols
        string_cols = [col for col in string_cols if col not in all_medical_code_cols]
        
        # Add remaining numeric columns to our numeric_cols list
        numeric_cols.extend(remaining_numeric)
        
        # 6. Standardize the data types
        for col in numeric_cols:
            if col in df1.columns and col in df2.columns:  # Verify column exists
                df1[col] = pd.to_numeric(df1[col], errors='coerce')
                df2[col] = pd.to_numeric(df2[col], errors='coerce')
        
        for col in string_cols:
            if col in df1.columns and col in df2.columns:  # Verify column exists
                df1[col] = df1[col].astype(str)
                df2[col] = df2[col].astype(str)
        
        return df1, df2, numeric_cols, string_cols
    
    def convert_timestamps_to_epoch(self, df, timestamp_cols, numeric_cols):
        """
        Convert timestamp columns to Unix epoch time (seconds since 1970-01-01).
        
        Args:
            df: DataFrame to process
            timestamp_cols: List of column names containing timestamp data
            numeric_cols: List of numeric column names for the dataframe
            
        Returns:
            Tuple containing (df, updated_numeric_cols) with updated values
        """
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Log problematic values
                    if df[col].isna().any():
                        problematic_values = df.loc[df[col].isna(), col].head(5).tolist()
                        logging.warning(f"Column '{col}' has {df[col].isna().sum()} values that couldn't be converted to datetime. Examples: {problematic_values}")
                        if problematic_values:
                            logging.warning(f"  Problem examples: {problematic_values}")
                    # Convert to Unix timestamp (seconds since epoch)
                    df[col] = df[col].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
                except Exception as e:
                    logging.error(f"Failed to convert column '{col}' to timestamp: {str(e)}")
                    # Force numeric conversion, set unconvertible values to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Add to numeric columns
                if col not in numeric_cols:
                    numeric_cols.append(col)
        
        return df, numeric_cols

    def categorize_columns(self, df1, df2, common_cols):
        """
        Categorize columns as numeric or string based on their content.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            common_cols: List of common column names
            
        Returns:
            Tuple of (numeric_cols, string_cols)
        """
        numeric_cols = []
        string_cols = []
        
        for col in common_cols:
            # Try to convert to numeric
            try:
                test1 = pd.to_numeric(df1[col].dropna().head(100), errors='raise')
                test2 = pd.to_numeric(df2[col].dropna().head(100), errors='raise')
                # If both convert successfully, it's a numeric column
                numeric_cols.append(col)
            except:
                # Otherwise, treat as string
                string_cols.append(col)
        
        logging.info(f"Categorized columns: {len(numeric_cols)} numeric, {len(string_cols)} string")
        return numeric_cols, string_cols

    def encode_medical_codes(self, df, code_columns):
        """
        Simplified encoding of medical codes by removing non-numeric characters,
        except for the initial letter in ICD codes.
        
        Args:
            df: DataFrame containing medical codes
            code_columns: List of code columns to encode
            
        Returns:
            DataFrame with encoded medical codes
        """
        df_encoded = df.copy()
        
        for col in code_columns:
            if col not in df.columns:
                continue
                
            # Create a new column for numeric representations
            numeric_col = f"{col}_numeric"
            
            # Convert to string and standardize
            df_encoded[col] = df_encoded[col].astype(str).str.upper().str.strip()
            
            # Initialize numeric column with float dtype from the start
            df_encoded[numeric_col] = -1.0  # This creates a float column
            
            valid_mask = ~df_encoded[col].isin(['UNKNOWN', 'UUU', 'NAN', 'NONE', ''])
            
            if valid_mask.any():
                valid_codes = df_encoded.loc[valid_mask, col]
                numeric_values = []
                
                for code in valid_codes:
                    # Remove dots and dashes
                    clean_code = code.replace('.', '').replace('-', '')
                    
                    if not clean_code:
                        numeric_values.append(-1.0)
                        continue
                    
                    # For ICD-10 codes (starting with letter)
                    if clean_code[0].isalpha():
                        # Get letter chapter (A=1, B=2, etc.)
                        chapter = ord(clean_code[0]) - ord('A') + 1
                        
                        # Extract only digits from the rest of the code
                        digits = ''.join(c for c in clean_code[1:] if c.isdigit())
                        decimal_part = float('0.' + digits) if digits else 0.0
                        
                        # Combine
                        numeric_values.append(float(chapter + decimal_part))
                    
                    # For procedure codes (starting with digit)
                    elif clean_code[0].isdigit():
                        # Extract only digits from the entire code
                        digits = ''.join(c for c in clean_code if c.isdigit())
                        
                        if digits:
                            # First digit is chapter, rest becomes decimal
                            chapter = int(digits[0])
                            decimal_part = float('0.' + digits[1:]) if len(digits) > 1 else 0.0
                            numeric_values.append(float(chapter + decimal_part))
                        else:
                            numeric_values.append(-1.0)
                    
                    else:
                        numeric_values.append(-1.0)
                
                # Explicitly convert to numpy array of float type before assignment
                import numpy as np
                numeric_array = np.array(numeric_values, dtype=float)
                df_encoded.loc[valid_mask, numeric_col] = numeric_array
        
        return df_encoded
    
    def get_sensitive_attributes_columns(all_columns, table_name):
        """
        Get columns containing sensitive attributes based on the table type.
        
        Args:
            all_columns: List of all column names available in the datasets
            table_name: Name of the table (used to determine which sensitive attributes to include)
            
        Returns:
            List of column names that match the sensitive attributes
        """
        # Define sensitive attributes based on table type
        if 'inpatient' in table_name.lower():
            sensitive_attributes = [
                'year_of_birth',
                'gender',
                'diagnosis_diagnosis',
                'procedure_code',
                'regional_code',
                'date_of_admission', 
                'date_of_discharge'
                # 'department_admission',
                # 'department_discharge',
                # 'cause_of_admission'
            ]
        elif 'outpatient' in table_name.lower():
            sensitive_attributes = [
                'year_of_birth',
                'gender',
                'diagnosis_diagnosis',
                'procedure_code',
                'regional_code',
                'practice_code',
                'from',  # outpatient_cases_from
                'to',    # outpatient_cases_to
                'year',
                'quarter'
            ]
        elif 'drugs' in table_name.lower():
            sensitive_attributes = [
                'year_of_birth',
                'gender',
                'regional_code',
                'date_of_prescription',
                'date_of_dispense',
                'pharma_central_number',
                'specialty_of_prescriber',
                'atc'
            ]
        
        # Get columns that contain any of the sensitive attribute substrings
        filtered_cols = []
        for column in all_columns:
            if any(attr in column.lower() for attr in sensitive_attributes):
                filtered_cols.append(column)
        
        if not filtered_cols:
            raise ValueError(f"No matching sensitive attributes found for table: {table_name}")
        
        print(f"Selected {len(filtered_cols)} sensitive columns for {table_name}")
        print(f"Columns: {', '.join(filtered_cols)}")
        
        return filtered_cols


    def calculate_metrics(self, db1_path, db2_path, table_name, sample_size=10000, use_pca=True, pca_variance=0.95):
        """
        Calculate privacy metrics (DCR and NNDR) between original and synthetic datasets.
        
        Args:
            db1_path: Path to the original DuckDB database
            db2_path: Path to the synthetic DuckDB database
            table_name: Name of the table to compare
            sample_size: Maximum number of records to use for the comparison
            use_pca: Whether to use PCA for dimensionality reduction
            pca_variance: Variance threshold for PCA (default: 0.95)
        
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
            raise ValueError("No common columns found between datasets")
        
        # Remove 'pid' and any columns containing 'caseID'
        # filtered_cols = [col for col in common_cols if col != 'pid' and 'caseID' not in col]
        # if len(filtered_cols) == 0:
            # logging.error("No common columns remain after filtering out pid and caseID columns")
            # raise ValueError("No common columns remain after filtering")
        # logging.info(f"Removed {len(common_cols) - len(filtered_cols)} columns containing 'pid' or 'caseID'")
        # logging.info(f"Proceeding with {len(filtered_cols)} common columns")

        all_columns = original_data.columns.tolist()
        filtered_cols = self.get_sensitive_attributes_columns(all_columns, table_name)
        original_data = original_data[filtered_cols]
        synthetic_data = synthetic_data[filtered_cols]
        
        # Sample the data to make computation feasible
        no_of_records = min(original_data.shape[0], synthetic_data.shape[0], sample_size)
        logging.info(f"Using {no_of_records} records for privacy metric calculations")
        
        # Sample from original and synthetic data
        original_sample = original_data.sample(n=no_of_records, random_state=42).reset_index(drop=True)
        synthetic_sample = synthetic_data.sample(n=no_of_records, random_state=42).reset_index(drop=True)
        
        # Determine column types
        original_sample, synthetic_sample, numeric_cols, string_cols = self.preprocess_dataframes(
            original_data, synthetic_sample
        )
        
        logging.info(f"After split - Numeric columns: {len(numeric_cols)}")
        logging.info(f"After split - String columns: {len(string_cols)}")
        
        # if use_pca:
        #     try:               
        #         # Run PCA to understand important features
        #         logging.info("Running PCA analysis to identify important features...")
        #         pca_results = analyze_principal_components(
        #             df=original_sample[numeric_cols],
        #             exclude_cols=[],  # Already filtered columns
        #             variance_threshold=pca_variance,
        #             display_plots=True,  # Set to True for debugging
        #             output_dir='pca_results_for_metrics'
        #         )
                
        #         # Log the most important features
        #         top_features = pca_results['feature_importance'].head(10)
        #         logging.info("Top 10 most important features:")
        #         for idx, row in top_features.iterrows():
        #             logging.info(f"  {idx}: {row['RelativeImportance']:.2f}% importance")
                
        #         # Get the reduced data
        #         original_sample = pca_results['reduced_data'].iloc[:no_of_records]
        #         synthetic_sample = pca_results['reduced_data'].iloc[no_of_records:]
                
        #         # Update column information after PCA
        #         numeric_cols = original_sample.columns.tolist()
        #         string_cols = []  # PCA output is all numeric
                
        #         logging.info(f"After PCA: Reduced from {len(filtered_cols)} to {len(numeric_cols)} dimensions")
        #         logging.info(f"Retained {pca_results['explained_variance'].sum()*100:.2f}% of variance")
                
        #     except Exception as e:
        #         logging.error(f"Error during PCA analysis: {str(e)}")
        #         logging.info("Continuing with original features (PCA disabled)")
        
        # Create transformer for feature encoding
        transformers = []
        if len(numeric_cols) > 0:
            transformers.append((SimpleImputer(missing_values=np.nan, strategy="constant", fill_value= -1.0), numeric_cols))
            
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
    calculator = DistanceMetricsCalculator(results_dir=args.results_dir)
    
    results = calculator.calculate_metrics(
        args.original, 
        args.synthetic, 
        args.table,
        args.sample_size,
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
        print("PRIVACY RISK: DCR of synthetic to original is lower than within-original distance")
        print("    This suggests synthetic records may be too similar to specific original records")
    else:
        print("DCR check passed: Synthetic records maintain safe distance from original records")
        
    if results['nndr']['synthetic_to_original_p5'] < results['nndr']['within_original_p5']:
        print("PRIVACY RISK: NNDR of synthetic to original is lower than within-original NNDR")
        print("    This suggests synthetic records may uniquely identify specific original records")
    else:
        print("NNDR check passed: Synthetic records don't uniquely identify specific original records")
    print("-" * 70)

if __name__ == "__main__":
    main()