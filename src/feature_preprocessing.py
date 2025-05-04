import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_preprocessing.log'),
        logging.StreamHandler()
    ]
)

class FeaturePreprocessor:
    """Process and encode features for privacy metrics calculations and data analysis."""
    
    @staticmethod
    def preprocess_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
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
        atc_col = next((col for col in common_cols if "atc" in col.lower()), None)
        
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
            df1 = FeaturePreprocessor.encode_medical_codes(df1, all_medical_code_cols)
            df2 = FeaturePreprocessor.encode_medical_codes(df2, all_medical_code_cols)
            
            # Update common_cols after medical code conversion (original cols removed, new numeric cols added)
            common_cols = list(set(df1.columns).intersection(set(df2.columns)))

        if atc_col:
            logging.info(f"Using simplified encoding for atc code columns")
            df1 = FeaturePreprocessor.encode_atc_codes(df1, atc_col)
            df2 = FeaturePreprocessor.encode_atc_codes(df2, atc_col)
        
        # Initialize numeric_cols list
        numeric_cols = []
        
        # 4. Convert timestamp columns to Unix timestamps
        if timestamp_cols:
            logging.info(f"Converting {len(timestamp_cols)} timestamp columns to Unix timestamps")
            df1, numeric_cols = FeaturePreprocessor.convert_timestamps_to_epoch(df1, timestamp_cols, numeric_cols)
            df2, numeric_cols = FeaturePreprocessor.convert_timestamps_to_epoch(df2, timestamp_cols, numeric_cols)
        
        # 5. Now categorize remaining columns as numeric or string
        remaining_cols = [col for col in common_cols if col not in timestamp_cols]
        remaining_numeric, string_cols = FeaturePreprocessor.categorize_columns(df1, df2, remaining_cols)

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
    

    @staticmethod
    def preprocess_single_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a single dataframe to convert columns to standardized types.
        Particularly useful for t-closeness calculations and other metrics that work with 
        one dataframe at a time.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Processed DataFrame with standardized types
        """
        processed_df = df.copy()
        
        # Identify special column types
        diagnosis_cols = [col for col in df.columns if "diagnosis_diagnosis" in col.lower()]
        procedure_cols = [col for col in df.columns if "procedure_code" in col.lower()]
        atc_col = next((col for col in df.columns if "atc" in col.lower()), None)
        
        # Identify timestamp columns
        timestamp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ["date", "time", "from", "to"]):
                timestamp_cols.append(col)
        
        # Encode medical codes
        all_medical_code_cols = diagnosis_cols + procedure_cols
        if all_medical_code_cols:
            logging.info(f"Using simplified encoding for {len(all_medical_code_cols)} medical code columns")
            processed_df = FeaturePreprocessor.encode_medical_codes(processed_df, all_medical_code_cols)
        
        # Convert timestamp columns
        if timestamp_cols:
            logging.info(f"Converting {len(timestamp_cols)} timestamp columns to Unix timestamps")
            numeric_cols = []  # Temporary list, not returned
            processed_df, _ = FeaturePreprocessor.convert_timestamps_to_epoch(processed_df, timestamp_cols, numeric_cols)

        if atc_col:
            logging.info(f"Using simplified encoding for atc code columns")
            processed_df = FeaturePreprocessor.encode_atc_codes(processed_df, atc_col)
        
        # Categorize remaining columns as numeric or string
        remaining_cols = [col for col in df.columns if col not in timestamp_cols]
        
        # For single dataframe, try to convert to numeric where possible
        for col in remaining_cols:
            try:
                # Try to convert the column to numeric
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # If conversion made everything NaN, revert to original values as strings
                if processed_df[col].isna().all() and not df[col].isna().all():
                    processed_df[col] = df[col].astype(str)
            except:
                # If conversion failed, treat as string
                processed_df[col] = df[col].astype(str)
        
        logging.info(f"Preprocessed single dataframe with {len(processed_df.columns)} columns")
        
        return processed_df
    
    @staticmethod
    def convert_timestamps_to_epoch(df: pd.DataFrame, timestamp_cols: List[str], numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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
                        logging.warning(f"Column '{col}' has {df[col].isna().sum()} values that couldn't be converted to datetime.")
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

    @staticmethod
    def categorize_columns(df1: pd.DataFrame, df2: pd.DataFrame, common_cols: List[str]) -> Tuple[List[str], List[str]]:
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

    @staticmethod
    def encode_medical_codes(df: pd.DataFrame, code_columns: List[str]) -> pd.DataFrame:
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
                numeric_array = np.array(numeric_values, dtype=float)
                df_encoded.loc[valid_mask, numeric_col] = numeric_array
        
        return df_encoded
    
    @staticmethod
    def encode_atc_codes(df: pd.DataFrame, atc_column: str) -> pd.DataFrame:
        """
        Comprehensive encoding for ATC codes that preserves the complete hierarchical structure.
        
        ATC code structure:
        - 1st level: Anatomical main group (1 letter)
        - 2nd level: Therapeutic subgroup (2 digits)
        - 3rd level: Pharmacological subgroup (1 letter)
        - 4th level: Chemical subgroup (1 letter)
        - 5th level: Chemical substance (2 digits)
        
        Args:
            df: DataFrame containing ATC codes
            atc_column: Name of the column containing ATC codes
            
        Returns:
            DataFrame with encoded ATC codes replacing the original column
        """
        df_encoded = df.copy()
        
        # Convert to string type
        df_encoded[atc_column] = df_encoded[atc_column].astype(str)
        
        # Create a temporary column to store encoded values
        temp_encoded_col = "_temp_encoded"
        df_encoded[temp_encoded_col] = -1.0
        
        valid_mask = ~df_encoded[atc_column].isin(['UNKNOWN', 'UUU', 'NAN', 'NONE', ''])
        
        if valid_mask.any():
            valid_codes = df_encoded.loc[valid_mask, atc_column]
            numeric_values = []
            
            for code in valid_codes:
                # Skip invalid codes
                if not code or not code[0].isalpha():
                    numeric_values.append(-1.0)
                    continue
                
                # Initialize value components
                lvl1_val = 0  # Anatomical main group (1-26)
                lvl2_val = 0  # Therapeutic subgroup (0.01-0.99)
                lvl3_val = 0  # Pharmacological subgroup (0.0001-0.0026)
                lvl4_val = 0  # Chemical subgroup (0.000001-0.000026)
                lvl5_val = 0  # Chemical substance (0.00000001-0.00000099)
                
                # Extract and convert level 1: Anatomical main group (letter A-Z)
                if len(code) >= 1 and code[0].isalpha():
                    lvl1_val = ord(code[0]) - ord('A') + 1  # A=1, B=2, ..., Z=26
                
                # Extract and convert level 2: Therapeutic subgroup (2 digits)
                if len(code) >= 3 and code[1:3].isdigit():
                    lvl2_val = int(code[1:3]) / 100  # 01-99 -> 0.01-0.99
                
                # Extract and convert level 3: Pharmacological subgroup (letter)
                if len(code) >= 4 and code[3].isalpha():
                    lvl3_val = (ord(code[3]) - ord('A') + 1) / 10000  # A=0.0001, B=0.0002, etc.
                
                # Extract and convert level 4: Chemical subgroup (letter)
                if len(code) >= 5 and code[4].isalpha():
                    lvl4_val = (ord(code[4]) - ord('A') + 1) / 1000000  # A=0.000001, etc.
                
                # Extract and convert level 5: Chemical substance (2 digits)
                if len(code) >= 7 and code[5:7].isdigit():
                    lvl5_val = int(code[5:7]) / 100000000  # 01-99 -> 0.00000001-0.00000099
                
                # Combine all levels into a single numeric value
                combined_value = lvl1_val + lvl2_val + lvl3_val + lvl4_val + lvl5_val
                numeric_values.append(float(combined_value))
            
            # Update the temporary column
            numeric_array = np.array(numeric_values, dtype=float)
            df_encoded.loc[valid_mask, temp_encoded_col] = numeric_array
        
        # Replace original column with encoded values and drop the temporary column
        df_encoded[atc_column] = df_encoded[temp_encoded_col]
        df_encoded = df_encoded.drop(columns=[temp_encoded_col])
        
        return df_encoded
    
    @staticmethod
    def get_sensitive_attributes_columns(all_columns: List[str], table_name: str) -> List[str]:
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
            logging.warning(f"No matching sensitive attributes found for table: {table_name}")
            # Return all columns if no sensitive attributes found
            return all_columns
        
        logging.info(f"Selected {len(filtered_cols)} sensitive columns for {table_name}")
        logging.info(f"Columns: {', '.join(filtered_cols)}")
        
        return filtered_cols
    
# Helper functions that can be used directly without class instantiation

def preprocess_dataframes(df1, df2):
    """Wrapper for FeaturePreprocessor.preprocess_dataframes"""
    return FeaturePreprocessor.preprocess_dataframes(df1, df2)

def preprocess_single_dataframe(df):
    """Wrapper for FeaturePreprocessor.preprocess_single_dataframe"""
    return FeaturePreprocessor.preprocess_single_dataframe(df)

def encode_medical_codes(df, code_columns):
    """Wrapper for FeaturePreprocessor.encode_medical_codes"""
    return FeaturePreprocessor.encode_medical_codes(df, code_columns)

def convert_timestamps_to_epoch(df, timestamp_cols, numeric_cols=None):
    """Wrapper for FeaturePreprocessor.convert_timestamps_to_epoch"""
    if numeric_cols is None:
        numeric_cols = []
    return FeaturePreprocessor.convert_timestamps_to_epoch(df, timestamp_cols, numeric_cols)

def categorize_columns(df1, df2, common_cols):
    """Wrapper for FeaturePreprocessor.categorize_columns"""
    return FeaturePreprocessor.categorize_columns(df1, df2, common_cols)

def get_sensitive_attributes_columns(all_columns, table_name):
    """Wrapper for FeaturePreprocessor.get_sensitive_attributes_columns"""
    return FeaturePreprocessor.get_sensitive_attributes_columns(all_columns, table_name)