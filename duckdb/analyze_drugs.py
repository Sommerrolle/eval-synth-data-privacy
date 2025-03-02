import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_drugs_csv_with_dtypes(filepath):
    """Read the drugs CSV file with specified dtypes and identify issues."""
    logging.info(f"Analyzing file: {filepath}")
    
    # Define dtypes as specified
    DTYPES = {
        'pid': 'int',
        'pharma central number': str,
        'specialty of prescriber': str,
        'physican code': str,
        'practice code': str,
        'quantity': float,
        'amount due': float,
        'atc': str,
        'ddd': float
    }
    
    PARSE_DATES = ['date of prescription', 'date of dispense']
    
    # Try reading each column individually with the specified dtype
    for column, dtype in DTYPES.items():
        try:
            logging.info(f"Trying to read column '{column}' with dtype {dtype}")
            
            # Read only the specific column
            col_data = pd.read_csv(
                filepath,
                sep='\t',
                encoding='utf-8',
                usecols=[column] if column != 'pid' else ['pid'],
                dtype={column: dtype},
                on_bad_lines='warn',
                encoding_errors='replace'
            )
            
            logging.info(f"Successfully read column '{column}' with dtype {dtype}")
            
            # Additional check for numeric types
            if dtype == 'int' or dtype == int:
                # Check if we can convert values to int without losing data
                original_values = pd.read_csv(
                    filepath, 
                    sep='\t',
                    usecols=[column],
                    on_bad_lines='warn',
                    encoding_errors='replace'
                )[column]
                
                numeric_values = pd.to_numeric(original_values, errors='coerce')
                if not numeric_values.isna().equals(original_values.isna()):
                    logging.warning(f"Column '{column}' has non-numeric values")
                
                # Check for decimals
                valid_values = numeric_values.dropna()
                if not (valid_values == valid_values.astype(int)).all():
                    logging.error(f"Column '{column}' has float values that cannot be safely cast to int")
                    logging.error(f"Example values: {valid_values[valid_values != valid_values.astype(int)].head().tolist()}")
            
        except Exception as e:
            logging.error(f"Error reading column '{column}' with dtype {dtype}: {str(e)}")
            
            # If error with int, try reading as float to see values
            if dtype == 'int' or dtype == int:
                try:
                    col_data = pd.read_csv(
                        filepath,
                        sep='\t',
                        encoding='utf-8',
                        usecols=[column],
                        dtype={column: float},
                        on_bad_lines='warn',
                        encoding_errors='replace'
                    )
                    
                    logging.info(f"Column '{column}' can be read as float")
                    logging.info(f"Sample values: {col_data[column].head().tolist()}")
                    
                    # Check if values have decimals
                    has_decimals = (col_data[column] != col_data[column].astype(int)).any()
                    if has_decimals:
                        logging.error(f"Column '{column}' contains decimal values")
                        decimal_examples = col_data[col_data[column] != col_data[column].astype(int)][column].head().tolist()
                        logging.error(f"Examples of decimal values: {decimal_examples}")
                        
                except Exception as e2:
                    logging.error(f"Could not read '{column}' as float either: {str(e2)}")
    
    # Now try to read the entire file with all dtypes
    try:
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8',
            dtype=DTYPES,
            parse_dates=PARSE_DATES,
            on_bad_lines='warn',
            encoding_errors='replace'
        )
        logging.info("Successfully read the entire file with specified dtypes")
        return df
    except Exception as e:
        logging.error(f"Failed to read the entire file with specified dtypes: {str(e)}")
        return None

if __name__ == "__main__":
    # Replace this with your actual file path
    file_path = r"D:\Benutzer\Cuong.VoTa\datasets\claims_data\sle.drugs.csv"
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
    else:
        df = read_drugs_csv_with_dtypes(file_path)
        
        if df is not None:
            logging.info(f"Successfully read the file. Shape: {df.shape}")
            logging.info(f"Column dtypes: {df.dtypes}")