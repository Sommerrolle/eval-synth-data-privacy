import os
import duckdb
import pandas as pd
from typing import Dict, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_creation.log'),
        logging.StreamHandler()
    ]
)

# Define constants
DUCKDB_DIR = 'duckdb'
CSV_PREFIX = 'sle.'
CSV_SUFFIX = '.csv'

# Data type definitions
DTYPES = {
    'insurance_data': {
        'pid': int,
        'death': 'Int64',
        'regional_code': 'Int64'
    },
    'insurants': {
        'pid': int,
        'year_of_birth': 'Int64',
        'gender': 'Int64',
    },
    'inpatient_cases': {
        'pid': int,
        'caseID': 'Int64',
        'cause of admission': 'str',
        'cause of discharge': 'str',
        'outpatient treatment': 'Int64',
        'department admission': str,
        'department discharge': str
    },
    'inpatient_diagnosis': {
        'pid': int,
        'caseID': 'Int64',
        'diagnosis': str,
        'type of diagnosis': str,
        'is main diagnosis': 'Int64',
        'localisation': 'Int64'
    },
    'inpatient_fees': {
        'pid': int,
        'caseID': 'Int64',
        'billing code': str,
        'amount due': float,
        'quantity': 'Int64'
    },
    'inpatient_procedures': {
        'pid': int,
        'caseID': 'Int64',
        'procedure code': str,
        'localisation': 'Int64',
    },
    'outpatient_cases': {
        'pid': int,
        'caseID': 'Int64',
        'practice code': str,
        'amount due': float,
        'year': 'Int64',
        'quarter': 'Int64'
    },
    'outpatient_diagnosis': {
        'pid': int,
        'caseID': 'Int64',
        'diagnosis': str,
        'qualification': str,
        'localisation': 'Int64'
    },
    'outpatient_fees': {
        'pid': int,
        'caseID': 'Int64',
        'physican code': str,
        'specialty code': str,
        'billing code': str,
        'quantity': 'Int64',
    },
    'outpatient_procedures': {
        'pid': int,
        'caseID': 'Int64',
        'procedure code': str,
        'localisation': 'Int64',
    },
    'drugs': {
        'pid': int,
        'pharma central number': str,
        'specialty of prescriber': str,
        'physican code': str,
        'practice code': str,
        'quantity': float,
        'amount due': float,
        'atc': str,
        'ddd': float
    }
}

PARSE_DATES = {
    'insurance_data': ['from', 'to'],
    'inpatient_cases': ['date of admission', 'date of discharge'],
    'inpatient_fees': ['from', 'to'],
    'inpatient_procedures': ['date of procedure'],
    'outpatient_cases': ['from', 'to'],
    'outpatient_fees': ['date'],
    'drugs': ['date of prescription', 'date of dispense']
}

def rename_columns(df: pd.DataFrame, prefix: str, exceptions: List[str] = None, 
                  dataset_type: str = None) -> pd.DataFrame:
    """Rename DataFrame columns with consistent formatting."""
    if exceptions is None:
        exceptions = []

    def rename_column(col_name: str) -> str:
        if col_name in exceptions:
            return col_name
        if col_name == "caseID":
            if dataset_type == "inpatient":
                return "inpatient_caseID"
            elif dataset_type == "outpatient":
                return "outpatient_caseID"
        return f"{prefix}_{col_name.strip().replace(' ', '_').lower()}"

    return df.rename(columns=rename_column)

def read_csv_file(filepath: str, col_types: Dict, table_name: str) -> pd.DataFrame:
    """Read a CSV file with proper error handling."""
    try:
        df = pd.read_csv(
            filepath,
            dtype=col_types,
            sep='\t',
            encoding='utf-8',
            on_bad_lines='warn',
            parse_dates=PARSE_DATES.get(table_name, None),
            encoding_errors='replace'
        )
        return df
    except Exception as e:
        logging.error(f"Error reading {filepath}: {str(e)}")
        return None

def sort_tables_by_pid(database_path: str) -> None:
    """Sort all tables in the database by pid column."""
    con = duckdb.connect(database=database_path, read_only=False)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            columns = con.execute(f"DESCRIBE {table_name}").fetchall()
            column_names = [col[0] for col in columns]
            
            if 'pid' in column_names:
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS
                    SELECT * FROM {table_name} ORDER BY pid
                """)
                logging.info(f"Table '{table_name}' sorted by 'pid'")
    except Exception as e:
        logging.error(f"Error sorting tables: {str(e)}")
    finally:
        con.close()

def create_database(data_dir: str, force_recreate: bool = False) -> str:
    """Create a DuckDB database from CSV files in the specified directory."""
    # Create database name from directory name
    db_name = Path(data_dir).name
    os.makedirs(DUCKDB_DIR, exist_ok=True)
    db_path = Path(DUCKDB_DIR) / f"{db_name}.duckdb"
    
    # Check if database already exists
    if db_path.exists() and not force_recreate:
        logging.info(f"Database {db_path} already exists. Skipping creation.")
        return str(db_path)
    
    # Scan directory for CSV files
    csv_files = {}
    for file in os.listdir(data_dir):
        if file.startswith(CSV_PREFIX) and file.endswith(CSV_SUFFIX):
            table_name = file[len(CSV_PREFIX):-len(CSV_SUFFIX)]
            csv_files[table_name] = Path(data_dir) / file
    
    if not csv_files:
        logging.warning(f"No CSV files found in {data_dir}")
        return None
    
    # Create database and tables
    con = duckdb.connect(database=str(db_path), read_only=False)
    try:
        for table_name, file_path in csv_files.items():
            logging.info(f"Processing table: {table_name}")
            
            dataset_type = ("inpatient" if "inpatient" in table_name 
                          else "outpatient" if "outpatient" in table_name 
                          else None)
            
            df = read_csv_file(str(file_path), DTYPES.get(table_name, None), table_name)
            if df is None or len(df) == 0:
                continue
                
            df = rename_columns(df, prefix=table_name, exceptions=['pid'], 
                              dataset_type=dataset_type)
            
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
            logging.info(f"Created table '{table_name}' with {len(df)} rows")
        
        # Sort tables by pid
        con.close()
        sort_tables_by_pid(str(db_path))
        
        return str(db_path)
    
    except Exception as e:
        logging.error(f"Error creating database: {str(e)}")
        con.close()
        return None

def process_all_datasets(base_dir: str, force_recreate: bool = False) -> Dict[str, str]:
    """Process all dataset directories and create corresponding databases."""
    database_paths = {}
    
    # Get all subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        logging.info(f"Processing dataset directory: {subdir}")
        
        db_path = create_database(full_path, force_recreate)
        if db_path:
            database_paths[subdir] = db_path
            logging.info(f"Successfully created database for {subdir} at {db_path}")
        else:
            logging.warning(f"Failed to create database for {subdir}")
    
    return database_paths

def main():
    """Main function to process all datasets."""
    # Specify your base directory containing all dataset folders
    base_dir = 'D:/Benutzer/Cuong.VoTa/datasets'
    
    # Process all datasets
    logging.info("Starting database creation process...")
    database_paths = process_all_datasets(base_dir, force_recreate=False)
    
    # Print summary
    logging.info("\nDatabase Creation Summary:")
    for dataset, path in database_paths.items():
        logging.info(f"{dataset}: {path}")

if __name__ == "__main__":
    main()