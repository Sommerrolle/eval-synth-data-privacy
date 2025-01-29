import os
import duckdb
import pandas as pd
from typing import Dict, List, Set
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
    """Read a CSV file with consistent handling of missing values across datasets."""
    try:
        # First read without dtype specification to check the data
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8',
            on_bad_lines='warn',
            parse_dates=PARSE_DATES.get(table_name, None),
            encoding_errors='replace'
        )
        
        # Handle numeric columns consistently across all tables
        numeric_columns = {
            'amount due': 0.0,  # Replace NaN with 0.0 for all amount columns
            'quantity': 0,      # Replace NaN with 0 for all quantity columns
            'ddd': 0.0,        # Replace NaN with 0.0 for Defined Daily Dose
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Handle amount columns
            if 'amount' in col_lower:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            # Handle quantity columns
            elif 'quantity' in col_lower:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('Int64')
            
            # Handle DDD (Defined Daily Dose)
            elif col_lower == 'ddd':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            # Handle categorical columns (keep NaN)
            elif col in ['diagnosis', 'procedure_code', 'atc', 'billing_code', 
                        'physican_code', 'practice_code', 'specialty_code']:
                # Convert to string but keep NaN as NaN
                df[col] = df[col].astype(str).replace('nan', pd.NA)
            
            # Apply original dtype if specified
            elif col_types and col in col_types:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(col_types[col])
                except Exception as type_error:
                    logging.warning(f"Could not convert column {col} to {col_types[col]} in {filepath}. Error: {str(type_error)}")
        
        # Log the number of NaN values in each column
        nan_counts = df.isna().sum()
        if nan_counts.any():
            logging.info(f"NaN counts in {table_name}:")
            for col, count in nan_counts[nan_counts > 0].items():
                logging.info(f"{col}: {count} NaN values")
        
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


def get_existing_databases(duckdb_dir: str = DUCKDB_DIR) -> Set[str]:
    """Get set of existing database names (without .duckdb extension)."""
    duckdb_path = Path(duckdb_dir)
    if not duckdb_path.exists():
        return set()
    return {f.stem for f in duckdb_path.glob('*.duckdb')}

def display_directory_options(base_dir: str, existing_dbs: Set[str]) -> Dict[int, str]:
    """Display available directories with their status."""
    subdirs = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d))]
    
    print("\nAvailable directories:")
    print("-" * 60)
    print(f"{'#':<4} {'Directory':<30} {'Status':<20}")
    print("-" * 60)
    
    options = {}
    for i, subdir in enumerate(subdirs, 1):
        status = "Exists" if subdir in existing_dbs else "Not created"
        print(f"{i:<4} {subdir:<30} {status:<20}")
        options[i] = subdir
    print("-" * 60)
    
    return options

def get_user_selection(options: Dict[int, str]) -> List[str]:
    """Get user selection of directories to process."""
    while True:
        print("\nEnter the numbers of directories to process (space-separated)")
        print("Or enter 'all' to process all directories")
        choice = input("> ").strip().lower()
        
        if choice == 'all':
            return list(options.values())
            
        try:
            numbers = [int(x) for x in choice.split()]
            if all(n in options for n in numbers):
                return [options[n] for n in numbers]
            print("Invalid numbers. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numbers or 'all'")

def get_force_recreate_choice(selected_dirs: List[str], existing_dbs: Set[str]) -> Dict[str, bool]:
    """Get force_recreate choice for each selected directory."""
    force_choices = {}
    
    for directory in selected_dirs:
        if directory in existing_dbs:
            print(f"\nDatabase for '{directory}' already exists.")
            while True:
                choice = input("Recreate it? (y/n): ").lower()
                if choice in ('y', 'n'):
                    force_choices[directory] = (choice == 'y')
                    break
                print("Please enter 'y' or 'n'")
        else:
            force_choices[directory] = False
    
    return force_choices

def process_selected_datasets(base_dir: str, selected_dirs: List[str], 
                            force_choices: Dict[str, bool]) -> Dict[str, str]:
    """Process selected dataset directories with specified force_recreate choices."""
    database_paths = {}
    
    for directory in selected_dirs:
        full_path = os.path.join(base_dir, directory)
        logging.info(f"\nProcessing dataset directory: {directory}")
        
        db_path = create_database(full_path, force_choices.get(directory, False))
        if db_path:
            database_paths[directory] = db_path
            logging.info(f"Successfully created/updated database for {directory} at {db_path}")
        else:
            logging.warning(f"Failed to create database for {directory}")
    
    return database_paths

def main():
    """Main function with interactive directory selection."""
    # Specify your base directory containing all dataset folders
    base_dir = 'D:/Benutzer/Cuong.VoTa/datasets'
    
    # Get existing databases
    existing_dbs = get_existing_databases()
    
    # Display options and get user selection
    options = display_directory_options(base_dir, existing_dbs)
    if not options:
        logging.error("No directories found in base directory")
        return
    
    # Get user selection
    selected_dirs = get_user_selection(options)
    if not selected_dirs:
        logging.error("No directories selected")
        return
    
    # Get force_recreate choices
    force_choices = get_force_recreate_choice(selected_dirs, existing_dbs)
    
    # Process selected datasets
    logging.info("\nStarting database creation process...")
    database_paths = process_selected_datasets(base_dir, selected_dirs, force_choices)
    
    # Print summary
    print("\nDatabase Creation Summary:")
    print("-" * 60)
    for dataset, path in database_paths.items():
        print(f"{dataset}: {path}")
    print("-" * 60)

if __name__ == "__main__":
    main()