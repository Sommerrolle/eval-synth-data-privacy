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

# Dictionary for column type conversions
# Format: {'table_name': {'column_name': 'new_type', ...}, ...}
COLUMN_TYPE_MAPPINGS = {
    'insurants': {
        'pid': 'INTEGER',
        'insurants_gender': 'INTEGER',
        'insurants_year_of_birth': 'BIGINT'
    },
    'insurance_data': {
        'pid': 'INTEGER',
        'insurance_data_from': 'DATE',
        'insurance_data_to': 'DATE',
        'insurance_data_death': 'INTEGER',
        'insurance_data_regional_code': 'INTEGER'
    },
    'drugs': {
        'pid': 'INTEGER',
        'drugs_date_of_prescription': 'DATE',
        'drugs_date_of_dispense': 'DATE',
        'drugs_pharma_central_number': 'VARCHAR(9)',
        'drugs_specialty_of_prescriber': 'VARCHAR(2)',
        'drugs_physican_code': 'VARCHAR(9)',
        'drugs_practice_code': 'VARCHAR(9)',
        'drugs_quantity': 'DOUBLE',
        'drugs_amount_due': 'DOUBLE',
        'drugs_atc': 'VARCHAR',
        'drugs_ddd': 'DOUBLE'
    },
    'outpatient_cases': {
        'outpatient_caseID': 'INTEGER',
        'pid': 'INTEGER',
        'outpatient_cases_practice_code': 'VARCHAR(9)',
        'outpatient_cases_from': 'DATE',
        'outpatient_cases_to': 'DATE',
        'outpatient_cases_amount_due': 'DOUBLE',
        'outpatient_cases_year': 'INTEGER',
        'outpatient_cases_quarter': 'INTEGER'
    },
    'outpatient_diagnosis': {  # Renamed from outpatients_diagnosis
        'pid': 'INTEGER',
        'outpatient_caseID': 'INTEGER',
        'outpatient_diagnosis_diagnosis': 'VARCHAR',
        'outpatient_diagnosis_qualification': 'VARCHAR(1)',
        'outpatient_diagnosis_localisation': 'INTEGER'
    },
    'outpatient_fees': {  # Renamed from outpatients_fees
        'pid': 'INTEGER',
        'outpatient_caseID': 'INTEGER',
        'outpatient_fees_physican_code': 'VARCHAR',
        'outpatient_fees_specialty_code': 'VARCHAR',
        'outpatient_fees_billing_code': 'VARCHAR',
        'outpatient_fees_quantity': 'DOUBLE',
        'outpatient_fees_date': 'DATE'
    },
    'outpatient_procedures': {  # Renamed from outpatients_procedures
        'pid': 'INTEGER',
        'outpatient_caseID': 'INTEGER',
        'outpatient_procedures_procedure_code': 'VARCHAR',
        'outpatient_procedures_localisation': 'INTEGER',
        'outpatient_procedures_date_of_procedure': 'DATE'
    },
    'inpatient_cases': {
        'inpatient_caseID': 'INTEGER',
        'pid': 'INTEGER',
        'inpatient_cases_date_of_admission': 'DATE',
        'inpatient_cases_date_of_discharge': 'DATE',
        'inpatient_cases_cause_of_admission': 'VARCHAR(4)',
        'inpatient_cases_cause_of_discharge': 'VARCHAR(2)',
        'inpatient_cases_outpatient_treatment': 'INTEGER',
        'inpatient_cases_department_admission': 'VARCHAR(4)',
        'inpatient_cases_department_discharge': 'VARCHAR(4)'  # Fixed spelling
    },
    'inpatient_diagnosis': {  # Renamed from inpatients_diagnosis
        'pid': 'INTEGER',
        'inpatient_caseID': 'INTEGER',
        'inpatient_diagnosis_diagnosis': 'VARCHAR',
        'inpatient_diagnosis_type_of_diagnosis': 'VARCHAR(2)',
        'inpatient_diagnosis_is_main_diagnosis': 'INTEGER',
        'inpatient_diagnosis_localisation': 'INTEGER'
    },
    'inpatient_fees': {  # Renamed from inpatients_fees
        'pid': 'INTEGER',
        'inpatient_caseID': 'INTEGER',
        'inpatient_fees_from': 'DATE',
        'inpatient_fees_to': 'DATE',
        'inpatient_fees_billing_code': 'VARCHAR',
        'inpatient_fees_amount_due': 'DOUBLE',
        'inpatient_fees_quantity': 'DOUBLE'
    },
    'inpatient_procedures': {  # Renamed from inpatients_procedures
        'pid': 'INTEGER',
        'inpatient_caseID': 'INTEGER',
        'inpatient_procedures_procedure_code': 'VARCHAR',
        'inpatient_procedures_localisation': 'INTEGER',
        'inpatient_procedures_date_of_procedure': 'DATE'
    }
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
    """Read a CSV file with consistent handling of missing values and correct data types."""
    try:
        # Prepare the dtype dictionary for pandas read_csv
        dtype_dict = {}
        if col_types:
            for col, col_type in col_types.items():
                # Skip date columns as they will be handled by parse_dates
                if col in PARSE_DATES.get(table_name, []):
                    continue
                # Add to dtype dictionary
                dtype_dict[col] = col_type
        
        logging.info(f"Reading {filepath}...")
        
        # Read CSV with dtype specification
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8',
            on_bad_lines='warn',
            parse_dates=PARSE_DATES.get(table_name, None),
            encoding_errors='replace',
            dtype=dtype_dict
        )
        
        #Handle specific column types that might need additional processing
        for col in df.columns:
            col_lower = col.lower()
            # Handle amount columns
            if 'amount' in col_lower:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            # Handle quantity columns
            elif 'quantity' in col_lower:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            # Handle DDD (Defined Daily Dose)
            elif col_lower == 'ddd':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            elif col_lower == 'localisation':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Handle categorical columns (keep NaN)
            elif col in ['diagnosis', 'procedure_code', 'atc', 'billing_code', 
                        'physican_code', 'practice_code', 'specialty_code', 'atc',
                        'pharma_central_number', 'speciality_of_prescriber',
                        'qualificatoin', '']:
                # Only convert to string if not already an object type
                if df[col].dtype != 'object':
                    df[col] = df[col].astype(str).replace('nan', pd.NA)
        
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

def change_column_types(database_path: str, type_mappings: Dict[str, Dict[str, str]]) -> bool:
    """
    Change the data types of specified columns in a DuckDB database.
    
    Args:
        database_path: Path to the DuckDB database file
        type_mappings: Dictionary mapping table names to column/type mappings
                       Format: {'table_name': {'column_name': 'new_type', ...}, ...}
    
    Returns:
        bool: True if all changes were successful, False otherwise
    """
    if not os.path.exists(database_path):
        logging.error(f"Database file not found: {database_path}")
        return False
        
    con = duckdb.connect(database=database_path, read_only=False)
    success = True
    
    try:
        # Get list of tables in the database
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        
        for table_name, column_mappings in type_mappings.items():
            if table_name not in tables:
                logging.warning(f"Table '{table_name}' not found in database")
                success = False
                continue
                
            # Get current columns and their types
            current_columns = {row[0]: row[1] for row in con.execute(f"DESCRIBE {table_name}").fetchall()}
            
            for column_name, new_type in column_mappings.items():
                if column_name not in current_columns:
                    logging.warning(f"Column '{column_name}' not found in table '{table_name}'")
                    success = False
                    continue
                
                current_type = current_columns[column_name]
                
                if current_type == new_type:
                    logging.info(f"Column '{column_name}' in table '{table_name}' already has type '{new_type}'")
                    continue
                
                logging.info(f"Changing type of column '{column_name}' in table '{table_name}' from '{current_type}' to '{new_type}'")
                
                try:
                    con.execute(f"ALTER TABLE {table_name} ALTER {column_name} TYPE {new_type}")
                    logging.info(f"Successfully changed type of '{column_name}' in '{table_name}' to '{new_type}'")
                
                except Exception as e:
                    logging.error(f"Error changing column type: {str(e)}")
                    success = False
    
    except Exception as e:
        logging.error(f"Error changing column types: {str(e)}")
        success = False
    
    finally:
        con.close()
        
    return success


def fix_column_types_in_database(db_path: str) -> bool:
    """Apply standard column type conversions to the database."""
    logging.info(f"Fixing column data types in database: {db_path}")
    result = change_column_types(db_path, COLUMN_TYPE_MAPPINGS)
    if result:
        logging.info("Successfully updated all column data types")
    else:
        logging.warning("Some column type conversions failed. Check the log for details.")
    return result


def fix_existing_database_types():
    """Interactive function to fix column types in an existing database."""
    # Get a list of existing DuckDB databases
    existing_dbs = get_existing_databases()
    if not existing_dbs:
        print("No existing databases found in the DuckDB directory.")
        return
    
    print("\nAvailable databases:")
    for i, db in enumerate(existing_dbs, 1):
        print(f"{i}. {db}")
    
    while True:
        try:
            print("\nSelect a database to fix column types (enter number):")
            selection = int(input("> ").strip())
            if 1 <= selection <= len(existing_dbs):
                selected_db = list(existing_dbs)[selection - 1]
                break
            print(f"Please enter a number between 1 and {len(existing_dbs)}")
        except ValueError:
            print("Please enter a valid number")
    
    db_path = os.path.join(DUCKDB_DIR, f"{selected_db}.duckdb")
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return
    
    print(f"\nFixing column types in database: {db_path}")
    result = fix_column_types_in_database(db_path)
    
    if result:
        print(f"\n✅ Successfully updated column types in {selected_db}")
    else:
        print(f"\n⚠️ Some column type conversions failed in {selected_db}. Check the log for details.")


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

        # Fix column types if requested
        logging.info(f"Fixing column types in database: {db_path}")
        if change_column_types(str(db_path), COLUMN_TYPE_MAPPINGS):
            logging.info("Successfully updated column data types")
        else:
            logging.warning("Some column type conversions failed. Check the log for details.")
        
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
    existing_dbs = get_existing_databases(duckdb_dir = "data/duckdb")
    
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