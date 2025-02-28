import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

# Import your DuckDBManager
from duckdb_manager import DuckDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log'),
        logging.StreamHandler()
    ]
)

class MedicalDataPreprocessor:
    """Preprocess medical data for privacy metrics calculations."""
    
    def __init__(self, results_dir: str = 'results/preprocessed'):
        """Initialize the preprocessor."""
        self.results_dir = results_dir
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        self.db_manager = DuckDBManager()
    
    def preprocess_table(self, db_path: str, table_name: str, output_table_name: Optional[str] = None, 
                         max_nulls: int = 3) -> str:
        """
        Preprocess a medical data table by:
        1. Removing ID columns (except pid)
        2. Removing rows with too many NULL values
        3. Handling NULL values based on data type and medical domain context
        
        Args:
            db_path: Path to the DuckDB database
            table_name: Name of the table to preprocess
            output_table_name: Name for the preprocessed table
            max_nulls: Maximum number of NULL values allowed per row
            
        Returns:
            Name of the preprocessed table
        """
        if output_table_name is None:
            output_table_name = f"{table_name}_preprocessed"
        
        logging.info(f"Preprocessing table '{table_name}' from {db_path}")
        
        try:
            # Use db_manager to execute queries
            # Get table schema
            columns_info = self.db_manager.get_table_column_info(db_path, table_name)
            
            if not columns_info:
                logging.error(f"Failed to get column information for table '{table_name}'")
                return None
                
            column_names = [col['name'] for col in columns_info]
            column_types = [col['type'] for col in columns_info]
            
            # Remove ID columns except pid
            filtered_columns = [col for col in column_names 
                               if not ('ID' in col.upper() or col.lower().endswith('id')) 
                               or col.lower() == 'pid']
            
            logging.info(f"Removed {len(column_names) - len(filtered_columns)} ID columns")
            
            # Create SQL query for preprocessing
            preprocessing_sql = []
            
            for col, col_type in zip(column_names, column_types):
                if col not in filtered_columns:
                    continue
                
                # Medical domain-specific handling
                if any(date_field in col.lower() for date_field in ['date_of_admission', 'date_of_discharge']):
                    # Keep date fields as is
                    preprocessing_sql.append(col)
                elif 'date_of_procedure' in col.lower():
                    # Check if this is inpatient or outpatient based on the procedure date column name
                    if 'inpatient' in col.lower() and 'inpatient_cases_date_of_admission' in column_names:
                        date_col = 'inpatient_fees_from'
                    elif 'outpatient' in col.lower() and 'outpatient_cases_from' in column_names:
                        date_col = 'outpatient_fees_from'
                    else:
                        date_col = None
                    if date_col:
                        preprocessing_sql.append(f"""
                            COALESCE({col}, CAST({date_col} AS TIMESTAMP_NS)) AS {col}
                        """)
                    else:
                        preprocessing_sql.append(col)  
                elif 'diagnosis' in col.lower() and ('VARCHAR' in col_type or 'TEXT' in col_type):
                    # Keep diagnosis codes as is
                    preprocessing_sql.append(col)
                elif 'procedure_code' in col.lower() and ('VARCHAR' in col_type or 'TEXT' in col_type):
                    preprocessing_sql.append(f"COALESCE({col}, 'UNKNOWN') AS {col}")
                elif any(amount_field in col.lower() for amount_field in ['amount_due', 'quantity']):
                    # Replace financial/quantity nulls with 0
                    preprocessing_sql.append(f"COALESCE({col}, 0) AS {col}")
                elif 'gender' in col.lower():
                    # For gender, replace with mode
                    preprocessing_sql.append(f"""
                        COALESCE({col}, 
                            (SELECT {col} FROM {table_name} 
                            WHERE {col} IS NOT NULL 
                            GROUP BY {col} 
                            ORDER BY COUNT(*) DESC LIMIT 1)) AS {col}
                    """)
                elif 'year_of_birth' in col.lower():
                    # For birth year, replace with median
                    preprocessing_sql.append(f"COALESCE({col}, (SELECT MEDIAN({col}) FROM {table_name})) AS {col}")
                else:
                    # Default handling based on type
                    if any(num_type in col_type.upper() for num_type in ['INT', 'BIGINT', 'DOUBLE', 'FLOAT']):
                        # Numeric columns: replace with median
                        preprocessing_sql.append(f"COALESCE({col}, (SELECT MEDIAN({col}) FROM {table_name})) AS {col}")
                    elif any(text_type in col_type.upper() for text_type in ['VARCHAR', 'CHAR', 'TEXT']):
                        # Text columns: replace with 'Unknown'
                        preprocessing_sql.append(f"COALESCE({col}, 'Unknown') AS {col}")
                    else:
                        # Keep other columns as is
                        preprocessing_sql.append(col)
            
            # Create null count SQL for filtering rows
            null_count_sql = " + ".join([
                f"CASE WHEN {col} IS NULL THEN 1 ELSE 0 END" 
                for col in filtered_columns if col != 'pid'
            ])
            
            # Create the preprocessed table SQL
            create_table_sql = f"""
                CREATE OR REPLACE TABLE {output_table_name} AS
                SELECT {', '.join(preprocessing_sql)}
                FROM {table_name}
                WHERE ({null_count_sql}) <= {max_nulls}
            """
            
            # Execute query using db_manager
            self.db_manager.execute_query(db_path, f"DROP TABLE IF EXISTS {output_table_name}")
            self.db_manager.execute_query(db_path, create_table_sql)
            
            self._log_table_statistics(db_path, table_name, output_table_name, max_nulls, filtered_columns)
            return output_table_name
            
        except Exception as e:
            logging.error(f"Error preprocessing table: {str(e)}")
            return None
        
    def _log_table_statistics(self, db_path: str, table_name: str, output_table_name: str, max_nulls: int, filtered_columns: list):
        # Get table statistics
        original_count = self.db_manager.get_table_count(db_path, table_name)
        processed_count = self.db_manager.get_table_count(db_path, output_table_name)
        
        logging.info(f"Original row count: {original_count}")
        logging.info(f"Processed row count: {processed_count}")
        logging.info(f"Removed {original_count - processed_count} rows with > {max_nulls} NULL values")
        
        # Check for remaining nulls in each column
        for col in filtered_columns:
            null_count = self.db_manager.execute_query(
                db_path, 
                f"SELECT COUNT(*) FROM {output_table_name} WHERE {col} IS NULL"
            )
            if null_count and null_count[0][0] > 0:
                logging.info(f"Column '{col}': {null_count[0][0]} NULL values remain")


def main():
    """Main function to run the preprocessing."""
    
    # Hardcoded default parameters
    DEFAULT_PARAMS = {
        'db_path': 'data/duckdb/claims_data_minimal.duckdb',
        'table_name': 'joined_1_4_5_6_7',
        'output_table': None,  # Will use {table_name}_preprocessed by default
        'max_nulls': 3
    }
    
    # Only parse arguments if they were provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Preprocess health claims data for privacy metrics')
        parser.add_argument('--db_path', help=f'Path to the DuckDB database (default: {DEFAULT_PARAMS["db_path"]})')
        parser.add_argument('--table_name', help=f'Name of the table to preprocess (default: {DEFAULT_PARAMS["table_name"]})')
        parser.add_argument('--output_table', help='Name for the output table (default: {table_name}_preprocessed)')
        parser.add_argument('--max_nulls', type=int, 
                            help=f'Maximum number of NULL values allowed per row (default: {DEFAULT_PARAMS["max_nulls"]})')
        
        args = parser.parse_args()
        
        # Use provided arguments, falling back to defaults when needed
        db_path = args.db_path or DEFAULT_PARAMS['db_path']
        table_name = args.table_name or DEFAULT_PARAMS['table_name']
        output_table = args.output_table or DEFAULT_PARAMS['output_table']
        max_nulls = args.max_nulls if args.max_nulls is not None else DEFAULT_PARAMS['max_nulls']
    else:
        # Use all default parameters if no arguments provided
        logging.info("No command-line arguments provided. Using default parameters.")
        db_path = DEFAULT_PARAMS['db_path']
        table_name = DEFAULT_PARAMS['table_name']
        output_table = DEFAULT_PARAMS['output_table']
        max_nulls = DEFAULT_PARAMS['max_nulls']
    
    # Initialize preprocessor and run preprocessing
    preprocessor = MedicalDataPreprocessor()
    
    logging.info(f"Processing with parameters:")
    logging.info(f"  Database: {db_path}")
    logging.info(f"  Table: {table_name}")
    logging.info(f"  Output table: {output_table or f'{table_name}_preprocessed'}")
    logging.info(f"  Max nulls: {max_nulls}")
    
    preprocessor = MedicalDataPreprocessor()
    
    output_table = preprocessor.preprocess_table(
        db_path, 
        table_name, 
        output_table,
        max_nulls
    )
    
    if output_table:
        print(f"Successfully preprocessed table. New table name: {output_table}")
    else:
        print("Preprocessing failed. Check logs for details.")

if __name__ == "__main__":
    main()