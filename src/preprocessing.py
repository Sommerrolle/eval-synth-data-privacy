import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import duckdb

# Import DuckDBManager
from duckdb_manager import DuckDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preprocessing.log'),
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
    
    def handle_nulls(self, column_name: str, column_type: str, table_name: str) -> str:
        """
        Generate SQL expression to handle NULL values based on column type and medical domain context.
        
        Args:
            column_name: Name of the column to preprocess
            column_type: Data type of the column
            table_name: Name of the source table (for statistical replacements)
            
        Returns:
            SQL expression for handling NULL values in the column
        """
        col = column_name
        col_type = column_type.upper()
        col_lower = col.lower()
        
        # Date handling with domain-specific logic
        if 'DATE' in col_type or 'TIMESTAMP' in col_type or any(date_field in col_lower for date_field in ['date', 'from', 'to']):
            
            # Inpatient admission dates
            if 'date_of_admission' in col_lower and 'inpatient' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT {col.replace('admission', 'discharge')} - INTERVAL '5 days' 
                         FROM {table_name} t2
                         WHERE t2.inpatient_caseID = {table_name}.inpatient_caseID 
                           AND {col.replace('admission', 'discharge')} IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Inpatient discharge dates
            elif 'date_of_discharge' in col_lower and 'inpatient' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT {col.replace('discharge', 'admission')} + INTERVAL '5 days' 
                         FROM {table_name} t2
                         WHERE t2.inpatient_caseID = {table_name}.inpatient_caseID 
                           AND {col.replace('discharge', 'admission')} IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Inpatient procedure dates
            elif 'date_of_procedure' in col_lower and 'inpatient' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT inpatient_cases_date_of_admission + INTERVAL '1 day' 
                         FROM inpatient_cases ic
                         WHERE ic.inpatient_caseID = {table_name}.inpatient_caseID 
                           AND ic.inpatient_cases_date_of_admission IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Outpatient from dates
            elif ('from' in col_lower and 'outpatient' in col_lower and 'cases' in col_lower):
                to_col = col.replace('from', 'to')
                
                return f"""
                    COALESCE({col}, 
                        (SELECT {to_col} - INTERVAL '14 days' 
                         FROM {table_name} t2
                         WHERE t2.outpatient_caseID = {table_name}.outpatient_caseID 
                           AND {to_col} IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Outpatient to dates
            elif ('to' in col_lower and 'outpatient' in col_lower and 'cases' in col_lower):
                from_col = col.replace('to', 'from')
                
                return f"""
                    COALESCE({col}, 
                        (SELECT {from_col} + INTERVAL '14 days' 
                         FROM {table_name} t2
                         WHERE t2.outpatient_caseID = {table_name}.outpatient_caseID 
                           AND {from_col} IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Outpatient procedure dates (VARCHAR format in the database)
            elif 'date_of_procedure' in col_lower and 'outpatient' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT TO_VARCHAR(outpatient_cases_from, 'YYYY-MM-DD')
                         FROM outpatient_cases oc
                         WHERE oc.outpatient_caseID = {table_name}.outpatient_caseID 
                           AND oc.outpatient_cases_from IS NOT NULL 
                         LIMIT 1),
                        '2000-01-01'
                    ) AS {col}
                """
                
            # Fee date handling for inpatient
            elif ('fees' in col_lower and 'from' in col_lower and 'inpatient' in col_lower):
                return f"""
                    COALESCE({col}, 
                        (SELECT TO_VARCHAR(inpatient_cases_date_of_admission, 'YYYY-MM-DD')
                         FROM inpatient_cases ic
                         WHERE ic.inpatient_caseID = {table_name}.inpatient_caseID 
                           AND ic.inpatient_cases_date_of_admission IS NOT NULL 
                         LIMIT 1),
                        '2000-01-01'
                    ) AS {col}
                """
                
            elif ('fees' in col_lower and 'to' in col_lower and 'inpatient' in col_lower):
                return f"""
                    COALESCE({col}, 
                        (SELECT TO_VARCHAR(inpatient_cases_date_of_discharge, 'YYYY-MM-DD')
                         FROM inpatient_cases ic
                         WHERE ic.inpatient_caseID = {table_name}.inpatient_caseID 
                           AND ic.inpatient_cases_date_of_discharge IS NOT NULL 
                         LIMIT 1),
                        '2000-01-01'
                    ) AS {col}
                """
                
            # Outpatient fee dates (VARCHAR format in the database)
            elif 'fees_date' in col_lower and 'outpatient' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT TO_VARCHAR(outpatient_cases_from, 'YYYY-MM-DD')
                         FROM outpatient_cases oc
                         WHERE oc.outpatient_caseID = {table_name}.outpatient_caseID 
                           AND oc.outpatient_cases_from IS NOT NULL 
                         LIMIT 1),
                        '2000-01-01'
                    ) AS {col}
                """
                
            # Drug dates
            elif 'date_of_prescription' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT drugs_date_of_dispense - INTERVAL '1 day' 
                         FROM {table_name} t2
                         WHERE t2.pid = {table_name}.pid 
                           AND t2.drugs_date_of_dispense IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            elif 'date_of_dispense' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT drugs_date_of_prescription + INTERVAL '1 day' 
                         FROM {table_name} t2
                         WHERE t2.pid = {table_name}.pid 
                           AND t2.drugs_date_of_prescription IS NOT NULL 
                         LIMIT 1),
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Insurance data dates
            elif 'from' in col_lower and 'insurance' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT MIN(outpatient_cases_from)
                         FROM outpatient_cases oc
                         WHERE oc.pid = {table_name}.pid 
                           AND oc.outpatient_cases_from IS NOT NULL),
                        (SELECT MIN(inpatient_cases_date_of_admission)
                         FROM inpatient_cases ic
                         WHERE ic.pid = {table_name}.pid 
                           AND ic.inpatient_cases_date_of_admission IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            elif 'to' in col_lower and 'insurance' in col_lower:
                return f"""
                    COALESCE({col}, 
                        (SELECT MAX(outpatient_cases_to)
                         FROM outpatient_cases oc
                         WHERE oc.pid = {table_name}.pid 
                           AND oc.outpatient_cases_to IS NOT NULL),
                        (SELECT MAX(inpatient_cases_date_of_discharge)
                         FROM inpatient_cases ic
                         WHERE ic.pid = {table_name}.pid 
                           AND ic.inpatient_cases_date_of_discharge IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
                
            # Generic date handling for any other date fields
            else:
                return f"""
                    COALESCE({col}, 
                        (SELECT MEDIAN({col}) 
                         FROM {table_name} 
                         WHERE {col} IS NOT NULL),
                        DATE '2000-01-01'
                    ) AS {col}
                """
        
        # Medical domain-specific categorical data handling
        elif 'diagnosis_diagnosis' in col_lower:
            # For diagnosis codes, use 'UNKNOWN'
            return f"COALESCE({col}, 'UNKNOWN') AS {col}"
        elif 'type_of_diagnosis' in col_lower:
            # For type of diagnosis, use '00', because values range form 01-99
            return f"COALESCE({col}, '00') AS {col}"
        
        elif 'procedure_code' in col_lower:
            # For procedure codes, use 'UNKNOWN'
            return f"COALESCE({col}, 'UNKNOWN') AS {col}"
        
        elif 'pharma_central_number' in col_lower:
            # For pharmacy central numbers, use '00000000', 8 digits
            return f"COALESCE({col}, '00000000') AS {col}"
        
        elif 'physician_code' in col_lower or 'physican_code' in col_lower:
            # For physician codes, use '000000000', 9 digits
            return f"COALESCE({col}, '000000000') AS {col}"
        
        elif 'practice_code' in col_lower:
            # For practice codes, use '000000000'
            return f"COALESCE({col}, '000000000') AS {col}"
        
        elif 'specialty_of_prescriber' in col_lower or 'specialty_code' in col_lower:
            # For specialty codes, use '00'
            return f"COALESCE({col}, '00') AS {col}"
        
        elif 'department_admission' in col_lower or 'department_discharge' in col_lower:
            # For department codes, use '0000'
            return f"COALESCE({col}, '0000') AS {col}"
        
        elif 'atc' in col_lower:
            # For ATC codes (Anatomical Therapeutic Chemical), use 'UNKNOWN'
            return f"COALESCE({col}, 'UNKNOWN') AS {col}"
        
        elif 'cause_of_admission' in col_lower:
            # For cause of admission, use '0000'
            return f"COALESCE({col}, '0000') AS {col}"
        
        elif 'cause_of_discharge' in col_lower:
            # For cause of discharge, use '00'
            return f"COALESCE({col}, '00') AS {col}"
        
        elif 'qualification' in col_lower:
            # For qualification codes, use 'U' for UNKNOWN hehe
            return f"COALESCE({col}, 'U') AS {col}"
        
        elif 'billing_code' in col_lower:
            # For billing codes, use 'UNKNOWN', 8 digits
            return f"COALESCE({col}, 'UNKNOWN') AS {col}"
        
        # Financial and quantity handling
        elif 'amount_due' in col_lower:
            # Replace financial nulls with median if available, else 0
            return f"""
                COALESCE({col}, 
                    (SELECT MEDIAN({col}) FROM {table_name} WHERE {col} IS NOT NULL),
                    0
                ) AS {col}
            """
        
        # elif 'quantity' in col_lower or 'ddd' in col_lower:
        #     # Replace quantity nulls with median if available, else 1
        #     return f"""
        #         COALESCE({col}, 
        #             (SELECT MEDIAN({col}) FROM {table_name} WHERE {col} IS NOT NULL),
        #             1
        #         ) AS {col}
        #     """
        
        # Demographic data handling
        elif 'gender' in col_lower:
            # For gender, replace with mode or default to 1 if no mode
            return f"""
                COALESCE({col}, 
                    (SELECT {col} FROM {table_name} 
                     WHERE {col} IS NOT NULL 
                     GROUP BY {col} 
                     ORDER BY COUNT(*) DESC LIMIT 1),
                    1
                ) AS {col}
            """
        
        elif 'year_of_birth' in col_lower:
            # For birth year, replace with median or 1970 as fallback
            return f"""
                COALESCE({col}, 
                    (SELECT MEDIAN({col}) FROM {table_name} WHERE {col} IS NOT NULL),
                    1970
                ) AS {col}
            """
        
        elif 'death' in col_lower:
            # For death indicator, default to 0 (alive)
            return f"COALESCE({col}, 0) AS {col}"
        
        elif 'regional_code' in col_lower:
            # For regional code, use mode or default to 1
            return f"""
                COALESCE({col}, 
                    (SELECT {col} FROM {table_name} 
                     WHERE {col} IS NOT NULL 
                     GROUP BY {col} 
                     ORDER BY COUNT(*) DESC LIMIT 1),
                    1
                ) AS {col}
            """
        
        elif 'localisation' in col_lower:
            # For localisation codes, use 0
            return f"COALESCE({col}, 0) AS {col}"
        
        elif 'is_main_diagnosis' in col_lower:
            # For main diagnosis flag, default to 0 (not main)
            return f"COALESCE({col}, 0) AS {col}"
        
        elif 'outpatient_treatment' in col_lower:
            # For outpatient treatment flag, default to 0
            return f"COALESCE({col}, 0) AS {col}"
        
        # Default handling based on type
        elif any(num_type in col_type for num_type in ['INT', 'BIGINT', 'INTEGER']):
            # Integer columns: replace with mode if low cardinality, median otherwise, or 0 as fallback
            return f"""
                COALESCE({col}, 
                    (SELECT CASE 
                        WHEN COUNT(DISTINCT {col}) > 10 THEN MEDIAN({col})
                        ELSE (SELECT {col} FROM {table_name} 
                              WHERE {col} IS NOT NULL 
                              GROUP BY {col} 
                              ORDER BY COUNT(*) DESC LIMIT 1)
                     END
                     FROM {table_name} 
                     WHERE {col} IS NOT NULL),
                    0
                ) AS {col}
            """
        
        elif any(float_type in col_type for float_type in ['DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC']):
            # Float columns: replace with median or 0
            return f"""
                COALESCE({col}, 
                    (SELECT MEDIAN({col}) FROM {table_name} WHERE {col} IS NOT NULL),
                    0
                ) AS {col}
            """
        
        elif any(text_type in col_type for text_type in ['VARCHAR', 'CHAR', 'TEXT', 'STRING']):
            # Default for text columns: replace with 'UNKNOWN'
            return f"COALESCE({col}, 'UNKNOWN') AS {col}"
        
        else:
            # Keep other columns as is
            return f"{col} AS {col}"
    
    def filter_id_columns(self, column_names: List[str]) -> List[str]:
        """
        Filter out ID columns except for 'pid'.
        
        Args:
            column_names: List of all column names
            
        Returns:
            List of filtered column names
        """
        return [col for col in column_names 
                if not ('ID' in col.upper() or col.lower().endswith('id')) 
                or col.lower() == 'pid']
    
    def get_table_column_info(self, db_path: str, table_name: str) -> List[Dict]:
        """
        Get information about columns in a table.
        
        Args:
            db_path: Path to the database file
            table_name: Name of the table
            
        Returns:
            List of dictionaries with column information
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            return [{"name": col[0], "type": col[1]} for col in columns]
        except Exception as e:
            logging.error(f"Error getting column info for {table_name} from {db_path}: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
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
            # Get table schema
            columns_info = self.get_table_column_info(db_path, table_name)
            
            if not columns_info:
                logging.error(f"Failed to get column information for table '{table_name}'")
                return None
                
            column_names = [col['name'] for col in columns_info]
            column_types = [col['type'] for col in columns_info]
            
            # Remove ID columns except pid
            filtered_columns = self.filter_id_columns(column_names)
            
            logging.info(f"Removed {len(column_names) - len(filtered_columns)} ID columns")
            
            # Count nulls in each column of the original table and log
            for col in filtered_columns:
                null_count = self.db_manager.execute_query(
                    db_path, 
                    f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL"
                )
                if null_count and null_count[0][0] > 0:
                    logging.info(f"Column '{col}': {null_count[0][0]} NULL values in original table")
        
            # Create SQL query for preprocessing with NULL handling
            preprocessing_sql = []
            
            for col, col_type in zip(column_names, column_types):
                if col not in filtered_columns:
                    continue
                
                # Handle NULL values based on column type and context
                preprocessing_sql.append(self.handle_nulls(col, col_type, table_name))
            
            # Create null count SQL for filtering rows with too many nulls
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
            
            # Execute the query in two steps to avoid potential issues with complex queries
            try:
                # First drop existing table if it exists
                self.db_manager.execute_query(db_path, f"DROP TABLE IF EXISTS {output_table_name}")
                
                # Try to execute the full preprocessing query
                logging.info("Executing main preprocessing SQL")
                self.db_manager.execute_query(db_path, create_table_sql)
                
            except Exception as e:
                logging.error(f"Error executing full preprocessing query: {str(e)}")
                logging.info("Falling back to step-by-step preprocessing approach")
                
                # Fallback approach: Create table with basic structure first
                simple_columns = []
                for col in filtered_columns:
                    simple_columns.append(f"{col}")
                
                # Create empty table with basic structure
                create_table_sql = f"""
                    CREATE TABLE {output_table_name} AS
                    SELECT {', '.join(simple_columns)}
                    FROM {table_name}
                    WHERE ({null_count_sql}) <= {max_nulls}
                """
                
                self.db_manager.execute_query(db_path, create_table_sql)
                
                # Then update each column one by one to handle nulls
                for col, col_type in zip(column_names, column_types):
                    if col not in filtered_columns:
                        continue
                    
                    # Skip pid or columns without nulls
                    if col == 'pid':
                        continue
                    
                    # Get appropriate update SQL based on column context
                    update_expr = self.handle_nulls(col, col_type, table_name).split(' AS ')[0]
                    
                    try:
                        # Check if there are still nulls
                        null_check_sql = f"SELECT COUNT(*) FROM {output_table_name} WHERE {col} IS NULL"
                        null_count = self.db_manager.execute_query(db_path, null_check_sql)
                        
                        if null_count and null_count[0][0] > 0:
                            update_sql = f"UPDATE {output_table_name} SET {col} = {update_expr} WHERE {col} IS NULL"
                            self.db_manager.execute_query(db_path, update_sql)
                    except Exception as inner_e:
                        logging.warning(f"Error updating column {col}: {str(inner_e)}")
                        
                        # Final fallback - use very simple defaults based on type
                        try:
                            if any(num_type in col_type.upper() for num_type in ['INT', 'BIGINT', 'DOUBLE', 'FLOAT']):
                                self.db_manager.execute_query(db_path, f"UPDATE {output_table_name} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")
                        except Exception as final_e:
                            logging.error(f"Final fallback failed for column {col}: {str(final_e)}")
            
            # Verify that all NULLs have been handled
            self._verify_null_handling(db_path, output_table_name, filtered_columns)
            
            # Log preprocessing statistics
            self._log_table_statistics(db_path, table_name, output_table_name, max_nulls, filtered_columns)
            return output_table_name
            
        except Exception as e:
            logging.error(f"Error preprocessing table: {str(e)}")
            return None
    
    def _verify_null_handling(self, db_path: str, table_name: str, columns: List[str]) -> None:
        """
        Verify that all NULL values have been handled in the preprocessed table.
        
        Args:
            db_path: Path to the database
            table_name: Table to check
            columns: List of columns to check
        """
        conn = duckdb.connect(db_path, read_only=False)
        try:
            # Check each column for remaining NULLs
            remaining_nulls = False
            for col in columns:
                null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
                
                if null_count > 0:
                    logging.warning(f"Column '{col}' still has {null_count} NULL values after preprocessing")
                    remaining_nulls = True
                    
                    # Final emergency fix attempt
                    try:
                        logging.info(f"Emergency fix attempt for remaining NULLs in {col}")
                        
                        # Get column type
                        col_type = conn.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{col}'").fetchone()[0]
                        
                        # Apply appropriate fix based on data type
                        if any(num_type in col_type.upper() for num_type in ['INT', 'BIGINT']):
                            conn.execute(f"UPDATE {table_name} SET {col} = 0 WHERE {col} IS NULL")
                        elif any(float_type in col_type.upper() for float_type in ['DOUBLE', 'FLOAT', 'REAL']):
                            conn.execute(f"UPDATE {table_name} SET {col} = 0.0 WHERE {col} IS NULL")
                        elif any(date_type in col_type.upper() for date_type in ['DATE', 'TIMESTAMP']):
                            conn.execute(f"UPDATE {table_name} SET {col} = '2000-01-01' WHERE {col} IS NULL")
                        else:
                            conn.execute(f"UPDATE {table_name} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")
                            
                        # Verify fix worked
                        fixed_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
                        if fixed_count > 0:
                            logging.error(f"Failed to fix all NULLs in column '{col}'. {fixed_count} remain.")
                        else:
                            logging.info(f"Successfully fixed all NULLs in column '{col}'")
                            
                    except Exception as e:
                        logging.error(f"Error during emergency NULL fix for column '{col}': {str(e)}")
            
            if not remaining_nulls:
                logging.info("All NULL values have been successfully handled in the preprocessed table")
                
        except Exception as e:
            logging.error(f"Error during NULL verification: {str(e)}")
        finally:
            conn.close()
            
    def _log_table_statistics(self, db_path: str, table_name: str, output_table_name: str, max_nulls: int, filtered_columns: list):
        """
        Log statistics about the preprocessing operation.
        
        Args:
            db_path: Path to the database
            table_name: Original table name
            output_table_name: Preprocessed table name
            max_nulls: Maximum number of NULL values allowed per row
            filtered_columns: Columns included in the preprocessed table
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            
            # Get table statistics
            original_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            processed_count = conn.execute(f"SELECT COUNT(*) FROM {output_table_name}").fetchone()[0]
            
            logging.info(f"Original row count: {original_count:,}")
            logging.info(f"Processed row count: {processed_count:,}")
            logging.info(f"Removed {original_count - processed_count:,} rows with > {max_nulls} NULL values")
            
            # Collect statistics on unique values for key columns
            for col in filtered_columns:
                try:
                    col_type = conn.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name = '{output_table_name}' AND column_name = '{col}'").fetchone()[0]
                    
                    # Only do cardinality checks for non-numeric columns
                    if not any(num_type in col_type.upper() for num_type in ['INT', 'BIGINT', 'DOUBLE', 'FLOAT', 'REAL']):
                        distinct_count = conn.execute(f"SELECT COUNT(DISTINCT {col}) FROM {output_table_name}").fetchone()[0]
                        logging.info(f"Column '{col}' has {distinct_count:,} distinct values")
                except Exception as e:
                    logging.warning(f"Could not get distinct count for column '{col}': {str(e)}")
            
            # Check data distribution for important demographic and medical columns
            important_columns = [
                'insurants_year_of_birth', 'insurants_gender', 
                'inpatient_diagnosis_diagnosis', 'outpatient_diagnosis_diagnosis', 
                'inpatient_procedures_procedure_code', 'outpatient_procedures_procedure_code'
            ]
            
            for col in important_columns:
                if col in filtered_columns:
                    try:
                        most_common = conn.execute(f"""
                            SELECT {col}, COUNT(*) as count 
                            FROM {output_table_name} 
                            GROUP BY {col} 
                            ORDER BY count DESC 
                            LIMIT 5
                        """).fetchall()
                        
                        logging.info(f"Most common values for '{col}':")
                        for value, count in most_common:
                            logging.info(f"    {value}: {count:,} occurrences")
                    except Exception as e:
                        logging.warning(f"Could not analyze distribution for column '{col}': {str(e)}")
                        
            # Check for potential issues with date logic
            date_columns = [col for col in filtered_columns if 'date' in col.lower() or 'from' in col.lower() or 'to' in col.lower()]
            
            for col in date_columns:
                try:
                    # Check if column type is date or timestamp
                    col_type = conn.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name = '{output_table_name}' AND column_name = '{col}'").fetchone()[0]
                    
                    if any(date_type in col_type.upper() for date_type in ['DATE', 'TIMESTAMP']):
                        # Check for default dates
                        default_count = conn.execute(f"SELECT COUNT(*) FROM {output_table_name} WHERE {col} = '2000-01-01'").fetchone()[0]
                        if default_count > 0:
                            logging.info(f"Column '{col}' has {default_count:,} default dates ('2000-01-01')")
                        
                        # Check date range
                        min_date = conn.execute(f"SELECT MIN({col}) FROM {output_table_name}").fetchone()[0]
                        max_date = conn.execute(f"SELECT MAX({col}) FROM {output_table_name}").fetchone()[0]
                        logging.info(f"Column '{col}' date range: {min_date} to {max_date}")
                except Exception as e:
                    logging.warning(f"Could not analyze date distribution for column '{col}': {str(e)}")
        
        except Exception as e:
            logging.error(f"Error logging table statistics: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()

    def analyze_column_types(self, db_path: str, table_name: str) -> Dict[str, int]:
        """
        Analyze column types in a table to identify problematic columns.
        
        Args:
            db_path: Path to the database file
            table_name: Name of the table to analyze
            
        Returns:
            Dictionary with counts of each column type
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            columns = conn.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """).fetchall()
            
            type_counts = {}
            for _, col_type in columns:
                normalized_type = col_type.upper().split('(')[0]  # Remove size specifiers like VARCHAR(255)
                type_counts[normalized_type] = type_counts.get(normalized_type, 0) + 1
                
            return type_counts
        except Exception as e:
            logging.error(f"Error analyzing column types: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()


def main():
    """Main function to run the preprocessing."""
    
    # Hardcoded default parameters
    DEFAULT_PARAMS = {
        'db_path': 'data/duckdb/claims_data.duckdb',
        'table_names': ['joined_1_4_5_6_7', 'joined_1_8_9_10_11', 'drugs', 'inpatient_cases', 'outpatient_cases'],
        'output_prefix': 'preprocessed_',
        'max_nulls': 5
    }
    
    # Only parse arguments if they were provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Preprocess health claims data for privacy metrics')
        parser.add_argument('--db_path', help=f'Path to the DuckDB database (default: {DEFAULT_PARAMS["db_path"]})')
        parser.add_argument('--table_name', help='Name of a specific table to preprocess (overrides default tables)')
        parser.add_argument('--output_prefix', help=f'Prefix for output tables (default: {DEFAULT_PARAMS["output_prefix"]})')
        parser.add_argument('--max_nulls', type=int, 
                            help=f'Maximum number of NULL values allowed per row (default: {DEFAULT_PARAMS["max_nulls"]})')
        parser.add_argument('--all_tables', action='store_true', help='Process all tables in the database')
        
        args = parser.parse_args()
        
        # Use provided arguments, falling back to defaults when needed
        db_path = args.db_path or DEFAULT_PARAMS['db_path']
        output_prefix = args.output_prefix or DEFAULT_PARAMS['output_prefix']
        max_nulls = args.max_nulls if args.max_nulls is not None else DEFAULT_PARAMS['max_nulls']
        
        # Determine which tables to process
        if args.all_tables:
            # Initialize DuckDBManager to get all tables
            db_manager = DuckDBManager()
            tables = db_manager.execute_query(db_path, "SHOW TABLES")
            table_names = [table[0] for table in tables if not table[0].startswith('preprocessed_')]
        elif args.table_name:
            table_names = [args.table_name]
        else:
            table_names = DEFAULT_PARAMS['table_names']
    else:
        # Use all default parameters if no arguments provided
        logging.info("No command-line arguments provided. Using default parameters.")
        db_path = DEFAULT_PARAMS['db_path']
        table_names = DEFAULT_PARAMS['table_names']
        output_prefix = DEFAULT_PARAMS['output_prefix']
        max_nulls = DEFAULT_PARAMS['max_nulls']
    
    # Initialize preprocessor and run preprocessing for each table
    preprocessor = MedicalDataPreprocessor()
    
    logging.info(f"Processing with parameters:")
    logging.info(f"  Database: {db_path}")
    logging.info(f"  Tables: {', '.join(table_names)}")
    logging.info(f"  Output prefix: {output_prefix}")
    logging.info(f"  Max nulls: {max_nulls}")
    
    result_tables = []
    for table_name in table_names:
        logging.info(f"\n{'='*80}\nProcessing table: {table_name}\n{'='*80}")
        output_table = preprocessor.preprocess_table(
            db_path, 
            table_name, 
            f"{output_prefix}{table_name}",
            max_nulls
        )
        
        if output_table:
            result_tables.append(output_table)
            print(f"✅ Successfully preprocessed table: {table_name} → {output_table}")
        else:
            print(f"❌ Failed to preprocess table: {table_name}")
    
    print("\nPreprocessing Summary:")
    print("-" * 70)
    print(f"Successfully preprocessed {len(result_tables)} out of {len(table_names)} tables")
    for table in result_tables:
        print(f"  - {table}")
    print("-" * 70)
    print("The preprocessed tables are ready for further analysis and privacy metric calculations.")

if __name__ == "__main__":
    main()