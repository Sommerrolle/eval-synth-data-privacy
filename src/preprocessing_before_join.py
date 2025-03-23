import argparse
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import duckdb
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preprocessing.log'),
        logging.StreamHandler()
    ]
)

class DuckDBManager:
    """Simple version of DuckDBManager for standalone use."""
    
    def __init__(self, duckdb_dir: str = 'data/duckdb'):
        """Initialize the DuckDBManager."""
        self.duckdb_dir = duckdb_dir
    
    def get_database_path(self, db_name: str) -> str:
        """Get the full path to a database file."""
        return os.path.join(self.duckdb_dir, db_name)
    
    def execute_query(self, db_path: str, query: str):
        """Execute a SQL query on a database."""
        try:
            conn = duckdb.connect(db_path, read_only=False)
            result = conn.execute(query).fetchall()
            conn.close()
            return result
        except Exception as e:
            logging.error(f"Error executing query on {db_path}: {str(e)}")
            logging.error(f"Query: {query}")
            return None
    
    def get_table_count(self, db_path: str, table_name: str) -> int:
        """Get the number of rows in a table."""
        try:
            conn = duckdb.connect(db_path, read_only=True)
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logging.error(f"Error counting rows in {table_name} from {db_path}: {str(e)}")
            return 0
    
    def get_table_column_info(self, db_path: str, table_name: str) -> List[Dict]:
        """Get information about columns in a table."""
        try:
            conn = duckdb.connect(db_path, read_only=True)
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            conn.close()
            return [{"name": col[0], "type": col[1]} for col in columns]
        except Exception as e:
            logging.error(f"Error getting column info for {table_name} from {db_path}: {str(e)}")
            return []
    
    def get_all_tables(self, db_path: str) -> List[str]:
        """Get a list of all tables in the database."""
        try:
            conn = duckdb.connect(db_path, read_only=True)
            tables = conn.execute("SHOW TABLES").fetchall()
            conn.close()
            return [table[0] for table in tables]
        except Exception as e:
            logging.error(f"Error getting tables from {db_path}: {str(e)}")
            return []

class HealthClaimsPreprocessor:
    """Preprocess health claims data by handling missing values in individual tables."""
    
    def __init__(self, db_path: str, output_prefix: str = "clean_"):
        """
        Initialize the preprocessor.
        
        Args:
            db_path: Path to the DuckDB database
            output_prefix: Prefix for cleaned tables
        """
        self.db_path = db_path
        self.output_prefix = output_prefix
        self.db_manager = DuckDBManager()
        
        # Define the order of processing (core tables first)
        self.table_order = [
            'insurants',           # Patient data (center of star schema)
            'insurance_data',      # Insurance coverage info
            'inpatient_cases',     # Inpatient visits
            'outpatient_cases',    # Outpatient visits
            'inpatient_diagnosis', # Inpatient diagnoses
            'outpatient_diagnosis', # Outpatient diagnoses
            'inpatient_procedures', # Inpatient procedures
            'outpatient_procedures', # Outpatient procedures
            'inpatient_fees',      # Inpatient billing
            'outpatient_fees',     # Outpatient billing
            'drugs'                # Medication data
        ]
    
    def preprocess_all_tables(self):
        """
        Preprocess all tables in the database in the specified order.
        
        Returns:
            Number of successfully preprocessed tables
        """
        logging.info(f"Starting preprocessing for database: {self.db_path}")
        
        # Discover all tables in the database
        conn = duckdb.connect(self.db_path, read_only=True)
        all_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        conn.close()
        
        # Filter to only include tables in our predefined order
        found_tables = []
        missing_tables = []
        
        for table in self.table_order:
            if table in all_tables:
                found_tables.append(table)
            else:
                missing_tables.append(table)
        
        # Check for tables in the database that aren't in our predefined list
        unknown_tables = [table for table in all_tables if table not in self.table_order and not table.startswith(self.output_prefix)]
        
        if unknown_tables:
            error_msg = f"Found tables in the database that aren't in the predefined list: {', '.join(unknown_tables)}"
            #logging.error(error_msg)
            logging.warning(error_msg)
            #raise ValueError(error_msg)
        
        if missing_tables:
            logging.warning(f"The following tables were not found in the database: {', '.join(missing_tables)}")
            sys.exit(1)
        
        logging.info(f"Found {len(found_tables)} tables to process: {', '.join(found_tables)}")
        
        # Process each table
        success_count = 0
        for table_name in found_tables:
            logging.info(f"\n{'='*80}\nProcessing table: {table_name}\n{'='*80}")
            try:
                output_table = self.preprocess_table(table_name)
                if output_table:
                    success_count += 1
                    print(f"Successfully preprocessed table: {table_name} → {output_table}")
                else:
                    print(f"Failed to preprocess table: {table_name}")
            except Exception as e:
                logging.error(f"Error preprocessing table {table_name}: {str(e)}")
                print(f"Error preprocessing table: {table_name}")
        
        # Print summary
        print("\nPreprocessing Summary:")
        print("-" * 70)
        print(f"Successfully preprocessed {success_count} out of {len(found_tables)} tables")
        print("-" * 70)

        return success_count
    
    def preprocess_table(self, table_name: str) -> str:
        """
        Preprocess a specific table using domain-specific knowledge.
        
        Args:
            table_name: Name of the table to preprocess
            
        Returns:
            Name of the preprocessed table
        """
        output_table = f"{self.output_prefix}{table_name}"
        
        # Get column information
        columns_info = self.db_manager.get_table_column_info(self.db_path, table_name)
        if not columns_info:
            logging.error(f"Failed to get column information for table '{table_name}'")
            return None
        
        # Count nulls in original table
        self._log_null_counts(table_name, [col['name'] for col in columns_info])
        
        # Generate imputation SQL based on table type
        imputation_sql = self._generate_imputation_sql(table_name, columns_info)
        
        # Create the preprocessed table
        create_table_sql = f"""
            CREATE OR REPLACE TABLE {output_table} AS
            {imputation_sql}
        """
        
        try:
            # Execute the query
            self.db_manager.execute_query(self.db_path, create_table_sql)
            
            # Verify no nulls remain
            self._verify_no_nulls(output_table, [col['name'] for col in columns_info])
            
            # Log statistics
            self._log_table_statistics(table_name, output_table)
            
            return output_table
        
        except Exception as e:
            logging.error(f"Error creating preprocessed table: {str(e)}")
            
            # Try fallback approach with column-by-column updates
            try:
                logging.info(f"Attempting fallback approach for {table_name}")
                return self._preprocess_table_fallback(table_name, columns_info)
            except Exception as fallback_e:
                logging.error(f"Fallback approach also failed: {str(fallback_e)}")
                return None
    
    def _preprocess_table_fallback(self, table_name: str, columns_info: List[Dict]) -> str:
        """
        Fallback approach for preprocessing: copy table first, then update columns one by one.
        
        Args:
            table_name: Name of the table to preprocess
            columns_info: Column information for the table
            
        Returns:
            Name of the preprocessed table
        """
        output_table = f"{self.output_prefix}{table_name}"
        
        # Copy table structure and data
        self.db_manager.execute_query(self.db_path, f"DROP TABLE IF EXISTS {output_table}")
        self.db_manager.execute_query(self.db_path, f"CREATE TABLE {output_table} AS SELECT * FROM {table_name}")
        
        # Update each column one by one
        for col_info in columns_info:
            col_name = col_info['name']
            col_type = col_info['type']
            
            # Skip columns with no nulls
            null_count = self.db_manager.execute_query(
                self.db_path, 
                f"SELECT COUNT(*) FROM {output_table} WHERE {col_name} IS NULL"
            )
            
            if null_count and null_count[0][0] > 0:
                # Get imputation value based on column and table type
                imputation_expr = self._get_imputation_value(table_name, col_name, col_type)
                
                # Update the column
                update_sql = f"UPDATE {output_table} SET {col_name} = {imputation_expr} WHERE {col_name} IS NULL"
                self.db_manager.execute_query(self.db_path, update_sql)
        
        # Verify no nulls remain
        self._verify_no_nulls(output_table, [col['name'] for col in columns_info])
        
        return output_table
    
    def _generate_imputation_sql(self, table_name: str, columns_info: List[Dict]) -> str:
        """
        Generate the SQL for imputing missing values in a table.
        
        Args:
            table_name: Name of the table
            columns_info: Information about the columns in the table
            
        Returns:
            SQL query string for imputation
        """
        # List of column expressions with COALESCE
        column_expressions = []
        
        for col_info in columns_info:
            col_name = col_info['name']
            col_type = col_info['type']
            
            # Generate appropriate imputation for this column in this table
            if table_name == 'insurants':
                column_expressions.append(self._impute_insurants_column(col_name, col_type))
            elif table_name == 'insurance_data':
                column_expressions.append(self._impute_insurance_data_column(col_name, col_type))
            elif table_name == 'drugs':
                column_expressions.append(self._impute_drugs_column(col_name, col_type))
            elif 'inpatient_cases' in table_name:
                column_expressions.append(self._impute_inpatient_cases_column(col_name, col_type))
            elif 'outpatient_cases' in table_name:
                column_expressions.append(self._impute_outpatient_cases_column(col_name, col_type))
            elif 'inpatient_diagnosis' in table_name:
                column_expressions.append(self._impute_inpatient_diagnosis_column(col_name, col_type))
            elif 'outpatient_diagnosis' in table_name:
                column_expressions.append(self._impute_outpatient_diagnosis_column(col_name, col_type))
            elif 'inpatient_procedures' in table_name:
                column_expressions.append(self._impute_inpatient_procedures_column(col_name, col_type))
            elif 'outpatient_procedures' in table_name:
                column_expressions.append(self._impute_outpatient_procedures_column(col_name, col_type))
            elif 'inpatient_fees' in table_name:
                column_expressions.append(self._impute_inpatient_fees_column(col_name, col_type))
            elif 'outpatient_fees' in table_name:
                column_expressions.append(self._impute_outpatient_fees_column(col_name, col_type))
            else:
                # Generic handling for other tables
                column_expressions.append(self._impute_generic_column(col_name, col_type, table_name))
        
        # Construct the full SELECT statement
        return f"SELECT {', '.join(column_expressions)} FROM {table_name}"
    
    def _get_imputation_value(self, table_name: str, col_name: str, col_type: str) -> str:
        """
        Get the appropriate imputation value for a column in fallback mode.
        
        Args:
            table_name: Name of the table
            col_name: Name of the column
            col_type: Type of the column
            
        Returns:
            SQL expression for the imputation value
        """
        col_type_upper = col_type.upper()
        
        # Handle different data types with appropriate defaults
        if 'INT' in col_type_upper or 'BIGINT' in col_type_upper:
            return "0"
        elif 'DOUBLE' in col_type_upper or 'FLOAT' in col_type_upper:
            return "0.0"
        elif 'DATE' in col_type_upper or 'TIMESTAMP' in col_type_upper:
            return "'2510-01-01'"
        elif 'VARCHAR' in col_type_upper or 'CHAR' in col_type_upper or 'TEXT' in col_type_upper:
            # Special handling for specific columns
            if 'diagnosis' in col_name.lower():
                return "'UNKNOWN'"
            elif 'procedure_code' in col_name.lower():
                return "'UNKNOWN'"
            elif 'pharma_central_number' in col_name.lower():
                return "'00000000'"
            elif any(code in col_name.lower() for code in ['physician_code', 'physican_code', 'practice_code']):
                return "'000000000'"
            elif 'specialty' in col_name.lower():
                return "'00'"
            elif 'department' in col_name.lower():
                return "'0000'"
            else:
                return "'UNKNOWN'"
        else:
            return "NULL"  # Fallback, shouldn't be reached if verify_no_nulls is working
    
    #
    # Table-specific imputation methods
    #
    
    def _impute_insurants_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for insurants table columns."""
        col_lower = col_name.lower()
        
        if 'year_of_birth' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM insurants WHERE {col_name} IS NOT NULL),
                    1970
                ) AS {col_name}
            """
        elif 'gender' in col_lower:
                # COALESCE({col_name}, 
                #     (SELECT {col_name} FROM insurants 
                #      WHERE {col_name} IS NOT NULL 
                #      GROUP BY {col_name} 
                #      ORDER BY COUNT(*) DESC LIMIT 1),
                #     1
                # ) AS {col_name}
            # use the most frequent gender
            return f"""
                COALESCE({col_name}, 
                    (SELECT MODE({col_name}) FROM insurants WHERE {col_name} IS NOT NULL),
                    1
                ) AS {col_name}
            """
        else:
            # For other columns, use generic approach
            return self._impute_generic_column(col_name, col_type, 'insurants')
    
    def _impute_insurance_data_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for insurance_data table columns."""
        col_lower = col_name.lower()
    
        if 'from' in col_lower:
            # For 'from' column: If we can't find values among patient's entries, 
            # mark the row for removal with a condition
            return f"""
                CASE 
                    WHEN {col_name} IS NULL AND 
                        NOT EXISTS (SELECT 1 FROM insurance_data 
                                    WHERE pid = insurance_data.pid AND insurance_data_to IS NOT NULL)
                    THEN NULL  -- This row will be filtered out later
                    ELSE COALESCE({col_name},
                        (SELECT MAX(insurance_data_to) + INTERVAL '1 day'
                        FROM insurance_data
                        WHERE pid = insurance_data.pid AND insurance_data_to IS NOT NULL)
                    )
                END AS {col_name}
            """
        elif 'to' in col_lower:
            # For 'to' column: Similar approach, mark for removal if we can't find values
            return f"""
                CASE 
                    WHEN {col_name} IS NULL AND 
                        NOT EXISTS (SELECT 1 FROM insurance_data 
                                    WHERE pid = insurance_data.pid AND insurance_data_from IS NOT NULL)
                    THEN NULL  -- This row will be filtered out later
                    ELSE COALESCE({col_name},
                        (SELECT MAX(insurance_data_to) + INTERVAL '1 year'
                        FROM insurance_data
                        WHERE pid = insurance_data.pid AND insurance_data_from IS NOT NULL)
                    )
                END AS {col_name}
            """
        elif 'death' in col_lower:
            # For death column: Keep earliest row with death=1 and assume alive (0) otherwise
            # TODO no filtering of earlierst death, codes not working, check back if its necessay
            return f"""
                CASE
                    WHEN {col_name} IS NULL THEN 0  -- Patient is alive if no death info
                  --  WHEN {col_name} = 1 AND 
                  --      insurance_data_from > (SELECT MIN(insurance_data_from) FROM insurance_data 
                  --                          WHERE pid = insurance_data.pid AND {col_name} = 1)
                  --  THEN NULL  -- Mark duplicate death records for removal (keep only earliest)
                    ELSE {col_name}
                END AS {col_name}
            """
        elif 'regional_code' in col_lower:
            # For regional_code: First try patient's most frequent value, then random value
            return f"""
                COALESCE({col_name},
                    (SELECT mode() WITHIN GROUP (ORDER BY {col_name}) 
                    FROM insurance_data
                    WHERE pid = insurance_data.pid AND {col_name} IS NOT NULL),
                    (SELECT {col_name} FROM insurance_data 
                    WHERE {col_name} IS NOT NULL 
                    ORDER BY random() LIMIT 1)
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'insurance_data')
    
    def _impute_drugs_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for drugs table columns."""
        col_lower = col_name.lower()
        
        if 'date_of_prescription' in col_lower:
            return f"""
                COALESCE({col_name}, DATE '2510-01-01') AS {col_name}
            """
        elif 'date_of_dispense' in col_lower:
            return f"""
                COALESCE({col_name}, DATE '2510-01-01') AS {col_name}
            """
        elif 'pharma_central_number' in col_lower:
            return f"""
                COALESCE({col_name}, '00000000') AS {col_name}
            """
        elif 'specialty_of_prescriber' in col_lower:
            return f"""
                COALESCE({col_name}, '00') AS {col_name}
            """
        elif 'physican_code' in col_lower:
            return f"""
                COALESCE({col_name}, '000000000') AS {col_name}
            """
        elif 'practice_code' in col_lower:
            return f"""
                COALESCE({col_name}, '000000000') AS {col_name}
            """
        elif 'quantity' in col_lower:
            return f"""
                COALESCE({col_name}, 0.0) AS {col_name}
            """
        elif 'amount_due' in col_lower:
            return f"""
                COALESCE({col_name}, 0.0) AS {col_name}
            """
        elif 'atc' in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'ddd' in col_lower:
            return f"""
                COALESCE({col_name}, 0.0) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'drugs')
    
    def _impute_inpatient_cases_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for inpatient_cases table columns."""
        col_lower = col_name.lower()
        
        if 'date_of_admission' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT inpatient_cases_date_of_discharge - INTERVAL '5 days' 
                     FROM inpatient_cases ic
                     WHERE ic.inpatient_caseID = inpatient_cases.inpatient_caseID 
                       AND ic.inpatient_cases_date_of_discharge IS NOT NULL 
                     LIMIT 1)
                ) AS {col_name}
            """
        elif 'date_of_discharge' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT inpatient_cases_date_of_admission + INTERVAL '5 days' 
                     FROM inpatient_cases ic
                     WHERE ic.inpatient_caseID = inpatient_cases.inpatient_caseID 
                       AND ic.inpatient_cases_date_of_admission IS NOT NULL 
                     LIMIT 1)
                ) AS {col_name}
            """
        elif 'cause_of_admission' in col_lower:
            return f"""
                COALESCE({col_name}, '0000') AS {col_name}
            """
        elif 'cause_of_discharge' in col_lower:
            return f"""
                COALESCE({col_name}, '00') AS {col_name}
            """
        elif 'outpatient_treatment' in col_lower:
            return f"""
                COALESCE({col_name}, 0) AS {col_name}
            """
        elif 'department_admission' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT inpatient_cases_department_discharge
                     FROM inpatient_cases ic
                     WHERE ic.inpatient_caseID = inpatient_cases.inpatient_caseID 
                       AND ic.inpatient_cases_department_discharge IS NOT NULL 
                     LIMIT 1),
                    '0000'
                ) AS {col_name}
            """
        elif 'department_discharge' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT inpatient_cases_department_admission
                     FROM inpatient_cases ic
                     WHERE ic.inpatient_caseID = inpatient_cases.inpatient_caseID 
                       AND ic.inpatient_cases_department_admission IS NOT NULL 
                     LIMIT 1),
                    '0000'
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'inpatient_cases')
    
    def _impute_outpatient_cases_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for outpatient_cases table columns."""
        col_lower = col_name.lower()
        
        if 'practice_code' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT oc.{col_name}
                    FROM outpatient_cases oc
                    WHERE oc.pid = outpatient_cases.pid
                    AND oc.outpatient_caseID = outpatient_cases.outpatient_caseID
                    AND oc.{col_name} IS NOT NULL
                    LIMIT 1),
                    (SELECT oc.{col_name}
                    FROM outpatient_cases oc
                    WHERE oc.pid = outpatient_cases.pid
                    AND oc.{col_name} IS NOT NULL
                    LIMIT 1),
                    '000000000'
                ) AS {col_name}
            """
        elif 'from' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT outpatient_cases_to - INTERVAL '14 days' 
                     FROM outpatient_cases oc
                     WHERE oc.outpatient_caseID = outpatient_cases.outpatient_caseID 
                       AND oc.outpatient_cases_to IS NOT NULL 
                     LIMIT 1)
                ) AS {col_name}
            """
        elif 'to' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT outpatient_cases_from + INTERVAL '14 days' 
                     FROM outpatient_cases oc
                     WHERE oc.outpatient_caseID = outpatient_cases.outpatient_caseID 
                       AND oc.outpatient_cases_from IS NOT NULL 
                     LIMIT 1)
                ) AS {col_name}
            """
        elif 'amount_due' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM outpatient_cases WHERE {col_name} IS NOT NULL),
                    0.0
                ) AS {col_name}
            """
        elif 'year' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT EXTRACT(YEAR FROM outpatient_cases_from) 
                     FROM outpatient_cases oc
                     WHERE oc.outpatient_caseID = outpatient_cases.outpatient_caseID 
                       AND oc.outpatient_cases_from IS NOT NULL 
                     LIMIT 1)
                ) AS {col_name}
            """
        elif 'quarter' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT CEIL(EXTRACT(MONTH FROM outpatient_cases_from) / 3) 
                     FROM outpatient_cases oc
                     WHERE oc.outpatient_caseID = outpatient_cases.outpatient_caseID 
                       AND oc.outpatient_cases_from IS NOT NULL 
                     LIMIT 1),
                    (1 + CAST(RANDOM() * 3 AS INTEGER))  -- Random number between 1-4
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'outpatient_cases')
    
    def _impute_inpatient_diagnosis_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for inpatient_diagnosis table columns."""
        col_lower = col_name.lower()
        
        if 'diagnosis' in col_lower and 'diagnosis_' not in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'type_of_diagnosis' in col_lower:
            return f"""
                COALESCE({col_name}, '00') AS {col_name}
            """
        elif 'is_main_diagnosis' in col_lower:
            return f"""
                COALESCE({col_name}, 0) AS {col_name}
            """
        elif 'localisation' in col_lower:
            return f"""
                COALESCE({col_name}, 0) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'inpatient_diagnosis')
    
    def _impute_outpatient_diagnosis_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for outpatient_diagnosis table columns."""
        col_lower = col_name.lower()
        
        if 'diagnosis' in col_lower and 'diagnosis_' not in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'qualification' in col_lower:
            return f"""
                COALESCE({col_name}, 'U') AS {col_name}
            """
        elif 'localisation' in col_lower:
            return f"""
                COALESCE({col_name}, 0) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'outpatient_diagnosis')
    
    def _impute_inpatient_procedures_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for inpatient_procedures table columns."""
        col_lower = col_name.lower()
        
        if 'procedure_code' in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'localisation' in col_lower:
            return f"""
                COALESCE({col_name}, 0) AS {col_name}
            """
        elif 'date_of_procedure' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT inpatient_cases_date_of_admission + INTERVAL '1 day' 
                     FROM inpatient_cases ic
                     WHERE ic.inpatient_caseID = inpatient_procedures.inpatient_caseID 
                       AND ic.inpatient_cases_date_of_admission IS NOT NULL 
                     LIMIT 1)
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'inpatient_procedures')
    
    def _impute_outpatient_procedures_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for outpatient_procedures table columns."""
        col_lower = col_name.lower()
        
        if 'procedure_code' in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'localisation' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT op.outpatient_procedures_localisation
                    FROM outpatient_procedures op
                    WHERE op.pid = outpatient_procedures.pid
                        AND op.outpatient_procedures_localisation IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT 1),
                    0
                ) AS {col_name}
            """
        elif 'date_of_procedure' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT outpatient_cases_from
                    FROM outpatient_cases oc
                    WHERE oc.outpatient_caseID = outpatient_procedures.outpatient_caseID
                    AND oc.outpatient_cases_from IS NOT NULL
                    LIMIT 1)
                ) AS {col_name}
            """
        elif 'specialty_code' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT op.outpatient_procedures_specialty_code
                    FROM outpatient_procedures op
                    WHERE op.pid = outpatient_procedures.pid
                        AND op.outpatient_procedures_procedure_code = outpatient_procedures.outpatient_procedures_procedure_code
                        AND op.outpatient_procedures_specialty_code IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT 1),
                    '00'
                ) AS {col_name}
            """
        elif 'physician_code' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT op.outpatient_procedures_physician_code
                    FROM outpatient_procedures op
                    WHERE op.pid = outpatient_procedures.pid
                        AND op.outpatient_procedures_procedure_code = outpatient_procedures.outpatient_procedures_procedure_code
                        AND op.outpatient_procedures_physician_code IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT 1),
                    '000000000'
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'outpatient_procedures')
    
    def _impute_inpatient_fees_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for inpatient_fees table columns."""
        col_lower = col_name.lower()
        
        if 'from' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT inpatient_cases_date_of_admission
                    FROM inpatient_cases ic
                    WHERE ic.inpatient_caseID = inpatient_fees.inpatient_caseID
                    AND ic.inpatient_cases_date_of_admission IS NOT NULL
                    LIMIT 1)
                ) AS {col_name}
            """
        elif 'to' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT inpatient_cases_date_of_discharge
                    FROM inpatient_cases ic
                    WHERE ic.inpatient_caseID = inpatient_fees.inpatient_caseID
                    AND ic.inpatient_cases_date_of_discharge IS NOT NULL
                    LIMIT 1)
                ) AS {col_name}
            """
        elif 'billing_code' in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'amount_due' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM inpatient_fees WHERE {col_name} IS NOT NULL),
                    0.0
                ) AS {col_name}
            """
        elif 'quantity' in col_lower:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM inpatient_fees WHERE {col_name} IS NOT NULL),
                    1.0
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'inpatient_fees')
    
    def _impute_outpatient_fees_column(self, col_name: str, col_type: str) -> str:
        """Generate imputation for outpatient_fees table columns."""
        col_lower = col_name.lower()
        
        if 'physician_code' in col_lower:
            return f"""
                COALESCE({col_name}, '000000000') AS {col_name}
            """
        elif 'specialty_code' in col_lower:
            return f"""
                COALESCE({col_name}, '00') AS {col_name}
            """
        elif 'billing_code' in col_lower:
            return f"""
                COALESCE({col_name}, 'UNKNOWN') AS {col_name}
            """
        elif 'quantity' in col_lower:
            return f"""
                COALESCE({col_name}, 1) AS {col_name}
            """
        elif 'date' in col_lower:
            return f"""
                COALESCE({col_name},
                    (SELECT outpatient_cases_from
                    FROM outpatient_cases oc
                    WHERE oc.outpatient_caseID = outpatient_fees.outpatient_caseID
                    AND oc.outpatient_cases_from IS NOT NULL
                    LIMIT 1),
                    DATE '2510-01-01'
                ) AS {col_name}
            """
        else:
            return self._impute_generic_column(col_name, col_type, 'outpatient_fees')
    
    def _impute_generic_column(self, col_name: str, col_type: str, table_name: str) -> str:
        """
        Generic imputation for any column based on data type.
        
        Args:
            col_name: Name of the column
            col_type: Data type of the column
            table_name: Name of the table
            
        Returns:
            SQL expression for imputation
        """
        col_type_upper = col_type.upper()
        
        if 'INT' in col_type_upper or 'BIGINT' in col_type_upper:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
                    0
                ) AS {col_name}
            """
        elif 'DOUBLE' in col_type_upper or 'FLOAT' in col_type_upper:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
                    0.0
                ) AS {col_name}
            """
        elif 'DATE' in col_type_upper or 'TIMESTAMP' in col_type_upper:
            return f"""
                COALESCE({col_name}, 
                    (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
                    DATE '2000-01-01'
                ) AS {col_name}
            """
        elif 'VARCHAR' in col_type_upper or 'CHAR' in col_type_upper or 'TEXT' in col_type_upper:
            # For low-cardinality categorical columns, use mode
            return f"""
                COALESCE({col_name}, 
                    (SELECT {col_name} FROM {table_name} 
                     WHERE {col_name} IS NOT NULL 
                     GROUP BY {col_name} 
                     ORDER BY COUNT(*) DESC 
                     LIMIT 1),
                    'UNKNOWN'
                ) AS {col_name}
            """
        else:
            # For any other type, just pass through the column
            return f"{col_name} AS {col_name}"
    
    def _log_null_counts(self, table_name: str, columns: List[str]) -> None:
        """
        Log the number of NULL values in each column of a table.
        
        Args:
            table_name: Name of the table
            columns: List of column names
        """
        logging.info(f"NULL counts in original table '{table_name}':")
        
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            for col in columns:
                null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
                if null_count > 0:
                    total_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    percent = (null_count / total_count) * 100 if total_count > 0 else 0
                    logging.info(f"  - {col}: {null_count:,} NULLs ({percent:.2f}%)")
        except Exception as e:
            logging.error(f"Error checking NULL counts: {str(e)}")
        finally:
            conn.close()
    
    def _verify_no_nulls(self, table_name: str, columns: List[str]) -> None:
        """
        Verify that no NULL values remain in the preprocessed table.
        
        Args:
            table_name: Name of the preprocessed table
            columns: List of column names to check
        """
        # conn = duckdb.connect(self.db_path, read_only=True)
        # try:
        #     for col in columns:
        #         null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
        #         if null_count > 0:
        #             logging.warning(f"Column '{col}' in '{table_name}' still has {null_count} NULL values!")
                    
        #             # Emergency fix - apply basic imputation based on type
        #             col_type = conn.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{col}'").fetchone()[0]
                    
        #             conn.close()  # Close to reopen in write mode
        #             conn = duckdb.connect(self.db_path, read_only=False)
                    
        #             if 'INT' in col_type.upper() or 'BIGINT' in col_type.upper():
        #                 conn.execute(f"UPDATE {table_name} SET {col} = 0 WHERE {col} IS NULL")
        #             elif 'DOUBLE' in col_type.upper() or 'FLOAT' in col_type.upper():
        #                 conn.execute(f"UPDATE {table_name} SET {col} = 0.0 WHERE {col} IS NULL")
        #             elif 'DATE' in col_type.upper() or 'TIMESTAMP' in col_type.upper():
        #                 conn.execute(f"UPDATE {table_name} SET {col} = '2000-01-01' WHERE {col} IS NULL")
        #             else:
        #                 conn.execute(f"UPDATE {table_name} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")
                    
        #             # Verify the fix worked
        #             conn.close()  # Close to reopen in read mode
        #             conn = duckdb.connect(self.db_path, read_only=True)
                    
        #             null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
        #             if null_count > 0:
        #                 logging.error(f"Emergency fix failed! Column '{col}' still has {null_count} NULL values.")
        #             else:
        #                 logging.info(f"Emergency fix successful for column '{col}'")
        
        # except Exception as e:
        #     logging.error(f"Error verifying NULL values: {str(e)}")
        # finally:
        #     conn.close()
            
        # Final verification
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            # This creates a single row with one column per column in your table
            # Each column contains the count of NULL values for that column
            null_counts_per_column = conn.execute(f"""
                SELECT {', '.join([f'SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_{col}' for col in columns])}
                FROM {table_name}
            """).fetchone()
            
            # Sum up all the NULL counts across all columns
            total_nulls = sum(null_counts_per_column)
            
            if total_nulls == 0:
                logging.info(f"Verification complete: No NULL values in table '{table_name}'")
            else:
                # Log which columns still have NULLs
                for i, col in enumerate(columns):
                    if null_counts_per_column[i] > 0:
                        logging.warning(f"Column '{col}' has {null_counts_per_column[i]} NULL values")
                logging.error(f"Verification failed: {total_nulls} total NULL values remain in '{table_name}'")
        except Exception as e:
            logging.error(f"Error in final NULL verification: {str(e)}")
        finally:
            conn.close()
    
    def _log_table_statistics(self, original_table: str, processed_table: str) -> None:
        """
        Log statistics about the preprocessing operation.
        
        Args:
            original_table: Name of the original table
            processed_table: Name of the preprocessed table
        """
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            # Get row counts
            orig_count = conn.execute(f"SELECT COUNT(*) FROM {original_table}").fetchone()[0]
            proc_count = conn.execute(f"SELECT COUNT(*) FROM {processed_table}").fetchone()[0]
            
            logging.info(f"Original table '{original_table}': {orig_count:,} rows")
            logging.info(f"Processed table '{processed_table}': {proc_count:,} rows")
            
            # Get column counts
            orig_cols = conn.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{original_table}'").fetchone()[0]
            proc_cols = conn.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{processed_table}'").fetchone()[0]
            
            logging.info(f"Original table '{original_table}': {orig_cols} columns")
            logging.info(f"Processed table '{processed_table}': {proc_cols} columns")
            
            # Check for important medical columns and log their statistics
            important_cols = ['diagnosis', 'procedure_code', 'date_of_admission', 'date_of_discharge', 
                            'from', 'to', 'amount_due', 'quantity', 'year_of_birth', 'gender']
            
            for col_pattern in important_cols:
                cols = conn.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{processed_table}' 
                      AND column_name LIKE '%{col_pattern}%'
                """).fetchall()
                
                for col in cols:
                    col_name = col[0]
                    try:
                        distinct_count = conn.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {processed_table}").fetchone()[0]
                        total_count = conn.execute(f"SELECT COUNT(*) FROM {processed_table}").fetchone()[0]
                        
                        if distinct_count < 100:  # Only show distribution for low-cardinality columns
                            top_values = conn.execute(f"""
                                SELECT {col_name}, COUNT(*) AS count
                                FROM {processed_table}
                                GROUP BY {col_name}
                                ORDER BY count DESC
                                LIMIT 5
                            """).fetchall()
                            
                            value_info = ", ".join([f"'{val}': {cnt:,}" for val, cnt in top_values])
                            logging.info(f"Column '{col_name}': {distinct_count:,} distinct values out of {total_count:,} rows. Top values: {value_info}")
                        else:
                            logging.info(f"Column '{col_name}': {distinct_count:,} distinct values out of {total_count:,} rows")
                            
                            # For date columns, show range
                            if any(date_type in col_name.lower() for date_type in ['date', 'from', 'to']):
                                try:
                                    min_date = conn.execute(f"SELECT MIN({col_name}) FROM {processed_table}").fetchone()[0]
                                    max_date = conn.execute(f"SELECT MAX({col_name}) FROM {processed_table}").fetchone()[0]
                                    logging.info(f"  - Date range: {min_date} to {max_date}")
                                except:
                                    pass  # Not a date column
                    except Exception as inner_e:
                        logging.warning(f"Could not get statistics for column '{col_name}': {str(inner_e)}")
        
        except Exception as e:
            logging.error(f"Error logging table statistics: {str(e)}")
        finally:
            conn.close()


    def move_preprocessed_tables_to_new_db(self, keep_original: bool = False) -> str:
        """
        Create a new database with a "_preprocessed" suffix and move all preprocessed 
        tables to this new database using SQL ATTACH statement.
        
        Args:
            keep_original: If True, keep the preprocessed tables in the original database
                        If False, drop the preprocessed tables from the original database
        
        Returns:
            Path to the new database
        """
        # Determine the new database name
        original_db_path = self.db_path
        original_db_name = os.path.basename(original_db_path)
        original_db_dir = os.path.dirname(original_db_path)
        
        # Strip .duckdb extension if present
        if original_db_name.endswith('.duckdb'):
            base_name = original_db_name[:-7]
        else:
            base_name = original_db_name
        
        # Create new database name
        new_db_name = f"{base_name}_preprocessed.duckdb"
        new_db_path = os.path.join(original_db_dir, new_db_name)

        # Delete the database if it already exists
        if os.path.exists(new_db_path):
            try:
                os.remove(new_db_path)
                logging.info(f"Existing database {new_db_path} has been deleted.")
            except Exception as e:
                logging.error(f"Error deleting existing database {new_db_path}: {e}")
                return None
        
        logging.info(f"Creating new database: {new_db_path}")
        
        # Get all preprocessed tables from the original database
        conn = duckdb.connect(original_db_path, read_only=True)
        all_tables = conn.execute("SHOW TABLES").fetchall()
        preprocessed_tables = [table[0] for table in all_tables if table[0].startswith(self.output_prefix)]
        conn.close()
        
        if not preprocessed_tables:
            logging.warning(f"No preprocessed tables found in {original_db_path}")
            return None
        
        # Create new database connection
        conn_new = duckdb.connect(new_db_path, read_only=False)
        
        try:
            # Attach the original database
            conn_new.execute(f"ATTACH '{original_db_path}' AS orig")
            
            # Copy each preprocessed table to the new database
            for table_name in preprocessed_tables:
                start_time = time.time()
                logging.info(f"Moving table {table_name} to new database...")
                table_name_new = table_name.replace('clean_', '')
                
                # Create the table in the new database from the original
                conn_new.execute(f"CREATE TABLE {table_name_new} AS SELECT * FROM orig.{table_name}")
                
                # Verify row count in new database matches original
                orig_count = conn_new.execute(f"SELECT COUNT(*) FROM orig.{table_name}").fetchone()[0]
                new_count = conn_new.execute(f"SELECT COUNT(*) FROM {table_name_new}").fetchone()[0]
                
                if orig_count == new_count:
                    logging.info(f"Successfully copied {table_name}: {new_count:,} rows in {time.time() - start_time:.2f} seconds")
                else:
                    logging.error(f"Row count mismatch for {table_name}: Original={orig_count:,}, New={new_count:,}")
            
            # Detach the original database
            conn_new.execute("DETACH orig")
            
            # If requested, drop the tables from the original database
            if not keep_original:
                conn_orig = duckdb.connect(original_db_path, read_only=False)
                for table_name in preprocessed_tables:
                    conn_orig.execute(f"DROP TABLE {table_name}")
                    logging.info(f"Dropped table {table_name} from original database")
                conn_orig.close()
            
            return new_db_path
            
        except Exception as e:
            logging.error(f"Error moving preprocessed tables: {str(e)}")
            return None
        finally:
            conn_new.close()


def main():
    """Main function to run the table-level preprocessing."""
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess health claims data at the table level')
    parser.add_argument('--db_path', required=True, help='Path to the DuckDB database')
    parser.add_argument('--output_prefix', default='clean_', help='Prefix for cleaned tables (default: clean_)')
    parser.add_argument('--new_db', action='store_true', help='Create a new database for preprocessed tables')
    parser.add_argument('--keep_original', action='store_true', 
                        help='When using --new_db, keep preprocessed tables in the original database')
    
    # Use default arguments if none provided
    if len(sys.argv) == 1:
        print("No arguments provided. Using example values.")
        # args = parser.parse_args(['--db_path', 'duckdb\claims_data.duckdb'])
        args = parser.parse_args(['--db_path', 'duckdb\claims_data.duckdb', '--new_db'])
    else:
        args = parser.parse_args()
    
    # Run preprocessing
    start_time = time.time()
    print(f"Starting preprocessing for {args.db_path}")
    
    preprocessor = HealthClaimsPreprocessor(args.db_path, args.output_prefix)
    
    # First, preprocess all tables
    success_count = preprocessor.preprocess_all_tables()
    
    # If requested, move preprocessed tables to a new database
    if args.new_db and success_count > 0:
        print("\nCreating new database for preprocessed tables...")
        new_db_path = preprocessor.move_preprocessed_tables_to_new_db(keep_original=args.keep_original)
        
        if new_db_path:
            print(f"Successfully created new database: {new_db_path}")
            if not args.keep_original:
                print(f"Removed preprocessed tables from original database")
        else:
            print(f"Failed to create new database")
    
    elapsed_time = time.time() - start_time
    print(f"\nPreprocessing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()