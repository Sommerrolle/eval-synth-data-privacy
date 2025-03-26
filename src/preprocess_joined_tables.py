#!/usr/bin/env python3
"""
Script to preprocess joined health claims tables by handling missing values.
This script is designed to work with the tables created by join_by_year_with_stats.py:
- join_{year}_inpatient
- join_{year}_outpatient
- join_{year}_drugs

It applies domain-specific imputation strategies to replace NULL values with appropriate
defaults based on data type and column meaning.
"""

import sys
import argparse
import logging
from pathlib import Path
import time
import duckdb
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/joined_preprocessing.log'),
        logging.StreamHandler()
    ]
)

class JoinedTablePreprocessor:
    """Preprocess joined health claims tables by handling missing values."""
    
    def __init__(self, db_path: str, year: int, output_prefix: str = "clean_"):
        """
        Initialize the preprocessor.
        
        Args:
            db_path: Path to the DuckDB database
            year: Year of the joined tables
            output_prefix: Prefix for cleaned tables
        """
        self.db_path = db_path
        self.year = year
        self.output_prefix = output_prefix
        
        # Expected joined table names based on year
        self.joined_tables = {
            'inpatient': f"join_{year}_inpatient",
            'outpatient': f"join_{year}_outpatient",
            'drugs': f"join_{year}_drugs"
        }
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
    
    def get_table_column_info(self, table_name: str) -> List[Dict]:
        """Get information about columns in a table."""
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            conn.close()
            return [{"name": col[0], "type": col[1]} for col in columns]
        except Exception as e:
            logging.error(f"Error getting column info for {table_name}: {str(e)}")
            return []
    
    def get_table_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logging.error(f"Error counting rows in {table_name}: {str(e)}")
            return 0
    
    def execute_query(self, query: str):
        """Execute a SQL query on the database."""
        try:
            conn = duckdb.connect(self.db_path, read_only=False)
            result = conn.execute(query).fetchall()
            conn.close()
            return result
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            logging.error(f"Query: {query}")
            return None
    
    def preprocess_all_joined_tables(self) -> Dict[str, str]:
        """
        Preprocess all joined tables for the specified year.
        
        Returns:
            Dictionary mapping table types to preprocessed table names
        """
        processed_tables = {}
        
        # Check if the joined tables exist
        conn = duckdb.connect(self.db_path, read_only=True)
        existing_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        conn.close()
        
        for table_type, table_name in self.joined_tables.items():
            if table_name not in existing_tables:
                logging.warning(f"Table {table_name} not found in database")
                continue
            
            logging.info(f"\n{'='*80}\nProcessing joined table: {table_name}\n{'='*80}")
            
            try:
                output_table = self.preprocess_joined_table(table_name, table_type)
                if output_table:
                    processed_tables[table_type] = output_table
                    print(f"Successfully preprocessed table: {table_name} → {output_table}")
                else:
                    print(f"Failed to preprocess table: {table_name}")
            except Exception as e:
                logging.error(f"Error preprocessing table {table_name}: {str(e)}")
                print(f"Error preprocessing table: {table_name}")
        
        # Print summary
        print("\nPreprocessing Summary:")
        print("-" * 70)
        print(f"Successfully preprocessed {len(processed_tables)} out of {len(self.joined_tables)} tables")
        for table_type, table_name in processed_tables.items():
            print(f"  {table_type}: {table_name}")
        print("-" * 70)
        
        return processed_tables
    
    def preprocess_joined_table(self, table_name: str, table_type: str) -> Optional[str]:
        """
        Preprocess a specific joined table.
        
        Args:
            table_name: Name of the table to preprocess
            table_type: Type of the table (inpatient, outpatient, drugs)
            
        Returns:
            Name of the preprocessed table
        """
        output_table = f"{self.output_prefix}{table_name}"
        
        # Get column information
        columns_info = self.get_table_column_info(table_name)
        if not columns_info:
            logging.error(f"Failed to get column information for table '{table_name}'")
            return None
        
        # Count nulls in original table
        self._log_null_counts(table_name, [col['name'] for col in columns_info])
        
        # Generate imputation SQL based on table type
        if table_type == 'inpatient':
            imputation_sql = self._generate_inpatient_imputation_sql(table_name, columns_info)
        elif table_type == 'outpatient':
            imputation_sql = self._generate_outpatient_imputation_sql(table_name, columns_info)
        elif table_type == 'drugs':
            imputation_sql = self._generate_drugs_imputation_sql(table_name, columns_info)
        else:
            logging.error(f"Unknown table type: {table_type}")
            return None
        
        # Create the preprocessed table
        create_table_sql = f"""
            CREATE OR REPLACE TABLE {output_table} AS
            {imputation_sql}
        """
        
        try:
            # Execute the query
            self.execute_query(create_table_sql)
            
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
    
    def _preprocess_table_fallback(self, table_name: str, columns_info: List[Dict]) -> Optional[str]:
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
        self.execute_query(f"DROP TABLE IF EXISTS {output_table}")
        self.execute_query(f"CREATE TABLE {output_table} AS SELECT * FROM {table_name}")
        
        # Update each column one by one
        conn = duckdb.connect(self.db_path, read_only=False)
        try:
            for col_info in columns_info:
                col_name = col_info['name']
                col_type = col_info['type']
                
                # Skip columns with no nulls
                null_count = conn.execute(
                    f"SELECT COUNT(*) FROM {output_table} WHERE {col_name} IS NULL"
                ).fetchone()[0]
                
                if null_count > 0:
                    # Get imputation value based on column and table type
                    imputation_expr = self._get_imputation_value(col_name, col_type)
                    
                    # Update the column
                    update_sql = f"UPDATE {output_table} SET {col_name} = {imputation_expr} WHERE {col_name} IS NULL"
                    conn.execute(update_sql)
                    
                    logging.info(f"Updated {null_count} NULL values in column '{col_name}'")
            
            # Verify no nulls remain
            self._verify_no_nulls(output_table, [col['name'] for col in columns_info])
            
            return output_table
            
        except Exception as e:
            logging.error(f"Error in fallback approach: {str(e)}")
            return None
        
        finally:
            conn.close()
    
    def _get_imputation_value(self, col_name: str, col_type: str) -> str:
        """
        Get an appropriate imputation value for a column.
        
        Args:
            col_name: Name of the column
            col_type: Type of the column
            
        Returns:
            SQL expression for the imputation value
        """
        col_name_lower = col_name.lower()
        col_type_upper = col_type.upper()
        
        # Handle different data types with appropriate defaults
        if 'pid' in col_name_lower:
            # For patient ID, we shouldn't have nulls, but use a placeholder
            return "-999"
        elif 'year_of_birth' in col_name_lower:
            return "1970"
        elif 'gender' in col_name_lower:
            return "1"  # Default to a standard code, usually 1=female, 2=male
        elif 'death' in col_name_lower:
            return "0"  # Default to alive
        elif 'regional_code' in col_name_lower:
            return "0"  # Default regional code
        elif 'from' in col_name_lower or 'date_of_admission' in col_name_lower or 'date_of_procedure' in col_name_lower:
            return "'2510-01-01'"  # Default date
        elif 'to' in col_name_lower or 'date_of_discharge' in col_name_lower:
            return "'2510-01-01'"  # Default date
        elif 'quarter' in col_name_lower:
            return "1"  # Default to first quarter
        elif 'year' in col_name_lower:
            return str(self.year)  # Use the year parameter
        elif 'diagnosis' in col_name_lower:
            return "'UNKNOWN'"
        elif 'procedure_code' in col_name_lower:
            return "'UNKNOWN'"
        elif 'billing_code' in col_name_lower:
            return "'UNKNOWN'"
        elif 'pharma_central_number' in col_name_lower:
            return "'00000000'"
        elif any(code in col_name_lower for code in ['physician_code', 'practice_code']):
            return "'000000000'"
        elif 'specialty' in col_name_lower:
            return "'00'"
        elif 'department' in col_name_lower:
            return "'0000'"
        elif 'cause_of_admission' in col_name_lower:
            return "'0000'"
        elif 'cause_of_discharge' in col_name_lower:
            return "'00'"
        elif 'amount_due' in col_name_lower or 'ddd' in col_name_lower:
            return "0.0"
        elif 'quantity' in col_name_lower:
            return "1.0"
        elif 'outpatient_treatment' in col_name_lower or 'is_main_diagnosis' in col_name_lower:
            return "0"
        elif 'localisation' in col_name_lower:
            return "0"
        elif 'qualification' in col_name_lower:
            return "'U'"  # Default qualification code
        
        # Generic type-based defaults if no specific rule above
        if 'INT' in col_type_upper or 'BIGINT' in col_type_upper:
            return "0"
        elif 'DOUBLE' in col_type_upper or 'FLOAT' in col_type_upper:
            return "0.0"
        elif 'DATE' in col_type_upper or 'TIMESTAMP' in col_type_upper:
            return "'2510-01-01'"
        elif 'VARCHAR' in col_type_upper or 'CHAR' in col_type_upper or 'TEXT' in col_type_upper:
            return "'UNKNOWN'"
        else:
            return "NULL"  # Fallback, shouldn't be reached
    
    def _generate_inpatient_imputation_sql(self, table_name: str, columns_info: List[Dict]) -> str:
        """Generate SQL to impute missing values in inpatient joined table."""
        # List of column expressions with COALESCE
        column_expressions = []
        
        for col_info in columns_info:
            col_name = col_info['name']
            col_type = col_info['type']
            col_lower = col_name.lower()
            
            # # Core patient information
            # if 'pid' in col_lower:
            #     column_expressions.append(f"COALESCE({col_name}, -999) AS {col_name}")
            # elif 'insurants_year_of_birth' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 
            #             (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
            #             1970
            #         ) AS {col_name}
            #     """)
            # elif 'insurants_gender' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 
            #             (SELECT MODE({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
            #             1
            #         ) AS {col_name}
            #     """)
            # elif 'insurance_data_from' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, DATE '2000-01-01') AS {col_name}
            #     """)
            # elif 'insurance_data_to' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, DATE '2030-12-31') AS {col_name}
            #     """)
            # elif 'insurance_data_death' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 0) AS {col_name}
            #     """)
            # elif 'insurance_data_regional_code' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 0) AS {col_name}
            #     """)
            
            # Inpatient case information
            if 'inpatient_caseid' in col_lower:
                column_expressions.append(f"""
                    {col_name} AS {col_name}
                """)
            elif 'inpatient_cases_date_of_admission' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT inpatient_cases_date_of_discharge - INTERVAL '5 days' 
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_date_of_discharge IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            elif 'inpatient_cases_date_of_discharge' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT inpatient_cases_date_of_admission + INTERVAL '5 days' 
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_date_of_admission IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            elif 'inpatient_cases_cause_of_admission' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '0000') AS {col_name}
                """)
            elif 'inpatient_cases_cause_of_discharge' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '00') AS {col_name}
                """)
            elif 'inpatient_cases_outpatient_treatment' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0) AS {col_name}
                """)
            elif 'inpatient_cases_department_admission' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT inpatient_cases_department_discharge
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_department_discharge IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        '0000'
                    ) AS {col_name}
                """)
            elif 'inpatient_cases_department_discharge' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT inpatient_cases_department_admission
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_department_admission IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        '0000'
                    ) AS {col_name}
                """)
            
            # Diagnosis information
            elif 'inpatient_diagnosis_diagnosis' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 'UNKNOWN') AS {col_name}
                """)
            elif 'inpatient_diagnosis_type_of_diagnosis' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '00') AS {col_name}
                """)
            elif 'inpatient_diagnosis_is_main_diagnosis' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0) AS {col_name}
                """)
            elif 'inpatient_diagnosis_localisation' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0) AS {col_name}
                """)
            
            # Procedures information
            elif 'inpatient_procedures_procedure_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 'UNKNOWN') AS {col_name}
                """)
            elif 'inpatient_procedures_localisation' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0) AS {col_name}
                """)
            elif 'inpatient_procedures_date_of_procedure' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT inpatient_cases_date_of_admission + INTERVAL '1 day'
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_date_of_admission IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            
            # Fees information
            elif 'inpatient_fees_billing_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 'UNKNOWN') AS {col_name}
                """)
            elif 'inpatient_fees_amount_due' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
                        0.0
                    ) AS {col_name}
                """)
            elif 'inpatient_fees_quantity' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 1.0) AS {col_name}
                """)
            elif 'inpatient_fees_from' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT inpatient_cases_date_of_admission
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_date_of_admission IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            elif 'inpatient_fees_to' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT inpatient_cases_date_of_discharge
                         FROM {table_name} t
                         WHERE t.pid = {table_name}.pid 
                           AND t.inpatient_cases_date_of_discharge IS NOT NULL 
                           AND t.inpatient_caseid = {table_name}.inpatient_caseid
                         LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            
            # Default case for any other columns
            else:
                column_expressions.append(self._generate_generic_column_expr(col_name, col_type, table_name))
        
        # Construct the full SELECT statement
        return f"SELECT {', '.join(column_expressions)} FROM {table_name}"

    def _generate_outpatient_imputation_sql(self, table_name: str, columns_info: List[Dict]) -> str:
        """Generate SQL to impute missing values in outpatient joined table."""
        # List of column expressions with COALESCE
        column_expressions = []
        
        for col_info in columns_info:
            col_name = col_info['name']
            col_type = col_info['type']
            col_lower = col_name.lower()
            
            # # Core patient information
            # if 'pid' in col_lower:
            #     column_expressions.append(f"COALESCE({col_name}, -999) AS {col_name}")
            # elif 'insurants_year_of_birth' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 
            #             (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
            #             1970
            #         ) AS {col_name}
            #     """)
            # elif 'insurants_gender' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 
            #             (SELECT MODE({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
            #             1
            #         ) AS {col_name}
            #     """)
            # elif 'insurance_data_from' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, DATE '2000-01-01') AS {col_name}
            #     """)
            # elif 'insurance_data_to' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, DATE '2030-12-31') AS {col_name}
            #     """)
            # elif 'insurance_data_death' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 0) AS {col_name}
            #     """)
            # elif 'insurance_data_regional_code' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 0) AS {col_name}
            #     """)
            
            # Outpatient case information
            if 'outpatient_caseid' in col_lower:
                column_expressions.append(f"""
                    {col_name} AS {col_name}
                """)
            elif 'outpatient_cases_practice_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT outpatient_cases_practice_code 
                            FROM {table_name} t
                            WHERE t.pid = {table_name}.pid 
                                AND t.outpatient_caseid = {table_name}.outpatient_caseid
                                AND t.outpatient_cases_practice_code IS NOT NULL
                            LIMIT 1),
                        '000000000'
                    ) AS {col_name}
                """)
            elif 'outpatient_cases_from' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT outpatient_cases_to - INTERVAL '14 days' 
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid 
                        AND t.outpatient_cases_to IS NOT NULL 
                        AND t.outpatient_caseid = {table_name}.outpatient_caseid
                        LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            elif 'outpatient_cases_to' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT outpatient_cases_from + INTERVAL '14 days' 
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid 
                        AND t.outpatient_cases_from IS NOT NULL 
                        AND t.outpatient_caseid = {table_name}.outpatient_caseid
                        LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            elif 'outpatient_cases_amount_due' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0.0) AS {col_name}
                """)
            elif 'outpatient_cases_year' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, {self.year}) AS {col_name}
                """)
            elif 'outpatient_cases_quarter' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT CEIL(EXTRACT(MONTH FROM outpatient_cases_from) / 3) 
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid 
                        AND t.outpatient_cases_from IS NOT NULL 
                        AND t.outpatient_caseid = {table_name}.outpatient_caseid
                        LIMIT 1),
                    (1 + CAST(RANDOM() * 3 AS INTEGER))
                    ) AS {col_name}
                """)
            
            # Diagnosis information
            elif 'outpatient_diagnosis_diagnosis' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 'UNKNOWN') AS {col_name}
                """)
            elif 'outpatient_diagnosis_qualification' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 'U') AS {col_name}
                """)
            elif 'outpatient_diagnosis_localisation' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0) AS {col_name}
                """)
            
            # Procedures information
            elif 'outpatient_procedures_procedure_code' in col_lower:
                column_expressions.append(f"""
                    {col_name} AS {col_name}
                """)
            elif 'outpatient_procedures_localisation' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 0) AS {col_name}
                """)
            elif 'outpatient_procedures_date_of_procedure' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT outpatient_cases_from
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.outpatient_caseid = {table_name}.outpatient_caseid
                        AND t.outpatient_cases_from IS NOT NULL
                        LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            elif 'outpatient_procedures_specialty_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '00') AS {col_name}
                """)
            elif 'outpatient_procedures_physician_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '000000000') AS {col_name}
                """)
            
            # Fees information
            elif 'outpatient_fees_physician_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '000000000') AS {col_name}
                """)
            elif 'outpatient_fees_specialty_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, '00') AS {col_name}
                """)
            elif 'outpatient_fees_billing_code' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 'UNKNOWN') AS {col_name}
                """)
            elif 'outpatient_fees_quantity' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 1.0) AS {col_name}
                """)
            elif 'outpatient_fees_date' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT outpatient_cases_from
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.outpatient_caseid = {table_name}.outpatient_caseid
                        AND t.outpatient_cases_from IS NOT NULL
                        LIMIT 1),
                        DATE '2510-01-01'
                    ) AS {col_name}
                """)
            
            # Default case for any other columns
            else:
                column_expressions.append(self._generate_generic_column_expr(col_name, col_type, table_name))
        
        # Construct the full SELECT statement
        return f"SELECT {', '.join(column_expressions)} FROM {table_name}"
    
    def _generate_drugs_imputation_sql(self, table_name: str, columns_info: List[Dict]) -> str:
        """Generate SQL to impute missing values in drugs joined table."""
        # List of column expressions with COALESCE
        column_expressions = []
        
        for col_info in columns_info:
            col_name = col_info['name']
            col_type = col_info['type']
            col_lower = col_name.lower()
            
            # # Core patient information
            # if 'pid' in col_lower:
            #     column_expressions.append(f"COALESCE({col_name}, -999) AS {col_name}")
            # elif 'insurants_year_of_birth' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 
            #             (SELECT MEDIAN({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
            #             1970
            #         ) AS {col_name}
            #     """)
            # elif 'insurants_gender' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 
            #             (SELECT MODE({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL),
            #             1
            #         ) AS {col_name}
            #     """)
            # elif 'insurance_data_from' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, DATE '2000-01-01') AS {col_name}
            #     """)
            # elif 'insurance_data_to' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, DATE '2030-12-31') AS {col_name}
            #     """)
            # elif 'insurance_data_death' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 0) AS {col_name}
            #     """)
            # elif 'insurance_data_regional_code' in col_lower:
            #     column_expressions.append(f"""
            #         COALESCE({col_name}, 0) AS {col_name}
            #     """)
            
            # Drugs information
            if 'drugs_date_of_prescription' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT drugs_date_of_dispense - INTERVAL '7 days'
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_date_of_dispense IS NOT NULL
                        AND t.drugs_pharma_central_number = {table_name}.drugs_pharma_central_number
                        LIMIT 1),
                        DATE '{self.year}-06-01'
                    ) AS {col_name}
                """)
            elif 'drugs_date_of_dispense' in col_lower:
                column_expressions.append(f"""
                    COALESCE({col_name}, 
                        (SELECT drugs_date_of_prescription + INTERVAL '7 days'
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_date_of_prescription IS NOT NULL
                        AND t.drugs_pharma_central_number = {table_name}.drugs_pharma_central_number
                        LIMIT 1),
                        DATE '{self.year}-06-15'
                    ) AS {col_name}
                """)
            elif 'drugs_pharma_central_number' in col_lower:
                # For missing pharma central numbers, use common value or placeholder
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT drugs_pharma_central_number
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_pharma_central_number IS NOT NULL
                        AND (
                            t.drugs_atc = {table_name}.drugs_atc
                            OR t.drugs_date_of_dispense = {table_name}.drugs_date_of_dispense
                        )
                        LIMIT 1),
                        '00000000'
                    ) AS {col_name}
                """)
            elif 'drugs_specialty_of_prescriber' in col_lower:
                # For specialty, use the most common specialty for this patient
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT mode() WITHIN GROUP (ORDER BY drugs_specialty_of_prescriber)
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_specialty_of_prescriber IS NOT NULL
                        LIMIT 1),
                        (SELECT mode() WITHIN GROUP (ORDER BY drugs_specialty_of_prescriber)
                        FROM {table_name}
                        WHERE drugs_specialty_of_prescriber IS NOT NULL
                        LIMIT 1),
                        '00'
                    ) AS {col_name}
                """)
            elif 'drugs_physician_code' in col_lower:
                # For physician code, try to find the most frequent physician for this patient
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT mode() WITHIN GROUP (ORDER BY drugs_physician_code)
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_physician_code IS NOT NULL
                        LIMIT 1),
                        '000000000'
                    ) AS {col_name}
                """)
            elif 'drugs_practice_code' in col_lower:
                # For practice code, try to find the practice associated with the physician
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT drugs_practice_code
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_practice_code IS NOT NULL
                        AND t.drugs_physician_code = {table_name}.drugs_physician_code
                        LIMIT 1),
                        (SELECT mode() WITHIN GROUP (ORDER BY drugs_practice_code)
                        FROM {table_name} t
                        WHERE t.pid = {table_name}.pid
                        AND t.drugs_practice_code IS NOT NULL
                        LIMIT 1),
                        '000000000'
                    ) AS {col_name}
                """)
            elif 'drugs_quantity' in col_lower:
                # For quantity, use median for this drug type if available
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT MEDIAN(drugs_quantity)
                        FROM {table_name} t
                        WHERE t.drugs_pharma_central_number = {table_name}.drugs_pharma_central_number
                        AND t.drugs_quantity IS NOT NULL
                        LIMIT 1),
                        (SELECT MEDIAN(drugs_quantity)
                        FROM {table_name}
                        WHERE drugs_quantity IS NOT NULL),
                        1.0
                    ) AS {col_name}
                """)
            elif 'drugs_amount_due' in col_lower:
                # For amount due, use median for this drug type if available
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT MEDIAN(drugs_amount_due)
                        FROM {table_name} t
                        WHERE t.drugs_pharma_central_number = {table_name}.drugs_pharma_central_number
                        AND t.drugs_amount_due IS NOT NULL
                        LIMIT 1),
                        (SELECT MEDIAN(drugs_amount_due)
                        FROM {table_name}
                        WHERE drugs_amount_due IS NOT NULL),
                        0.0
                    ) AS {col_name}
                """)
            elif 'drugs_atc' in col_lower:
                # For ATC code, try to find it based on the pharma central number
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT mode() WITHIN GROUP (ORDER BY drugs_atc)
                        FROM {table_name} t
                        WHERE t.drugs_pharma_central_number = {table_name}.drugs_pharma_central_number
                        AND t.drugs_atc IS NOT NULL
                        LIMIT 1),
                        'UNKNOWN'
                    ) AS {col_name}
                """)
            elif 'drugs_ddd' in col_lower:
                # For DDD (Defined Daily Dose), use median for this ATC code if available
                column_expressions.append(f"""
                    COALESCE({col_name},
                        (SELECT MEDIAN(drugs_ddd)
                        FROM {table_name} t
                        WHERE t.drugs_atc = {table_name}.drugs_atc
                        AND t.drugs_ddd IS NOT NULL
                        LIMIT 1),
                        (SELECT MEDIAN(drugs_ddd)
                        FROM {table_name}
                        WHERE drugs_ddd IS NOT NULL),
                        0.0
                    ) AS {col_name}
                """)
            
            # Default case for any other columns
            else:
                column_expressions.append(self._generate_generic_column_expr(col_name, col_type, table_name))
        
        # Construct the full SELECT statement
        return f"SELECT {', '.join(column_expressions)} FROM {table_name}"
        
    def _generate_generic_column_expr(self, col_name: str, col_type: str, table_name: str) -> str:
        """
        Generate a generic imputation expression based on data type.
        
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
                    DATE '2510-01-01'
                ) AS {col_name}
            """
        elif 'VARCHAR' in col_type_upper or 'CHAR' in col_type_upper or 'TEXT' in col_type_upper:
            # For low-cardinality categorical columns, use mode (most frequent value)
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

    # def _log_null_counts(self, table_name: str, columns: List[str], csv_path: str = "stats_preprocess_joined_tables.csv") -> None:
    #     """
    #     Log the number of NULL values in each column of a table and append to a CSV file.
        
    #     Args:
    #         table_name: Name of the table
    #         columns: List of column names
    #         csv_path: Path to the CSV file to save results
    #     """
    #     logging.info(f"NULL counts in original table '{table_name}':")
        
    #     # Create a list to store results
    #     null_counts_data = []
        
    #     conn = duckdb.connect(self.db_path, read_only=True)
    #     try:
    #         # Get total row count for the table
    #         total_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
    #         for col in columns:
    #             null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
                
    #             # Only record columns with nulls
    #             if null_count > 0:
    #                 percent = (null_count / total_count) * 100 if total_count > 0 else 0
    #                 logging.info(f"  - {col}: {null_count:,} NULLs ({percent:.2f}%)")
                    
    #                 # Add to results list
    #                 null_counts_data.append({
    #                     'table_name': table_name,
    #                     'column_name': col,
    #                     'null_count': null_count,
    #                     'total_rows': total_count,
    #                     'null_percentage': round(percent, 2),
    #                     'year': self.year,
    #                 })
    #     except Exception as e:
    #         logging.error(f"Error checking NULL counts: {str(e)}")
    #     finally:
    #         conn.close()
        
    #     # If we have results, save to CSV
    #     if null_counts_data:
    #         try:
                
    #             # Check if file exists
    #             if os.path.exists(csv_path):
    #                 # Load existing CSV
    #                 df_existing = pd.read_csv(csv_path)
                    
    #                 # Create DataFrame from new data
    #                 df_new = pd.DataFrame(null_counts_data)
                    
    #                 # Check if this table has been processed before
    #                 mask = df_existing['table_name'] == table_name
    #                 if mask.any():
    #                     # Remove existing entries for this table
    #                     df_existing = df_existing[~mask]
                    
    #                 # Append new data
    #                 df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    
    #                 # Save combined data
    #                 df_combined.to_csv(csv_path, index=False)
    #                 logging.info(f"Updated NULL counts for table '{table_name}' in {csv_path}")
    #             else:
    #                 # Create new CSV file
    #                 df_new = pd.DataFrame(null_counts_data)
    #                 df_new.to_csv(csv_path, index=False)
    #                 logging.info(f"Created new NULL counts CSV at {csv_path}")
    #         except Exception as e:
    #             logging.error(f"Error saving NULL counts to CSV: {str(e)}")
        
    #     return null_counts_data
    
    def _verify_no_nulls(self, table_name: str, columns: List[str]) -> None:
        """
        Verify that no NULL values remain in the preprocessed table.
        
        Args:
            table_name: Name of the preprocessed table
            columns: List of column names to check
        """
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
            logging.error(f"Error in NULL verification: {str(e)}")
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
            
            # Get patient counts
            orig_patients = conn.execute(f"SELECT COUNT(DISTINCT pid) FROM {original_table}").fetchone()[0]
            proc_patients = conn.execute(f"SELECT COUNT(DISTINCT pid) FROM {processed_table}").fetchone()[0]
            
            logging.info(f"Original table '{original_table}': {orig_patients:,} unique patients")
            logging.info(f"Processed table '{processed_table}': {proc_patients:,} unique patients")
            
            # For case IDs
            case_id_col = None
            if "inpatient" in original_table:
                case_id_col = "inpatient_caseid"
            elif "outpatient" in original_table:
                case_id_col = "outpatient_caseid"
                
            if case_id_col:
                try:
                    orig_cases = conn.execute(f"SELECT COUNT(DISTINCT {case_id_col}) FROM {original_table} WHERE {case_id_col} IS NOT NULL").fetchone()[0]
                    proc_cases = conn.execute(f"SELECT COUNT(DISTINCT {case_id_col}) FROM {processed_table} WHERE {case_id_col} IS NOT NULL").fetchone()[0]
                    
                    logging.info(f"Original table '{original_table}': {orig_cases:,} unique cases")
                    logging.info(f"Processed table '{processed_table}': {proc_cases:,} unique cases")
                except:
                    pass
                
        except Exception as e:
            logging.error(f"Error logging table statistics: {str(e)}")
        finally:
            conn.close()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess joined health claims tables')
    parser.add_argument('--db_path', required=True, help='Path to the DuckDB database')
    parser.add_argument('--year', type=int, required=True, help='Year of the joined tables')
    parser.add_argument('--output_prefix', default='clean_', help='Prefix for cleaned tables')
    
    # Use default arguments if none provided (for testing)
    if len(sys.argv) == 1:
        print("No arguments provided. Using example values.")
        args = parser.parse_args(['--db_path', 'duckdb/claims_data.duckdb', '--year', '2017'])
    else:
        args = parser.parse_args()
    
    return args

def main():
    """Main function to run the preprocessing script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print banner
    print("\n" + "=" * 80)
    print(f"Joined Health Claims Tables Preprocessor - Year: {args.year}")
    print("=" * 80 + "\n")
    
    # Create and run the preprocessor
    start_time = time.time()
    print(f"Starting preprocessing for {args.db_path}")
    
    preprocessor = JoinedTablePreprocessor(
        db_path=args.db_path,
        year=args.year,
        output_prefix=args.output_prefix
    )
    
    processed_tables = preprocessor.preprocess_all_joined_tables()
    
    elapsed_time = time.time() - start_time
    print(f"\nPreprocessing completed in {elapsed_time:.2f} seconds")
    
    return processed_tables

if __name__ == "__main__":
    main()