#!/usr/bin/env python3
"""
Script to join health claims tables across all years and create three comprehensive tables:
1. join_all_inpatient
2. join_all_outpatient
3. join_all_drugs

This script combines the insurants and insurance_data tables with other domain-specific
tables, ensuring that events occur within the insurance period.
"""

import os
import sys
import argparse
import logging
import csv
from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('table_joining.log'),
        logging.StreamHandler()
    ]
)

def create_inpatient_join(conn):
    """
    Create a comprehensive inpatient data join across all years.
    
    Args:
        conn: DuckDB connection
    
    Returns:
        Name of the created table
    """
    table_name = "join_all_inpatient"
    logging.info(f"Creating {table_name} table...")
    
    # Create the join query
    create_table_query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT 
        ins.pid,
        ins.insurants_year_of_birth,
        ins.insurants_gender,
        insd.insurance_data_from,
        insd.insurance_data_to,
        insd.insurance_data_death,
        insd.insurance_data_regional_code,
        ic.inpatient_caseID,
        ic.inpatient_cases_date_of_admission,
        ic.inpatient_cases_date_of_discharge,
        ic.inpatient_cases_cause_of_admission,
        ic.inpatient_cases_cause_of_discharge,
        ic.inpatient_cases_outpatient_treatment,
        ic.inpatient_cases_department_admission,
        ic.inpatient_cases_department_discharge,
        id.inpatient_diagnosis_diagnosis,
        id.inpatient_diagnosis_type_of_diagnosis,
        id.inpatient_diagnosis_is_main_diagnosis,
        id.inpatient_diagnosis_localisation,
        ip.inpatient_procedures_procedure_code,
        ip.inpatient_procedures_localisation,
        ip.inpatient_procedures_date_of_procedure,
        ifees.inpatient_fees_billing_code,
        ifees.inpatient_fees_amount_due,
        ifees.inpatient_fees_quantity,
        ifees.inpatient_fees_from,
        ifees.inpatient_fees_to
    FROM 
        insurants ins
    JOIN 
        insurance_data insd ON ins.pid = insd.pid
    LEFT JOIN 
        inpatient_cases ic ON ins.pid = ic.pid
        AND (
            -- Event occurs within insurance period
            ic.inpatient_cases_date_of_admission >= insd.insurance_data_from
            AND (ic.inpatient_cases_date_of_admission <= insd.insurance_data_to OR insd.insurance_data_to IS NULL)
        )
    LEFT JOIN 
        inpatient_diagnosis id ON ic.pid = id.pid AND ic.inpatient_caseID = id.inpatient_caseID
    LEFT JOIN 
        inpatient_procedures ip ON ic.pid = ip.pid AND ic.inpatient_caseID = ip.inpatient_caseID
        AND (
            -- Procedure occurs within insurance period
            (ip.inpatient_procedures_date_of_procedure IS NULL OR 
             (ip.inpatient_procedures_date_of_procedure >= insd.insurance_data_from
              AND (ip.inpatient_procedures_date_of_procedure <= insd.insurance_data_to OR insd.insurance_data_to IS NULL)))
        )
    LEFT JOIN 
        inpatient_fees ifees ON ic.pid = ifees.pid AND ic.inpatient_caseID = ifees.inpatient_caseID
        AND (
            -- Fee period overlaps with insurance period
            (ifees.inpatient_fees_from >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR ifees.inpatient_fees_from IS NULL)
            AND (ifees.inpatient_fees_from <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR ifees.inpatient_fees_from IS NULL)
            OR (ifees.inpatient_fees_to >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR ifees.inpatient_fees_to IS NULL)
            AND (ifees.inpatient_fees_to <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR ifees.inpatient_fees_to IS NULL)
        )
    ORDER BY 
        ins.pid, 
        ic.inpatient_caseID, 
        id.inpatient_diagnosis_is_main_diagnosis DESC,
        ip.inpatient_procedures_date_of_procedure,
        ifees.inpatient_fees_from
    """
    
    # Execute the query to create the table
    conn.execute(create_table_query)
    
    # Get comprehensive statistics about the join
    statistics_query = f"""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT pid) as unique_patients,
        COUNT(DISTINCT inpatient_caseID) as unique_cases,
        SUM(CASE WHEN inpatient_caseID IS NOT NULL THEN 1 ELSE 0 END) as rows_with_cases,
        SUM(CASE WHEN inpatient_diagnosis_diagnosis IS NOT NULL THEN 1 ELSE 0 END) as rows_with_diagnoses,
        SUM(CASE WHEN inpatient_procedures_procedure_code IS NOT NULL THEN 1 ELSE 0 END) as rows_with_procedures,
        SUM(CASE WHEN inpatient_fees_billing_code IS NOT NULL THEN 1 ELSE 0 END) as rows_with_fees,
        COUNT(DISTINCT CASE WHEN inpatient_caseID IS NOT NULL THEN pid END) as patients_with_cases,
        COUNT(DISTINCT CASE WHEN inpatient_diagnosis_diagnosis IS NOT NULL THEN pid END) as patients_with_diagnoses,
        COUNT(DISTINCT CASE WHEN inpatient_procedures_procedure_code IS NOT NULL THEN pid END) as patients_with_procedures,
        COUNT(DISTINCT CASE WHEN inpatient_fees_billing_code IS NOT NULL THEN pid END) as patients_with_fees,
        MIN(inpatient_cases_date_of_admission) as earliest_admission,
        MAX(inpatient_cases_date_of_admission) as latest_admission
    FROM
        {table_name}
    """
    stats = conn.execute(statistics_query).fetchdf()
    
    # Log statistics
    logging.info(f"Statistics for {table_name}:")
    for col in stats.columns:
        logging.info(f"  {col}: {stats[col].iloc[0]:,}")
    
    # Get row count
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logging.info(f"Created table '{table_name}' with {row_count:,} rows")
    
    return table_name

def create_outpatient_join(conn):
    """
    Create a comprehensive outpatient data join across all years.
    
    Args:
        conn: DuckDB connection
    
    Returns:
        Name of the created table
    """
    table_name = "join_all_outpatient"
    logging.info(f"Creating {table_name} table...")
    
    # Create the join query
    create_table_query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT 
        ins.pid,
        ins.insurants_year_of_birth,
        ins.insurants_gender,
        insd.insurance_data_from,
        insd.insurance_data_to,
        insd.insurance_data_death,
        insd.insurance_data_regional_code,
        oc.outpatient_caseID,
        oc.outpatient_cases_practice_code,
        oc.outpatient_cases_from,
        oc.outpatient_cases_to,
        oc.outpatient_cases_amount_due,
        oc.outpatient_cases_year,
        oc.outpatient_cases_quarter,
        od.outpatient_diagnosis_diagnosis,
        od.outpatient_diagnosis_qualification,
        od.outpatient_diagnosis_localisation,
        op.outpatient_procedures_procedure_code,
        op.outpatient_procedures_localisation,
        op.outpatient_procedures_date_of_procedure,
        op.outpatient_procedures_specialty_code,
        op.outpatient_procedures_physician_code,
        ofees.outpatient_fees_physician_code,
        ofees.outpatient_fees_specialty_code,
        ofees.outpatient_fees_billing_code,
        ofees.outpatient_fees_quantity,
        ofees.outpatient_fees_date
    FROM 
        insurants ins
    JOIN 
        insurance_data insd ON ins.pid = insd.pid
    LEFT JOIN 
        outpatient_cases oc ON ins.pid = oc.pid
        AND (
            -- Case period overlaps with insurance period
            (oc.outpatient_cases_from >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR oc.outpatient_cases_from IS NULL)
            AND (oc.outpatient_cases_from <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR oc.outpatient_cases_from IS NULL)
            OR (oc.outpatient_cases_to >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR oc.outpatient_cases_to IS NULL)
            AND (oc.outpatient_cases_to <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR oc.outpatient_cases_to IS NULL)
        )
    LEFT JOIN 
        outpatient_diagnosis od ON oc.pid = od.pid AND oc.outpatient_caseID = od.outpatient_caseID
    LEFT JOIN 
        outpatient_procedures op ON oc.pid = op.pid AND oc.outpatient_caseID = op.outpatient_caseID
        AND (
            -- Procedure occurs within insurance period
            (op.outpatient_procedures_date_of_procedure >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR op.outpatient_procedures_date_of_procedure IS NULL)
            AND (op.outpatient_procedures_date_of_procedure <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR op.outpatient_procedures_date_of_procedure IS NULL)
        )
    LEFT JOIN 
        outpatient_fees ofees ON oc.pid = ofees.pid AND oc.outpatient_caseID = ofees.outpatient_caseID
        AND (
            -- Fee occurs within insurance period
            (ofees.outpatient_fees_date >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR ofees.outpatient_fees_date IS NULL)
            AND (ofees.outpatient_fees_date <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR ofees.outpatient_fees_date IS NULL)
        )
    ORDER BY 
        ins.pid, 
        oc.outpatient_caseID, 
        oc.outpatient_cases_from,
        op.outpatient_procedures_date_of_procedure,
        ofees.outpatient_fees_date
    """
    
    # Execute the query to create the table
    conn.execute(create_table_query)
    
    # Get comprehensive statistics about the join
    statistics_query = f"""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT pid) as unique_patients,
        COUNT(DISTINCT outpatient_caseID) as unique_cases,
        SUM(CASE WHEN outpatient_caseID IS NOT NULL THEN 1 ELSE 0 END) as rows_with_cases,
        SUM(CASE WHEN outpatient_diagnosis_diagnosis IS NOT NULL THEN 1 ELSE 0 END) as rows_with_diagnoses,
        SUM(CASE WHEN outpatient_procedures_procedure_code IS NOT NULL THEN 1 ELSE 0 END) as rows_with_procedures,
        SUM(CASE WHEN outpatient_fees_billing_code IS NOT NULL THEN 1 ELSE 0 END) as rows_with_fees,
        COUNT(DISTINCT CASE WHEN outpatient_caseID IS NOT NULL THEN pid END) as patients_with_cases,
        COUNT(DISTINCT CASE WHEN outpatient_diagnosis_diagnosis IS NOT NULL THEN pid END) as patients_with_diagnoses,
        COUNT(DISTINCT CASE WHEN outpatient_procedures_procedure_code IS NOT NULL THEN pid END) as patients_with_procedures,
        COUNT(DISTINCT CASE WHEN outpatient_fees_billing_code IS NOT NULL THEN pid END) as patients_with_fees,
        MIN(outpatient_cases_from) as earliest_case,
        MAX(outpatient_cases_to) as latest_case,
        MIN(outpatient_cases_year) as earliest_year,
        MAX(outpatient_cases_year) as latest_year
    FROM
        {table_name}
    """
    stats = conn.execute(statistics_query).fetchdf()
    
    # Log statistics
    logging.info(f"Statistics for {table_name}:")
    for col in stats.columns:
        logging.info(f"  {col}: {stats[col].iloc[0]:,}")
    
    # Get row count
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logging.info(f"Created table '{table_name}' with {row_count:,} rows")
    
    return table_name

def create_drugs_join(conn):
    """
    Create a drugs data join across all years.
    
    Args:
        conn: DuckDB connection
    
    Returns:
        Name of the created table
    """
    table_name = "join_all_drugs"
    logging.info(f"Creating {table_name} table...")
    
    # Create the join query
    create_table_query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT 
        ins.pid,
        ins.insurants_year_of_birth,
        ins.insurants_gender,
        insd.insurance_data_from,
        insd.insurance_data_to,
        insd.insurance_data_death,
        insd.insurance_data_regional_code,
        d.drugs_date_of_prescription,
        d.drugs_date_of_dispense,
        d.drugs_pharma_central_number,
        d.drugs_specialty_of_prescriber,
        d.drugs_physician_code,
        d.drugs_practice_code,
        d.drugs_quantity,
        d.drugs_amount_due,
        d.drugs_atc,
        d.drugs_ddd
    FROM 
        insurants ins
    JOIN 
        insurance_data insd ON ins.pid = insd.pid
    LEFT JOIN 
        drugs d ON ins.pid = d.pid
        AND (
            -- Drug dispensed within insurance period
            (d.drugs_date_of_dispense >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR d.drugs_date_of_dispense IS NULL)
            AND (d.drugs_date_of_dispense <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR d.drugs_date_of_dispense IS NULL)
            OR
            -- Drug prescribed within insurance period
            (d.drugs_date_of_prescription >= insd.insurance_data_from OR insd.insurance_data_from IS NULL OR d.drugs_date_of_prescription IS NULL)
            AND (d.drugs_date_of_prescription <= insd.insurance_data_to OR insd.insurance_data_to IS NULL OR d.drugs_date_of_prescription IS NULL)
        )
    WHERE
        d.drugs_date_of_dispense IS NOT NULL
        OR d.drugs_date_of_prescription IS NOT NULL
    ORDER BY 
        ins.pid, 
        d.drugs_date_of_dispense
    """
    
    # Execute the query to create the table
    conn.execute(create_table_query)
    
    # Get comprehensive statistics about the drugs join
    statistics_query = f"""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT pid) as unique_patients,
        SUM(CASE WHEN drugs_pharma_central_number IS NOT NULL THEN 1 ELSE 0 END) as rows_with_drugs,
        COUNT(DISTINCT CASE WHEN drugs_pharma_central_number IS NOT NULL THEN pid END) as patients_with_drugs,
        COUNT(DISTINCT drugs_pharma_central_number) as unique_drug_codes,
        COUNT(DISTINCT drugs_atc) as unique_atc_codes,
        AVG(drugs_quantity) as avg_drug_quantity,
        AVG(drugs_amount_due) as avg_drug_cost,
        MIN(drugs_date_of_dispense) as earliest_dispense_date,
        MAX(drugs_date_of_dispense) as latest_dispense_date,
        COUNT(DISTINCT drugs_physician_code) as unique_prescribers,
        COUNT(DISTINCT drugs_practice_code) as unique_practices
    FROM
        {table_name}
    """
    stats = conn.execute(statistics_query).fetchdf()
    
    # Log statistics
    logging.info(f"Statistics for {table_name}:")
    for col in stats.columns:
        if col in ['avg_drug_quantity', 'avg_drug_cost']:
            logging.info(f"  {col}: {stats[col].iloc[0]:.4f}")
        elif col in ['earliest_dispense_date', 'latest_dispense_date']:
            logging.info(f"  {col}: {stats[col].iloc[0]}")
        else:
            logging.info(f"  {col}: {stats[col].iloc[0]:,}")
    
    # Get row count
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logging.info(f"Created table '{table_name}' with {row_count:,} rows")
    
    return table_name

def get_table_row_counts(conn):
    """
    Get row counts for all tables in the database.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        Dictionary with table names as keys and row counts as values
    """
    # Get list of all tables
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [table[0] for table in tables]
    
    # Get row count for each table
    row_counts = {}
    for table in table_names:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            row_counts[table] = count
        except Exception as e:
            logging.error(f"Error getting row count for table {table}: {str(e)}")
            row_counts[table] = -1  # Indicate error
    
    return row_counts

def save_row_counts_to_csv(db_path, row_counts, csv_filename="table_counts.csv"):
    """
    Save table row counts to a CSV file, appending to existing file if it exists.
    If a row for the database already exists, it will be updated.
    
    Args:
        db_path: Path to the database
        row_counts: Dictionary with table names as keys and row counts as values
        csv_filename: Name of the CSV file to save to
        
    Returns:
        Path to the CSV file
    """
    db_name = Path(db_path).stem
    
    # Get base table names and joined table names
    base_tables = sorted([table for table in row_counts.keys() if not table.startswith('join_')])
    joined_tables = sorted([table for table in row_counts.keys() if table.startswith('join_')])
    all_columns = base_tables + joined_tables
    
    # Ensure the three main joined tables are represented
    for table_type in ['inpatient', 'outpatient', 'drugs']:
        table_name = f"join_all_{table_type}"
        if table_name not in all_columns:
            all_columns.append(table_name)
            row_counts[table_name] = 0  # Use 0 to indicate table doesn't exist
    
    # Check if file exists and load it
    if os.path.exists(csv_filename):
        try:
            # Try to load existing file
            df = pd.read_csv(csv_filename, index_col=0)
            logging.info(f"Loaded existing CSV file: {csv_filename}")
        except Exception as e:
            logging.warning(f"Error loading existing CSV file: {str(e)}. Creating new file.")
            df = pd.DataFrame(index=[])
    else:
        # Create new DataFrame
        df = pd.DataFrame(index=[])
    
    # Create or update row for this database
    if db_name in df.index:
        logging.info(f"Updating existing row for database: {db_name}")
        # Update existing columns
        for col in all_columns:
            if col in df.columns:
                df.at[db_name, col] = row_counts.get(col, 0)
            else:
                # Add new column if it doesn't exist
                df[col] = pd.Series(dtype='int64')
                df.at[db_name, col] = row_counts.get(col, 0)
    else:
        logging.info(f"Adding new row for database: {db_name}")
        # Create new row
        new_row = pd.DataFrame({col: [row_counts.get(col, 0)] for col in all_columns}, index=[db_name])
        
        # Add new columns to existing DataFrame if needed
        for col in all_columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype='int64')
        
        # Append new row
        df = pd.concat([df, new_row])
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Convert all values to integers
    for col in df.columns:
        df[col] = df[col].astype(int)
    
    # Save to CSV
    df.to_csv(csv_filename)
    logging.info(f"Table row counts saved to {csv_filename}")
    
    return csv_filename

def check_database_tables(conn):
    """
    Check if the required tables exist in the database.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        True if all required tables exist, False otherwise
    """
    required_tables = [
        'insurants',
        'insurance_data',
        'inpatient_cases',
        'inpatient_diagnosis',
        'inpatient_procedures',
        'inpatient_fees',
        'outpatient_cases',
        'outpatient_diagnosis',
        'outpatient_procedures',
        'outpatient_fees',
        'drugs'
    ]
    
    # Get existing tables
    tables = conn.execute("SHOW TABLES").fetchall()
    existing_tables = set(table[0] for table in tables)
    
    # Check if all required tables exist
    missing_tables = [table for table in required_tables if table not in existing_tables]
    
    if missing_tables:
        logging.error(f"Missing required tables: {', '.join(missing_tables)}")
        return False
    
    return True

def create_joined_tables(db_path, csv_path="table_counts.csv"):
    """
    Create the three joined tables across all years.
    
    Args:
        db_path: Path to the DuckDB database
        csv_path: Path to the CSV file for row counts
        
    Returns:
        Dictionary with the names of the created tables
    """
    # Ensure database file exists
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return None
    
    logging.info(f"Opening database: {db_path}")
    conn = duckdb.connect(database=db_path, read_only=False)
    
    try:
        # Check if required tables exist
        if not check_database_tables(conn):
            conn.close()
            return None
        
        # Get initial row counts (before creating new tables)
        initial_row_counts = get_table_row_counts(conn)
        
        # Create the three joined tables
        inpatient_table = create_inpatient_join(conn)
        outpatient_table = create_outpatient_join(conn)
        drugs_table = create_drugs_join(conn)
        
        # Get updated row counts (after creating new tables)
        final_row_counts = get_table_row_counts(conn)
        
        # Save row counts to CSV
        csv_file = save_row_counts_to_csv(db_path, final_row_counts, csv_path)
        logging.info(f"Table row counts saved to {csv_file}")
        
        return {
            'inpatient': inpatient_table,
            'outpatient': outpatient_table,
            'drugs': drugs_table,
            'csv_path': csv_file
        }
    
    except Exception as e:
        logging.error(f"Error creating joined tables: {str(e)}")
        return None
    
    finally:
        conn.close()

def main():
    """Main function to run the table joining script."""
    parser = argparse.ArgumentParser(description='Create joined tables for health claims data across all years')
    parser.add_argument('--db_path', required=True, help='Path to the DuckDB database')
    parser.add_argument('--count_only', action='store_true', help='Only count rows without creating joined tables')
    parser.add_argument('--csv_path', default='table_counts.csv', help='Path to the CSV file for row counts (default: table_counts.csv)')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 80)
    print(f"Health Claims Data Table Joiner - All Years")
    print("=" * 80 + "\n")
    
    # If count_only flag is set, just get the row counts without creating joins
    if args.count_only:
        if not os.path.exists(args.db_path):
            print(f"Database file not found: {args.db_path}")
            return
        
        print(f"Counting rows in database: {args.db_path}")
        conn = duckdb.connect(database=args.db_path, read_only=True)
        row_counts = get_table_row_counts(conn)
        conn.close()
        
        csv_path = save_row_counts_to_csv(args.db_path, row_counts, args.csv_path)
        print(f"\nTable row counts saved to {csv_path}")
        return
    
    # Create the joined tables
    tables = create_joined_tables(db_path=args.db_path, csv_path=args.csv_path)
    
    if tables:
        print("\nSuccessfully created the following tables:")
        print("-" * 40)
        for table_type, table_name in tables.items():
            if table_type != 'csv_path':
                print(f"{table_type}: {table_name}")
        print("-" * 40)
        print(f"\nTable row counts saved to: {tables.get('csv_path', 'N/A')}")
    else:
        print("\nFailed to create joined tables. Check the log for details.")

if __name__ == "__main__":
    main()