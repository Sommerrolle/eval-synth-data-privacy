#!/usr/bin/env python3
"""
Script to combine yearly joined tables into comprehensive tables across all years.
This script should be run after using join_by_year_with_stats.py and preprocess_joined_tables.py
to create and clean tables for each year.

Steps:
1. Identify all year-specific joined tables in the database (with optional prefix filtering)
2. Combine them using UNION ALL into three master tables:
   - [prefix]_all_inpatient
   - [prefix]_all_outpatient
   - [prefix]_all_drugs
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('table_combining.log'),
        logging.StreamHandler()
    ]
)

def identify_yearly_tables(conn, prefix=""):
    """
    Identify all year-specific joined tables in the database with the given prefix.
    
    Args:
        conn: DuckDB connection
        prefix: Optional prefix to filter tables (e.g., "clean_")
        
    Returns:
        Dictionary with categories (inpatient, outpatient, drugs) as keys and lists of table names as values
    """
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [table[0] for table in tables]
    
    # Pattern to match year-specific tables (e.g., clean_join_2018_inpatient)
    pattern = f'^{re.escape(prefix)}join_(\\d{{4}})_(inpatient|outpatient|drugs)$'
    
    # Categorize tables by type
    categorized_tables = {
        'inpatient': [],
        'outpatient': [],
        'drugs': []
    }
    
    for table in table_names:
        match = re.match(pattern, table)
        if match:
            year = match.group(1)
            category = match.group(2)
            categorized_tables[category].append({
                'name': table,
                'year': int(year)
            })
    
    # Sort tables by year
    for category in categorized_tables:
        categorized_tables[category].sort(key=lambda x: x['year'])
    
    return categorized_tables

def combine_tables(conn, tables, category, output_prefix=""):
    """
    Combine year-specific tables into one comprehensive table.
    
    Args:
        conn: DuckDB connection
        tables: List of table info dictionaries (with 'name' and 'year' keys)
        category: Category name (inpatient, outpatient, or drugs)
        output_prefix: Prefix for the output table name
        
    Returns:
        Name of the created combined table
    """
    if not tables:
        logging.warning(f"No {category} tables found to combine")
        return None
    
    output_table = f"{output_prefix}all_{category}"
    logging.info(f"Creating {output_table} by combining {len(tables)} tables...")
    
    # Get the list of table names
    table_names = [table['name'] for table in tables]
    years = [table['year'] for table in tables]
    
    # Log the tables being combined
    logging.info(f"Combining tables for years: {', '.join(map(str, years))}")
    
    # Create the UNION ALL query
    union_query = " UNION ALL ".join([f"SELECT * FROM {table}" for table in table_names])
    create_query = f"CREATE OR REPLACE TABLE {output_table} AS {union_query}"
    
    # Execute the query to create the combined table
    conn.execute(create_query)
    
    # Get row count and statistics
    row_count = conn.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]
    unique_patients = conn.execute(f"SELECT COUNT(DISTINCT pid) FROM {output_table}").fetchone()[0]
    
    logging.info(f"Created {output_table} with {row_count:,} rows and {unique_patients:,} unique patients")
    
    return output_table

def create_combined_tables(db_path, input_prefix="", output_prefix="", filter_years=None):
    """
    Create comprehensive tables by combining year-specific tables.
    
    Args:
        db_path: Path to the DuckDB database
        input_prefix: Prefix of input tables to combine (e.g., "clean_")
        output_prefix: Prefix for output combined tables
        filter_years: Optional list of years to include (if None, include all years)
        
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
        # Identify yearly tables with the given prefix
        categorized_tables = identify_yearly_tables(conn, input_prefix)
        
        # Filter tables by year if specified
        if filter_years:
            logging.info(f"Filtering for years: {', '.join(map(str, filter_years))}")
            for category in categorized_tables:
                categorized_tables[category] = [
                    table for table in categorized_tables[category]
                    if table['year'] in filter_years
                ]
        
        # Log found tables
        total_tables = sum(len(tables) for tables in categorized_tables.values())
        if total_tables == 0:
            logging.warning(f"No tables with prefix '{input_prefix}' found in the database")
            return {}
            
        logging.info(f"Found {total_tables} year-specific tables with prefix '{input_prefix}' to combine")
        for category, tables in categorized_tables.items():
            years_str = ', '.join(str(table['year']) for table in tables)
            logging.info(f"  {category}: {len(tables)} tables ({years_str})")
        
        # Combine tables for each category
        result = {}
        for category, tables in categorized_tables.items():
            combined_table = combine_tables(conn, tables, category, output_prefix)
            if combined_table:
                result[category] = combined_table
        
        return result
    
    except Exception as e:
        logging.error(f"Error combining tables: {str(e)}")
        return None
    
    finally:
        conn.close()

def main():
    """Main function to run the table combining script."""
    parser = argparse.ArgumentParser(description='Combine yearly joined tables into comprehensive tables')
    parser.add_argument('--db_path', required=True, help='Path to the DuckDB database')
    parser.add_argument('--input_prefix', default="clean_", help='Prefix of input tables to combine (default: "clean_")')
    parser.add_argument('--output_prefix', default="", help='Prefix for output combined tables (default: "")')
    parser.add_argument('--years', nargs='+', type=int, help='Years to include (space-separated, e.g., 2018 2019 2020)')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 80)
    print(f"Health Claims Data Table Combiner")
    if args.input_prefix:
        print(f"Input prefix: '{args.input_prefix}'")
    print("=" * 80 + "\n")
    
    # Create the combined tables
    tables = create_combined_tables(
        db_path=args.db_path, 
        input_prefix=args.input_prefix, 
        output_prefix=args.output_prefix,
        filter_years=args.years
    )
    
    if tables:
        print("\nSuccessfully created the following combined tables:")
        print("-" * 40)
        for category, table_name in tables.items():
            print(f"{category}: {table_name}")
        print("-" * 40)
    else:
        print("\nFailed to create combined tables. Check the log for details.")

if __name__ == "__main__":
    main()