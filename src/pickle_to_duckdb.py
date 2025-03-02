import pickle
import pandas as pd
import duckdb
import os
import sys
import numpy as np
from pathlib import Path

def process_dataframe(df, name):
    """Process a single dataframe, count missing values, and return the dataframe."""
    print(f"\n--- Table: {name} ---")
    print(f"Total rows: {len(df)}")
    
    missing_counts = {}
    for column in df.columns:
        # Count missing values (None, NaN, NA, etc.)
        na_count = df[column].isna().sum()
        
        # For object/string columns, also check for empty strings and "NA" strings
        if df[column].dtype == 'object':
            empty_str_count = (df[column] == "").sum()
            na_str_count = (df[column] == "NA").sum() + (df[column] == "<NA>").sum()
            total_missing = na_count + empty_str_count + na_str_count
            
            if empty_str_count > 0 or na_str_count > 0:
                print(f"  {column}: {total_missing} missing values ({na_count} NaN, {empty_str_count} empty strings, {na_str_count} 'NA' strings)")
            else:
                print(f"  {column}: {na_count} missing values")
        else:
            print(f"  {column}: {na_count} missing values")
        
        missing_counts[column] = na_count
    
    return df, missing_counts

def pickle_to_duckdb(pickle_path, duckdb_path):
    """
    Read a pickle file containing a dictionary of dataframes,
    count missing values, and save to DuckDB.
    """
    # Load the pickle file
    print(f"Loading pickle file: {pickle_path}")
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Check if the data is a dictionary (as expected from the R script)
    if not isinstance(data, dict):
        print("Warning: Pickle file does not contain a dictionary of dataframes")
        if isinstance(data, pd.DataFrame):
            print("Found a single DataFrame. Processing it directly.")
            data = {"main_table": data}
        else:
            print(f"Unexpected data type: {type(data)}")
            return
    
    # Create DuckDB connection
    print(f"Creating DuckDB database: {duckdb_path}")
    if os.path.exists(duckdb_path):
        print(f"Warning: {duckdb_path} already exists, it will be overwritten")
    
    con = duckdb.connect(duckdb_path)
    
    # Process each dataframe in the dictionary
    total_missing = {}
    for table_name, df in data.items():
        if not isinstance(df, pd.DataFrame):
            print(f"Warning: {table_name} is not a DataFrame, skipping")
            continue
            
        # Process the dataframe and count missing values
        processed_df, missing_counts = process_dataframe(df, table_name)
        total_missing[table_name] = missing_counts
        
        # Write to DuckDB
        try:
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM processed_df")
            print(f"Table '{table_name}' created in DuckDB with {len(processed_df)} rows")
        except Exception as e:
            print(f"Error writing {table_name} to DuckDB: {e}")
    
    # Close the connection
    con.close()
    print(f"\nDuckDB database created at: {duckdb_path}")
    
    # Summary of missing values
    print("\n=== Missing Values Summary ===")
    for table_name, missing_dict in total_missing.items():
        total = sum(missing_dict.values())
        if total > 0:
            print(f"{table_name}: {total} total missing values across all columns")
        else:
            print(f"{table_name}: No missing values")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <pickle_file_path> [output_duckdb_path]")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    
    # Default output path is the same as input but with .duckdb extension
    if len(sys.argv) >= 3:
        duckdb_path = sys.argv[2]
    else:
        duckdb_path = str(Path(pickle_path).with_suffix('.duckdb'))
    
    pickle_to_duckdb(pickle_path, duckdb_path)