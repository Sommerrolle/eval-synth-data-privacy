import duckdb
import os
import pandas as pd
from typing import List, Dict, Tuple

# Define base path for the input CSV files
# base_path = '/home/cvt/Documents/masterarbeit/claims_data'
testdata_path = 'D:\Benutzer\Cuong.VoTa\datasets\claims_data'

# Define file paths for the input CSV files
file_paths = {
    'insurance_data': f'{testdata_path}/sle.insurance_data.csv',
    #'insurants': f'{base_path}/sle.insurants_500.csv',
    'insurants': f'{testdata_path}/sle.insurants.csv',
    'inpatient_cases': f'{testdata_path}/sle.inpatient_cases.csv',
    'inpatient_diagnosis': f'{testdata_path}/sle.inpatient_diagnosis.csv',
    'inpatient_procedures': f'{testdata_path}/sle.inpatient_procedures.csv',
    'inpatient_fees': f'{testdata_path}/sle.inpatient_fees.csv',
    'outpatient_cases': f'{testdata_path}/sle.outpatient_cases.csv',
    'outpatient_diagnosis': f'{testdata_path}/sle.outpatient_diagnosis.csv',
    'outpatient_fees': f'{testdata_path}/sle.outpatient_fees.csv',
    'outpatient_procedures': f'{testdata_path}/sle.outpatient_procedures.csv',
    'drugs': f'{testdata_path}/sle.drugs.csv'
}

# Define dtypes for each file
dtypes = {
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

parse_dates = {
    'insurance_data': ['from', 'to'],
    'inpatient_cases': ['date of admission', 'date of discharge'],
    'inpatient_fees': ['from', 'to'],
    'inpatient_procedures': ['date of procedure'],
    'outpatient_cases': ['from', 'to'],
    'outpatient_fees': ['date'],
    'drugs': ['date of prescription', 'date of dispense']
}

# Table mapping for naming convention
TABLE_MAPPING = {
    'insurants': 1,
    'insurance_data': 2,
    'drugs': 3,
    'inpatient_cases': 4,
    'inpatient_diagnosis': 5,
    'inpatient_procedures': 6,
    'inpatient_fees': 7,
    'outpatient_cases': 8,
    'outpatient_diagnosis': 9,
    'outpatient_procedures': 10,
    'outpatient_fees': 11
}

def display_table_options():
    """Display available tables with their corresponding numbers."""
    print("\nAvailable tables:")
    print("-" * 40)
    for table_name, number in TABLE_MAPPING.items():
        print(f"{number}. {table_name}")
    print("-" * 40)


def get_user_selection() -> List[str]:
    """Get user input for table selection."""
    while True:
        try:
            print("\nEnter the numbers of the tables you want to join (separated by spaces):")
            numbers = input("> ").strip().split()
            numbers = [int(n) for n in numbers]
            
            # Validate input
            if not all(1 <= n <= 11 for n in numbers):
                print("Error: Please enter numbers between 1 and 11.")
                continue
                
            # Convert numbers back to table names
            reverse_mapping = {v: k for k, v in TABLE_MAPPING.items()}
            selected_tables = [reverse_mapping[n] for n in numbers]
            
            if 'insurants' not in selected_tables:
                print("Warning: 'insurants' (1) should typically be included as it's the primary table.")
                if input("Do you want to continue anyway? (y/n): ").lower() != 'y':
                    continue
            
            return selected_tables
            
        except ValueError:
            print("Error: Please enter valid numbers separated by spaces.")


def get_join_keys(selected_tables: List[str]) -> Dict[str, List[str]]:
    """Create a dictionary of join keys for selected tables."""
    keys = {
        'insurants': ['pid'],
        'insurance_data': ['pid'],
        'drugs': ['pid'],
        'inpatient_cases': ['pid'],
        'inpatient_diagnosis': ['pid', 'inpatient_caseID'],
        'inpatient_procedures': ['pid', 'inpatient_caseID'],
        'inpatient_fees': ['pid', 'inpatient_caseID'],
        'outpatient_cases': ['pid'],
        'outpatient_diagnosis': ['pid', 'outpatient_caseID'],
        'outpatient_procedures': ['pid', 'outpatient_caseID'],
        'outpatient_fees': ['pid', 'outpatient_caseID']
    }
    return {table: keys[table] for table in selected_tables}


def generate_output_table_name(selected_tables: List[str]) -> str:
    """Generate the output table name based on selected tables."""
    # Get numbers and sort them numerically
    numbers = [TABLE_MAPPING[table] for table in selected_tables]
    sorted_numbers = sorted(numbers)  # Simple numerical sort
    return 'joined_' + '_'.join(str(n) for n in sorted_numbers)


def rename_columns(df, prefix, exceptions=None, dataset_type=None):
    """
    Renames the columns of a DataFrame by adding a prefix, replacing spaces with underscores, and converting to lowercase.
    Special handling for renaming 'caseID' based on dataset type (inpatient or outpatient).

    Parameters:
        df (DataFrame): The input DataFrame.
        prefix (str): Prefix to add to column names.
        exceptions (list, optional): List of column names to exclude from renaming.
        dataset_type (str, optional): Type of dataset ('inpatient' or 'outpatient') for special 'caseID' handling.

    Returns:
        DataFrame: The updated DataFrame with renamed columns.
    """
    if exceptions is None:
        exceptions = []

    def rename_column(col_name):
        if col_name in exceptions:
            return col_name
        if col_name == "caseID":
            if dataset_type == "inpatient":
                return "inpatient_caseID"
            elif dataset_type == "outpatient":
                return "outpatient_caseID"
        return f"{prefix}_{col_name.strip().replace(' ', '_').lower()}"

    df = df.rename(columns=rename_column)
    return df


def read_data(filename, col_types, table_name, parse_dates):

    try:
        # try reading in the data with pandas
        df = pd.read_csv(filename,
                         dtype=col_types,
                         sep='\t',
                         encoding='utf-8',
                         on_bad_lines='warn',
                         parse_dates=parse_dates.get(table_name, None),
                         encoding_errors='replace')
        return df
    except Exception as e:
        print(f"Error reading in {filename}: {e}")


def sort_tables_by_pid(database_path):
    """
    Sorts all tables in the DuckDB database by their 'pid' column and saves the order persistently.

    Parameters:
        database_path (str): Path to the DuckDB database file.

    Returns:
        None
    """
    # Connect to the DuckDB database
    con = duckdb.connect(database=database_path, read_only=False)

    # Get a list of all tables
    tables = con.execute("SHOW TABLES").fetchall()

    for table in tables:
        table_name = table[0]

        # Check if the table has a 'pid' column
        columns = con.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [col[0] for col in columns]

        if 'pid' not in column_names:
            print(f"Skipping table '{table_name}' as it does not have a 'pid' column.")
            continue

        # Sort the table by 'pid' and overwrite it
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM {table_name} ORDER BY pid
        """)
        print(f"Table '{table_name}' sorted by 'pid' and saved persistently.")

    # Close the connection
    con.close()
    print("All tables have been sorted by 'pid' and updated in the database.")


def chunkwise_join_and_save(db_path, keys, output_table, chunk_size=200000):
    """
    Performs chunkwise joins of large tables in a DuckDB database and saves intermediate results.

    Parameters:
        db_path (str): Path to the DuckDB database file.
        keys (dict): Dictionary of join keys for each table.
        output_table (str): Name of the table to save the final joined results.
        chunk_size (int): Number of rows to process per chunk.

    Returns:
        None
    """
    # Connect to the DuckDB database
    con = duckdb.connect(db_path)
    print(f"Connected to DuckDB database at {db_path}.")

    # Debug tables
    # Get a list of all tables
    tables = con.execute("SHOW TABLES").fetchall()
    print(tables)

    # Ensure the output table doesn't already exist
    con.execute(f"DROP TABLE IF EXISTS {output_table};")
    print(f"Cleared previous output table '{output_table}' if it existed.")

    # Get the total number of rows in the primary table
    primary_table = 'insurants'
    total_rows_query = f"SELECT COUNT(*) FROM {primary_table};"
    total_rows = con.execute(total_rows_query).fetchone()[0]
    print(f"Total rows in '{primary_table}': {total_rows}")

    # Process the primary table in chunks
    for offset in range(0, total_rows, chunk_size):
        print(f"Processing chunk from row {offset} to {offset + chunk_size - 1}...")
        
        # Fetch the chunk from the primary table
        primary_chunk_query = f"""
            SELECT *
            FROM {primary_table}
            LIMIT {chunk_size} OFFSET {offset};
        """
        primary_chunk = con.execute(primary_chunk_query).fetch_df()
        print(f"Fetched {len(primary_chunk)} rows from '{primary_table}'.")

        # Join the chunk with other tables
        result_chunk = primary_chunk
        for table_name, join_keys in keys.items():
            if table_name == primary_table:
                continue  # Skip the primary table
            
            print(f"Joining with table '{table_name}' on keys {join_keys}...")
            
            # Construct the join query
            join_query = f"""
                SELECT *
                FROM {table_name}
                WHERE {join_keys[0]} IN (
                    SELECT DISTINCT {join_keys[0]} FROM primary_chunk
                );
            """
            filtered_table = con.execute(join_query).fetch_df()
            print(f"Fetched {len(filtered_table)} rows from '{table_name}' for joining.")

            # Merge with the current result chunk
            result_chunk = result_chunk.merge(
                filtered_table,
                on=join_keys,
                how='left'
            )
            print(f"Resulting chunk now has {len(result_chunk)} rows.")

        # Write or append the results to the output table
        if offset == 0:
            print(f"Creating output table '{output_table}'...")
            con.register("result_chunk", result_chunk)
            con.execute(f"CREATE TABLE {output_table} AS SELECT * FROM result_chunk;")
        else:
            print(f"Appending to output table '{output_table}'...")
            con.register("result_chunk", result_chunk)
            con.execute(f"INSERT INTO {output_table} SELECT * FROM result_chunk;")
        print(f"Chunk from row {offset} to {offset + chunk_size - 1} processed and saved.")

    print(f"Chunkwise joining complete. Results saved to table '{output_table}'.")


def explain_join(db_path, keys, primary_table='insurants'):
    """
    Analyzes the join relationships and estimates the complexity and time required.

    Parameters:
        db_path (str): Path to the DuckDB database.
        keys (dict): Dictionary of join keys for each table.
        primary_table (str): The primary table for the join process.

    Returns:
        None
    """
    con = duckdb.connect(db_path)
    print(f"Connected to DuckDB database at {db_path}.")

    # Get the size of the primary table
    primary_size = con.execute(f"SELECT COUNT(*) FROM {primary_table};").fetchone()[0]
    print(f"Primary table '{primary_table}' has {primary_size} rows.")

    # Initialize the cumulative size
    cumulative_size = primary_size

    # Iterate through all tables and analyze relationships
    for table_name, join_keys in keys.items():
        if table_name == primary_table:
            continue  # Skip the primary table

        # Get the size of the secondary table
        table_size = con.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
        print(f"Table '{table_name}' has {table_size} rows.")

        # Estimate the one-to-many relationship
        # Count how many rows in the secondary table match each row in the primary table
        relationship_query = f"""
            SELECT AVG(match_count)
            FROM (
                SELECT COUNT(*) AS match_count
                FROM {table_name}
                GROUP BY {', '.join(join_keys)}
            );
        """
        avg_relationship = con.execute(relationship_query).fetchone()[0] or 1
        print(f"On average, each row in '{primary_table}' matches {avg_relationship:.2f} rows in '{table_name}'.")

        # Update the cumulative size
        cumulative_size *= avg_relationship
        print(f"Estimated cumulative size after joining with '{table_name}': {cumulative_size:.0f} rows.")

    # Estimate the time based on cumulative size
    estimated_time = cumulative_size / 1e6  # Assume 1 million rows take ~1 second
    print(f"Estimated final output size: {cumulative_size:.0f} rows.")
    print(f"Estimated processing time: {estimated_time:.2f} seconds.")

    con.close()


def perform_join(db_path: str, selected_tables: List[str], chunk_size: int = 200000) -> None:
    """Perform the join operation with the selected tables."""
    keys = get_join_keys(selected_tables)
    output_table = generate_output_table_name(selected_tables)
    
    # First explain the join
    print("\nAnalyzing join complexity...")
    explain_join(db_path, keys)
    
    # Ask for confirmation
    if input("\nDo you want to proceed with the join? (y/n): ").lower() != 'y':
        print("Join operation cancelled.")
        return
    
    print(f"\nProceeding with join operation. Output table will be: {output_table}")
    chunkwise_join_and_save(db_path, keys, output_table, chunk_size)


def main():
    db_path = 'duckdb/claims_data.duckdb'
    
    print("Welcome to the Interactive Table Join Tool")
    print("========================================")
    
    while True:
        display_table_options()
        selected_tables = get_user_selection()
        
        print("\nYou selected the following tables:")
        for table in selected_tables:
            print(f"- {table}")
        
        if input("\nConfirm selection? (y/n): ").lower() == 'y':
            perform_join(db_path, selected_tables)
        
        if input("\nDo you want to create another joined table? (y/n): ").lower() != 'y':
            break
    
    print("\nThank you for using the Interactive Table Join Tool!")


# def main():
#     # Path to the DuckDB database
#     db_path = 'duckdb/claims_data.duckdb'

#     ##############################
#     ##### Working code ###########
#     ##############################

#     # # Define DuckDB connection
#     # con = duckdb.connect(database=db_path, read_only=False)

#     # # Process and create tables
#     # for table_name, file_path in file_paths.items():
#     #     # Determine dataset type
#     #     dataset_type = "inpatient" if "inpatient" in table_name else "outpatient" if "outpatient" in table_name else None

#     #     # Read the CSV data
#     #     pandas_df = read_data(file_path, dtypes.get(table_name, None), table_name, parse_dates)

#     #     # Rename columns
#     #     renamed_df = rename_columns(pandas_df, prefix=table_name, exceptions=['pid'], dataset_type=dataset_type)

#     #     # Create a DuckDB table
#     #     con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM renamed_df")

#     #     # Print debug info
#     #     print(f"Table '{table_name}' created in DuckDB with {len(renamed_df)} rows.")

#     # # Close the connection
#     # con.close()
#     # print("All tables have been created in DuckDB.")

#     # Sorting all tables by pid
#     # sort_tables_by_pid('claims_data.duckdb')

#     ##############################
#     ##### End of working code ####
#     ##############################

#     # Define keys for joining
#     # keys = {
#     #     'insurance_data': ['pid'],
#     #     'drugs': ['pid']
#         # 'inpatient_cases': ['pid'],
#         # 'inpatient_diagnosis': ['pid', 'inpatient_caseID'],
#         # 'inpatient_procedures': ['pid', 'inpatient_caseID'],
#         # 'inpatient_fees': ['pid', 'inpatient_caseID'],
#         # 'outpatient_cases': ['pid'],
#         # 'outpatient_diagnosis': ['pid', 'outpatient_caseID'],
#         # 'outpatient_procedures': ['pid', 'outpatient_caseID'],
#         # 'outpatient_fees': ['pid', 'outpatient_caseID'],
#     }

#     # Output table name
#     output_table = 'joined_data'

#     # Chunk size
#     chunk_size = 200

#     # Run the pipeline
#     # took too long
#     # chunkwise_join_and_save(db_path, keys, output_table, chunk_size)

#     explain_join(db_path, keys)

if __name__ == "__main__":
    main()
