import pandas as pd
import os


def rename_columns(df, prefix, exceptions=None, dataset_type=None):
    """
    Renames the columns of a DataFrame by adding a prefix, replacing spaces with underscores, and converting to lowercase.
    Special handling for renaming 'caseID' based on dataset type (inpatient or outpatient).
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
    """
    Reads a CSV file with the given column types and date parsing.

    Parameters:
        filename (str): Path to the CSV file.
        col_types (dict): Dictionary of column types for the file.
        table_name (str): Name of the table being read.
        parse_dates (dict): Dictionary mapping table names to date columns.

    Returns:
        pd.DataFrame: Loaded and parsed DataFrame.
    """
    try:
        return pd.read_csv(
            filename,
            dtype=col_types,
            sep='\t',
            encoding='utf-8',
            on_bad_lines='warn',
            parse_dates=parse_dates.get(table_name, None),
            encoding_errors='replace'
        )
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return pd.DataFrame()


def split_by_pid(insurants_file, chunk_size, dtypes, parse_dates):
    """
    Splits the insurants CSV file into chunks by the pid column.

    Parameters:
        insurants_file (str): Path to the insurants CSV file.
        chunk_size (int): Number of pids per split.
        dtypes (dict): Dictionary of column types for the file.
        parse_dates (dict): Dictionary mapping table names to date columns.

    Yields:
        pd.DataFrame: A DataFrame containing a chunk of the insurants data.
    """
    insurants = read_data(insurants_file, dtypes['insurants'], 'insurants', parse_dates)
    unique_pids = insurants['pid'].unique()

    for i in range(0, len(unique_pids), chunk_size):
        pid_chunk = unique_pids[i:i + chunk_size]
        yield insurants[insurants['pid'].isin(pid_chunk)]


import dask.dataframe as dd

def filter_and_merge_with_dask(pid_chunk_df, other_files, dtypes, parse_dates, keys):
    """
    Filters all other CSVs by the given pid chunk using Dask and merges them efficiently.

    Parameters:
        pid_chunk_df (pd.DataFrame): The chunk of insurants filtered by pid.
        other_files (dict): Dictionary of file paths for other CSVs.
        dtypes (dict): Dictionary of column types for each file.
        parse_dates (dict): Dictionary mapping table names to date columns.
        keys (dict): Dictionary mapping table names to join keys.

    Returns:
        dd.DataFrame: The merged result for the current pid chunk.
    """
    # Convert the pid_chunk_df to a Dask DataFrame
    pid_chunk_ddf = dd.from_pandas(pid_chunk_df, npartitions=1)

    # Initialize the result with the Dask DataFrame of pid_chunk
    result_ddf = pid_chunk_ddf

    # Perform parallelized filtering and merging for each file
    for table_name, file_path in other_files.items():
        dataset_type = "inpatient" if "inpatient" in table_name else "outpatient" if "outpatient" in table_name else None

        pandas_df = read_data(file_path, dtypes.get(table_name, None), table_name, parse_dates)
        table_ddf = dd.from_pandas(pandas_df, npartitions=1)

        # Rename columns
        table_ddf = table_ddf.map_partitions(rename_columns, prefix=table_name, exceptions=['pid'], dataset_type=dataset_type)

        # Filter the table by the pids in the current chunk
        filtered_table_ddf = table_ddf[table_ddf['pid'].isin(pid_chunk_df['pid'])]

        # Merge the filtered table with the current result
        result_ddf = dd.merge(result_ddf, filtered_table_ddf, how='left', on=keys[table_name])

    # Return the computed Dask DataFrame
    return result_ddf.compute()



def process_insurants_and_merge(insurants_file, other_files, output_file, chunk_size, dtypes, parse_dates, keys):
    """
    Processes the insurants CSV file, splits it by pid, filters and merges other CSVs,
    and appends the results to a single output file.

    Parameters:
        insurants_file (str): Path to the insurants CSV file.
        other_files (dict): Dictionary of file paths for other CSVs.
        output_file (str): Path to save the final merged result.
        chunk_size (int): Number of pids per split.
        dtypes (dict): Dictionary of column types for each file.
        parse_dates (dict): Dictionary mapping table names to date columns.
        keys (dict): Dictionary mapping table names to join keys.

    Returns:
    Returns:
        None
    """
    # Ensure the output file does not already exist
    if os.path.exists(output_file):
        os.remove(output_file)

    # Process each pid chunk
    for pid_chunk_df in split_by_pid(insurants_file, chunk_size, dtypes, parse_dates):
        # Filter and merge with other files
        merged_chunk = filter_and_merge_with_dask(pid_chunk_df, other_files, dtypes, parse_dates, keys)
        # Append the result to the output file
        merged_chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    print(f"Processing complete. Final result saved to {output_file}.")

# def filter_and_merge(pid_chunk_df, other_files, dtypes, parse_dates, keys):
#     """
#     Filters all other CSVs by the given pid chunk and merges them.
#
#     Parameters:
#         pid_chunk_df (pd.DataFrame): The chunk of insurants filtered by pid.
#         other_files (dict): Dictionary of file paths for other CSVs.
#         dtypes (dict): Dictionary of column types for each file.
#         parse_dates (dict): Dictionary mapping table names to date columns.
#         keys (dict): Dictionary mapping table names to join keys.
#
#     Returns:
#         pd.DataFrame: The merged result for the current pid chunk.
#     """
#     result_df = pid_chunk_df
#
#     for table_name, file_path in other_files.items():
#         dataset_type = "inpatient" if "inpatient" in table_name else "outpatient" if "outpatient" in table_name else None
#         table_df = read_data(file_path, dtypes.get(table_name, None), table_name, parse_dates)
#         table_df = rename_columns(table_df, prefix=table_name, exceptions=['pid'], dataset_type=dataset_type)
#         filtered_table = table_df[table_df['pid'].isin(pid_chunk_df['pid'])]
#         result_df = pd.merge(result_df, filtered_table, how='left', on=keys.get(table_name, ['pid']))
#
#     return result_df


if __name__ == "__main__":
    # Define the base path for the input CSV files
    base_path = '/home/cvt/Documents/masterarbeit/claims_data'

    # Define file paths for the input CSV files
    file_paths = {
        'insurance_data': f'{base_path}/sle.insurance_data.csv',
        'insurants': f'{base_path}/sle.insurants.csv',
        'inpatient_cases': f'{base_path}/sle.inpatient_cases.csv',
        'inpatient_diagnosis': f'{base_path}/sle.inpatient_diagnosis.csv',
        'inpatient_procedures': f'{base_path}/sle.inpatient_procedures.csv',
        'inpatient_fees': f'{base_path}/sle.inpatient_fees.csv',
        'outpatient_cases': f'{base_path}/sle.outpatient_cases.csv',
        'outpatient_diagnosis': f'{base_path}/sle.outpatient_diagnosis.csv',
        'outpatient_fees': f'{base_path}/sle.outpatient_fees.csv',
        'outpatient_procedures': f'{base_path}/sle.outpatient_procedures.csv',
        'drugs': f'{base_path}/sle.drugs.csv'
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
            'year of birth': 'Int64',
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
            'outpatient_diagnosis': str,
            'outpatient_diagnosis_qualification': str,
            'outpatient_diagnosis_localisation': 'Int64',
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

    # Define keys for joining
    keys = {
        'insurance_data': ['pid'],
        'inpatient_cases': ['pid'],
        'inpatient_diagnosis': ['pid', 'inpatient_caseID'],
        'inpatient_procedures': ['pid', 'inpatient_caseID'],
        'inpatient_fees': ['pid', 'inpatient_caseID'],
        'outpatient_cases': ['pid'],
        'outpatient_diagnosis': ['pid', 'outpatient_caseID'],
        'outpatient_procedures': ['pid', 'outpatient_caseID'],
        'outpatient_fees': ['pid', 'outpatient_caseID'],
        'drugs': ['pid']
    }

    # Output file path
    output_file = './final_result.csv'

    # Chunk size (number of pids per split)
    chunk_size = 50

    # Run the pipeline
    process_insurants_and_merge(file_paths['insurants'], file_paths, output_file, chunk_size, dtypes, parse_dates, keys)
