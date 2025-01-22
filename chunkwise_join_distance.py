import pandas as pd
import numpy as np
import os


def chunkwise_join_and_process(original_files, synthetic_files, keys, output_path, original_chunk_size=200000, synthetic_chunk_size=100000):
    """
    Handles chunkwise joining of multiple CSVs and calculates DCR.

    Parameters:
        original_files (dict): Paths to the original dataset CSV files.
        synthetic_files (dict): Paths to the synthetic dataset CSV files.
        keys (list): List of keys to use for joins (e.g., ['pid', 'inpatient_caseID']).
        output_path (str): Directory to save the results.
        original_chunk_size (int): Number of rows per chunk for the original dataset.
        synthetic_chunk_size (int): Number of rows per chunk for the synthetic dataset.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Process the original dataset chunk by chunk
    chunk_id = 0
    for original_chunk in read_and_join_chunkwise(original_files, keys, chunk_size=original_chunk_size):
        process_chunk_with_dcr(
            original_chunk, synthetic_files, keys, output_path, chunk_id, synthetic_chunk_size
        )
        chunk_id += 1

    print("Chunkwise DCR processing complete. Results saved to:", output_path)


def read_and_join_chunkwise(files, keys, chunk_size):
    """
    Reads and joins multiple tables chunkwise.

    Parameters:
        files (dict): Dictionary of file paths for the tables to be joined.
        keys (list): Keys to join the tables on.
        chunk_size (int): Number of rows per chunk.

    Yields:
        pd.DataFrame: A joined chunk of data.
    """
    # Load the primary file (e.g., 'insurance_data')
    primary_file = files.pop('primary')
    for primary_chunk in pd.read_csv(primary_file, chunksize=chunk_size):
        result_chunk = primary_chunk
        # Iteratively join with other tables
        for table_name, file_path in files.items():
            result_chunk = pd.merge(result_chunk, pd.read_csv(file_path), how='left', on=keys)
        yield result_chunk


def process_chunk_with_dcr(original_chunk, synthetic_files, keys, output_path, chunk_id, synthetic_chunk_size):
    """
    Processes a single chunk of the original dataset by calculating the DCR against
    chunks of the synthetic dataset.

    Parameters:
        original_chunk (pd.DataFrame): A chunk of the original dataset.
        synthetic_files (dict): Paths to the synthetic dataset tables.
        keys (list): Keys to join synthetic tables on.
        output_path (str): Path to save the results.
        chunk_id (int): Identifier for the chunk being processed.
        synthetic_chunk_size (int): Number of rows to process per synthetic chunk.

    Returns:
        None
    """
    results = []

    # Join the synthetic tables chunkwise
    for synthetic_chunk in read_and_join_chunkwise(synthetic_files, keys, chunk_size=synthetic_chunk_size):
        # Calculate DCR for the joined synthetic chunk
        chunk_results = calculate_dcr_for_chunk(original_chunk, synthetic_chunk)
        results.extend(chunk_results)

    # Convert results to a DataFrame and save
    result_df = pd.DataFrame(results, columns=['original_pid', 'synthetic_pid', 'DCR'])
    output_file = os.path.join(output_path, f'chunk_{chunk_id}_dcr.csv')
    result_df.to_csv(output_file, index=False)


def calculate_dcr_for_chunk(original_chunk, synthetic_chunk):
    """
    Calculates the Distance to Closest Record (DCR) for each record in the original chunk
    based on the current synthetic chunk.

    Parameters:
        original_chunk (pd.DataFrame): A chunk of the original dataset.
        synthetic_chunk (pd.DataFrame): A chunk of the synthetic dataset.

    Returns:
        list: A list of tuples (original_pid, synthetic_pid, DCR).
    """
    results = []

    # Iterate through rows in the original chunk
    for original_idx, original_row in original_chunk.iterrows():
        # Calculate distances to all records in the synthetic chunk
        diff = synthetic_chunk.drop(columns=['pid']).values - original_row.drop(['pid']).values
        euclidean_distances = np.linalg.norm(diff, axis=1)

        # Find the index of the closest record
        closest_idx = np.argmin(euclidean_distances)
        closest_distance = euclidean_distances[closest_idx]
        closest_synthetic_pid = synthetic_chunk.iloc[closest_idx]['pid']

        # Append the result as (original_pid, synthetic_pid, DCR)
        results.append((original_row['pid'], closest_synthetic_pid, closest_distance))

    return results


if __name__ == "__main__":
    # Define paths to the CSV files
    original_files = {
        'primary': 'original_insurance_data.csv',
        'insurants': 'original_insurants.csv',
        'inpatient_cases': 'original_inpatient_cases.csv',
        # Add other tables...
    }
    synthetic_files = {
        'primary': 'synthetic_insurance_data.csv',
        'insurants': 'synthetic_insurants.csv',
        'inpatient_cases': 'synthetic_inpatient_cases.csv',
        # Add other tables...
    }

    # Keys to join on
    keys = ['pid', 'inpatient_caseID']  # Adjust as necessary

    # Output path for processed results
    output_path = './processed_dcr_chunks'

    # Chunk sizes
    original_chunk_size = 200000
    synthetic_chunk_size = 100000

    # Run the pipeline
    chunkwise_join_and_process(original_files, synthetic_files, keys, output_path, original_chunk_size, synthetic_chunk_size)
