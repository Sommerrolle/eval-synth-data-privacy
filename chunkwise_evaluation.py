import pandas as pd
import numpy as np
import os

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

def process_original_chunk(original_chunk, synthetic_file, output_path, chunk_id, synthetic_chunk_size=100000):
    """
    Processes a single chunk of the original dataset bDistance to Closest Record (DCR)the original dataset.
        synthetic_file (str): Path to the synthetic dataset file.
        output_path (str): Path to save the results.
        chunk_id (int): Identifier for the chunk being processed.
        synthetic_chunk_size (int): Number of rows to process per synthetic chunk.

    Returns:
        None
    """
    results = []

    # Process the synthetic dataset chunk by chunk
    for synthetic_chunk in pd.read_csv(synthetic_file, chunksize=synthetic_chunk_size):
        # Calculate distances for the current synthetic chunk
        chunk_results = calculate_dcr_for_chunk(original_chunk, synthetic_chunk)
        results.extend(chunk_results)

    # Convert results to a DataFrame and save
    result_df = pd.DataFrame(results, columns=['original_pid', 'synthetic_pid', 'DCR'])
    output_file = os.path.join(output_path, f'chunk_{chunk_id}_dcr.csv')
    result_df.to_csv(output_file, index=False)

def chunkwise_dcr_pipeline(original_file, synthetic_file, output_path, original_chunk_size=200000, synthetic_chunk_size=100000):
    """
    Processes the original and synthetic datasets chunkwise to calculate DCR.

    Parameters:
        original_file (str): Path to the original dataset file.
        synthetic_file (str): Path to the synthetic dataset file.
        output_path (str): Directory to save intermediate results.
        original_chunk_size (int): Number of rows per original dataset chunk.
        synthetic_chunk_size (int): Number of rows per synthetic dataset chunk.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Process the original dataset chunk by chunk
    chunk_id = 0
    for original_chunk in pd.read_csv(original_file, chunksize=original_chunk_size):
        process_original_chunk(original_chunk, synthetic_file, output_path, chunk_id, synthetic_chunk_size)
        chunk_id += 1

    print("Chunkwise DCR processing complete. Results saved to:", output_path)

if __name__ == "__main__":
    # Paths to the datasets
    original_file = 'original_dataset.csv'
    synthetic_file = 'synthetic_dataset.csv'

    # Output path for processed chunks
    output_path = './processed_dcr_chunks'

    # Chunk sizes
    original_chunk_size = 200000
    synthetic_chunk_size = 100000

    # Run the pipeline
    chunkwise_dcr_pipeline(original_file, synthetic_file, output_path, original_chunk_size, synthetic_chunk_size)
