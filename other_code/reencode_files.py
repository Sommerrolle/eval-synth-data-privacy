import os
import pandas as pd

def reencode_csv_files(directory):
    """
    Reads all .csv files in the specified directory, re-encodes them to UTF-8,
    and overwrites the original files.

    Parameters:
    - directory (str): Path to the directory containing .csv files.
    """
    try:
        # List all files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]

        if not files:
            print("No .csv files found in the directory.")
            return

        for file in files:
            file_path = os.path.join(directory, file)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, encoding='latin1', sep='\t')  # Assuming input files might have different encodings
                # Save the CSV file with UTF-8 encoding
                df.to_csv(file_path, index=False, encoding='utf-8')
                print(f"Re-encoded and saved: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    except Exception as e:
        print(f"An error occurred while processing the directory: {e}")

# Example usage
directory_path = "/home/cvt/Documents/masterarbeit/Claims Data"  # Replace with the path to your directory
reencode_csv_files(directory_path)
