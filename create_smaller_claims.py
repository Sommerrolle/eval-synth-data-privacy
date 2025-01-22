import pandas as pd

def process_csv(input_file, output_file, rows=500):
    """
    Reads the first `rows` rows of a CSV file and saves them to a new file.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to the output CSV file.
    - rows (int): Number of rows to read from the input file. Default is 500.
    """
    try:
        # Define the data types for each column
        dtype = {
            'pid': 'int',
            'year of birth': 'Int64',
            'gender': 'Int64',
        }
        
        # Read the first `rows` rows with specific dtypes
        df = pd.read_csv(input_file, nrows=rows, sep='\t', dtype=dtype)
        # Save to a new CSV
        df.to_csv(output_file, index=False, sep='\t')
        print(f"Saved the first {rows} rows from {input_file} to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_csv = "/home/cvt/Documents/masterarbeit/claims_data/sle.insurants.csv"  # Replace with your input CSV file path
output_csv = "/home/cvt/Documents/masterarbeit/claims_data/sle.insurants_500.csv"  # Replace with your desired output file path
process_csv(input_csv, output_csv)