import duckdb
import pandas as pd

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


def main():
    # Define DuckDB connection
    con = duckdb.connect(database='claims_data.duckdb', read_only=False)

    # Process and create tables
    for table_name, file_path in file_paths.items():
        # Determine dataset type
        dataset_type = "inpatient" if "inpatient" in table_name else "outpatient" if "outpatient" in table_name else None

        # Read the CSV data
        pandas_df = read_data(file_path, dtypes.get(table_name, None), table_name, parse_dates)

        # Rename columns
        renamed_df = rename_columns(pandas_df, prefix=table_name, exceptions=['pid'], dataset_type=dataset_type)

        # Create a DuckDB table
        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM renamed_df")

        # Print debug info
        print(f"Table '{table_name}' created in DuckDB with {len(renamed_df)} rows.")

    # Close the connection
    con.close()
    print("All tables have been created in DuckDB.")

if __name__ == "__main__":
    main()
