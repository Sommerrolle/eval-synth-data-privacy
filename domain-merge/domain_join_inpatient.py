import argparse
import dask.dataframe as dd
import os
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

def main(testdata_path, output_path):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Define file paths for the input CSV files
    file_paths = {
        'insurance_data': f'{testdata_path}/sle.insurance_data.csv',
        'insurants': f'{testdata_path}/sle.insurants.csv',
        'inpatient_cases': f'{testdata_path}/sle.inpatient_cases.csv',
        'inpatient_diagnosis': f'{testdata_path}/sle.inpatient_diagnosis.csv',
        'inpatient_procedures': f'{testdata_path}/sle.inpatient_procedures.csv',
        'inpatient_fees': f'{testdata_path}/sle.inpatient_fees.csv',
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
            'year_of_birth': int,
            'gender': int,
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
        #'inpatient_fees': ['from', 'to'],
        'inpatient_procedures': ['date of procedure'],
        'outpatient_cases': ['from', 'to'],
        'outpatient_fees': ['date'],
        'drugs': ['date of prescription', 'date of dispense']
    }

    # Read and process each CSV
    dataframes = {}
    for table_name, file_path in file_paths.items():
        # Determine dataset type
        dataset_type = "inpatient" if "inpatient" in table_name else "outpatient" if "outpatient" in table_name else None

        # Read CSV
        df = dd.read_csv(
            file_path,
            sep='\t',
            dtype=dtypes.get(table_name, None),
            parse_dates=parse_dates.get(table_name, None),
            assume_missing=True
        )

        # Rename columns
        df = rename_columns(df, prefix=table_name, exceptions=['pid'], dataset_type=dataset_type)

        # Store the processed DataFrame
        dataframes[table_name] = df

    # Merge datasets step by step
    # Example: Merging 'insurance_data' with 'insurants'
    df_merged = dd.merge(dataframes['insurance_data'], dataframes['insurants'], on='pid', how='left')
    # Merge inpatient_cases
    df_merged = dd.merge(df_merged, dataframes['inpatient_cases'], on=['pid'], how='left')
    # Merge inpatient_diagnosis on both pid and inpatient_caseID
    df_merged = dd.merge(df_merged, dataframes['inpatient_diagnosis'], on=['pid', 'inpatient_caseID'], how='left')
    # Merge inpatient_procedures on both pid and inpatient_caseID
    df_merged = dd.merge(df_merged, dataframes['inpatient_procedures'], on=['pid', 'inpatient_caseID'], how='left')
    # Merge inpatient_fees on both pid and inpatient_caseID
    # df_merged = dd.merge(df_merged, dataframes['inpatient_fees'], on=['pid', 'inpatient_caseID'], how='left')
    # Merge drugs on pid only (no caseID in drugs)
    # df_merged = dd.merge(df_merged, dataframes['drugs'], on='pid', how='left')

    print('Merge done.. now save that shit')

    # Compute and save the result to CSV
    result_path = os.path.join(output_path, 'inpatient_with_drugs.csv')
    df_merged.compute().to_csv(result_path, index=False)
    print(f"Merging complete. Output saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV datasets using Dask")
    parser.add_argument(
        '--testdata_path',
        type=str,
        required=True,
        help="Base path to the testdata folder containing input CSV files"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path to the folder where the output should be saved"
    )
    args = parser.parse_args()

    main(args.testdata_path, args.output_path)
