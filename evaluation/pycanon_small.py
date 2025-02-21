# import pycanon as pc
from pycanon.report import pdf
import duckdb
import pandas as pd

def export_table_to_csv(db_path: str, table_name: str, output_path: str):
   """
   Export a DuckDB table to CSV file.
   """
   try:
       # Connect to database
       con = duckdb.connect(db_path)
       
       # Export table
       con.execute(f"COPY {table_name} TO '{output_path}' (HEADER, DELIMITER ',')")
       
       print(f"Successfully exported {table_name} to {output_path}")
   
   except Exception as e:
       print(f"Error exporting table: {str(e)}")
   
   finally:
       con.close()



def main():
    ##export_table_to_csv("duckdb\claims_data.duckdb", "joined_1_8_9_10_11", "claims_data_joined_1_8_9_10_11.csv")
    #FILE_NAME = "claims_data_joined_1_8_9_10_11.csv"
    FILE_NAME = "testdata\healthcare-dataset-stroke-data.csv"
    # QI = [
    #     "insurants_year_of_birth",
    #     "insurants_gender",
    #     "outpatient_cases_from",
    #     "outpatient_cases_practice_code",
    #     "outpatient_cases_year",
    #     "outpatient_cases_quarter"
    #   ]
    # SA = ["outpatient_diagnosis_diagnosis",
    #       "outpatient_procedures_procedure_code"]
    
    QI = [
            'gender',
            'age',
            'hypertension',
            'heart_disease',
            'ever_married',
            'work_type',
            'Residence_type',
            'smoking_status'
        ]
        
    SA = ['stroke']
    
    DATA = pd.read_csv(FILE_NAME)

    # # Calculate k for k-anonymity:
    # k = pc.anonymity.k_anonymity(DATA, QI)

    # # Print the anonymity report:
    # pc.report.print_report(DATA, QI, SA)

    #FILE_PDF = "report_claimsdata.pdf"
    FILE_PDF = "report_stroke_data.pdf"
    pdf.get_pdf_report(DATA, QI, SA, file_pdf = FILE_PDF)

if __name__ == "__main__":
    main()

