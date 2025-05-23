import os
import json
from datetime import datetime
from scipy.stats import wasserstein_distance
from typing import List, Dict
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import logging
from duckdb_manager.duckdb_manager import DuckDBManager
from feature_preprocessing import preprocess_single_dataframe


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/privacy_metrics.log'),
        logging.StreamHandler()
    ]
)

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class PrivacyMetricsCalculator:
    """Calculate privacy metrics for synthetic health claims data"""
    
    def __init__(self, results_dir: str = 'results/privacy_calculator', qi_inpatient=None, qi_outpatient=None, qi_drugs=None, sensitive_attributes=None, sample_size=5000000):
        """
        Initialize the PrivacyMetricsCalculator with the specified parameters.
        
        Args:
            results_dir: Directory to store results
            qi_inpatient: List of quasi-identifiers for inpatient data
            qi_outpatient: List of quasi-identifiers for outpatient data
            sensitive_attributes: List of sensitive attributes
            sample_size: Maximum number of samples to use for calculations
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.selected_metrics = self.get_privacy_metrics_selection()
        self.qi_inpatient = qi_inpatient or []
        self.qi_outpatient = qi_outpatient or []
        self.qi_drugs = qi_drugs or []
        self.sensitive_attributes = sensitive_attributes or []
        self.db_manager = DuckDBManager()
        self.sample_size = self.get_sample_size_input(default=sample_size)
        

    def calculate_l_diversity(self, df: pd.DataFrame, quasi_identifiers: List[str], sensitive_attributes: List[str]) -> Dict:
        """
        Calculate l-diversity aligned with the external library implementation pyCANON.
        
        Args:
            df: DataFrame containing the data
            quasi_identifiers: List of quasi-identifier column names
            sensitive_attributes: List of sensitive attribute column names
            
        Returns:
            Dictionary containing l-diversity metrics including entropy l-diversity
        """
        if not quasi_identifiers or not sensitive_attributes:
            return {"error": "Missing identifiers or sensitive attributes"}
        
        results = {}
        
        for sensitive_attr in sensitive_attributes:
            if sensitive_attr not in df.columns:
                continue
            
            groups = df.groupby(quasi_identifiers)
            l_values = []
            entropy_values = []
            entropy_l_values = []
            problematic_groups = 0
            total_groups = 0
            
            for _, group in groups:
                total_groups += 1
                
                # Get values of sensitive attribute in this group and handle None values
                values = group[sensitive_attr].fillna('NULL_VALUE').values
                
                # Count distinct values for traditional l-diversity
                distinct_count = len(np.unique(values))
                l_values.append(distinct_count)
                
                if distinct_count == 1:
                    problematic_groups += 1
                    entropy_values.append(0)  # No diversity means zero entropy
                    entropy_l_values.append(1)  # e^0 = 1
                else:
                    # Calculate entropy using natural logarithm (as in the library)
                    unique_values, counts = np.unique(values, return_counts=True)
                    p = counts / len(values)
                    entropy = -np.sum(p * np.log(p))  # Natural log
                    entropy_values.append(entropy)
                    
                    # Calculate entropy l-diversity value: e^entropy
                    entropy_l = np.exp(entropy)
                    entropy_l_values.append(entropy_l)
            
            # Calculate metrics
            min_l = min(l_values) if l_values else 0
            min_entropy = min(entropy_values) if entropy_values else 0
            min_entropy_l = min(entropy_l_values) if entropy_l_values else 1
            
            results[sensitive_attr] = {
                "l_diversity": min_l,
                "entropy_raw": float(min_entropy),
                "entropy_l_diversity": int(min_entropy_l),  # Convert to integer as in the library pyCANON
                "average_entropy": float(np.mean(entropy_values)) if entropy_values else None,
                "average_distinct_values": float(np.mean(l_values)) if l_values else 0,
                "problematic_groups_count": problematic_groups,
                "total_groups": total_groups,
                "problematic_groups_percentage": (problematic_groups / total_groups * 100) if total_groups > 0 else 0,
            }
        
        logging.info('Calculated l-diversity')
        return results

    def calculate_t_closeness(self, df: pd.DataFrame, quasi_identifiers: List[str],
                            sensitive_attributes: List[str], t_threshold: float = 0.15) -> Dict:
        """
        Calculate t-closeness for the dataset using preprocessed features.
        
        Args:
            df: DataFrame containing the data
            quasi_identifiers: List of quasi-identifier column names
            sensitive_attributes: List of sensitive attribute column names
            t_threshold: Threshold for t-closeness violation
            
        Returns:
            Dictionary containing t-closeness metrics
        """
        if not quasi_identifiers or not sensitive_attributes:
            return {"error": "Missing identifiers or sensitive attributes"}
        
        results = {}
        
        for sensitive_attr in sensitive_attributes:
            if sensitive_attr not in df.columns:
                continue
            
            # Create a copy of the dataframe with only the columns we need
            columns_to_use = quasi_identifiers + [sensitive_attr]
            df_subset = df[columns_to_use].copy()
            
            # Create a new dataframe with only the sensitive attribute for preprocessing
            df_sensitive = df[[sensitive_attr]].copy()
            
            # Use the new preprocessing function instead of LabelEncoder
            df_sensitive_processed = preprocess_single_dataframe(df_sensitive)
            
            # The processed column might have a different name if it's a medical code
            # If it's a medical code, the preprocessor would add '_numeric' suffix
            if f"{sensitive_attr}_numeric" in df_sensitive_processed.columns:
                processed_attr_name = f"{sensitive_attr}_numeric"
            else:
                processed_attr_name = sensitive_attr
            
            # Replace the original column with the processed version
            df_subset[sensitive_attr] = df_sensitive_processed[processed_attr_name]
            
            # Compute the global distribution of the sensitive attribute
            global_dist = df_subset[sensitive_attr].value_counts(normalize=True).sort_index()
            
            groups = df_subset.groupby(quasi_identifiers)
            t_values = []
            
            for _, group in groups:
                group_dist = group[sensitive_attr].value_counts(normalize=True).sort_index()
                
                # Align global and group distributions
                all_values = sorted(set(global_dist.index) | set(group_dist.index))
                global_aligned = global_dist.reindex(all_values, fill_value=0)
                group_aligned = group_dist.reindex(all_values, fill_value=0)
                
                # Compute Wasserstein Distance (EMD) using numerical values
                emd = wasserstein_distance(
                    all_values, all_values,  # Use the processed numerical values
                    u_weights=global_aligned.values,
                    v_weights=group_aligned.values
                )
                t_values.append(emd)
            
            results[sensitive_attr] = {
                "t_closeness": max(t_values) if t_values else None,
                "average_distance": float(np.mean(t_values)) if t_values else None,
                "groups_violating_t": sum(1 for t in t_values if t > t_threshold),
                "total_groups": len(t_values)
            }
        
        logging.info('Calculated t-closeness')
        return results

    def calculate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> Dict:
        """
        Calculate k-anonymity for the dataset.
        
        Args:
            df: DataFrame containing the data
            quasi_identifiers: List of quasi-identifier column names
            
        Returns:
            Dictionary containing k-anonymity metrics
        """
        if df.empty or not quasi_identifiers or not all(qi in df.columns for qi in quasi_identifiers):
            return {"error": "Missing, empty dataset, or invalid quasi-identifiers"}
        
        # Group by quasi-identifiers and count occurrences
        group_counts = df.groupby(quasi_identifiers).size()

        if group_counts.empty:
            return {"error": "No valid groups found after grouping by quasi-identifiers"}

        # Compute k-anonymity metrics
        k_value = group_counts.min()  # Smallest group size defines k
        avg_group_size = group_counts.mean()
        total_groups = len(group_counts)
        unique_records = (group_counts == 1).sum()
        vulnerable_groups = (group_counts < 5).sum()
        high_risk_percentage = (unique_records / len(df)) * 100 if len(df) > 0 else 0

        # Optimized group size distribution (1-10 and >10)
        group_size_counts = group_counts.value_counts()

        group_size_distribution = {
            **{f"groups_of_size_{i}": int(group_size_counts.get(i, 0)) for i in range(1, 11)},  # Sizes 1-10
            "groups_larger_than_10": int(group_size_counts[group_size_counts.index > 10].sum())
        }

        # Compute entropy for measuring diversity of group sizes
        probabilities = group_counts / len(df)
        entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0

        # Privacy score: A lower variance in group sizes indicates better anonymization
        privacy_score = (1 - (unique_records / len(df))) * 100 if len(df) > 0 else 100

        logging.info('Calculated k-anonymity')

        return {
            "k_anonymity": int(k_value),
            "average_group_size": float(avg_group_size),
            "total_groups": int(total_groups),
            "unique_records": int(unique_records),
            "vulnerable_groups": int(vulnerable_groups),
            "high_risk_percentage": float(high_risk_percentage),
            "group_size_distribution": group_size_distribution,
            "entropy": float(entropy),
            "total_records": len(df),
            "privacy_score": float(privacy_score)
        }
    
    # def sample_data_using_duckdb(self, db_path: str, table_name: str, sample_size: int) -> pd.DataFrame:
    #     """
    #     Sample data from a table using DuckDB's sampling capabilities.
        
    #     Args:
    #         db_path: Path to the database
    #         table_name: Name of the table to sample from
    #         sample_size: Desired number of samples
            
    #     Returns:
    #         DataFrame containing the sampled data
    #     """
    #     # Get the total row count first
    #     total_count = self.db_manager.get_table_count(db_path, table_name)
        
    #     if sample_size >= total_count:
    #         # If sample size is larger than or equal to total count, just load the entire table
    #         df = self.db_manager.load_table_data(db_path, table_name)
    #         logging.info(f"Loaded entire table with {len(df):,} rows")
    #         return df
        
    #     # Try using TABLESAMPLE to get a random sample
    #     sampling_percentage = (sample_size / total_count) * 100
    #     query = f"SELECT * FROM {table_name} USING SAMPLE {sampling_percentage}% (reservoir)"
        
    #     try:
    #         result = self.db_manager.execute_query(db_path, query)
    #         df = pd.DataFrame(result)
            
    #         # Check if we got approximately the right number of rows
    #         if len(df) < sample_size * 0.9:
    #             logging.warning(f"Sample size too small: requested {sample_size}, got {len(df)}")
                
    #             # Try using ORDER BY RANDOM() LIMIT instead
    #             query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
    #             result = self.db_manager.execute_query(db_path, query)
    #             df = pd.DataFrame(result)
    #             logging.info(f"Resampled to {len(df):,} rows using RANDOM() LIMIT")
    #         elif len(df) > sample_size * 1.1:
    #             logging.warning(f"Sample size too large: requested {sample_size}, got {len(df)}")
    #             # If we got too many rows, subsample using pandas
    #             df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    #             logging.info(f"Subsampled to {len(df):,} rows using pandas")
                
    #         # Check for duplicates
    #         duplicates = df.duplicated().sum()
    #         logging.info(f"Duplicates in sampled data: {duplicates:,}")
            
    #         return df
        
    #     except Exception as e:
    #         logging.error(f"Error sampling data: {str(e)}")
    #         # Fall back to loading the entire table and sampling with pandas
    #         logging.info("Falling back to loading entire table and sampling with pandas")
    #         df = self.db_manager.load_table_data(db_path, table_name)
            
    #         if len(df) > sample_size:
    #             df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    #             logging.info(f"Sampled to {len(df):,} rows using pandas")
            
    #         duplicates = df.duplicated().sum()
    #         logging.info(f"Duplicates in sampled data: {duplicates:,}")
            
    #         return df

    def analyze_table(self, db_path: str, table_name: str) -> Dict:
        """
        Analyze privacy metrics for a specific table in a database.
        
        Args:
            db_path: Path to the database
            table_name: Name of the table to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Use the DuckDBManager to load the data
        df = self.db_manager.load_table_data(db_path, table_name)
        # df = self.sample_data_using_duckdb(db_path, table_name, self.sample_size)
        
        if df.empty:
            logging.error(f"Failed to load data from {table_name}")
            return {"error": f"Failed to load data from {table_name}"}
        
        total_count = len(df)
        actual_sample_size = min(self.sample_size, total_count)
        
        logging.info(f"Table {table_name} has {total_count:,} rows, using {actual_sample_size:,} for analysis")
        print(f"Table {table_name} has {total_count:,} rows, using {actual_sample_size:,} for analysis")
        
        # Sample the data to make computation feasible if necessary
        if actual_sample_size < total_count:
            df = df.sample(n=actual_sample_size, random_state=42).reset_index(drop=True)
            
            # Check for duplicates in the sample
            duplicates = df.duplicated().sum()
            logging.info(f"Duplicates in sampled data: {duplicates}")
        
        logging.info(f"Analyzing table: {table_name} with {len(df):,} rows")
        
        # Select appropriate quasi-identifiers based on table name
        if 'inpatient' in table_name:  # inpatient table
            quasi_identifiers = self.qi_inpatient
        elif 'outpatient' in table_name:  # outpatient table
            quasi_identifiers = self.qi_outpatient
        elif 'drugs' in table_name:  # drugs table
            quasi_identifiers = self.qi_drugs
        else:
            raise ValueError(f"Unknown table type: {table_name}")
            
        results = {}
        
        if 'k-anonymity' in self.selected_metrics:
            results.update(self.calculate_k_anonymity(df, quasi_identifiers))
            
        if 'l-diversity' in self.selected_metrics:
            results['l_diversity'] = self.calculate_l_diversity(df, quasi_identifiers, self.sensitive_attributes)
            
        if 't-closeness' in self.selected_metrics:
            results['t_closeness'] = self.calculate_t_closeness(df, quasi_identifiers, self.sensitive_attributes)
        
        return {
            "table_name": table_name,
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": self.sensitive_attributes,
            "sample_size_used": len(df),
            "total_table_size": total_count,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    def get_privacy_metrics_selection(self) -> List[str]:
        """
        Get user input for which privacy metrics to calculate.
        
        Returns:
            List of selected metric names
        """
        metrics = {
            '1': 'k-anonymity',
            '2': 'l-diversity',
            '3': 't-closeness'
        }
        
        print("\nAvailable privacy metrics:")
        for key, metric in metrics.items():
            print(f"{key}. {metric}")
        
        while True:
            print("\nEnter numbers for desired metrics (space-separated):")
            selections = input("> ").strip().split()
            if all(s in metrics for s in selections):
                return [metrics[s] for s in selections]
            print("Invalid selection. Please try again.")

    def save_results(self, results: Dict, db_name: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            results: Dictionary containing analysis results
            db_name: Name of the database
        """
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"privacy_metrics_{db_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(
                obj=results,
                fp=f,
                cls=NpEncoder,
                indent=2
            )
        logging.info(f"Results saved to: {filepath}")

    def get_sample_size_input(self, default=5000000):
        """
        Get user input for the sample size to use for calculations.
        
        Args:
            default: Default sample size if no input is provided
            
        Returns:
            Integer sample size
        """
        while True:
            try:
                print(f"\nEnter maximum number of samples to use for calculations (default: {default:,}):")
                user_input = input("> ").strip()
                
                if not user_input:
                    print(f"Using default sample size of {default:,}")
                    return default
                
                sample_size = int(user_input.replace(',', ''))
                if sample_size <= 0:
                    print("Sample size must be positive. Please try again.")
                    continue
                
                print(f"Using sample size of {sample_size:,}")
                return sample_size
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def run_analysis(self):
        """Run the analysis workflow with user interaction"""
        # Use the DuckDBManager to get the database list
        databases = self.db_manager.get_database_list()
        if not databases:
            print("No databases found in the directory.")
            return
        
        print("\nAvailable databases:")
        for i, db in enumerate(databases, 1):
            print(f"{i}. {db}")
        
        while True:
            try:
                print("\nSelect a database to analyze (enter number):")
                db_idx = int(input("> ").strip())
                if 1 <= db_idx <= len(databases):
                    selected_db = databases[db_idx - 1]
                    break
                print(f"Please enter a number between 1 and {len(databases)}")
            except (ValueError, IndexError):
                print("Invalid selection. Please enter a valid number.")
        
        db_path = self.db_manager.get_database_path(selected_db)
        
        # Use the DuckDBManager to get tables
        tables = self.db_manager.get_joined_tables(db_path)
        if not tables:
            print("No tables found in the database.")
            return
        
        # Filter for relevant tables if desired
        # For example, only comprehensive tables or tables matching certain patterns
        # filtered_tables = [table for table in tables if "all" in table.lower()]
        filtered_tables = tables
        
        if not filtered_tables:
            print("No matching tables found in the database.")
            return
        
        print("\nFound the following tables:")
        for i, table in enumerate(filtered_tables, 1):
            table_count = self.db_manager.get_table_count(db_path, table)
            print(f"{i}. {table} ({table_count:,} rows)")
        
        while True:
            try:
                print("\nSelect tables to analyze (space-separated numbers, or 'all' for all tables):")
                selection = input("> ").strip()
                if selection.lower() == 'all':
                    selected_tables = filtered_tables
                    break
                
                table_indices = [int(idx) for idx in selection.split()]
                if all(1 <= idx <= len(filtered_tables) for idx in table_indices):
                    selected_tables = [filtered_tables[idx - 1] for idx in table_indices]
                    break
                print(f"Please enter numbers between 1 and {len(filtered_tables)}")
            except (ValueError, IndexError):
                print("Invalid selection. Please enter valid numbers or 'all'.")
            
        results = {
            "database": selected_db,
            "sample_size_requested": self.sample_size,
            "analyses": []
        }
        
        for table in selected_tables:
            logging.info(f"Starting analysis for table: {table}")
            analysis = self.analyze_table(db_path, table)
            results["analyses"].append(analysis)
        
        db_name = selected_db.replace('.duckdb', '')
        self.save_results(results, db_name)


def main():
    """Main function to run the privacy metrics calculation"""
    # Define quasi-identifiers and sensitive attributes
    qi_inpatient = [
        "insurants_year_of_birth",
        "insurants_gender",
        "insurance_data_regional_code",
        "inpatient_cases_date_of_admission",
        "inpatient_cases_department_admission"
    ]
    
    qi_outpatient = [
        "insurants_year_of_birth",
        "insurants_gender",
        "insurance_data_regional_code",
        "outpatient_cases_from",
        "outpatient_cases_to",
        "outpatient_cases_practice_code",
        "outpatient_cases_year",
        "outpatient_cases_quarter"
    ]

    qi_drugs = [
        "insurants_year_of_birth",
        "insurants_gender",
        "insurance_data_regional_code",
        "drugs_date_of_prescription",
        "drugs_date_of_dispense"
    ]
    
    sensitive_attributes = [
        "inpatient_diagnosis_diagnosis",
        "inpatient_procedures_procedure_code",
        "outpatient_diagnosis_diagnosis",
        "outpatient_procedures_procedure_code",
        'drugs_pharma_central_number',
        'drugs_specialty_of_prescriber',
        'drugs_atc'   
    ]
    
    calculator = PrivacyMetricsCalculator(
        qi_inpatient=qi_inpatient,
        qi_outpatient=qi_outpatient,
        qi_drugs=qi_drugs,
        sensitive_attributes=sensitive_attributes
    )
    
    calculator.run_analysis()

if __name__ == "__main__":
    main()