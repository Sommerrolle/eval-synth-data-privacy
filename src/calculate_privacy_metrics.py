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
from duckdb_manager import DuckDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('privacy_metrics.log'),
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
    
    def __init__(self, results_dir: str = 'results/privacy_calculator', qi_inpatient=None, qi_outpatient=None, sensitive_attributes=None):
        """
        Initialize the PrivacyMetricsCalculator with the specified parameters.
        
        Args:
            results_dir: Directory to store results
            qi_inpatient: List of quasi-identifiers for inpatient data
            qi_outpatient: List of quasi-identifiers for outpatient data
            sensitive_attributes: List of sensitive attributes
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.selected_metrics = self.get_privacy_metrics_selection()
        self.qi_inpatient = qi_inpatient or []
        self.qi_outpatient = qi_outpatient or []
        self.sensitive_attributes = sensitive_attributes or []
        self.db_manager = DuckDBManager()
        

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
        Calculate t-closeness for the dataset.
        
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
            
            # Encode categorical sensitive attributes
            encoder = LabelEncoder()
            df[sensitive_attr] = encoder.fit_transform(df[sensitive_attr])

            # Compute the global distribution of the sensitive attribute
            global_dist = df[sensitive_attr].value_counts(normalize=True).sort_index()
            
            groups = df.groupby(quasi_identifiers)
            t_values = []
            
            for _, group in groups:
                group_dist = group[sensitive_attr].value_counts(normalize=True).sort_index()
                
                # Align global and group distributions
                all_values = sorted(set(global_dist.index) | set(group_dist.index))
                global_aligned = global_dist.reindex(all_values, fill_value=0)
                group_aligned = group_dist.reindex(all_values, fill_value=0)
                
                # Compute Wasserstein Distance (EMD) using numerical values
                emd = wasserstein_distance(
                    all_values, all_values,  # Use the encoded numerical values
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

    def compare_tables(self, db1_path: str, db2_path: str, table_name: str) -> Dict:
        """
        Compare privacy metrics between two versions of the same table in different databases.
        
        Args:
            db1_path: Path to the first database
            db2_path: Path to the second database
            table_name: Name of the table to compare
            
        Returns:
            Dictionary containing comparison results
        """
        # Use the DuckDBManager to load the data
        df1 = self.db_manager.load_table_data(db1_path, table_name)
        df2 = self.db_manager.load_table_data(db2_path, table_name)
        
        if df1.empty or df2.empty:
            logging.error(f"Failed to load data from {table_name}")
            return {"error": f"Failed to load data from {table_name}"}
        
        logging.info(f"Analyzing table: {table_name}")
        
        # Select appropriate quasi-identifiers based on table name
        if '_4_5_6_7' in table_name:  # inpatient table
            quasi_identifiers = self.qi_inpatient
        elif '_8_9_10_11' in table_name:  # outpatient table
            quasi_identifiers = self.qi_outpatient
        else:
            raise ValueError(f"Unknown table type: {table_name}")
            
        results1 = {}
        results2 = {}
        
        if 'k-anonymity' in self.selected_metrics:
            results1.update(self.calculate_k_anonymity(df1, quasi_identifiers))
            results2.update(self.calculate_k_anonymity(df2, quasi_identifiers))
            
        if 'l-diversity' in self.selected_metrics:
            results1['l_diversity'] = self.calculate_l_diversity(df1, quasi_identifiers, self.sensitive_attributes)
            results2['l_diversity'] = self.calculate_l_diversity(df2, quasi_identifiers, self.sensitive_attributes)
            
        if 't-closeness' in self.selected_metrics:
            results1['t_closeness'] = self.calculate_t_closeness(df1, quasi_identifiers, self.sensitive_attributes)
            results2['t_closeness'] = self.calculate_t_closeness(df2, quasi_identifiers, self.sensitive_attributes)
        
        return {
            "table_name": table_name,
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": self.sensitive_attributes,
            "dataset1_results": results1,
            "dataset2_results": results2,
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

    def save_results(self, results: Dict, db1_name: str, db2_name: str):
        """
        Save comparison results to a JSON file.
        
        Args:
            results: Dictionary containing comparison results
            db1_name: Name of the first database
            db2_name: Name of the second database
        """
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"privacy_metrics_{db1_name}_{db2_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(
                obj=results,
                fp=f,
                cls=NpEncoder,
                indent=2
            )
        logging.info(f"Results saved to: {filepath}")

    def run_comparison(self):
        """Run the comparison workflow with user interaction"""
        # Use the DuckDBManager to get the database list
        databases = self.db_manager.get_database_list()
        if len(databases) < 2:
            print("Need at least 2 databases to compare.")
            return
        
        print("\nAvailable databases:")
        for i, db in enumerate(databases, 1):
            print(f"{i}. {db}")
        
        while True:
            try:
                print("\nSelect two databases to compare (enter two numbers):")
                db1_idx, db2_idx = map(int, input("> ").strip().split())
                db1 = databases[db1_idx - 1]
                db2 = databases[db2_idx - 1]
                break
            except (ValueError, IndexError):
                print("Invalid selection. Please enter two valid numbers.")
        
        db1_path = self.db_manager.get_database_path(db1)
        db2_path = self.db_manager.get_database_path(db2)
        
        # Use the DuckDBManager to get common tables
        common_tables = self.db_manager.get_common_tables(db1_path, db2_path)
        if not common_tables:
            print("No matching joined tables found between the databases.")
            return
        
        print("\nFound the following common tables:")
        for i, table in enumerate(common_tables, 1):
            table_count1 = self.db_manager.get_table_count(db1_path, table)
            table_count2 = self.db_manager.get_table_count(db2_path, table)
            print(f"{i}. {table} ({table_count1:,} rows vs {table_count2:,} rows)")
            
        results = {
            "database1": db1,
            "database2": db2,
            "comparisons": []
        }
        
        for table in common_tables:
            logging.info(f"Starting comparison for table: {table}")
            comparison = self.compare_tables(db1_path, db2_path, table)
            results["comparisons"].append(comparison)
        
        db1_name = db1.replace('.duckdb', '')
        db2_name = db2.replace('.duckdb', '')
        self.save_results(results, db1_name, db2_name)


def main():
    """Main function to run the privacy metrics calculation"""
    # Define quasi-identifiers and sensitive attributes
    qi_inpatient = [
        "insurants_year_of_birth",
        "insurants_gender",
        "inpatient_cases_date_of_admission",
        "inpatient_cases_department_admission"
    ]
    
    qi_outpatient = [
        "insurants_year_of_birth",
        "insurants_gender",
        "outpatient_cases_from",
        "outpatient_cases_practice_code",
        "outpatient_cases_year",
        "outpatient_cases_quarter"
    ]
    
    sensitive_attributes = [
        "inpatient_diagnosis_diagnosis",
        "inpatient_procedures_procedure_code",
        "outpatient_diagnosis_diagnosis",
        "outpatient_procedures_procedure_code",    
    ]
    
    calculator = PrivacyMetricsCalculator(
        qi_inpatient=qi_inpatient,
        qi_outpatient=qi_outpatient,
        sensitive_attributes=sensitive_attributes
    )
    
    calculator.run_comparison()

if __name__ == "__main__":
    main()