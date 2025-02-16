import duckdb
import os
import json
from datetime import datetime
from scipy.stats import wasserstein_distance
from typing import List, Dict
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
import time

class NpEncoder(json.JSONEncoder):
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
    def __init__(self, results_dir: str = 'results', qi_inpatient=None, qi_outpatient=None, sensitive_attributes=None):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.selected_metrics = self.get_privacy_metrics_selection()
        self.qi_inpatient = qi_inpatient or []
        self.qi_outpatient = qi_outpatient or []
        self.sensitive_attributes = sensitive_attributes or []
        
    def get_database_list(self) -> List[str]:
        duckdb_dir = 'duckdb'
        if not os.path.exists(duckdb_dir):
            raise FileNotFoundError("duckdb directory not found")
        return [f for f in os.listdir(duckdb_dir) if f.endswith('.duckdb')]

    def get_joined_tables(self, db_path: str) -> List[str]:
        con = duckdb.connect(db_path, read_only=True)
        tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'joined%'").fetchall()
        con.close()
        return [table[0] for table in tables]

    def calculate_l_diversity(self, df: pd.DataFrame, quasi_identifiers: List[str], sensitive_attributes: List[str]) -> Dict:
        if not quasi_identifiers or not sensitive_attributes:
            return {"error": "Missing identifiers or sensitive attributes"}
        
        results = {}
        
        for sensitive_attr in sensitive_attributes:
            if sensitive_attr not in df.columns:
                continue
            
            groups = df.groupby(quasi_identifiers)
            l_values = []
            entropies = []
            
            for _, group in groups:
                value_counts = group[sensitive_attr].value_counts()
                distinct_count = value_counts.nunique()
                l_values.append(distinct_count)
                
                probabilities = value_counts / len(group)
                entropy = -np.sum(probabilities * np.log2(probabilities))
                entropies.append(entropy)
            
            # Avoid returning misleading zero values when no data is available
            results[sensitive_attr] = {
                "l_diversity": min(l_values) if l_values else None,
                "average_distinct_values": float(np.mean(l_values)) if l_values else None,
                "entropy_l_diversity": min(entropies) if entropies else None,
                "average_entropy": float(np.mean(entropies)) if entropies else None
            }
        
        print('Calculated l-diversity...')
        return results

    def calculate_t_closeness(self, df: pd.DataFrame, quasi_identifiers: List[str], 
                            sensitive_attributes: List[str], t_threshold: float = 0.15) -> Dict:
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

        print('Calculated t-closeness...')
        return results

    def calculate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> Dict:
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

        # Optimized group size distribution (1–10 and >10)
        group_size_counts = group_counts.value_counts()

        group_size_distribution = {
            **{i: group_size_counts.get(i, 0) for i in range(1, 11)},  # Sizes 1-10
            "groups_larger_than_10": group_size_counts[group_size_counts.index > 10].sum()
        }

        # Compute entropy for measuring diversity of group sizes
        probabilities = group_counts / len(df)
        entropy = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0

        # Privacy score: A lower variance in group sizes indicates better anonymization
        privacy_score = (1 - (unique_records / len(df))) * 100 if len(df) > 0 else 100

        print('Calculated k-anonymity...')

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
        con1 = duckdb.connect(db1_path, read_only=True)
        con2 = duckdb.connect(db2_path, read_only=True)
        
        df1 = con1.execute(f"SELECT * FROM {table_name}").fetch_df()
        df2 = con2.execute(f"SELECT * FROM {table_name}").fetch_df()
        
        con1.close()
        con2.close()
        
        print(f"\nAnalyzing table: {table_name}")
        
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"privacy_metrics_{db1_name}_{db2_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(
                obj=results,       # The object to serialize
                fp=f,              # File object to write into
                cls=NpEncoder,     # Optional: Custom encoder for NumPy types
                indent=2           # Optional: Pretty print with 2 spaces
            )
        print(f"\nResults saved to: {filepath}")

    def run_comparison(self):
        databases = self.get_database_list()
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
        
        db1_path = os.path.join('duckdb', db1)
        db2_path = os.path.join('duckdb', db2)
        
        tables1 = set(self.get_joined_tables(db1_path))
        tables2 = set(self.get_joined_tables(db2_path))
        
        common_tables = tables1 & tables2
        if not common_tables:
            print("No matching joined tables found between the databases.")
            return
        
        results = {
            "database1": db1,
            "database2": db2,
            "comparisons": []
        }
        
        for table in common_tables:
            comparison = self.compare_tables(db1_path, db2_path, table)
            results["comparisons"].append(comparison)
        
        self.save_results(results, db1.replace('.duckdb', ''), db2.replace('.duckdb', ''))

def timed_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.4f} seconds")
    return result

def main():
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
        "inpatient_procedure_procedure_code",
        "outpatient_diagnosis_diagnosis",
        "outpatient_procedure_procedure_code",    
    ]
    
    calculator = PrivacyMetricsCalculator(
        qi_inpatient=qi_inpatient,
        qi_outpatient=qi_outpatient,
        sensitive_attributes=sensitive_attributes
    )
    calculator.run_comparison()

if __name__ == "__main__":
    main()
   #timed_execution(main)