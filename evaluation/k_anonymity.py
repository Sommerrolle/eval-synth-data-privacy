import duckdb
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd

class KAnonymityCalculator:
    def __init__(self, results_dir: str = 'results'):
        """Initialize the calculator with a results directory."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def get_database_list(self) -> List[str]:
        """Get list of available DuckDB databases in the duckdb directory."""
        duckdb_dir = 'duckdb'
        if not os.path.exists(duckdb_dir):
            raise FileNotFoundError("duckdb directory not found")
        
        databases = [f for f in os.listdir(duckdb_dir) if f.endswith('.duckdb')]
        return databases

    def get_joined_tables(self, db_path: str) -> List[str]:
        """Get list of tables starting with 'joined' from a database."""
        con = duckdb.connect(db_path, read_only=True)
        tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'joined%'").fetchall()
        con.close()
        return [table[0] for table in tables]

    def get_quasi_identifiers(self, table_columns: List[str]) -> List[str]:
        """Let user select quasi-identifiers from available columns."""
        print("\nAvailable columns:")
        for i, col in enumerate(table_columns, 1):
            print(f"{i}. {col}")
        
        print("\nEnter the numbers of the columns to use as quasi-identifiers (space-separated):")
        while True:
            try:
                selections = input("> ").strip().split()
                indices = [int(s) - 1 for s in selections]
                if all(0 <= i < len(table_columns) for i in indices):
                    return [table_columns[i] for i in indices]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter valid numbers separated by spaces.")

    def calculate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> Dict:
        """Calculate detailed k-anonymity metrics for given dataframe and quasi-identifiers."""
        if not quasi_identifiers:
            return {"error": "No quasi-identifiers selected"}
        
        # Group by quasi-identifiers and count group sizes
        group_counts = df.groupby(quasi_identifiers).size()
        
        # Calculate detailed k-anonymity metrics
        k_value = group_counts.min()  # minimum group size
        avg_group_size = group_counts.mean()
        total_groups = len(group_counts)
        
        # Calculate distribution of group sizes
        group_size_distribution = {}
        for k in range(1, 11):  # Count groups of size 1-10
            count = len(group_counts[group_counts == k])
            group_size_distribution[f"groups_of_size_{k}"] = count
        
        # Count groups larger than 10
        group_size_distribution["groups_larger_than_10"] = len(group_counts[group_counts > 10])
        
        # Calculate risk metrics
        unique_records = len(group_counts[group_counts == 1])
        vulnerable_groups = len(group_counts[group_counts < 5])
        high_risk_percentage = (unique_records / len(df)) * 100
        
        # Calculate entropy of group size distribution
        from math import log2
        probabilities = group_counts / len(df)
        entropy = -sum(p * log2(p) for p in probabilities if p > 0)
        
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
            "privacy_score": float((1 - (unique_records / len(df))) * 100)  # Higher is better
        }

    def compare_tables(self, db1_path: str, db2_path: str, table_name: str) -> Dict:
        """Compare k-anonymity between corresponding tables in two databases."""
        con1 = duckdb.connect(db1_path, read_only=True)
        con2 = duckdb.connect(db2_path, read_only=True)
        
        # Get column information
        columns1 = con1.execute(f"DESCRIBE {table_name}").fetchall()
        columns2 = con2.execute(f"DESCRIBE {table_name}").fetchall()
        
        # Convert to dataframes for easier processing
        df1 = con1.execute(f"SELECT * FROM {table_name}").fetch_df()
        df2 = con2.execute(f"SELECT * FROM {table_name}").fetch_df()
        
        con1.close()
        con2.close()
        
        # Get common columns
        common_columns = list(set(df1.columns) & set(df2.columns))
        print(f"\nAnalyzing table: {table_name}")
        print(f"Common columns found: {len(common_columns)}")
        
        # Let user select quasi-identifiers from common columns
        quasi_identifiers = self.get_quasi_identifiers(common_columns)
        
        # Calculate k-anonymity for both datasets
        results1 = self.calculate_k_anonymity(df1, quasi_identifiers)
        results2 = self.calculate_k_anonymity(df2, quasi_identifiers)
        
        return {
            "table_name": table_name,
            "quasi_identifiers": quasi_identifiers,
            "dataset1_results": results1,
            "dataset2_results": results2,
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, results: Dict, db1_name: str, db2_name: str):
        """Save comparison results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"k_anonymity_{db1_name}_{db2_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")

    def run_comparison(self):
        """Run the complete comparison process with user interaction."""
        # Get available databases
        databases = self.get_database_list()
        if len(databases) < 2:
            print("Need at least 2 databases to compare.")
            return
        
        # Display available databases
        print("\nAvailable databases:")
        for i, db in enumerate(databases, 1):
            print(f"{i}. {db}")
        
        # Get user selection
        while True:
            try:
                print("\nSelect two databases to compare (enter two numbers):")
                db1_idx, db2_idx = map(int, input("> ").strip().split())
                db1 = databases[db1_idx - 1]
                db2 = databases[db2_idx - 1]
                break
            except (ValueError, IndexError):
                print("Invalid selection. Please enter two valid numbers.")
        
        # Get joined tables from both databases
        db1_path = os.path.join('duckdb', db1)
        db2_path = os.path.join('duckdb', db2)
        
        tables1 = set(self.get_joined_tables(db1_path))
        tables2 = set(self.get_joined_tables(db2_path))
        
        # Find common tables
        common_tables = tables1 & tables2
        if not common_tables:
            print("No matching joined tables found between the databases.")
            return
        
        # Compare each common table
        results = {
            "database1": db1,
            "database2": db2,
            "comparisons": []
        }
        
        for table in common_tables:
            comparison = self.compare_tables(db1_path, db2_path, table)
            results["comparisons"].append(comparison)
        
        # Save results
        self.save_results(results, db1.replace('.duckdb', ''), db2.replace('.duckdb', ''))

def main():
    calculator = KAnonymityCalculator()
    calculator.run_comparison()

if __name__ == "__main__":
    main()