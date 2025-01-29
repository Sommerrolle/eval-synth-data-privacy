import duckdb
import logging
from typing import List, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('table_joining.log'),
        logging.StreamHandler()
    ]
)

# Table mapping for naming convention
TABLE_MAPPING = {
    'insurants': 1,
    'insurance_data': 2,
    'drugs': 3,
    'inpatient_cases': 4,
    'inpatient_diagnosis': 5,
    'inpatient_procedures': 6,
    'inpatient_fees': 7,
    'outpatient_cases': 8,
    'outpatient_diagnosis': 9,
    'outpatient_procedures': 10,
    'outpatient_fees': 11
}

def get_join_keys(selected_tables: List[str]) -> Dict[str, List[str]]:
    """Create a dictionary of join keys for selected tables."""
    keys = {
        'insurants': ['pid'],
        'insurance_data': ['pid'],
        'drugs': ['pid'],
        'inpatient_cases': ['pid'],
        'inpatient_diagnosis': ['pid', 'inpatient_caseID'],
        'inpatient_procedures': ['pid', 'inpatient_caseID'],
        'inpatient_fees': ['pid', 'inpatient_caseID'],
        'outpatient_cases': ['pid'],
        'outpatient_diagnosis': ['pid', 'outpatient_caseID'],
        'outpatient_procedures': ['pid', 'outpatient_caseID'],
        'outpatient_fees': ['pid', 'outpatient_caseID']
    }
    return {table: keys[table] for table in selected_tables}

def generate_output_table_name(selected_tables: List[str]) -> str:
    """Generate the output table name based on selected tables."""
    numbers = [TABLE_MAPPING[table] for table in selected_tables]
    return 'joined_' + '_'.join(str(n) for n in sorted(numbers))

def analyze_join_complexity(con: duckdb.DuckDBPyConnection, keys: Dict[str, List[str]], 
                          primary_table: str = 'insurants') -> Dict:
    """Analyze the complexity of the join operation."""
    analysis = {}
    
    # Get primary table size
    primary_size = con.execute(f"SELECT COUNT(*) FROM {primary_table};").fetchone()[0]
    analysis['primary_size'] = primary_size
    analysis['cumulative_size'] = primary_size
    
    # Analyze each table
    for table_name, join_keys in keys.items():
        if table_name == primary_table:
            continue
            
        table_size = con.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
        
        # Calculate average relationships
        relationship_query = f"""
            SELECT AVG(match_count)
            FROM (
                SELECT COUNT(*) AS match_count
                FROM {table_name}
                GROUP BY {', '.join(join_keys)}
            );
        """
        avg_relationship = con.execute(relationship_query).fetchone()[0] or 1
        
        analysis[table_name] = {
            'size': table_size,
            'avg_matches': avg_relationship
        }
        
        analysis['cumulative_size'] *= avg_relationship
    
    # Estimate processing time (rough estimate)
    analysis['estimated_time'] = analysis['cumulative_size'] / 1e6
    
    return analysis

def display_table_options():
    """Display available tables with their corresponding numbers."""
    print("\nAvailable tables:")
    print("-" * 40)
    for table_name, number in TABLE_MAPPING.items():
        print(f"{number}. {table_name}")
    print("-" * 40)

def get_user_selection() -> List[str]:
    """Get user input for table selection."""
    while True:
        try:
            print("\nEnter the numbers of the tables you want to join (separated by spaces):")
            numbers = input("> ").strip().split()
            numbers = [int(n) for n in numbers]
            
            if not all(1 <= n <= len(TABLE_MAPPING) for n in numbers):
                print(f"Error: Please enter numbers between 1 and {len(TABLE_MAPPING)}.")
                continue
            
            reverse_mapping = {v: k for k, v in TABLE_MAPPING.items()}
            selected_tables = [reverse_mapping[n] for n in numbers]
            
            if 'insurants' not in selected_tables:
                print("Warning: 'insurants' (1) should typically be included as it's the primary table.")
                if input("Do you want to continue anyway? (y/n): ").lower() != 'y':
                    continue
            
            return selected_tables
            
        except ValueError:
            print("Error: Please enter valid numbers separated by spaces.")

def perform_join(db_path: str, selected_tables: List[str], chunk_size: int = 200000, 
                force: bool = False) -> Optional[str]:
    """Perform the join operation with selected tables."""
    con = duckdb.connect(db_path)
    try:
        keys = get_join_keys(selected_tables)
        output_table = generate_output_table_name(selected_tables)
        
        # Check if output table already exists
        existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        if output_table in existing_tables and not force:
            logging.info(f"Table {output_table} already exists. Use force=True to recreate.")
            return output_table
        
        # Analyze join complexity
        analysis = analyze_join_complexity(con, keys)
        logging.info("\nJoin Analysis:")
        logging.info(f"Estimated final size: {analysis['cumulative_size']:,.0f} rows")
        logging.info(f"Estimated time: {analysis['estimated_time']:.2f} seconds")
        
        if not force and analysis['estimated_time'] > 300:  # 5 minutes
            response = input("Join operation might take a while. Continue? (y/n): ")
            if response.lower() != 'y':
                return None
        
        # Perform the join
        con.execute(f"DROP TABLE IF EXISTS {output_table};")
        
        # Process in chunks
        primary_table = 'insurants'
        total_rows = con.execute(f"SELECT COUNT(*) FROM {primary_table};").fetchone()[0]
        
        for offset in range(0, total_rows, chunk_size):
            logging.info(f"Processing chunk starting at row {offset}")
            
            # Create temporary tables for the chunk
            chunk_query = f"""
                CREATE TEMPORARY TABLE temp_chunk AS
                SELECT * FROM {primary_table}
                LIMIT {chunk_size} OFFSET {offset};
            """
            con.execute(chunk_query)
            
            # Build the join query
            join_query = ["SELECT DISTINCT * FROM temp_chunk"]
            
            for table_name, join_keys in keys.items():
                if table_name == primary_table:
                    continue
                
                join_conditions = " AND ".join(
                    f"t.{key} = temp_chunk.{key}" for key in join_keys
                )
                
                join_query.append(f"""
                    LEFT JOIN {table_name} t
                    ON {join_conditions}
                """)
            
            final_query = "\n".join(join_query)
            
            # Execute the join for this chunk
            if offset == 0:
                con.execute(f"CREATE TABLE {output_table} AS {final_query}")
            else:
                con.execute(f"INSERT INTO {output_table} {final_query}")
            
            # Clean up temporary table
            con.execute("DROP TABLE temp_chunk")
            
            logging.info(f"Processed {min(offset + chunk_size, total_rows)} out of {total_rows} rows")
        
        logging.info(f"Join completed successfully. Result saved in table '{output_table}'")
        return output_table
    
    except Exception as e:
        logging.error(f"Error during join operation: {str(e)}")
        return None
    
    finally:
        con.close()

def list_available_databases(duckdb_dir: str = 'duckdb') -> Dict[str, Path]:
    """List all available DuckDB databases."""
    duckdb_path = Path(duckdb_dir)
    if not duckdb_path.exists():
        logging.error(f"DuckDB directory {duckdb_dir} does not exist")
        return {}
        
    databases = {}
    for file in duckdb_path.glob('*.duckdb'):
        databases[file.stem] = file
    
    return databases

def interactive_join_session(db_path: str):
    """Run an interactive session for joining tables."""
    print(f"\nConnected to database: {db_path}")
    
    while True:
        display_table_options()
        selected_tables = get_user_selection()
        
        print("\nSelected tables:")
        for table in selected_tables:
            print(f"- {table}")
        
        if input("\nConfirm selection? (y/n): ").lower() == 'y':
            result_table = perform_join(db_path, selected_tables)
            if result_table:
                print(f"\nSuccessfully created joined table: {result_table}")
        
        if input("\nWould you like to create another joined table? (y/n): ").lower() != 'y':
            break

def main():
    """Main function to run the table joining tool."""
    print("DuckDB Table Join Tool")
    print("=====================")
    
    # List available databases
    databases = list_available_databases()
    if not databases:
        print("No DuckDB databases found in the duckdb directory.")
        return
    
    print("\nAvailable databases:")
    for i, (name, path) in enumerate(databases.items(), 1):
        print(f"{i}. {name}")
    
    # Let user select database
    while True:
        try:
            selection = int(input("\nSelect a database number: "))
            if 1 <= selection <= len(databases):
                selected_db = list(databases.values())[selection - 1]
                break
            print(f"Please enter a number between 1 and {len(databases)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Run interactive session
    interactive_join_session(str(selected_db))

if __name__ == "__main__":
    main()