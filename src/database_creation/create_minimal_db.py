import os
import duckdb
from pathlib import Path
from typing import List, Tuple, Optional

def find_duckdb_files(base_path: str) -> List[Path]:
    """Find all DuckDB database files in the given directory and its subdirectories."""
    duckdb_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.duckdb'):
                duckdb_files.append(Path(root) / file)
    return duckdb_files

def select_database(databases: List[Path]) -> Optional[Path]:
    """Let user select a database from the list."""
    if not databases:
        print("No DuckDB databases found!")
        return None
        
    print("\nAvailable databases:")
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db}")
        
    while True:
        try:
            choice = int(input("\nSelect a database (enter number): "))
            if 1 <= choice <= len(databases):
                return databases[choice - 1]
            print(f"Please enter a number between 1 and {len(databases)}")
        except ValueError:
            print("Please enter a valid number")

def get_tables(db_path: Path) -> List[str]:
    """Get list of tables in the database."""
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        con.close()
        return tables
    except Exception as e:
        print(f"Error accessing database: {e}")
        return []

def select_table(tables: List[str]) -> Optional[str]:
    """Let user select a table from the list."""
    if not tables:
        print("No tables found in database!")
        return None
        
    print("\nAvailable tables:")
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table}")
        
    while True:
        try:
            choice = int(input("\nSelect a table (enter number): "))
            if 1 <= choice <= len(tables):
                return tables[choice - 1]
            print(f"Please enter a number between 1 and {len(tables)}")
        except ValueError:
            print("Please enter a valid number")

def get_row_count(db_path: Path, table: str) -> int:
    """Get the total number of rows in the selected table."""
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.close()
        return count
    except Exception as e:
        print(f"Error getting row count: {e}")
        return 0

def get_sample_size(total_rows: int) -> int:
    """Let user specify how many rows to extract."""
    while True:
        try:
            size = int(input(f"\nEnter number of rows to extract (max {total_rows}): "))
            if 1 <= size <= total_rows:
                return size
            print(f"Please enter a number between 1 and {total_rows}")
        except ValueError:
            print("Please enter a valid number")

def create_minimal_database(source_path: Path, table: str, sample_size: int) -> Optional[Path]:
    """Create a new database with sampled data."""
    try:
        # Create new database name
        new_path = source_path.parent / f"{source_path.stem}_minimal{source_path.suffix}"
        
        # Create a connection to the new database
        con = duckdb.connect(str(new_path))
        
        try:
            # Attach source database
            con.execute(f"ATTACH '{str(source_path)}' AS source (READ_ONLY)")
            
            # Create new table with sampled data from source
            query = f"""
            CREATE TABLE {table} AS 
            SELECT * FROM source.{table} 
            ORDER BY RANDOM() 
            LIMIT {sample_size}
            """
            con.execute(query)
            
            # Detach source database
            con.execute("DETACH source")
            
            return new_path
            
        finally:
            # Ensure connection is closed even if an error occurs
            con.close()
            
    except Exception as e:
        print(f"Error creating minimal database: {e}")
        return None
    

def main():
    # Get current directory
    current_dir = os.getcwd()
    
    # Find all DuckDB databases
    print("Searching for DuckDB databases...")
    databases = find_duckdb_files(current_dir)
    
    # Let user select database
    selected_db = select_database(databases)
    if not selected_db:
        return
    
    # Get and display tables
    tables = get_tables(selected_db)
    selected_table = select_table(tables)
    if not selected_table:
        return
    
    # Get row count and sample size
    total_rows = get_row_count(selected_db, selected_table)
    if total_rows == 0:
        print("Selected table is empty!")
        return
    
    sample_size = get_sample_size(total_rows)
    
    # Create minimal database
    print("\nCreating minimal database...")
    new_db_path = create_minimal_database(selected_db, selected_table, sample_size)
    
    if new_db_path:
        print(f"\nSuccessfully created minimal database: {new_db_path}")
        print(f"Extracted {sample_size} rows from {selected_table}")
    else:
        print("Failed to create minimal database!")

if __name__ == "__main__":
    main()