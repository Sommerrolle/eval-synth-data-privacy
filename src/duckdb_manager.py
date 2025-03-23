import duckdb
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

class DuckDBManager:
    """
    A class to manage DuckDB database connections and operations for privacy metrics calculations.
    This centralizes all database operations and provides a clean interface for the PrivacyMetricsCalculator.
    """
    
    def __init__(self, duckdb_dir: str = 'data/duckdb'):
        """
        Initialize the DuckDBManager with the directory containing DuckDB databases.
        
        Args:
            duckdb_dir: Directory containing DuckDB database files
        """
        self.duckdb_dir = duckdb_dir
        if not os.path.exists(duckdb_dir):
            logging.warning(f"DuckDB directory {duckdb_dir} does not exist")
            os.makedirs(duckdb_dir, exist_ok=True)
    
    def get_database_list(self) -> List[str]:
        """
        List all available DuckDB databases in the configured directory.
        
        Returns:
            List of database filenames
        """
        if not os.path.exists(self.duckdb_dir):
            raise FileNotFoundError(f"DuckDB directory {self.duckdb_dir} does not exist")
        return [f for f in os.listdir(self.duckdb_dir) if f.endswith('.duckdb')]
    
    def get_database_path(self, db_name: str) -> str:
        """
        Get the full path to a database file.
        
        Args:
            db_name: Database filename
            
        Returns:
            Full path to the database file
        """
        return os.path.join(self.duckdb_dir, db_name)
    
    def get_joined_tables(self, db_path: str) -> List[str]:
        """
        Get a list of all joined tables in the specified database.
        
        Args:
            db_path: Path to the DuckDB database file
            
        Returns:
            List of table names starting with 'joined'
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'join%'").fetchall()
            return [table[0] for table in tables]
        except Exception as e:
            logging.error(f"Error retrieving joined tables from {db_path}: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_common_tables(self, db1_path: str, db2_path: str) -> List[str]:
        """
        Find common joined tables between two databases.
        
        Args:
            db1_path: Path to the first database
            db2_path: Path to the second database
            
        Returns:
            List of table names present in both databases
        """
        tables1 = set(self.get_joined_tables(db1_path))
        tables2 = set(self.get_joined_tables(db2_path))
        return list(tables1 & tables2)
    
    def load_table_data(self, db_path: str, table_name: str) -> pd.DataFrame:
        """
        Load data from a specified table into a pandas DataFrame.
        
        Args:
            db_path: Path to the database file
            table_name: Name of the table to load
            
        Returns:
            DataFrame containing the table data
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            df = conn.execute(f"SELECT * FROM {table_name}").fetch_df()
            return df
        except Exception as e:
            logging.error(f"Error loading table {table_name} from {db_path}: {str(e)}")
            return pd.DataFrame()
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_table_column_info(self, db_path: str, table_name: str) -> List[Dict]:
        """
        Get information about columns in a table.
        
        Args:
            db_path: Path to the database file
            table_name: Name of the table
            
        Returns:
            List of dictionaries with column information
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            return [{"name": col[0], "type": col[1]} for col in columns]
        except Exception as e:
            logging.error(f"Error getting column info for {table_name} from {db_path}: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def execute_query(self, db_path: str, query: str) -> Optional[List[Tuple]]:
        """
        Execute a SQL query on a database.
        
        Args:
            db_path: Path to the database file
            query: SQL query to execute
            
        Returns:
            Query results as a list of tuples, or None if an error occurs
        """
        try:
            conn = duckdb.connect(db_path)
            results = conn.execute(query).fetchall()
            return results
        except Exception as e:
            logging.error(f"Error executing query on {db_path}: {str(e)}")
            logging.error(f"Query: {query}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_table_count(self, db_path: str, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            db_path: Path to the database file
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            return count
        except Exception as e:
            logging.error(f"Error counting rows in {table_name} from {db_path}: {str(e)}")
            return 0
        finally:
            if 'conn' in locals():
                conn.close()

    def get_database_size(self, db_path: str) -> int:
        """
        Get the file size of a database.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Size of the database file in bytes
        """
        try:
            return os.path.getsize(db_path)
        except Exception as e:
            logging.error(f"Error getting file size for {db_path}: {str(e)}")
            return 0
            
    def get_distinct_values(self, db_path: str, table_name: str, column_name: str) -> List:
        """
        Get distinct values for a column in a table.
        
        Args:
            db_path: Path to the database file
            table_name: Name of the table
            column_name: Name of the column
            
        Returns:
            List of distinct values in the column
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)
            results = conn.execute(f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name}").fetchall()
            return [r[0] for r in results]
        except Exception as e:
            logging.error(f"Error getting distinct values for {column_name} in {table_name} from {db_path}: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()