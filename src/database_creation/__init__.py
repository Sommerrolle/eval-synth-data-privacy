"""
Database Creation Package

This package contains modules for creating DuckDB databases from CSV files and
performing database operations. It provides functionality for database creation,
table joining, and database management.

Modules:
- create_databases: Main database creation from CSV files
- join_tables: Table joining operations
- create_minimal_db: Creating minimal databases with sampled data
"""

from .create_databases import create_database, process_all_datasets
from .join_tables import perform_join, interactive_join_session
from .create_minimal_db import create_minimal_database

__all__ = [
    'create_database',
    'process_all_datasets',
    'perform_join',
    'interactive_join_session',
    'create_minimal_database'
] 