# Database Creation Package

This package contains modules for creating DuckDB databases from CSV files and performing database operations. It provides functionality for database creation, table joining, and database management.

## Package Structure

```
database_creation/
├── __init__.py                    # Package initialization and exports
├── create_databases.py           # Main database creation from CSV files
├── join_tables.py                # Table joining operations
├── create_minimal_db.py          # Creating minimal databases with sampled data
└── README.md                     # This file
```

## Modules

### create_databases.py
The main module for creating DuckDB databases from CSV files. It handles CSV file reading, data type conversion, and database creation.

**Key Features:**
- CSV file reading with proper data types
- Automatic column renaming and prefixing
- Database creation with table sorting
- Column type optimization
- Batch processing of multiple datasets

**Main Functions:**
- `create_database(data_dir, force_recreate=False)`: Create a database from CSV files in a directory
- `process_all_datasets(base_dir, force_recreate=False)`: Process all dataset directories
- `fix_column_types_in_database(db_path)`: Optimize column data types

**Usage:**
```python
from database_creation import create_database, process_all_datasets

# Create a single database
db_path = create_database("path/to/csv/files")

# Process all datasets in a directory
databases = process_all_datasets("path/to/datasets")
```

### join_tables.py
Provides functionality for joining tables within a DuckDB database. It supports complex join operations with performance optimization.

**Key Features:**
- Multi-table join operations
- Join complexity analysis
- Chunked processing for large datasets
- Interactive table selection
- Performance estimation

**Main Functions:**
- `perform_join(db_path, selected_tables, chunk_size=200000, force=False)`: Perform table joins
- `interactive_join_session(db_path)`: Interactive join session
- `analyze_join_complexity(con, keys, primary_table='insurants')`: Analyze join performance

**Usage:**
```python
from database_creation import perform_join, interactive_join_session

# Perform a join operation
result_table = perform_join("database.duckdb", ["insurants", "inpatient_cases"])

# Interactive join session
interactive_join_session("database.duckdb")
```

### create_minimal_db.py
Creates minimal databases with sampled data for testing and development purposes.

**Key Features:**
- Database sampling and subset creation
- Interactive database and table selection
- Row count analysis
- Sample size configuration

**Main Functions:**
- `create_minimal_database(source_path, table, sample_size)`: Create a minimal database
- `find_duckdb_files(base_path)`: Find all DuckDB files in a directory
- `select_database(databases)`: Interactive database selection

**Usage:**
```python
from database_creation import create_minimal_database

# Create a minimal database with sampled data
minimal_db = create_minimal_database("source.duckdb", "table_name", 1000)
```

## Usage

### Importing the Package

```python
# Import main functions
from database_creation import create_database, perform_join, create_minimal_database

# Import for batch processing
from database_creation import process_all_datasets, interactive_join_session
```

### Creating Databases from CSV Files

```python
from database_creation import create_database

# Create a database from CSV files
db_path = create_database("path/to/csv/files")

# Force recreate existing database
db_path = create_database("path/to/csv/files", force_recreate=True)
```

### Joining Tables

```python
from database_creation import perform_join

# Join multiple tables
result = perform_join(
    db_path="database.duckdb",
    selected_tables=["insurants", "inpatient_cases", "inpatient_diagnosis"],
    chunk_size=100000
)
```

### Creating Minimal Databases

```python
from database_creation import create_minimal_database

# Create a minimal database for testing
minimal_db = create_minimal_database(
    source_path="large_database.duckdb",
    table="inpatient_cases",
    sample_size=1000
)
```

## Command Line Usage

### Creating Databases

```bash
# Create database from CSV files
python -m database_creation.create_databases

# Create minimal database
python -m database_creation.create_minimal_db

# Join tables
python -m database_creation.join_tables
```

## Dependencies

This package depends on:
- `duckdb` - Database operations
- `pandas` - Data manipulation and CSV reading
- `numpy` - Numerical operations
- `pathlib` - Path handling
- `logging` - Logging functionality

## Input Requirements

### CSV Files
- Files should be named with the pattern: `{prefix}_{table_name}.csv`
- Default prefix: `claims_data_`
- Default suffix: `.csv`
- Files should be tab-separated (TSV format)
- UTF-8 encoding is expected

### Data Types
The package automatically handles common data types:
- Integer columns (pid, etc.)
- Float columns (amounts, quantities)
- String columns (codes, descriptions)
- Date columns (prescription dates, admission dates)

## Output

### Database Files
- Created databases are saved in the `duckdb/` directory
- Database names are derived from the source directory name
- Tables are automatically sorted by primary key (pid)

### Logging
- Comprehensive logging is provided for all operations
- Log files are created in the `logs/` directory
- Progress information is displayed during long operations

## Error Handling

The package includes comprehensive error handling:
- CSV reading error recovery
- Database connection management
- Memory-efficient chunked processing
- Graceful failure recovery
- Detailed error logging 