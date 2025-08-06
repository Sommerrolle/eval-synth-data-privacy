# Data Processing Package

This package contains all the modules needed for the health claims data processing pipeline. It includes preprocessing, joining, and combining operations for health data analysis.

## Package Structure

```
data_processing/
├── __init__.py                    # Package initialization and exports
├── data_pipeline.py              # Main pipeline orchestrator
├── preprocessing_before_join.py  # Raw table preprocessing
├── join_by_year_with_stats.py    # Year-specific table joining
├── preprocess_joined_tables.py   # Joined table preprocessing
├── combine_yearly_joins.py       # Yearly table combining
├── join_all_by_year.py          # Comprehensive table joining across all years
└── README.md                     # This file
```

## Modules

### data_pipeline.py
The main orchestrator for the entire data processing pipeline. It coordinates the execution of all other modules in the correct order.

**Key Features:**
- Orchestrates the complete data processing workflow
- Handles error recovery and logging
- Provides comprehensive status reporting
- Supports flexible pipeline execution modes

**Usage:**
```python
from data_processing import ProcessingPipeline

pipeline = ProcessingPipeline(db_path="path/to/database.duckdb")
pipeline.run_pipeline()
```

### preprocessing_before_join.py
Handles the preprocessing of raw tables before joining operations. It applies domain-specific imputation strategies to replace NULL values.

**Key Features:**
- Automatic missing value imputation
- Table-specific preprocessing rules
- Comprehensive logging and statistics
- Support for creating cleaned database copies

### join_by_year_with_stats.py
Creates year-specific joined tables by combining insurants and insurance data with domain-specific tables.

**Key Features:**
- Year-specific data filtering
- Comprehensive table joining with statistics
- Support for inpatient, outpatient, and drugs data
- Automatic row count reporting

### preprocess_joined_tables.py
Preprocesses joined tables by handling missing values in the combined data.

**Key Features:**
- Domain-specific imputation for joined data
- Comprehensive null value handling
- Detailed table statistics
- Support for year-specific processing

### combine_yearly_joins.py
Combines year-specific tables into comprehensive tables across all years.

**Key Features:**
- Multi-year data combination
- Configurable year filtering
- Comprehensive table statistics
- Support for different output prefixes

### join_all_by_year.py
Creates comprehensive joined tables across all years in a single operation.

**Key Features:**
- All-year data processing
- Comprehensive table joining
- Detailed statistics reporting
- Support for multiple data domains

## Usage

### Importing the Package

```python
# Import the main pipeline
from data_processing import ProcessingPipeline

# Import individual components
from data_processing import HealthClaimsPreprocessor
from data_processing import create_joined_tables
from data_processing import JoinedTablePreprocessor
from data_processing import create_combined_tables
```

### Running the Complete Pipeline

```python
from data_processing import ProcessingPipeline

# Initialize the pipeline
pipeline = ProcessingPipeline(
    db_path="path/to/database.duckdb",
    years=range(2014, 2022),
    output_prefix="clean_",
    combine_prefix=""
)

# Run the complete pipeline
pipeline.run_pipeline()
```

### Running Individual Steps

```python
from data_processing import ProcessingPipeline

pipeline = ProcessingPipeline(db_path="path/to/database.duckdb")

# Run only specific steps
pipeline.preprocess_raw_tables()
pipeline.join_tables_by_year(2017)
pipeline.preprocess_joined_tables(2017)
pipeline.combine_yearly_tables()
```

## Command Line Usage

The main pipeline can be run from the command line:

```bash
# Run the complete pipeline
python -m data_processing.data_pipeline --db_path path/to/database.duckdb

# Run with specific options
python -m data_processing.data_pipeline \
    --db_path path/to/database.duckdb \
    --start_year 2017 \
    --end_year 2020 \
    --skip_preprocess \
    --skip_combine
```

## Dependencies

This package depends on:
- `duckdb` - Database operations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `pathlib` - Path handling
- `logging` - Logging functionality

## Output

The pipeline generates:
- Cleaned individual tables with the specified prefix
- Year-specific joined tables
- Comprehensive tables combining all years
- Detailed logs and statistics
- CSV files with table row counts

## Error Handling

The pipeline includes comprehensive error handling:
- Script execution monitoring
- Detailed error logging
- Graceful failure recovery
- Status reporting for each step 