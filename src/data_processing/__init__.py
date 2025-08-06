"""
Data Processing Package

This package contains all the modules needed for the health claims data processing pipeline.
It includes preprocessing, joining, and combining operations for health data analysis.

Modules:
- data_pipeline: Main pipeline orchestrator
- preprocessing_before_join: Raw table preprocessing
- join_by_year_with_stats: Year-specific table joining
- preprocess_joined_tables: Joined table preprocessing
- combine_yearly_joins: Yearly table combining
- join_all_by_year: Comprehensive table joining across all years
"""

from .data_pipeline import ProcessingPipeline
from .preprocessing_before_join import HealthClaimsPreprocessor
from .join_by_year_with_stats import create_joined_tables
from .preprocess_joined_tables import JoinedTablePreprocessor
from .combine_yearly_joins import create_combined_tables
from .join_all_by_year import create_joined_tables as create_all_joined_tables

__all__ = [
    'ProcessingPipeline',
    'HealthClaimsPreprocessor', 
    'create_joined_tables',
    'JoinedTablePreprocessor',
    'create_combined_tables',
    'create_all_joined_tables'
] 