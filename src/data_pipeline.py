#!/usr/bin/env python3
"""
Health Claims Data Processing Pipeline

This script orchestrates the full data processing pipeline for health claims data:
1. Preprocessing raw tables (preprocessing_before_join.py)
2. Joining tables by year (join_by_year_with_stats.py)
3. Preprocessing joined tables (preprocess_joined_tables.py)
4. Combining year-specific tables into comprehensive tables (combine_yearly_joins.py)

The pipeline processes data for years 2014-2021.
"""

import os
import sys
import subprocess
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

class ProcessingPipeline:
    """Orchestrates the health claims data processing pipeline."""
    
    def __init__(self, db_path, years=range(2014, 2022), output_prefix="clean_", combine_prefix=""):
        """
        Initialize the pipeline.
        
        Args:
            db_path: Path to the DuckDB database
            years: Years to process (default: 2014-2021)
            output_prefix: Prefix for cleaned tables
            combine_prefix: Prefix for combined tables
        """
        self.db_path = db_path
        self.years = years
        self.output_prefix = output_prefix
        self.combine_prefix = combine_prefix
        
        # Verify that the database file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Define scripts with absolute paths
        self.scripts = {
            "preprocess": str(SCRIPT_DIR / "preprocessing_before_join.py"),
            "join": str(SCRIPT_DIR / "join_by_year_with_stats.py"),
            "postprocess": str(SCRIPT_DIR / "preprocess_joined_tables.py"),
            "combine": str(SCRIPT_DIR / "combine_yearly_joins.py")
        }
        
        # Verify that all required scripts exist
        for name, script in self.scripts.items():
            if not os.path.exists(script):
                # Try to find script in current directory if not found at expected path
                script_name = Path(script).name
                if os.path.exists(script_name):
                    self.scripts[name] = script_name
                    logging.info(f"Found script in current directory: {script_name}")
                else:
                    raise FileNotFoundError(f"Required script not found: {script}")
    
    def run_script(self, script_name, args, stage_name=None):
        """
        Run a script with the given arguments.
        
        Args:
            script_name: Name of the script to run
            args: List of arguments to pass to the script
            stage_name: Optional name for the stage (for logging)
            
        Returns:
            True if successful, False otherwise
        """
        if stage_name is None:
            stage_name = script_name
            
        # Check if script exists
        if not os.path.exists(script_name):
            logging.error(f"Script not found: {script_name}")
            print(f"ERROR: Script not found: {script_name}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory used: {SCRIPT_DIR}")
            return False
            
        cmd = [sys.executable, script_name] + args
        cmd_str = " ".join(cmd)
        
        logging.info(f"Running {stage_name}: {cmd_str}")
        print(f"\n{'='*80}\nRunning {stage_name}\n{'='*80}")
        
        try:
            # Run the script and capture output
            start_time = time.time()
            process = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Log and print stdout
            for line in process.stdout.splitlines():
                print(line)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Successfully completed {stage_name} in {elapsed_time:.2f} seconds")
            print(f"\nCompleted {stage_name} in {elapsed_time:.2f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {stage_name}: {e}")
            logging.error(f"Command output: {e.stdout}")
            logging.error(f"Command error: {e.stderr}")
            print(f"\nERROR: Failed to run {stage_name}")
            print(f"Error details: {e}")
            return False
            
        except Exception as e:
            logging.error(f"Unexpected error running {stage_name}: {str(e)}")
            print(f"\nERROR: Failed to run {stage_name} due to unexpected error")
            print(f"Error details: {str(e)}")
            return False
    
    def preprocess_raw_tables(self):
        """Run the preprocessing step on raw tables."""
        args = [
            "--db_path", self.db_path,
            "--output_prefix", self.output_prefix,
            "--new_db"  # Create a new database with preprocessed tables
        ]
        
        return self.run_script(
            self.scripts["preprocess"],
            args,
            "Preprocessing Raw Tables"
        )
    
    def join_tables_by_year(self, year):
        """
        Run the table joining step for a specific year.
        
        Args:
            year: The year to process
            
        Returns:
            True if successful, False otherwise
        """
        args = [
            "--db_path", self.db_path,
            "--year", str(year)
        ]
        
        return self.run_script(
            self.scripts["join"],
            args,
            f"Joining Tables for Year {year}"
        )
    
    def preprocess_joined_tables(self, year):
        """
        Run the preprocessing step on joined tables for a specific year.
        
        Args:
            year: The year to process
            
        Returns:
            True if successful, False otherwise
        """
        args = [
            "--db_path", self.db_path,
            "--year", str(year),
            "--output_prefix", self.output_prefix
        ]
        
        return self.run_script(
            self.scripts["postprocess"],
            args,
            f"Preprocessing Joined Tables for Year {year}"
        )
    
    def combine_yearly_tables(self):
        """
        Run the step to combine year-specific tables into comprehensive tables.
        
        Returns:
            True if successful, False otherwise
        """
        args = [
            "--db_path", self.db_path,
            "--input_prefix", self.output_prefix,
            "--output_prefix", self.combine_prefix
        ]
        
        # Add years if not processing all years
        if len(self.years) < 8:  # Assuming default is 2014-2021 (8 years)
            args.extend(["--years"] + [str(year) for year in self.years])
        
        return self.run_script(
            self.scripts["combine"],
            args,
            "Combining Year-Specific Tables"
        )
    
    def run_pipeline(self):
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary with results for each stage
        """
        results = {
            "preprocess_raw": False,
            "join_by_year": {},
            "preprocess_joined": {},
            "combine_yearly": False
        }
        
        # Step 1: Preprocess raw tables
        logging.info("Starting Step 1: Preprocessing Raw Tables")
        results["preprocess_raw"] = self.preprocess_raw_tables()
        
        if not results["preprocess_raw"]:
            logging.error("Failed at Step 1: Preprocessing Raw Tables. Aborting pipeline.")
            return results
        
        # Steps 2 & 3: For each year, join tables and preprocess the joined tables
        successful_years = []
        for year in self.years:
            logging.info(f"Starting Steps 2 & 3 for year {year}")
            
            # Step 2: Join tables by year
            join_result = self.join_tables_by_year(year)
            results["join_by_year"][year] = join_result
            
            if not join_result:
                logging.error(f"Failed at Step 2: Joining Tables for Year {year}. Skipping preprocessing for this year.")
                continue
            
            # Step 3: Preprocess joined tables
            preprocess_result = self.preprocess_joined_tables(year)
            results["preprocess_joined"][year] = preprocess_result
            
            if not preprocess_result:
                logging.error(f"Failed at Step 3: Preprocessing Joined Tables for Year {year}.")
            else:
                successful_years.append(year)
        
        # Step 4: Combine year-specific tables into comprehensive tables
        if successful_years:
            logging.info(f"Starting Step 4: Combining Tables for Years {successful_years}")
            # Temporarily update years to only those that were successfully processed
            original_years = self.years
            self.years = successful_years
            
            results["combine_yearly"] = self.combine_yearly_tables()
            
            # Restore original years
            self.years = original_years
            
            if not results["combine_yearly"]:
                logging.error("Failed at Step 4: Combining Year-Specific Tables.")
        else:
            logging.error("No successful yearly processing to combine. Skipping Step 4.")
            results["combine_yearly"] = False
        
        # Summarize results
        self._summarize_results(results)
        
        return results
    
    def _summarize_results(self, results):
        """
        Summarize the pipeline results.
        
        Args:
            results: Dictionary with results for each stage
        """
        print("\n" + "=" * 80)
        print("Health Claims Data Processing Pipeline - Summary")
        print("=" * 80)
        
        # Step 1 summary
        status = "✅ Success" if results["preprocess_raw"] else "❌ Failed"
        print(f"Step 1: Preprocessing Raw Tables - {status}")
        
        # Step 2 & 3 summary
        print("\nYearly Processing Results:")
        print("-" * 60)
        print(f"{'Year':<10} {'Join Tables':<20} {'Preprocess Joined':<20}")
        print("-" * 60)
        
        successful_years = []
        for year in self.years:
            join_status = "✅ Success" if results["join_by_year"].get(year, False) else "❌ Failed"
            preprocess_status = "✅ Success" if results["preprocess_joined"].get(year, False) else "❌ Failed"
            print(f"{year:<10} {join_status:<20} {preprocess_status:<20}")
            
            if results["preprocess_joined"].get(year, False):
                successful_years.append(year)
        
        print("-" * 60)
        
        # Step 4 summary
        if "combine_yearly" in results:
            status = "✅ Success" if results["combine_yearly"] else "❌ Failed"
            print(f"\nStep 4: Combining Year-Specific Tables ({', '.join(map(str, successful_years))}) - {status}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Health Claims Data Processing Pipeline')
    parser.add_argument('--db_path', required=True, help='Path to the DuckDB database')
    parser.add_argument('--start_year', type=int, default=2014, help='Start year (inclusive)')
    parser.add_argument('--end_year', type=int, default=2021, help='End year (inclusive)')
    parser.add_argument('--output_prefix', default='clean_', help='Prefix for cleaned tables')
    parser.add_argument('--combine_prefix', default='', help='Prefix for combined tables')
    parser.add_argument('--skip_preprocess', action='store_true', help='Skip preprocessing raw tables')
    parser.add_argument('--skip_combine', action='store_true', help='Skip combining year-specific tables')
    parser.add_argument('--only_year', type=int, help='Process only this specific year')
    parser.add_argument('--only_combine', action='store_true', help='Only run the combine step')
    parser.add_argument('--scripts_dir', type=str, help='Directory containing the processing scripts')
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    # Override SCRIPT_DIR if --scripts_dir is provided
    global SCRIPT_DIR
    if args.scripts_dir:
        SCRIPT_DIR = Path(args.scripts_dir)
        logging.info(f"Using scripts from directory: {SCRIPT_DIR}")
    
    # Determine years to process
    if args.only_year:
        years = [args.only_year]
    else:
        years = list(range(args.start_year, args.end_year + 1))
    
    # Create the pipeline
    try:
        pipeline = ProcessingPipeline(
            db_path=args.db_path,
            years=years,
            output_prefix=args.output_prefix,
            combine_prefix=args.combine_prefix
        )
        
        # Run only the combine step if requested
        if args.only_combine:
            logging.info("Running only the combine step")
            result = pipeline.combine_yearly_tables()
            status = "✅ Success" if result else "❌ Failed"
            print(f"\nStep 4: Combining Year-Specific Tables - {status}")
            return 0 if result else 1
        
        # Run the full pipeline or selected steps
        if args.skip_preprocess and args.skip_combine:
            # Run only steps 2 & 3
            logging.info("Skipping preprocessing of raw tables and combining tables")
            results = {
                "preprocess_raw": True,  # Pretend it succeeded
                "join_by_year": {},
                "preprocess_joined": {}
            }
            
            for year in years:
                logging.info(f"Starting Steps 2 & 3 for year {year}")
                
                # Step 2: Join tables by year
                join_result = pipeline.join_tables_by_year(year)
                results["join_by_year"][year] = join_result
                
                if not join_result:
                    logging.error(f"Failed at Step 2: Joining Tables for Year {year}. Skipping preprocessing for this year.")
                    continue
                
                # Step 3: Preprocess joined tables
                preprocess_result = pipeline.preprocess_joined_tables(year)
                results["preprocess_joined"][year] = preprocess_result
            
            # Summarize results
            pipeline._summarize_results(results)
            
        elif args.skip_preprocess:
            # Run steps 2, 3, & 4
            logging.info("Skipping preprocessing of raw tables")
            results = {
                "preprocess_raw": True,  # Pretend it succeeded
                "join_by_year": {},
                "preprocess_joined": {},
                "combine_yearly": False
            }
            
            successful_years = []
            for year in years:
                logging.info(f"Starting Steps 2 & 3 for year {year}")
                
                # Step 2: Join tables by year
                join_result = pipeline.join_tables_by_year(year)
                results["join_by_year"][year] = join_result
                
                if not join_result:
                    logging.error(f"Failed at Step 2: Joining Tables for Year {year}. Skipping preprocessing for this year.")
                    continue
                
                # Step 3: Preprocess joined tables
                preprocess_result = pipeline.preprocess_joined_tables(year)
                results["preprocess_joined"][year] = preprocess_result
                
                if preprocess_result:
                    successful_years.append(year)
            
            # Step 4: Combine year-specific tables
            if not args.skip_combine and successful_years:
                logging.info(f"Starting Step 4: Combining Tables for Years {successful_years}")
                # Temporarily update years to only those that were successfully processed
                original_years = pipeline.years
                pipeline.years = successful_years
                
                results["combine_yearly"] = pipeline.combine_yearly_tables()
                
                # Restore original years
                pipeline.years = original_years
            
            # Summarize results
            pipeline._summarize_results(results)
            
        elif args.skip_combine:
            # Run steps 1, 2, & 3
            logging.info("Skipping combining of year-specific tables")
            results = {
                "preprocess_raw": False,
                "join_by_year": {},
                "preprocess_joined": {}
            }
            
            # Step 1: Preprocess raw tables
            logging.info("Starting Step 1: Preprocessing Raw Tables")
            results["preprocess_raw"] = pipeline.preprocess_raw_tables()
            
            if not results["preprocess_raw"]:
                logging.error("Failed at Step 1: Preprocessing Raw Tables. Aborting pipeline.")
                pipeline._summarize_results(results)
                return 1
            
            # Steps 2 & 3
            for year in years:
                logging.info(f"Starting Steps 2 & 3 for year {year}")
                
                # Step 2: Join tables by year
                join_result = pipeline.join_tables_by_year(year)
                results["join_by_year"][year] = join_result
                
                if not join_result:
                    logging.error(f"Failed at Step 2: Joining Tables for Year {year}. Skipping preprocessing for this year.")
                    continue
                
                # Step 3: Preprocess joined tables
                preprocess_result = pipeline.preprocess_joined_tables(year)
                results["preprocess_joined"][year] = preprocess_result
            
            # Summarize results
            pipeline._summarize_results(results)
            
        else:
            # Run the complete pipeline
            pipeline.run_pipeline()
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"ERROR: Pipeline failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()