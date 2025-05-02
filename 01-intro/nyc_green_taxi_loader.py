import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NYCGreenTaxiLoader:
    def __init__(self, years=(2023, 2024), start_month=1, end_month=12):
        """Initialize the NYC Green Taxi data loader"""
        self.base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
        self.years = years if isinstance(years, tuple) else (years,)
        self.start_month = start_month
        self.end_month = end_month
        
        # Create output directories
        self.data_dir = Path("green_taxi_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Directory for raw data
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        
        # Directory for processed data
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Directory for reports
        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def generate_urls(self):
        """Generate all URLs based on the specified years and months"""
        urls = []
        
        for year in self.years:
            for month in range(self.start_month, self.end_month + 1):
                url = self.base_url.format(year=year, month=month)
                urls.append((year, month, url))
        
        return urls
    
    def download_parquet(self, url_info):
        """Download parquet file from URL and save to raw directory"""
        year, month, url = url_info
        try:
            # Define output file path
            file_name = f"green_taxi_{year}_{month:02d}.parquet"
            output_path = self.raw_dir / file_name
            
            # Skip if file already exists
            if output_path.exists():
                logger.info(f"File already exists: {output_path}")
                return year, month, output_path
            
            # Download the file
            logger.info(f"Downloading: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Save the raw file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {output_path}")
            return year, month, output_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download from {url}: {str(e)}")
            return year, month, None
    
    def process_file(self, file_info):
        """Process a downloaded parquet file"""
        year, month, file_path = file_info
        
        if file_path is None or not file_path.exists():
            logger.warning(f"Skipping processing for {year}-{month:02d} - file not available")
            return
        
        try:
            # Define output file path
            processed_path = self.processed_dir / f"green_taxi_{year}_{month:02d}.parquet"
            
            # Skip if processed file already exists
            if processed_path.exists():
                logger.info(f"Processed file already exists: {processed_path}")
                return
            
            # Read the parquet file
            logger.info(f"Processing: {file_path}")
            df = pd.read_parquet(file_path)
            
            # Add metadata columns
            df['data_year'] = year
            df['data_month'] = month
            
            # Data cleaning and transformation
            # For example, you might want to:
            # - Convert datetime columns
            # - Fix data types
            # - Handle missing values
            
            if 'lpep_pickup_datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['lpep_pickup_datetime']):
                df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
            
            if 'lpep_dropoff_datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['lpep_dropoff_datetime']):
                df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
            
            # Save the processed file
            df.to_parquet(processed_path, index=False)
            logger.info(f"Saved processed data: {processed_path}")
            
            # Create a monthly summary
            self._create_monthly_summary(df, year, month)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    def _create_monthly_summary(self, df, year, month):
        """Create a summary for the monthly data"""
        try:
            # Create summary statistics
            summary = {
                'year': year,
                'month': month,
                'file_date': f"{year}-{month:02d}",
                'record_count': len(df),
                'avg_fare': df['fare_amount'].mean() if 'fare_amount' in df.columns else None,
                'avg_distance': df['trip_distance'].mean() if 'trip_distance' in df.columns else None,
                'total_revenue': df['total_amount'].sum() if 'total_amount' in df.columns else None,
                'min_pickup_date': df['lpep_pickup_datetime'].min() if 'lpep_pickup_datetime' in df.columns else None,
                'max_pickup_date': df['lpep_pickup_datetime'].max() if 'lpep_pickup_datetime' in df.columns else None,
            }
            
            # Save to monthly summary file
            summary_file = self.reports_dir / "monthly_summaries.csv"
            
            # Append to existing summary file or create new one
            summary_df = pd.DataFrame([summary])
            
            if summary_file.exists():
                existing_df = pd.read_csv(summary_file)
                
                # Check if this month already exists
                mask = (existing_df['year'] == year) & (existing_df['month'] == month)
                if mask.any():
                    # Update existing entry
                    existing_df.loc[mask] = summary_df.iloc[0]
                else:
                    # Append new entry
                    existing_df = pd.concat([existing_df, summary_df], ignore_index=True)
                
                # Sort by year and month
                existing_df = existing_df.sort_values(['year', 'month'])
                
                # Save the updated summary
                existing_df.to_csv(summary_file, index=False)
            else:
                # Create new summary file
                summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"Updated monthly summary for {year}-{month:02d}")
            
        except Exception as e:
            logger.error(f"Error creating monthly summary for {year}-{month:02d}: {str(e)}")
    
    def create_combined_dataset(self):
        """Combine all processed data into a single dataset"""
        logger.info("Creating combined dataset...")
        
        # List all processed files
        processed_files = list(self.processed_dir.glob("*.parquet"))
        
        if not processed_files:
            logger.warning("No processed files found to combine")
            return
        
        # Combine all files
        combined_df = pd.concat([pd.read_parquet(file) for file in processed_files], ignore_index=True)
        
        # Save combined dataset
        combined_path = self.data_dir / "combined_green_taxi_data.parquet"
        combined_df.to_parquet(combined_path, index=False)
        
        logger.info(f"Created combined dataset with {len(combined_df):,} records: {combined_path}")
        
        # Create a CSV version for easier access
        csv_path = self.data_dir / "combined_green_taxi_data.csv"
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Created CSV version: {csv_path}")
        
        return combined_df
    
    def create_yearly_summaries(self):
        """Create yearly summary files"""
        logger.info("Creating yearly summaries...")
        
        # Check if monthly summary exists
        summary_file = self.reports_dir / "monthly_summaries.csv"
        if not summary_file.exists():
            logger.warning("No monthly summaries found. Run the pipeline first.")
            return
        
        # Load the monthly summary
        monthly_df = pd.read_csv(summary_file)
        
        # Group by year and create yearly summaries
        yearly_summaries = []
        
        for year in monthly_df['year'].unique():
            year_data = monthly_df[monthly_df['year'] == year]
            
            yearly_summary = {
                'year': year,
                'total_records': year_data['record_count'].sum(),
                'avg_fare': year_data['avg_fare'].mean(),
                'avg_distance': year_data['avg_distance'].mean(),
                'total_revenue': year_data['total_revenue'].sum(),
                'months_covered': len(year_data),
            }
            
            yearly_summaries.append(yearly_summary)
        
        # Create yearly summary file
        yearly_df = pd.DataFrame(yearly_summaries)
        yearly_file = self.reports_dir / "yearly_summaries.csv"
        yearly_df.to_csv(yearly_file, index=False)
        
        logger.info(f"Created yearly summaries: {yearly_file}")
    
    def run(self, max_workers=4):
        """Run the complete data pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting NYC Green Taxi data pipeline at {start_time}")
        
        # Generate URLs
        url_list = self.generate_urls()
        logger.info(f"Processing {len(url_list)} files")
        
        # Download files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            downloaded_files = list(executor.map(self.download_parquet, url_list))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_file, downloaded_files)
        
        # Create combined dataset
        self.create_combined_dataset()
        
        # Create yearly summaries
        self.create_yearly_summaries()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Pipeline completed in {duration}")
        logger.info(f"Data is available in '{self.data_dir}'")
        logger.info(f"Raw data: '{self.raw_dir}'")
        logger.info(f"Processed data: '{self.processed_dir}'")
        logger.info(f"Reports: '{self.reports_dir}'")

if __name__ == "__main__":
    ...