import pandas as pd
import os
from src.config import DATA_PATH

def load_data(months=None):
    """
    Load trip data from either local CSV file or remote parquet files.
    
    Args:
        months (list, optional): List of month-year combinations (e.g., ["01-2024", "02-2024"]).
            If provided, data will be loaded from remote parquet files.
    
    Returns:
        pd.DataFrame: Processed trip data
    """
    if months is None:
        # Load data from local CSV file (original behavior)
        if os.path.exists(DATA_PATH):
            print(f"Loading data from local file: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH, parse_dates=["lpep_pickup_datetime", "lpep_dropoff_datetime"])
        else:
            raise FileNotFoundError(f"Local data file not found: {DATA_PATH}")
    else:
        # Load data from remote parquet files
        print(f"Loading data from remote parquet files for months: {months}")
        dfs = []
        
        for month_year in months:
            month, year = month_year.split("-")
            url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month}.parquet"
            print(f"Fetching data from: {url}")
            try:
                month_df = pd.read_parquet(url)
                print(f"Successfully loaded data for {month}-{year}, shape: {month_df.shape}")
                dfs.append(month_df)
            except Exception as e:
                print(f"Error loading data for {month}-{year}: {e}")
        
        if not dfs:
            raise ValueError("No data was successfully loaded from the specified months")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Combined data shape: {df.shape}")
    
    # Common processing for both data sources
    df = df[df["lpep_dropoff_datetime"] > df["lpep_pickup_datetime"]]
    df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60.0  # minutes
    
    return df