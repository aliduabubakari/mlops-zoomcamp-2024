import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class TaxiFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that performs feature engineering
    on taxi trip data. This class encapsulates all the feature engineering
    logic to enable clean use in any scikit-learn Pipeline and allows
    for pickling/logging.
    
    This transformer:
    - Cleans data (removing invalid trips, etc.)
    - Creates datetime features
    - Engineers distance and speed features
    - Handles categorical features
    - Creates financial features
    
    The transformer is stateless (no fit required) and returns a clean
    feature matrix ready for modeling.
    """
    
    def __init__(self):
        # Store the feature column names for later retrieval
        self.feature_cols_ = []
    
    def fit(self, X, y=None):
        """
        Nothing to learn in this stateless feature engineering.
        
        Args:
            X (pd.DataFrame): Raw taxi trip data
            y: Ignored, exists for scikit-learn compatibility
            
        Returns:
            self: Returns self for method chaining
        """
        return self
    
    def transform(self, df):
        """
        Transform the raw taxi data into engineered features.
        
        Args:
            df (pd.DataFrame): Raw taxi trip data
            
        Returns:
            pd.DataFrame: Engineered feature matrix
        """
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # ========== Basic Data Cleaning ==========
        # Drop rows with missing key datetime columns
        df = df[df["lpep_pickup_datetime"].notna() & df["lpep_dropoff_datetime"].notna()]
        
        # Drop rows where dropoff time is before pickup time
        invalid_times = df["lpep_dropoff_datetime"] <= df["lpep_pickup_datetime"]
        if invalid_times.sum() > 0:
            df = df[~invalid_times]
        
        # ========== Drop Unused Columns ==========
        # Drop ehail_fee as it's all null
        if 'ehail_fee' in df.columns:
            df.drop(columns=['ehail_fee'], inplace=True)
        
        # ========== Handle Financial Columns ==========
        # Define financial columns
        financial_cols = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                        'tolls_amount', 'improvement_surcharge', 'total_amount',
                        'congestion_surcharge']
        
        # Ensure all financial columns exist
        financial_cols = [col for col in financial_cols if col in df.columns]
        
        # Handle negative values in financial columns
        for col in financial_cols:
            if col in df.columns:
                neg_mask = df[col] < 0
                
                # Add flag column to indicate refund
                df[f'{col}_was_refund'] = neg_mask
                
                # Replace negative values with 0
                df.loc[neg_mask, col] = 0
            else:
                # If column doesn't exist, still add a flag column for consistency (all False)
                df[f'{col}_was_refund'] = False
        
        # ========== Create DateTime Features ==========
        # Extract datetime features from pickup time
        df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['lpep_pickup_datetime'].dt.dayofweek
        df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
        df['pickup_year'] = df['lpep_pickup_datetime'].dt.year
        df['pickup_weekday'] = df['lpep_pickup_datetime'].dt.dayofweek < 5  # True for weekdays
        df['pickup_weekend'] = ~df['pickup_weekday']
        
        # Time-of-day categories
        df['is_morning'] = (df['pickup_hour'] >= 6) & (df['pickup_hour'] < 10)   # 6am - 10am
        df['is_day'] = (df['pickup_hour'] >= 10) & (df['pickup_hour'] < 16)      # 10am - 4pm
        df['is_evening'] = (df['pickup_hour'] >= 16) & (df['pickup_hour'] < 20)  # 4pm - 8pm
        df['is_night'] = (df['pickup_hour'] >= 20) | (df['pickup_hour'] < 6)     # 8pm - 6am
        
        # ========== Calculate Trip Duration ==========
        # Calculate trip duration in seconds
        df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds()
        
        # Filter out unreasonable trip durations (less than 60 seconds or more than 3 hours)
        duration_mask = (df['trip_duration'] >= 60) & (df['trip_duration'] <= 10800)  # 1 min to 3 hours
        df = df[duration_mask]
        
        # ========== Handle Categorical Features ==========
        # Handle store_and_fwd_flag
        if 'store_and_fwd_flag' in df.columns:
            df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).fillna(0)
        
        # Convert payment_type to categorical features
        if 'payment_type' in df.columns:
            payment_dummies = pd.get_dummies(df['payment_type'], prefix='payment', dtype=int)
            df = pd.concat([df, payment_dummies], axis=1)
        
        # Convert RatecodeID to categorical features if present
        if 'RatecodeID' in df.columns:
            ratecode_dummies = pd.get_dummies(df['RatecodeID'], prefix='ratecode', dtype=int)
            df = pd.concat([df, ratecode_dummies], axis=1)
        
        # ========== Distance and Speed Features ==========
        # Handle trip_distance
        if 'trip_distance' in df.columns:
            # Flag potential GPS errors (very short distances)
            df['zero_distance'] = df['trip_distance'] < 0.1
            
            # Calculate average speed (mph) - handle division by zero
            df['avg_speed'] = np.where(
                df['trip_duration'] > 0, 
                (df['trip_distance'] / (df['trip_duration'] / 3600)),  # trip_duration is in seconds
                0
            )
            
            # Filter out unrealistic speeds (> 80 mph)
            speed_mask = (df['avg_speed'] <= 80)
            df = df[speed_mask]
        
        # ========== Create Special Features ==========
        # Calculate total surcharges
        surcharge_cols = [col for col in ['extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge'] 
                          if col in df.columns]
        if surcharge_cols:
            df['total_surcharges'] = df[surcharge_cols].sum(axis=1)
        
        # Calculate tip percentage where applicable
        if 'tip_amount' in df.columns and 'fare_amount' in df.columns:
            df['tip_percentage'] = np.where(
                df['fare_amount'] > 0,
                (df['tip_amount'] / df['fare_amount']) * 100,
                0
            )
        
        # ========== Prepare Final Features ==========
        # List of numerical features to keep
        numerical_features = [
            'passenger_count', 'trip_distance', 'fare_amount', 
            'pickup_hour', 'pickup_day', 'pickup_month',
            'store_and_fwd_flag', 'avg_speed', 'total_surcharges', 'tip_percentage'
        ]
        
        # Filter to only include columns that exist in the dataframe
        numerical_features = [col for col in numerical_features if col in df.columns]
        
        # Boolean features
        boolean_features = [col for col in df.columns if df[col].dtype == bool]
        
        # Get dummy column names
        dummy_cols = [col for col in df.columns if col.startswith(('payment_', 'ratecode_'))]
        
        # Combine all feature columns
        feature_cols = numerical_features + boolean_features + dummy_cols
        
        # Select only columns that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Drop rows with any missing values in feature columns
        df_clean = df.dropna(subset=feature_cols)
        
        # Create feature matrix
        X = df_clean[feature_cols]
        
        # Store feature column names for later retrieval
        self.feature_cols_ = list(X.columns)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Ignored, exists for scikit-learn compatibility
            
        Returns:
            list: Feature names created by this transformer
        """
        return self.feature_cols_


# For backward compatibility with the original function API
def engineer_features(df):
    """
    Perform feature engineering on the taxi trip data.
    
    Args:
        df (pd.DataFrame): Raw taxi trip data
    
    Returns:
        tuple: (X, y) - feature matrix and target vector
    """
    print("Starting feature engineering process...")
    print(f"Initial data shape: {df.shape}")
    
    # Use TaxiFeatureEngineer for feature engineering
    transformer = TaxiFeatureEngineer()
    X = transformer.transform(df)
    
    # Get trip_duration for all rows that survived the cleaning
    y = df.loc[X.index, 'trip_duration']
    
    # Log data reduction
    print(f"Initial rows: {df.shape[0]}")
    print(f"Rows after cleaning: {X.shape[0]}")
    print(f"Data retention: {X.shape[0] / df.shape[0]:.2%}")
    print(f"Final feature set: {X.shape[1]} features")
    print(f"Feature list: {', '.join(X.columns)}")
    
    return X, y