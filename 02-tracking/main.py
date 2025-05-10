import argparse
import os
import json
from datetime import datetime
import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

from src.data_loader import load_data
from src.feature_engineering import engineer_features, TaxiFeatureEngineer
from src.model_training import train_model
from src.config import (
    MODEL_SAVE_PATH,
    HYPEROPT_ROOT_DIR,
    HYPEROPT_DEFAULT_EVALS,
    TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
    TRIP_DURATION_MIN_SECONDS,
    TRIP_DURATION_MAX_SECONDS,
    MAX_SPEED_MPH,
    FINANCIAL_COLUMNS,
    DEFAULT_MODEL_PARAMS,
)

# Import hyperparameter optimization (if available)
try:
    from src.hyperparameter import optimize_hyperparameters
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: hyperopt is not installed. Hyperparameter optimization will not be available.")

# Define constant experiment name instead of creating timestamped ones
EXPERIMENT_NAME = "taxi_trip_duration_prediction"

def main():
    """Main function to run the taxi trip duration prediction model"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train taxi trip duration prediction models")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["lr", "rf", "xgb", "lgbm"], 
        required=True, 
        help="Model type: 'lr' (Linear Regression), 'rf' (Random Forest), 'xgb' (XGBoost), or 'lgbm' (LightGBM)"
    )
    parser.add_argument(
        "--months", 
        type=str, 
        nargs="*", 
        help="List of months to retrieve data from (format: MM-YYYY). Example: '01-2024 02-2024'"
    )
    parser.add_argument(
        "--params", 
        type=str, 
        help="JSON string of model parameters. Example: '{\"n_estimators\": 200}'"
    )
    parser.add_argument(
        "--save-name", 
        type=str, 
        help="Custom name for saving the model"
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only perform feature engineering and save the processed data without training"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Custom name for the MLflow run (default: model type + timestamp)"
    )
    parser.add_argument(
        "--save-transformer",
        action="store_true",
        help="Save the feature engineering transformer for later use in prediction pipelines"
    )
    
    # Add hyperparameter optimization arguments
    if HYPEROPT_AVAILABLE:
        parser.add_argument(
            "--hyperopt",
            action="store_true",
            help="Perform hyperparameter optimization using hyperopt"
        )
        parser.add_argument(
            "--max-evals",
            type=int,
            default=50,
            help="Maximum number of hyperparameter combinations to try (default: 50)"
        )
        parser.add_argument(
            "--hyperopt-save-name",
            type=str,
            help="Custom name for saving the best model from hyperparameter optimization"
        )
    
    args = parser.parse_args()
    
    # Check if hyperopt is requested but not available
    if not HYPEROPT_AVAILABLE and hasattr(args, 'hyperopt') and args.hyperopt:
        print("Error: Hyperparameter optimization requested but hyperopt is not installed.")
        print("Please install it using: pip install hyperopt")
        return
    
    # Create directory structure
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Create run ID with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_timestamp}"
    
    # Set up MLflow tracking
    if TRACKING_URI.startswith("http"):
        mlflow.set_tracking_uri(TRACKING_URI)
        print(f"Using remote MLflow tracking server: {TRACKING_URI}")
    else:
        # Use file-based tracking
        tracking_uri = "file://" + os.path.abspath("mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using local file-based MLflow tracking: {tracking_uri}")
    
    # Set experiment name - use constant name instead of timestamped one
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data based on provided months or from local file
    if args.months:
        df = load_data(months=args.months)
        data_source = f"Remote parquet ({', '.join(args.months)})"
    else:
        df = load_data()
        data_source = "Local CSV"
    
    print(f"Data loaded from {data_source}, shape: {df.shape}")
    
    # Save raw data summary
    run_dir = os.path.join(MODEL_SAVE_PATH, run_id)
    data_summary_path = os.path.join(run_dir, "data_summary.txt")
    os.makedirs(os.path.dirname(data_summary_path), exist_ok=True)
    
    with open(data_summary_path, "w") as f:
        f.write(f"Data Summary\n")
        f.write(f"===========\n\n")
        f.write(f"Source: {data_source}\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"  - {col}: {dtype}\n")
        f.write("\nMissing Values:\n")
        for col, missing in df.isna().sum().items():
            f.write(f"  - {col}: {missing} ({missing/len(df):.2%})\n")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    
    # Create the feature engineering transformer
    transformer = TaxiFeatureEngineer()
    
    # Apply feature engineering
    X = transformer.transform(df)
    
    # Get target variable for rows that survived the cleaning
    y = df.loc[X.index, 'trip_duration']
    
    print(f"Feature engineering complete. Features: {X.shape}, Target: {y.shape}")
    
    # Save the transformer for later use in prediction pipelines
    if args.save_transformer:
        transformer_path = os.path.join(run_dir, "feature_transformer.joblib")
        joblib.dump(transformer, transformer_path)
        print(f"Feature transformer saved to: {transformer_path}")
    
    # Save processed features
    processed_data_dir = os.path.join(run_dir, "processed_data")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    X.to_csv(os.path.join(processed_data_dir, "X_features.csv"), index=False)
    y.to_csv(os.path.join(processed_data_dir, "y_target.csv"), index=False)
    
    # Feature stats summary
    feature_stats_path = os.path.join(processed_data_dir, "feature_stats.txt")
    with open(feature_stats_path, "w") as f:
        f.write("Feature Statistics\n")
        f.write("=================\n\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Number of samples: {X.shape[0]}\n\n")
        f.write("Feature list:\n")
        for col in X.columns:
            f.write(f"  - {col}\n")
        f.write("\nNumeric feature statistics:\n")
        f.write(X.describe().to_string())
        f.write("\n\nTarget variable statistics:\n")
        f.write(y.describe().to_string())
    
    if args.features_only:
        print("\nFeature engineering completed. Processed data saved to:")
        print(f"  - {os.path.join(processed_data_dir, 'X_features.csv')}")
        print(f"  - {os.path.join(processed_data_dir, 'y_target.csv')}")
        print("Exiting without training as --features-only flag was used.")
        return
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train-validation split: Train: {X_train.shape}, Validation: {X_val.shape}")
    
    # Save train/val splits
    X_train.to_csv(os.path.join(processed_data_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(processed_data_dir, "X_val.csv"), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(processed_data_dir, "y_val.csv"), index=False)
    
    # Parse custom parameters if provided
    if args.params:
        try:
            custom_params = json.loads(args.params)
            print(f"Using custom parameters: {custom_params}")
        except json.JSONDecodeError:
            print(f"Error parsing parameters JSON. Using default parameters.")
            custom_params = DEFAULT_MODEL_PARAMS.get(args.model, {})
    else:
        custom_params = DEFAULT_MODEL_PARAMS.get(args.model, {})
        print(f"Using default parameters for {args.model}: {custom_params}")
    
    # Define run name - either custom or auto-generated
    run_name = args.run_name if args.run_name else f"{args.model}_{run_timestamp}"
    
    # Start MLflow run explicitly with proper name
    with mlflow.start_run(run_name=run_name) as run:
        # Log data metadata
        mlflow.log_param("data_source", data_source)
        mlflow.log_param("data_rows", len(df))
        mlflow.log_param("feature_count", X.shape[1])
        
        # Log feature names
        feature_names = list(X.columns)
        mlflow.log_param("features", str(feature_names))
        
        # Log the feature engineering transformer if requested
        if args.save_transformer:
            mlflow.log_artifact(transformer_path, "feature_engineering")
        
        # Train the selected model
        print(f"\nTraining {args.model} model...")
        model, metrics = train_model(
            X_train, y_train, X_val, y_val, 
            model_type=args.model, 
            params=custom_params,
            active_run=run  # Pass the active run to avoid nested runs
        )
        
        # If hyperparameter tuning is requested
        if HYPEROPT_AVAILABLE and hasattr(args, 'hyperopt') and args.hyperopt:
            print("\n" + "="*50)
            print(f"Starting hyperparameter optimization for {args.model}")
            print("="*50)
            
            # Run hyperparameter optimization (passing the current run to keep it nested)
            best_params, best_model, best_metrics, trials = optimize_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type=args.model,
                max_evals=args.max_evals,
                parent_run=run
            )
            
            # Log improvement from base model to optimized model
            rmse_improvement = metrics['val_rmse'] - best_metrics['val_rmse']
            improvement_percentage = (rmse_improvement / metrics['val_rmse']) * 100
            
            mlflow.log_metric("base_to_optimized_rmse_improvement", rmse_improvement)
            mlflow.log_metric("base_to_optimized_improvement_percentage", improvement_percentage)
            
            # Print comparison
            print("\n" + "="*50)
            print("Base Model vs Optimized Model Comparison")
            print("="*50)
            print(f"Base Model Validation RMSE:      {metrics['val_rmse']:.4f}")
            print(f"Optimized Model Validation RMSE: {best_metrics['val_rmse']:.4f}")
            print(f"Improvement:                     {rmse_improvement:.4f} ({improvement_percentage:.2f}%)")
            print("="*50)
            
            # Save the best model with custom name if provided
            if hasattr(args, 'hyperopt_save_name') and args.hyperopt_save_name:
                best_model_path = os.path.join(MODEL_SAVE_PATH, f"{args.hyperopt_save_name}.joblib")
                joblib.dump(best_model, best_model_path)
                print(f"Best model from hyperparameter optimization saved to: {best_model_path}")
                
                # If saving transformer, create a complete pipeline with the best model
                if args.save_transformer:
                    from sklearn.pipeline import Pipeline
                    pipeline = Pipeline([
                        ('features', transformer),
                        ('model', best_model)
                    ])
                    pipeline_path = os.path.join(MODEL_SAVE_PATH, f"{args.hyperopt_save_name}_pipeline.joblib")
                    joblib.dump(pipeline, pipeline_path)
                    print(f"Complete pipeline with best model saved to: {pipeline_path}")
        
        # Add custom model name if provided
        if args.save_name:
            custom_model_path = os.path.join(MODEL_SAVE_PATH, f"{args.save_name}.joblib")
            joblib.dump(model, custom_model_path)
            print(f"Base model saved with custom name: {custom_model_path}")
            
            # If saving transformer, create a complete pipeline with the model
            if args.save_transformer:
                from sklearn.pipeline import Pipeline
                pipeline = Pipeline([
                    ('features', transformer),
                    ('model', model)
                ])
                pipeline_path = os.path.join(MODEL_SAVE_PATH, f"{args.save_name}_pipeline.joblib")
                joblib.dump(pipeline, pipeline_path)
                print(f"Complete pipeline saved to: {pipeline_path}")
    
    print("\nTraining pipeline complete!")
    print(f"All artifacts saved in: {os.path.join(MODEL_SAVE_PATH, run_id)}")
    print(f"MLflow run ID: {run.info.run_id}")
    print(f"View experiment details in MLflow UI: mlflow ui")
    
if __name__ == "__main__":
    main()