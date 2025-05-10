def train_model(X_train, y_train, X_val, y_val, model_type="rf", params=None, active_run=None):
    """
    Train a regression model and log metrics with MLflow
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        model_type (str): Type of model to train ('lr', 'rf', 'xgb', 'lgbm')
        params (dict): Parameters for the model
        active_run (mlflow.ActiveRun, optional): Active MLflow run to use instead of creating a new one
    
    Returns:
        tuple: (trained model, metrics dictionary)
    """
    import mlflow
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    import joblib
    import os
    from datetime import datetime
    from src.config import MODEL_SAVE_PATH
    
    # Import additional models
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
        print("XGBoost not installed. XGBoost models will not be available.")

    try:
        import lightgbm as lgbm
        LGBM_AVAILABLE = True
    except ImportError:
        LGBM_AVAILABLE = False
        print("LightGBM not installed. LightGBM models will not be available.")
    
    def root_mean_squared_error(y_true, y_pred):
        """Calculate RMSE"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def create_model_artifacts_dir():
        """Create model artifacts directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = os.path.join(MODEL_SAVE_PATH, f"run_{timestamp}")
        os.makedirs(artifacts_dir, exist_ok=True)
        return artifacts_dir
    
    if params is None:
        params = {}
    
    # Create timestamped artifacts directory
    artifacts_dir = create_model_artifacts_dir()
    print(f"Model artifacts will be saved to: {artifacts_dir}")
    
    # Use the parent run if provided (avoiding nested runs)
    if active_run is None:
        # Start MLflow run with tags
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        own_run = mlflow.start_run(run_name=run_name)
        in_own_run = True
    else:
        own_run = active_run
        in_own_run = False
    
    # Log run metadata
    mlflow.set_tags({
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "feature_count": X_train.shape[1],
        "training_samples": X_train.shape[0],
    })
    
    # Initialize the appropriate model based on model_type
    if model_type == "lr":
        model = LinearRegression(**params)
    elif model_type == "rf":
        model = RandomForestRegressor(**params)
    elif model_type == "xgb":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it using 'pip install xgboost'")
        model = xgb.XGBRegressor(**params)
    elif model_type == "lgbm":
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install it using 'pip install lightgbm'")
        model = lgbm.LGBMRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Log feature names
    feature_names_path = os.path.join(artifacts_dir, "feature_names.txt")
    with open(feature_names_path, "w") as f:
        f.write("\n".join(X_train.columns))
    mlflow.log_artifact(feature_names_path)
    
    # Train the model
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Compute metrics
    metrics = {
        "train_rmse": root_mean_squared_error(y_train, train_preds),
        "train_mae": mean_absolute_error(y_train, train_preds),
        "train_r2": r2_score(y_train, train_preds),
        "val_rmse": root_mean_squared_error(y_val, val_preds),
        "val_mae": mean_absolute_error(y_val, val_preds),
        "val_r2": r2_score(y_val, val_preds)
    }
    
    # Log parameters and metrics
    mlflow.log_param("model_type", model_type)
    for param, value in params.items():
        mlflow.log_param(param, value)
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Save and log model artifact
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{model_type}_{timestamp}.joblib"
    model_path = os.path.join(artifacts_dir, model_filename)
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    
    # Log model directly to MLflow model registry
    if model_type == "lr":
        mlflow.sklearn.log_model(model, f"{model_type}_model")
    elif model_type == "rf":
        mlflow.sklearn.log_model(model, f"{model_type}_model")
    elif model_type == "xgb" and XGB_AVAILABLE:
        mlflow.xgboost.log_model(model, f"{model_type}_model")
    elif model_type == "lgbm" and LGBM_AVAILABLE:
        mlflow.lightgbm.log_model(model, f"{model_type}_model")
    
    # Create and log model metadata
    metadata = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "features": list(X_train.columns),
        "parameters": params,
        "metrics": metrics
    }
    
    metadata_path = os.path.join(artifacts_dir, "model_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            if key == "features":
                f.write(f"{key}:\n")
                for feature in value:
                    f.write(f"  - {feature}\n")
            elif key == "parameters":
                f.write(f"{key}:\n")
                for param_key, param_value in value.items():
                    f.write(f"  {param_key}: {param_value}\n")
            elif key == "metrics":
                f.write(f"{key}:\n")
                for metric_key, metric_value in value.items():
                    f.write(f"  {metric_key}: {metric_value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    mlflow.log_artifact(metadata_path)
    
    # Log predictions for analysis
    predictions_df = pd.DataFrame({
        'actual': y_val,
        'predicted': val_preds,
        'error': y_val - val_preds,
        'abs_error': np.abs(y_val - val_preds),
        'squared_error': (y_val - val_preds) ** 2
    })
    
    predictions_path = os.path.join(artifacts_dir, "validation_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    mlflow.log_artifact(predictions_path)
    
    # Log feature importances if available
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(artifacts_dir, f"{model_type}_feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Also create a text report of feature importance rankings
        importance_report_path = os.path.join(artifacts_dir, f"{model_type}_feature_ranking.txt")
        with open(importance_report_path, "w") as f:
            f.write("Feature Importance Ranking:\n")
            for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance'])):
                f.write(f"{i+1}. {feature}: {importance:.6f}\n")
        
        mlflow.log_artifact(importance_report_path)
    
    # End run only if we created it
    if in_own_run:
        mlflow.end_run()
    
    print(f"\nModel Training Results ({model_type}):")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
    print(f"Validation R²: {metrics['val_r2']:.4f}")
    print(f"Model saved to: {model_path}")
    
    return model, metrics