import mlflow
import numpy as np
import pandas as pd
import os
import json
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import MODEL_SAVE_PATH

# Define hyperparameter search spaces for each model type
SEARCH_SPACES = {
    "lr": {
        # Linear regression has minimal hyperparameters, but we can include:
        "fit_intercept": hp.choice("fit_intercept", [True, False]),
        "normalize": hp.choice("normalize", [True, False])
    },
    "rf": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
        "max_depth": scope.int(hp.quniform("max_depth", 5, 30, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": hp.choice("bootstrap", [True, False]),
        "random_state": 42
    },
    "xgb": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "gamma": hp.uniform("gamma", 0, 5),
        "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
        "random_state": 42
    },
    "lgbm": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "num_leaves": scope.int(hp.quniform("num_leaves", 20, 150, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-10), np.log(1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-10), np.log(1)),
        "random_state": 42
    }
}

def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type, max_evals=50, parent_run=None):
    """
    Optimize hyperparameters for the selected model using hyperopt.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        model_type (str): Type of model to optimize ('lr', 'rf', 'xgb', 'lgbm')
        max_evals (int): Maximum number of hyperparameter combinations to try
        parent_run (mlflow.ActiveRun): Parent MLflow run
    
    Returns:
        dict: Best hyperparameters found
        object: Trained model with best hyperparameters
        dict: Performance metrics for the best model
        Trials: hyperopt trials object with all optimization information
    """
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Unsupported model type for hyperparameter optimization: {model_type}")
        
    # Import necessary model classes
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
        
    try:
        import lightgbm as lgbm
        LGBM_AVAILABLE = True
    except ImportError:
        LGBM_AVAILABLE = False
    
    # Create a timestamp for this optimization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimization_dir = os.path.join(MODEL_SAVE_PATH, f"hyperopt_{model_type}_{timestamp}")
    os.makedirs(optimization_dir, exist_ok=True)
    
    # If no parent run is provided, create a new experiment
    if parent_run is None:
        run_name = f"hyperopt_{model_type}_{timestamp}"
        with mlflow.start_run(run_name=run_name) as run:
            parent_run = run
            own_run = True
    else:
        own_run = False
        
    # Log the hyperparameter search space
    search_space_path = os.path.join(optimization_dir, "search_space.json")
    with open(search_space_path, "w") as f:
        json.dump(SEARCH_SPACES[model_type], f, indent=2, default=str)
    mlflow.log_artifact(search_space_path)
    
    # Track all trials
    trials = Trials()
    best_model = None
    best_metrics = None
    
    # Define the objective function for hyperopt
    def objective(params):
        # Log iteration start
        iteration = len(trials) + 1
        iteration_start_time = datetime.now()
        
        # For integer parameters, ensure they are integers
        for param in ["n_estimators", "max_depth", "min_samples_split", 
                      "min_samples_leaf", "min_child_weight", "num_leaves"]:
            if param in params:
                params[param] = int(params[param])
                
        # Create child MLflow run for this iteration
        with mlflow.start_run(run_name=f"iteration_{iteration}", nested=True):
            # Log current parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Initialize and train the model
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
                
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            
            # Calculate metrics
            train_rmse = root_mean_squared_error(y_train, train_preds)
            train_mae = mean_absolute_error(y_train, train_preds)
            train_r2 = r2_score(y_train, train_preds)
            
            val_rmse = root_mean_squared_error(y_val, val_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("val_r2", val_r2)
            
            # Log feature importances if available
            if hasattr(model, "feature_importances_"):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                iteration_dir = os.path.join(optimization_dir, f"iteration_{iteration}")
                os.makedirs(iteration_dir, exist_ok=True)
                
                importance_path = os.path.join(iteration_dir, "feature_importance.csv")
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            
            # Log iteration time
            iteration_time = (datetime.now() - iteration_start_time).total_seconds()
            mlflow.log_metric("iteration_time_seconds", iteration_time)
            
            # Log iteration summary
            iteration_summary = {
                "iteration": iteration,
                "params": params,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "train_rmse": train_rmse,
                "iteration_time_seconds": iteration_time
            }
            
            iteration_summary_path = os.path.join(optimization_dir, f"iteration_{iteration}.json")
            with open(iteration_summary_path, "w") as f:
                json.dump(iteration_summary, f, indent=2, default=str)
            mlflow.log_artifact(iteration_summary_path)
            
            # Return the result to hyperopt
            result = {
                'loss': val_rmse,  # Minimize validation RMSE
                'status': STATUS_OK,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'train_rmse': train_rmse,
                'model': model,  # Save the trained model
                'iteration': iteration
            }
            
            print(f"Iteration {iteration}/{max_evals}: val_rmse={val_rmse:.4f}, val_r2={val_r2:.4f}")
            
            nonlocal best_model, best_metrics
            if best_model is None or val_rmse < best_metrics.get('val_rmse', float('inf')):
                best_model = model
                best_metrics = {
                    'train_rmse': train_rmse,
                    'train_mae': train_mae,
                    'train_r2': train_r2,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_r2': val_r2
                }
            
            return result
    
    print(f"\nStarting hyperparameter optimization for {model_type} model...")
    print(f"Search space: {json.dumps(SEARCH_SPACES[model_type], indent=2, default=str)}")
    print(f"Max evaluations: {max_evals}")
    
    # Run the optimization
    start_time = datetime.now()
    best_params = fmin(
        fn=objective,
        space=SEARCH_SPACES[model_type],
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        show_progressbar=True
    )
    
    optimization_time = (datetime.now() - start_time).total_seconds()
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    
    # Get the parameters in the correct format (converting indices to actual values)
    from hyperopt import space_eval
    best_params = space_eval(SEARCH_SPACES[model_type], best_params)
    
    # Log the best parameters and metrics
    with mlflow.start_run(run_name="best_model", nested=True):
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log best metrics
        for metric_name, metric_value in best_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log the best model
        if model_type == "lr":
            mlflow.sklearn.log_model(best_model, f"best_{model_type}_model")
        elif model_type == "rf":
            mlflow.sklearn.log_model(best_model, f"best_{model_type}_model")
        elif model_type == "xgb" and XGB_AVAILABLE:
            mlflow.xgboost.log_model(best_model, f"best_{model_type}_model")
        elif model_type == "lgbm" and LGBM_AVAILABLE:
            mlflow.lightgbm.log_model(best_model, f"best_{model_type}_model")
    
    # Create optimization summary
    optimization_summary = {
        "model_type": model_type,
        "max_evaluations": max_evals,
        "total_time_seconds": optimization_time,
        "best_parameters": best_params,
        "best_metrics": best_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save optimization summary
    summary_path = os.path.join(optimization_dir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(optimization_summary, f, indent=2, default=str)
    mlflow.log_artifact(summary_path)
    
    # Create a comprehensive results CSV with all trials
    results_df = pd.DataFrame()
    for i, trial in enumerate(trials.trials):
        if trial['result']['status'] == 'ok':
            trial_params = trial['misc']['vals']
            # Convert hyperopt's internal format to actual parameter values
            actual_params = space_eval(SEARCH_SPACES[model_type], {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in trial_params.items()})
            
            # Create a row with all information
            row = {
                'iteration': i + 1,
                'val_rmse': trial['result']['val_rmse'],
                'val_mae': trial['result']['val_mae'],
                'val_r2': trial['result']['val_r2'],
                'train_rmse': trial['result']['train_rmse']
            }
            
            # Add all parameters to the row
            for param, value in actual_params.items():
                row[param] = value
                
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    # Sort by validation RMSE (best first)
    results_df = results_df.sort_values('val_rmse')
    
    # Save and log the results
    results_path = os.path.join(optimization_dir, "all_trials_results.csv")
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)
    
    print("\n--- Hyperparameter Optimization Results ---")
    print(f"Best validation RMSE: {best_metrics['val_rmse']:.4f}")
    print(f"Best validation R²: {best_metrics['val_r2']:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # End the run only if we created it
    if own_run:
        mlflow.end_run()
    
    return best_params, best_model, best_metrics, trials