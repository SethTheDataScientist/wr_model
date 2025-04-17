import sys
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)
import model_functions



# set seed for reproducibility
np.random.seed(123)
verbose = False


path = os.path.abspath("../model_training/data")
# Load the data
data = model_functions.DataLoader(path)    

[data.X, data.y,
data.train_X, data.train_y, data.train_sw, 
data.val_X, data.val_y, data.val_sw,
data.test_X, data.test_y, data.test_sw] = model_functions.prepare_model_splits(data.train_df, data.select_features, sample_weights_col  = 'clean_scaling')

data.plot_prediction_set = data.prediction_set[data.select_features]

data.full_predictions = data.model_df[data.select_features]

data.eval_model_df = data.model_df.copy()

data.best_model, data.feature_importances, data.test_predictions, data.best_grid, data.results = model_functions.xgboost_regression_model(data.train_X, data.train_y, data.val_X, data.val_y, data.test_X, data.test_y, 
                            val_sw  = data.val_sw, train_sw  = data.train_sw, test_sw  = data.test_sw,
                            monotonic_constraints=data.monotonic_constraints)



# Create a new MLflow Experiment
mlflow.set_experiment("rb_model_experiment")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(data.best_grid)

    # Log the metrics
    for metric, value in data.results.items():
        mlflow.log_metric(metric, value)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Switching to only players with nfl drafted data")

    # Infer the model signature
    signature = infer_signature(data.train_X, data.best_model.predict(data.train_X))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=data.best_model,
        artifact_path="rb_model_V1",
        signature=signature,
        input_example=data.train_X,
        registered_model_name="rb_model_V1",
    )

    
    run = mlflow.active_run()
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    data.run_data  = [run_id, experiment_id]

# After processing your data, save all dataframes and check against an existing directory
result = data.save_dataframes(save_dir=os.path.abspath("../model_evaluation/data"))

# See what was saved and what was skipped
if verbose:
    print(f"Saved dataframes: {result['saved']}")
    print(f"Skipped dataframes: {result['skipped']}")