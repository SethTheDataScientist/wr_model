import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models import Model
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)
import model_functions



# set seed for reproducibility
np.random.seed(123)


# Load the data
path = os.path.abspath("../model_evaluation/data")
data = model_functions.DataLoader(path)    

save_dir=os.path.abspath(f"../mlruns/{data.run_data[1]}/{data.run_data[0]}/output")

model_functions.shap_analysis(data.best_model, data.train_X, data.test_X, model_type='regression',
                                save_dir=save_dir, prefix='shap_')

output_predictions = data.best_model.predict(data.full_predictions)
output_predictions = pd.Series(output_predictions, index=data.full_predictions.index)
updated_prediction_set = data.full_predictions.copy()
updated_prediction_set['predicted_label'] = output_predictions
updated_prediction_set = updated_prediction_set[['predicted_label'] + [col for col in updated_prediction_set.columns if col != 'predicted_label']]
updated_prediction_set = pd.merge(updated_prediction_set, data.eval_model_df['target'], how = 'left', left_index = True, right_index = True)
updated_prediction_set = updated_prediction_set[['target'] + [col for col in updated_prediction_set.columns if col != 'target']]


plot_predictions = data.best_model.predict(data.plot_prediction_set)
plot_predictions = pd.Series(plot_predictions, index=data.plot_prediction_set.index) 
plot_df = data.plot_prediction_set.copy()
plot_df['predicted_label'] = plot_predictions
plot_df = pd.merge(plot_df, data.eval_model_df['target'], how = 'left', left_index = True, right_index = True)


updated_prediction_set = updated_prediction_set.sort_values(by='predicted_label', ascending=False)
updated_prediction_set = pd.merge(updated_prediction_set, data.season_context, how = 'left', left_index = True, right_index = True)


# Plot the predicted_label against the target variable
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='predicted_label', y='target')
plt.xlabel("Predicted Label")
plt.ylabel("Target Variable")
plt.title("Predicted Label vs Target Variable")
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
sns.regplot(data=plot_df, x='predicted_label', y='target', scatter=False, color='blue')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_scatter_plot.png'), dpi=300, bbox_inches='tight')
plt.close()


current_year = updated_prediction_set[updated_prediction_set['Last_Season'] == 2024].copy()
other_prospects = updated_prediction_set[updated_prediction_set['Last_Season'] != 2024].copy()
current_year.to_csv(f"../mlruns/{data.run_data[1]}/{data.run_data[0]}/output/rb_model_predictions_2024.csv", index = True)