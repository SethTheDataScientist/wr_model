import sys
import os
import pandas as pd
import numpy as np
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)
import model_functions


# set seed for reproducibility
np.random.seed(123)
verbose = False

path = os.path.abspath("../data_ingestion/data")
# Load the data
data = model_functions.DataLoader(data_dir = path)    
# model_functions.DataLoader.get_dataframe_info(data)

# Filter the wr_model_df to only include last season of 2020 and before
data.wr_model_df['Last_Season'] = data.wr_model_df['Last_Season'].astype('int')
data.wr_model_df['Seasons'] = data.wr_model_df['Seasons'].astype('int')

data.filtered_df = pd.merge(data.wr_model_df, data.PFF_College_conference_addition, how='left', on='team_name')

data.working_df = data.College_PFF_player_mapping.copy()

data.working_df = (
    data.working_df
    .sort_values(by='score', ascending=False)
    .groupby('player_name', as_index=False)
    .head(1)
    .reset_index(drop=True)
)

data.filtered_df = pd.merge(data.filtered_df, data.working_df, how = 'left', left_on = 'player_id', right_on = 'pff_id')

data.filtered_df = data.filtered_df[(data.filtered_df['nic_year'] == data.filtered_df['Last_Season'] + 1) | (data.filtered_df['nic_year'].isna())]


data.filtered_df = pd.merge(data.filtered_df, data.College_NGS_Player_Scores, how = 'left', on = ['player_name', 'nic_year', 'pro_pos', 'school'])
data.filtered_df = pd.merge(data.filtered_df, data.Imputed_Big_Board_Ranks, how = 'left', on = ['player_name', 'nic_year', 'pro_pos', 'school'])

data.filtered_df = data.filtered_df.drop_duplicates()

data.filtered_df = data.filtered_df.drop(columns=['team_name', 'pos_team', 'Conference', 'player_id_y', 'player'])

data.filtered_df = pd.get_dummies(data.filtered_df, columns = ['Strength', 'Filter'])

data.player_ids = data.filtered_df['player_id_x'].unique()

data.select_features = ['ID', 'player_id_x', 'Last_Season', 'Seasons',
                    'ContestedTile', 
                    'Value', 
                    'TotalNonSepSeasons', 
                    'NonSepPercent',
                    'Slot_rate', 'best_Slot_rate', 'worst_Slot_rate',
                    'RR', 'TPRR', 'YPRR', 'TDPRR', 'ADOT', 'YAC',                     
                    'best_RR', 'best_TPRR',  'best_YPRR', 'best_TDPRR', 'best_ADOT', 'best_YAC',
                    'worst_RR', 'worst_TPRR',  'worst_YPRR', 'worst_TDPRR', 'worst_ADOT', 'worst_YAC',
                    'Strength_Power 5', 'Filter_NonSeparator', 'Filter_Solid', 'Filter_Gadget',
                    'ht_in', 'wt', 'arm_in', 'wing_in',
                    'c_reps', 'c_10y', 'c_40y', 'c_vj_in', 'c_bj_in', 'c_3c', 'c_ss20', 'est_40y', 'WAR',
                    'athleticism_score',
                    #   'production_score'
                    ]


data.monotonic_constraints = {
    'Seasons': 0, 
    'ContestedTile': 1,
    'athleticism_score': 1,
    # 'production_score': 1,
    'Value': 1, 
    'TotalNonSepSeasons': -1, 
    'NonSepPercent': -1,
    'Slot_rate': 0, 'best_Slot_rate': 0, 'worst_Slot_rate': 0,
    'RR': 1, 'TPRR': 1, 'YPRR': 1, 'TDPRR': 1, 'ADOT': 0, 'YAC': 1,
    'best_RR': 1, 'best_TPRR': 1,  'best_YPRR': 1, 'best_TDPRR': 1, 'best_ADOT': 0, 'best_YAC': 1,
    'worst_RR': 1, 'worst_TPRR': 1,  'worst_YPRR': 1, 'worst_TDPRR': 1, 'worst_ADOT': 0, 'worst_YAC': 1,
    'Strength_Power 5': 0,
    'Filter_NonSeparator': -1, 'Filter_Solid': 1, 'Filter_Gadget': -1,
    'ht_in': 1, 'wt': 1, 'arm_in': 1, 'wing_in': 1,
    'c_reps': 1, 'c_10y': -1, 'c_40y': -1, 'c_vj_in': 1, 'c_bj_in': 1,
    'c_3c': -1, 'c_ss20': -1, 'est_40y': -1, 'WAR': 1
}

data.filtered_df['draft_day'] = np.where(data.filtered_df['Last_Season'] == 2024, 7, data.filtered_df['draft_day'])
data.filtered_df = data.filtered_df[data.filtered_df['draft_day'].notna()]
data.model_df = data.filtered_df[data.select_features]

# Identify boolean columns
bool_cols = data.model_df.select_dtypes(include='bool').columns

# One-hot encode those columns (True → 1, False → 0)
data.model_df[bool_cols] = data.model_df[bool_cols].astype(int)

# Target Variable
nfl = data.nfl_target.copy()

nfl = nfl[nfl['player_id'].isin(data.player_ids)][['player_id', 'new_close_score', 'clean_scaling']]

data.model_df = pd.merge(data.model_df, nfl, how = 'left', left_on = 'player_id_x', right_on = 'player_id')

data.model_df = data.model_df.set_index('ID')
data.select_features.remove('ID')
data.select_features.remove('player_id_x')
data.select_features.remove('Last_Season')

data.model_df = data.model_df.drop(columns=['player_id_x', 'player_id'])

data.model_df = data.model_df.rename(columns={'new_close_score': 'target'})

# Convert target to percentile
data.model_df['target'] = data.model_df['target'].astype('float')
data.model_df = model_functions.convert_to_percentile(data.model_df, 'target')

data.model_df['target'] = data.model_df['target'].apply(
    lambda x: np.random.uniform(0, 0.1) if pd.isna(x) else x
)


# Fill in missing athletic information with knn values
data.model_df = model_functions.knn_impute_columns(data.model_df, target_columns = [
                    'ht_in', 'wt', 'arm_in', 'wing_in',
                    'c_reps', 'c_10y', 'c_40y', 'c_vj_in',
                      'c_bj_in', 'c_3c', 'c_ss20', 'est_40y'],
                      feature_columns = [
                    'ContestedTile', 
                    'Value', 
                    'NonSepPercent',
                    'Slot_rate', 'best_Slot_rate', 'worst_Slot_rate',
                    'RR', 'TPRR', 'YPRR', 'TDPRR', 'ADOT', 'YAC',                     
                    'best_RR', 'best_TPRR',  'best_YPRR', 'best_TDPRR', 'best_ADOT', 'best_YAC',
                    'worst_RR', 'worst_TPRR',  'worst_YPRR', 'worst_TDPRR', 'worst_ADOT', 'worst_YAC',
                    'Strength_Power 5', 'Filter_NonSeparator', 'Filter_Solid', 'Filter_Gadget',
                      ], n_neighbors = 5)


data.model_df = model_functions.impute_all_missing_values(data.model_df, method='median')

data.model_df = data.model_df.drop_duplicates()

# Split into training and prediction sets
data.train_df = data.model_df[data.model_df['Last_Season'] != 2024] 
data.prediction_set = data.model_df[data.model_df['Last_Season'] != 2024]

data.season_context = data.model_df[['Last_Season']] 

data.train_df = data.train_df.drop(columns=['Last_Season'])
data.prediction_set = data.prediction_set.drop(columns=['Last_Season'])

# After processing your data, save all dataframes and check against an existing directory
result = data.save_dataframes(save_dir=os.path.abspath("../model_training/data"), check_dir=path)

# See what was saved and what was skipped
if verbose:
    print(f"Saved dataframes: {result['saved']}")
    print(f"Skipped dataframes: {result['skipped']}")