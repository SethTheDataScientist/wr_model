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
save_path = os.path.abspath("../model_training/data")
# Load the data
def run_data_ingestion(path, save_path):
    data = model_functions.DataLoader(data_dir = path)    
    # model_functions.DataLoader.get_dataframe_info(data)

    data.rb_model_df['Last_Season'] = data.rb_model_df['Last_Season'].astype('int')
    data.rb_model_df['Seasons'] = data.rb_model_df['Seasons'].astype('int')

    data.filtered_df = data.rb_model_df.copy()

    data.working_df = data.College_PFF_player_mapping.copy()

    data.working_df = (
        data.working_df
        .sort_values(by='score', ascending=False)
        .groupby('player_name', as_index=False)
        .head(1)
        .reset_index(drop=True)
    ).drop(columns=['player_id'])

    data.filtered_df = pd.merge(data.filtered_df, data.working_df, how = 'left', left_on = 'player_id', right_on = 'pff_id')

    # data.filtered_df = data.filtered_df[(data.filtered_df['nic_year'] == data.filtered_df['Last_Season'] + 1) | (data.filtered_df['nic_year'].isna())]

    # print(data.filtered_df[data.filtered_df['ID.x'] == 'Alvin Kamara - 11822'])

    data.filtered_df = data.filtered_df.drop_duplicates()

    data.filtered_df = pd.get_dummies(data.filtered_df, columns = ['Strength'])

    data.player_ids = data.filtered_df['player_id'].unique()

    data.select_features = ['ID.x', 'player_id', 'Last_Season', 'Seasons',
                            'Attempts', 'ForcedMissedTackleRate', 'ExplosiveRate', 'TDP', 'YardsAfterContact', 'YPRR', 'WAR', 'RushWAR', 'RecWAR',
                            'best_Attempts', 'best_ForcedMissedTackleRate', 'best_ExplosiveRate', 'best_TDP', 'best_YardsAfterContact', 'best_YPRR', 'best_WAR', 'best_RushWAR', 'best_RecWAR',
                            'worst_Attempts', 'worst_ForcedMissedTackleRate', 'worst_ExplosiveRate', 'worst_TDP', 'worst_YardsAfterContact', 'worst_YPRR', 'worst_WAR', 'worst_RushWAR', 'worst_RecWAR',
                            'Strength_Power 5',
                        'ht_in', 'wt', 'arm_in', 'wing_in',
                        'c_reps', 'c_10y', 'c_40y', 'c_vj_in', 'c_bj_in', 'c_3c', 'c_ss20', 'est_40y',
                        ]


    data.monotonic_constraints = {
        'Seasons': 0, 'Attempts': 1, 
        'ForcedMissedTackleRate': 1, 'ExplosiveRate': 1, 'TDP': 1, 'YardsAfterContact': 1, 'YPRR': 1, 'WAR': 1, 'RushWAR': 1, 'RecWAR': 1,
        'best_Attempts': 1, 'best_ForcedMissedTackleRate': 1, 'best_ExplosiveRate': 1, 'best_TDP': 1, 'best_YardsAfterContact': 1, 'best_YPRR': 1, 'best_WAR': 1, 'best_RushWAR': 1, 'best_RecWAR': 1,
        'worst_Attempts': 1, 'worst_ForcedMissedTackleRate': 1, 'worst_ExplosiveRate': 1, 'worst_TDP': 1, 'worst_YardsAfterContact': 1, 'worst_YPRR': 1, 'worst_WAR': 1, 'worst_RushWAR': 1, 'worst_RecWAR': 1,
        'Strength_Power 5': 0,
        'ht_in': 0, 'wt': 0, 'arm_in': 0, 'wing_in': 0,
        'c_reps': 1, 'c_10y': -1, 'c_40y': -1, 'c_vj_in': 1, 'c_bj_in': 1,
        'c_3c': -1, 'c_ss20': -1, 'est_40y': -1
    }

    data.filtered_df['draft_day'] = np.where(data.filtered_df['Last_Season'] == 2024, 7, data.filtered_df['draft_day'])
    data.filtered_df = data.filtered_df[data.filtered_df['draft_day'].notna()]
    data.model_df = data.filtered_df[data.select_features]

    # Identify boolean columns
    bool_cols = data.model_df.select_dtypes(include='bool').columns

    # One-hot encode those columns (True → 1, False → 0)
    data.model_df[bool_cols] = data.model_df[bool_cols].astype(int)

    # Target Variable
    nfl = data.rb_nfl_target.copy()

    nfl = nfl[nfl['player_id'].isin(data.player_ids)][['player_id', 'new_close_score', 'clean_scaling']]

    data.model_df = pd.merge(data.model_df, nfl, how = 'left', left_on = 'player_id', right_on = 'player_id')

    data.model_df = data.model_df.set_index('ID.x')
    data.select_features.remove('ID.x')
    data.select_features.remove('player_id')
    data.select_features.remove('Last_Season')

    data.model_df = data.model_df.drop(columns=['player_id'])

    data.model_df = data.model_df.rename(columns={'new_close_score': 'target'})

    # Convert target to percentile
    data.model_df['target'] = data.model_df['target'].astype('float')
    data.model_df = model_functions.convert_to_percentile(data.model_df, 'target')

    data.model_df['target'] = data.model_df['target'].apply(
        lambda x: np.random.uniform(0, 0.1) if pd.isna(x) else x
    )

    data.model_df['target'] = data.model_df['target'].apply(
        lambda x: 0.001 if x == 0 else (0.999 if x == 1 else x)
    )

    data.model_df['clean_scaling'] = np.log(data.model_df['target']) - np.log((1 - data.model_df['target']))
    data.model_df['clean_scaling'] = data.model_df['clean_scaling'] + 12

    data.model_df = model_functions.impute_all_missing_values(data.model_df, method='median')

    data.model_df = data.model_df.drop_duplicates()

    # Split into training and prediction sets
    data.train_df = data.model_df[data.model_df['Last_Season'] != 2024] 
    data.prediction_set = data.model_df[data.model_df['Last_Season'] != 2024]

    data.season_context = data.model_df[['Last_Season']] 

    data.train_df = data.train_df.drop(columns=['Last_Season'])
    data.prediction_set = data.prediction_set.drop(columns=['Last_Season'])

    # After processing your data, save all dataframes and check against an existing directory
    result = data.save_dataframes(save_dir=save_path, check_dir=path)

    # See what was saved and what was skipped
    if verbose:
        print(f"Saved dataframes: {result['saved']}")
        print(f"Skipped dataframes: {result['skipped']}")


# path = os.path.abspath("../rb_model/data_ingestion/data")
# save_path = os.path.abspath("../rb_model/model_training/data")
run_data_ingestion(path, save_path)