import os
import pandas as pd
import pickle

from src.constants import DATA_DIR
from src.data_utils.race import prepare_racepp_dataset
from src.data_utils.arc import prepare_arc_dataset
from src.data_utils.am import prepare_assistments_dataset
from src.data_utils.text2props_mapping import get_difficulty_dict_for_text2props, get_df_for_text2props

# # RACE++
print("DOING RACE++")
race_data_dir = 'data/raw/RACE'
race_c_data_dir = 'data/raw/race-c-master/data'
prepare_racepp_dataset(race_data_dir, race_c_data_dir, DATA_DIR)
# conversion to text2props format
df_train = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_test.csv'))
df_dev = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_dev.csv'))
pickle.dump(get_difficulty_dict_for_text2props(pd.concat([df_train, df_test, df_dev])),
            open(os.path.join(DATA_DIR, 't2p_race_pp_difficulty_dict.p'), 'wb'))
get_df_for_text2props(df_train).to_csv(os.path.join(DATA_DIR, f't2p_race_pp_train.csv'), index=False)
get_df_for_text2props(df_test).to_csv(os.path.join(DATA_DIR, f't2p_race_pp_test.csv'), index=False)
get_df_for_text2props(df_dev).to_csv(os.path.join(DATA_DIR, f't2p_race_pp_dev.csv'), index=False)

# ARC
print("DOING ARC")
arc_data_dir = 'data/raw/ARC-V1-Feb2018'
prepare_arc_dataset(arc_data_dir, 'data/processed')
# conversion to text2props format
df_train = pd.read_csv(os.path.join(DATA_DIR, 'arc_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'arc_test.csv'))
df_dev = pd.read_csv(os.path.join(DATA_DIR, 'arc_dev.csv'))
pickle.dump(get_difficulty_dict_for_text2props(pd.concat([df_train, df_test, df_dev])),
            open(os.path.join(DATA_DIR, 't2p_arc_difficulty_dict.p'), 'wb'))
get_df_for_text2props(df_train).to_csv(os.path.join(DATA_DIR, f't2p_arc_train.csv'), index=False)
get_df_for_text2props(df_test).to_csv(os.path.join(DATA_DIR, f't2p_arc_test.csv'), index=False)
get_df_for_text2props(df_dev).to_csv(os.path.join(DATA_DIR, f't2p_arc_dev.csv'), index=False)

# ASSISTments
print("[INFO] DOING AM...")
am_data_dir = 'data/interim/assistments'
prepare_assistments_dataset(am_data_dir, 'data/processed')
# conversion to text2props format
df_train = pd.read_csv(os.path.join(DATA_DIR, 'am_train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'am_test.csv'))
df_dev = pd.read_csv(os.path.join(DATA_DIR, 'am_dev.csv'))
pickle.dump(get_difficulty_dict_for_text2props(pd.concat([df_train, df_test, df_dev])),
            open(os.path.join(DATA_DIR, 't2p_am_difficulty_dict.p'), 'wb'))
get_df_for_text2props(df_train).to_csv(os.path.join(DATA_DIR, f't2p_am_train.csv'), index=False)
get_df_for_text2props(df_test).to_csv(os.path.join(DATA_DIR, f't2p_am_test.csv'), index=False)
get_df_for_text2props(df_dev).to_csv(os.path.join(DATA_DIR, f't2p_am_dev.csv'), index=False)
