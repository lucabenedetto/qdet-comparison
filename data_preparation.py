import logging
import os
import pandas as pd

from qdet_utils.constants import DATA_DIR
from qdet_utils.data_utils.race import prepare_racepp_dataset, prepare_subsampled_racepp_dataset
from qdet_utils.data_utils.arc import prepare_arc_dataset
from qdet_utils.data_utils.am import prepare_assistments_dataset
from qdet_utils.data_utils.mapping_text2props import convert_to_text2props_format_and_store_data
from qdet_utils.data_utils.mapping_r2de import convert_to_r2de_format_and_store_data


# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # RACE++
    logger.info("Starting preparation RACE++")
    race_data_dir = 'data/raw/RACE'
    race_c_data_dir = 'data/raw/race-c-master/data'
    # whole RACE++
    df_train_rpp, df_dev_rpp, df_test_rpp = prepare_racepp_dataset(race_data_dir, race_c_data_dir, DATA_DIR)
    convert_to_r2de_format_and_store_data(df_train_rpp, df_dev_rpp, df_test_rpp, DATA_DIR, 'race_pp')
    convert_to_text2props_format_and_store_data(df_train_rpp, df_dev_rpp, df_test_rpp, DATA_DIR, 'race_pp')
    # sub-sampled datasets
    for training_size in [4_000, 8_000, 12_000]:
        df_train, df_dev, df_test = prepare_subsampled_racepp_dataset(df_train_rpp, df_dev_rpp, df_test_rpp, training_size, DATA_DIR)
        convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, f'race_pp_{training_size}')
        convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, f'race_pp_{training_size}')

    # ARC
    logger.info("Starting preparation ARC")
    arc_data_dir = 'data/raw/ARC-V1-Feb2018'
    df_train, df_dev, df_test = prepare_arc_dataset(arc_data_dir, 'data/processed')
    convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'arc')
    convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'arc')

    logger.info("Starting preparation ARC Balanced")
    balanced_df_train = pd.DataFrame(columns=df_train.columns)
    for diff in range(3, 10):
        if diff in {5, 8}:
            balanced_df_train = pd.concat([balanced_df_train, df_train[df_train['difficulty'] == diff].sample(n=500)], ignore_index=True)
        else:
            balanced_df_train = pd.concat([balanced_df_train, df_train[df_train['difficulty'] == diff]], ignore_index=True)
    balanced_df_train = balanced_df_train.sample(frac=1.0)
    balanced_df_train.to_csv(os.path.join('data/processed', f'arc_balanced_train.csv'), index=False)
    df_dev.to_csv(os.path.join('data/processed', f'arc_balanced_dev.csv'), index=False)
    df_test.to_csv(os.path.join('data/processed', f'arc_balanced_test.csv'), index=False)
    convert_to_r2de_format_and_store_data(balanced_df_train, df_dev, df_test, DATA_DIR, 'arc_balanced')
    convert_to_text2props_format_and_store_data(balanced_df_train, df_dev, df_test, DATA_DIR, 'arc_balanced')

    # ASSISTments
    logger.info("Starting preparation AM")
    am_data_dir = 'data/interim/assistments'
    df_train, df_dev, df_test = prepare_assistments_dataset(am_data_dir, 'data/processed')
    convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'am')
    convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'am')


if __name__ == "__main__":
    main()
