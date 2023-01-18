import logging
import os
import pandas as pd

from qdet_utils.constants import DATA_DIR
from qdet_utils.data_manager import (
    RaceDatamanager,
    ArcDataManager,
    AmDataManager,
)

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # RACE++
    logger.info("Starting preparation RACE++")
    race_data_dir = 'data/raw/RACE'
    race_c_data_dir = 'data/raw/race-c-master/data'
    racepp_dm = RaceDatamanager()
    dataset = racepp_dm.get_racepp_dataset(race_data_dir, race_c_data_dir, DATA_DIR)
    # whole RACE++
    racepp_dm.convert_to_r2de_format_and_store_dataset(dataset, DATA_DIR, 'race_pp')
    racepp_dm.convert_to_text2props_format_and_store_dataset(dataset, DATA_DIR, 'race_pp')
    racepp_dm.convert_to_transformers_format_and_store_dataset(dataset, DATA_DIR, 'race_pp', skip_answers_texts=False)
    # sub-sampled datasets
    for training_size in [4_000, 8_000, 12_000]:
        subsampled_dataset = racepp_dm.get_subsampled_racepp_dataset(dataset, training_size, DATA_DIR)
        racepp_dm.convert_to_r2de_format_and_store_dataset(subsampled_dataset, DATA_DIR, f'race_pp_{training_size}')
        racepp_dm.convert_to_text2props_format_and_store_dataset(subsampled_dataset, DATA_DIR, f'race_pp_{training_size}')
        racepp_dm.convert_to_transformers_format_and_store_dataset(subsampled_dataset, DATA_DIR, f'race_pp_{training_size}', skip_answers_texts=False)

    # ARC
    logger.info("Starting preparation ARC")
    arc_data_dir = 'data/raw/ARC-V1-Feb2018'
    arc_dm = ArcDataManager()
    dataset = arc_dm.get_arc_dataset(arc_data_dir, DATA_DIR)
    arc_dm.convert_to_r2de_format_and_store_dataset(dataset, DATA_DIR, 'arc')
    arc_dm.convert_to_text2props_format_and_store_dataset(dataset, DATA_DIR, 'arc')
    arc_dm.convert_to_transformers_format_and_store_dataset(dataset, DATA_DIR, 'arc', skip_answers_texts=False)

    # TODO
    # logger.info("Starting preparation ARC Balanced")
    # balanced_df_train = pd.DataFrame(columns=df_train.columns)
    # for diff in range(3, 10):
    #     if diff in {5, 8}:
    #         balanced_df_train = pd.concat([balanced_df_train, df_train[df_train['difficulty'] == diff].sample(n=500)], ignore_index=True)
    #     else:
    #         balanced_df_train = pd.concat([balanced_df_train, df_train[df_train['difficulty'] == diff]], ignore_index=True)
    # balanced_df_train = balanced_df_train.sample(frac=1.0)
    # balanced_df_train.to_csv(os.path.join('data/processed', f'arc_balanced_train.csv'), index=False)
    # df_dev.to_csv(os.path.join('data/processed', f'arc_balanced_dev.csv'), index=False)
    # df_test.to_csv(os.path.join('data/processed', f'arc_balanced_test.csv'), index=False)
    # convert_to_r2de_format_and_store_dataset(balanced_df_train, df_dev, df_test, DATA_DIR, 'arc_balanced')
    # convert_to_text2props_format_and_store_dataset(balanced_df_train, df_dev, df_test, DATA_DIR, 'arc_balanced')

    # ASSISTments
    logger.info("Starting preparation AM")
    am_data_dir = 'data/interim/assistments'
    am_dm = AmDataManager()
    dataset = am_dm.get_assistments_dataset(am_data_dir, DATA_DIR)
    am_dm.convert_to_r2de_format_and_store_dataset(dataset, DATA_DIR, 'am')
    am_dm.convert_to_text2props_format_and_store_dataset(dataset, DATA_DIR, 'am')
    am_dm.convert_to_transformers_format_and_store_dataset(dataset, DATA_DIR, 'am', skip_answers_texts=True)


if __name__ == "__main__":
    main()
