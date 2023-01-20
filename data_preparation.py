import logging

from qdet_utils.constants import (
    DATA_DIR,
    RACE_PP,
    ARC,
    ARC_BALANCED,
    AM,
)
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
    race_pp_dm = RaceDatamanager()
    dataset = race_pp_dm.get_racepp_dataset(race_data_dir, race_c_data_dir, DATA_DIR)
    # whole RACE++
    race_pp_dm.convert_to_r2de_format_and_store_dataset(dataset, DATA_DIR, RACE_PP)
    race_pp_dm.convert_to_text2props_format_and_store_dataset(dataset, DATA_DIR, RACE_PP)
    race_pp_dm.convert_to_transformers_format_and_store_dataset(dataset, DATA_DIR, RACE_PP, skip_answers_texts=False)

    # sub-sampled datasets
    for training_size in [4_000, 8_000, 12_000]:
        sub_sampled_dataset = race_pp_dm.get_subsampled_racepp_dataset(DATA_DIR, training_size, DATA_DIR)
        dataset_name = f'{RACE_PP}_{training_size}'
        race_pp_dm.convert_to_r2de_format_and_store_dataset(sub_sampled_dataset, DATA_DIR, dataset_name)
        race_pp_dm.convert_to_text2props_format_and_store_dataset(sub_sampled_dataset, DATA_DIR, dataset_name)
        race_pp_dm.convert_to_transformers_format_and_store_dataset(sub_sampled_dataset, DATA_DIR, dataset_name, False)

    # ARC
    logger.info("Starting preparation ARC")
    arc_data_dir = 'data/raw/ARC-V1-Feb2018'
    arc_dm = ArcDataManager()
    dataset = arc_dm.get_arc_dataset(arc_data_dir, DATA_DIR)
    arc_dm.convert_to_r2de_format_and_store_dataset(dataset, DATA_DIR, ARC)
    arc_dm.convert_to_text2props_format_and_store_dataset(dataset, DATA_DIR, ARC)
    arc_dm.convert_to_transformers_format_and_store_dataset(dataset, DATA_DIR, ARC, skip_answers_texts=False)

    logger.info("Starting preparation ARC Balanced")
    balanced_dataset = arc_dm.get_arc_balanced_dataset(dataset, DATA_DIR)
    arc_dm.convert_to_r2de_format_and_store_dataset(balanced_dataset, DATA_DIR, ARC_BALANCED)
    arc_dm.convert_to_text2props_format_and_store_dataset(balanced_dataset, DATA_DIR, ARC_BALANCED)
    arc_dm.convert_to_transformers_format_and_store_dataset(balanced_dataset, DATA_DIR, ARC_BALANCED, skip_answers_texts=False)

    # ASSISTments
    logger.info("Starting preparation AM")
    am_data_dir = 'data/interim/assistments'
    am_dm = AmDataManager()
    dataset = am_dm.get_assistments_dataset(am_data_dir, DATA_DIR)
    am_dm.convert_to_r2de_format_and_store_dataset(dataset, DATA_DIR, AM)
    am_dm.convert_to_text2props_format_and_store_dataset(dataset, DATA_DIR, AM)
    am_dm.convert_to_transformers_format_and_store_dataset(dataset, DATA_DIR, AM, skip_answers_texts=True)


if __name__ == "__main__":
    main()
