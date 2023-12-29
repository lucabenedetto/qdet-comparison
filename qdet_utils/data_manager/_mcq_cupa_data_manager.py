from typing import Dict, Optional

import logging
import numpy as np
import os
import pandas as pd

from ._data_manager import DataManager

from qdet_utils.constants import (
    DF_COLS,
    CORRECT_ANSWER,
    OPTIONS,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    QUESTION,
    CONTEXT,
    CONTEXT_ID,
    Q_ID,
    SPLIT,
    DIFFICULTY,
    DEV,
    TEST,
    TRAIN,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McqCupaDatamanager(DataManager):

    def get_mcq_cupa_dataset(
            self,
            data_dir: str,
            output_data_dir: str,
            save_dataset: bool = True,
            train_size: Optional[float] = 0.6,
            test_size: Optional[float] = 0.25,
    ) -> Dict[str, pd.DataFrame]:
        # TODO: clean this
        df = self._get_mcq_cupa_dataset(data_dir=data_dir)
        logger.info(f"Size of df = {len(df)}.")
        tmp_series_context_ids = df.sort_values(CONTEXT_ID)[CONTEXT_ID].drop_duplicates().sample(frac=1)
        logger.info(f"Num of context IDs = {len(tmp_series_context_ids)}.")

        context_ids = dict()
        context_ids[TRAIN] = set(tmp_series_context_ids.iloc[:int(len(tmp_series_context_ids) * train_size)].values)
        context_ids[TEST] = set(tmp_series_context_ids.iloc[int(len(tmp_series_context_ids) * train_size):int(len(tmp_series_context_ids) * train_size)+int(len(tmp_series_context_ids) * test_size)].values)
        context_ids[DEV] = set(tmp_series_context_ids.iloc[int(len(tmp_series_context_ids) * train_size)+int(len(tmp_series_context_ids) * test_size):].values)

        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            logger.info(f"Num of {split} context IDs = {len(context_ids[split])}.")
            dataset[split] = df[df[CONTEXT_ID].isin(context_ids[split])].copy()
            logger.info(f"Len of {split} df = {len(dataset[split])}")
            if save_dataset:
                dataset[split].to_csv(os.path.join(output_data_dir, f'mcq_cupa_{split}.csv'), index=False)
        return dataset

    def _get_mcq_cupa_dataset(self, data_dir) -> pd.DataFrame:

        df = pd.read_csv(os.path.join(data_dir, 'mcq_data_cupa.csv'), encoding='ISO-8859-1')
        assert len(df) == df['Task_ID'].nunique()

        out_df = pd.DataFrame(columns=DF_COLS)
        dict_map_task_id_to_task_difficulty = dict()
        dict_map_task_id_to_product = dict()

        for idx, row in df.iterrows():
            context = row['Text'].encode('ascii', 'ignore').decode("ascii")  # to fix issue with encoding
            context_id = row['Task_ID']
            split = None

            dict_map_task_id_to_product[context_id] = row['Product']
            dict_map_task_id_to_task_difficulty[context_id] = row['task_v0_diff']

            for q_idx in range(1, 10):
                if row[f'Q{q_idx}'] is np.nan:
                    print(f"[INFO] skipping Task_ID {context_id} - Q{q_idx}, because it is null.")
                else:
                    q_id = f"{context_id}_Q{q_idx}"
                    question = row[f'Q{q_idx}'].encode('ascii', 'ignore').decode("ascii")  # to fix issue with encoding
                    option_0 = row[f'Q{q_idx}A'].encode('ascii', 'ignore').decode("ascii")  # to fix issue with encoding
                    option_1 = row[f'Q{q_idx}B'].encode('ascii', 'ignore').decode("ascii")  # to fix issue with encoding
                    option_2 = row[f'Q{q_idx}C'].encode('ascii', 'ignore').decode("ascii")  # to fix issue with encoding
                    option_3 = row[f'Q{q_idx}D'].encode('ascii', 'ignore').decode("ascii")  # to fix issue with encoding
                    options = [option_0, option_1, option_2, option_3]
                    correct_answer = ord(row[f'Q{q_idx}_answer']) - ord('A')
                    difficulty = row[f'Q{q_idx}_diff']
                    new_row_df = pd.DataFrame([{
                        CORRECT_ANSWER: correct_answer,
                        OPTIONS: options,
                        OPTION_0: option_0,
                        OPTION_1: option_1,
                        OPTION_2: option_2,
                        OPTION_3: option_3,
                        QUESTION: question,
                        CONTEXT: context,
                        CONTEXT_ID: context_id,
                        Q_ID: q_id,
                        SPLIT: split,
                        DIFFICULTY: difficulty,
                    }])
                    out_df = pd.concat([out_df, new_row_df], ignore_index=True)
        return out_df
