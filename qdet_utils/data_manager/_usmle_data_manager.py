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
    ITEM_TYPE,
    EXAM_TYPE,
    RESPONSE_TIME,
    DEV,
    TEST,
    TRAIN,
)
from qdet_utils.constants_usmle import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UsmleDataManager(DataManager):

    def get_usmle_dataset(
            self,
            data_dir: str,
            output_data_dir: str,
            save_dataset: bool = True,
            train_size: Optional[float] = 0.6,
            test_size: Optional[float] = 0.25,
    ) -> Dict[str, pd.DataFrame]:
        # TODO: clean this
        df = self._get_usmle_dataset(data_dir=data_dir)
        logger.info(f"Size of df = {len(df)}.")
        # tmp_series_context_ids = df.sort_values(CONTEXT_ID)[CONTEXT_ID].drop_duplicates().sample(frac=1)
        # logger.info(f"Num of context IDs = {len(tmp_series_context_ids)}.")

        # context_ids = dict()
        # context_ids[TRAIN] = set(tmp_series_context_ids.iloc[:int(len(tmp_series_context_ids) * train_size)].values)
        # context_ids[TEST] = set(tmp_series_context_ids.iloc[int(len(tmp_series_context_ids) * train_size):int(len(tmp_series_context_ids) * train_size)+int(len(tmp_series_context_ids) * test_size)].values)
        # context_ids[DEV] = set(tmp_series_context_ids.iloc[int(len(tmp_series_context_ids) * train_size)+int(len(tmp_series_context_ids) * test_size):].values)

        # TODO: split the dataset
        dataset = dict()
        dataset[TRAIN] = df.copy()
        if save_dataset:
            dataset[TRAIN].to_csv(os.path.join(output_data_dir, f'usmle_{TRAIN}.csv'), index=False)
        # for split in [TRAIN, DEV, TEST]:
        #     logger.info(f"Num of {split} context IDs = {len(context_ids[split])}.")
        #     dataset[split] = df[df[CONTEXT_ID].isin(context_ids[split])].copy()
        #     logger.info(f"Len of {split} df = {len(dataset[split])}")
        #     if save_dataset:
        #         dataset[split].to_csv(os.path.join(output_data_dir, f'mcq_cupa_{split}.csv'), index=False)
        return dataset

    def _get_usmle_dataset(self, data_dir) -> pd.DataFrame:

        df = pd.read_csv(os.path.join(data_dir, 'train_final.csv'), encoding='utf-8')

        out_df = pd.DataFrame(columns=DF_COLS)

        for idx, row in df.iterrows():
            split = TRAIN
            # option_0 = row[USMLE_ANSWER__A]
            options = [row[f'Answer__{opt_idx}']
                       for opt_idx in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                       if row[f'Answer__{opt_idx}'] not in {None, ''}]
            # options = [option_0, option_1, option_2, option_3, ...]
            new_row_df = pd.DataFrame([{
                CORRECT_ANSWER: ord(row[USMLE_ANSWER_KEY]) - ord('A'),
                OPTIONS: options,
                OPTION_0: None,  # the point is that they can have more than 4 options, so I'd have to change the cols
                OPTION_1: None,
                OPTION_2: None,
                OPTION_3: None,
                QUESTION: row[USMLE_ITEMSTEM_TEXT],
                CONTEXT: '',
                CONTEXT_ID: '',
                Q_ID: row[USMLE_ITEM_NUM],
                SPLIT: split,
                DIFFICULTY: row[USMLE_DIFFICULTY],
                ITEM_TYPE: row[USMLE_ITEM_TYPE],  # If pic or text
                EXAM_TYPE: row[USMLE_EXAM],  # which of the three steps
                RESPONSE_TIME: row[USMLE_RESPONSE_TIME],
            }])
            out_df = pd.concat([out_df, new_row_df], ignore_index=True)
        return out_df
