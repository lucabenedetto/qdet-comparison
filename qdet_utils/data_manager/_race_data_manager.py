from typing import Dict
import pandas as pd
import json
import os
from os import listdir

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

from ._data_manager import DataManager


class RaceDatamanager(DataManager):
    ANSWERS = 'answers'
    OPTIONS = 'options'
    QUESTIONS = 'questions'
    ARTICLE = 'article'
    ID = 'id'

    HIGH = 'high'
    MIDDLE = 'middle'
    COLLEGE = 'college'

    LEVEL_TO_INT_DIFFICULTY_MAP = {MIDDLE: 0, HIGH: 1, COLLEGE: 2}

    def get_racepp_dataset(
            self,
            race_data_dir: str,
            race_c_data_dir: str,
            output_data_dir: str,
            save_dataset: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df_race = self.get_raw_race_df(data_dir=race_data_dir, split=split)
            df_race_c = self.get_raw_race_c_df(data_dir=race_c_data_dir, split=split)
            df = pd.concat([df_race, df_race_c])
            if save_dataset:
                df.to_csv(os.path.join(output_data_dir, f'race_pp_{split}.csv'), index=False)
            dataset[split] = df.copy()
        return dataset

    def get_subsampled_racepp_dataset(
            self,
            dataset: Dict[str, pd.DataFrame],
            training_size: int,
            output_data_dir: str,
            random_state: int = None,
            balanced_sampling: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        subsampled_dataset = dict()
        if balanced_sampling:
            df_train = dataset[TRAIN].copy()
            df_train = pd.concat([df_train[df_train[DIFFICULTY] == 0].sample(training_size, random_state=random_state),
                                  df_train[df_train[DIFFICULTY] == 1].sample(training_size, random_state=random_state),
                                  df_train[df_train[DIFFICULTY] == 2].sample(training_size, random_state=random_state)])
        else:
            df_train = dataset[TRAIN].sample(training_size, random_state=random_state)
        df_train.to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{TRAIN}.csv'), index=False)
        dataset[DEV].to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{DEV}.csv'), index=False)
        dataset[TEST].to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{TEST}.csv'), index=False)
        subsampled_dataset[TRAIN] = df_train.copy()
        subsampled_dataset[DEV] = dataset[DEV].copy()
        subsampled_dataset[TEST] = dataset[TEST].copy()
        return subsampled_dataset

    def get_race_dataset(self, data_dir: str, out_data_dir: str, save_dataset: bool = True) -> Dict[str, pd.DataFrame]:
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df = self.get_raw_race_df(data_dir=data_dir, split=split)
            if save_dataset:
                df.to_csv(os.path.join(out_data_dir, f'race_{split}.csv'), index=False)
            dataset[split] = df.copy()
        return dataset

    def get_race_c_dataset(self, data_dir: str, out_data_dir: str, save_dataset: bool = True) -> Dict[str, pd.DataFrame]:
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df = self.get_raw_race_c_df(data_dir=data_dir, split=split)
            if save_dataset:
                df.to_csv(os.path.join(out_data_dir, f'race_c_{split}.csv'), index=False)
            dataset[split] = df.copy()
        return dataset

    def _append_new_reading_passage_to_df(
            self,
            df: pd.DataFrame,
            reading_passage_data,
            split: str,
            level: str,
    ) -> pd.DataFrame:
        answers = reading_passage_data[self.ANSWERS]
        options = reading_passage_data[self.OPTIONS]
        questions = reading_passage_data[self.QUESTIONS]
        article = reading_passage_data[self.ARTICLE]
        context_id = reading_passage_data[self.ID]

        for idx in range(len(questions)):
            # this is just to check that there are no anomalies
            assert ord('A') <= ord(answers[idx]) <= ord('Z')
            df = pd.concat([df, pd.DataFrame([{CORRECT_ANSWER: ord(answers[idx])-ord('A'),
                                               OPTIONS: options[idx],
                                               OPTION_0: options[idx][0],
                                               OPTION_1: options[idx][1],
                                               OPTION_2: options[idx][2],
                                               OPTION_3: options[idx][3],
                                               QUESTION: questions[idx],
                                               CONTEXT: article,
                                               CONTEXT_ID: context_id[:-4],
                                               Q_ID: '%s_q%d' % (context_id[:-4], idx),
                                               SPLIT: split,
                                               DIFFICULTY: self.LEVEL_TO_INT_DIFFICULTY_MAP[level]}])])
        return df

    def get_raw_race_df(self, data_dir: str, split: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=DF_COLS)
        for level in [self.HIGH, self.MIDDLE]:
            for filename in listdir(os.path.join(data_dir, split, level)):
                with open(os.path.join(data_dir, split, level, filename), 'r') as f:
                    reading_passage_data = json.load(f)
                df = self._append_new_reading_passage_to_df(df, reading_passage_data, split, level)
        assert set(df.columns) == set(DF_COLS)
        return df

    def get_raw_race_c_df(self, data_dir: str, split: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=DF_COLS)
        for filename in listdir(os.path.join(data_dir, split)):
            with open(os.path.join(data_dir, split, filename), 'r') as f:
                reading_passage_data = json.load(f)
            df = self._append_new_reading_passage_to_df(df, reading_passage_data, split, level=self.COLLEGE)
        assert set(df.columns) == set(DF_COLS)
        return df
