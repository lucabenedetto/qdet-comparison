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
    HIGH = 'high'
    MIDDLE = 'middle'
    COLLEGE = 'college'

    LEVEL_TO_DIFFICULTY_MAP = {MIDDLE: 0, HIGH: 1, COLLEGE: 2}

    def get_racepp_dataset(self, race_data_dir, race_c_data_dir, output_data_dir, save_dataset=True):
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df_race = self.get_raw_race_df(data_dir=race_data_dir, split=split)
            df_race_c = self.get_raw_race_c_df(data_dir=race_c_data_dir, split=split)
            df = pd.concat([df_race, df_race_c])
            assert set(df.columns) == set(DF_COLS)
            if save_dataset:
                df.to_csv(os.path.join(output_data_dir, f'race_pp_{split}.csv'), index=False)
            dataset[split] = df.copy()
        return dataset

    def get_subsampled_racepp_dataset(self, dataset, training_size, output_data_dir, random_state=None):
        # TODO:
        #  this is the sampling per class (in the original implementation, it is balanced!!), I should also
        #  implement the sampling "unbalanced"
        df_train = dataset[TRAIN].copy()
        subsampled_dataset = dict()
        df_train = pd.concat([df_train[df_train[DIFFICULTY] == 0].sample(training_size, random_state=random_state),
                              df_train[df_train[DIFFICULTY] == 1].sample(training_size, random_state=random_state),
                              df_train[df_train[DIFFICULTY] == 2].sample(training_size, random_state=random_state)])
        df_train.to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{TRAIN}.csv'), index=False)
        dataset[DEV].to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{DEV}.csv'), index=False)
        dataset[TEST].to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{TEST}.csv'), index=False)
        subsampled_dataset[TRAIN] = df_train.copy()
        subsampled_dataset[DEV] = dataset[DEV].copy()
        subsampled_dataset[TEST] = dataset[TEST].copy()
        return subsampled_dataset

    def prepare_race_dataset(self, race_data_dir, output_data_dir):
        for split in [TRAIN, DEV, TEST]:
            df_race = self.get_raw_race_df(data_dir=race_data_dir, split=split)
            assert set(df_race.columns) == set(DF_COLS)
            df_race.to_csv(os.path.join(output_data_dir, f'race_{split}.csv'), index=False)

    def prepare_race_c_dataset(self, race_c_data_dir, output_data_dir):
        for split in [TRAIN, DEV, TEST]:
            df_race_c = self.get_raw_race_c_df(data_dir=race_c_data_dir, split=split)
            assert set(df_race_c.columns) == set(DF_COLS)
            df_race_c.to_csv(os.path.join(output_data_dir, f'race_c_{split}.csv'), index=False)

    def _append_new_reading_passage_to_df(self, df, reading_passage_data, split, level):
        answers = reading_passage_data['answers']
        options = reading_passage_data['options']
        questions = reading_passage_data['questions']
        article = reading_passage_data['article']
        context_id = reading_passage_data['id']

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
                                               DIFFICULTY: self.LEVEL_TO_DIFFICULTY_MAP[level]}])])
        return df

    def get_raw_race_df(self, data_dir: str, split: str):
        df = pd.DataFrame(columns=DF_COLS)
        for level in [self.HIGH, self.MIDDLE]:
            for filename in listdir(os.path.join(data_dir, split, level)):
                with open(os.path.join(data_dir, split, level, filename), 'r') as f:
                    reading_passage_data = json.load(f)
                df = self._append_new_reading_passage_to_df(df, reading_passage_data, split, level)
        return df

    def get_raw_race_c_df(self, data_dir: str, split: str):
        df = pd.DataFrame(columns=DF_COLS)
        for filename in listdir(os.path.join(data_dir, split)):
            with open(os.path.join(data_dir, split, filename), 'r') as f:
                reading_passage_data = json.load(f)
            df = self._append_new_reading_passage_to_df(df, reading_passage_data, split, level=self.COLLEGE)
        return df
