from typing import Dict
import logging
import pandas as pd
import os

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcDataManager(DataManager):
    ARC_DEV = 'Dev'
    ARC_TEST = 'Test'
    ARC_TRAIN = 'Train'
    MAP_TO_PROCESSED_SPLIT_NAMES = {ARC_DEV: DEV, ARC_TEST: TEST, ARC_TRAIN: TRAIN}

    ARC_IS_MULTIPLE_CHOICE_QUESTION = 'isMultipleChoiceQuestion'
    ARC_INCLUDES_DIAGRAM = 'includesDiagram'
    ARC_QUESTION_ID = 'questionID'
    ARC_ANSWER_KEY = 'AnswerKey'
    ARC_SCHOOL_GRADE = 'schoolGrade'
    ARC_QUESTION = 'question'
    ARC_CATEGORY = 'category'

    def get_arc_dataset(self, data_dir: str, out_data_dir: str) -> Dict[str, pd.DataFrame]:
        dataset = dict()
        for split in [self.ARC_TRAIN, self.ARC_DEV, self.ARC_TEST]:
            df = self.get_arc_df_by_split(data_dir, split)
            df.to_csv(os.path.join(out_data_dir, f'arc_{self.MAP_TO_PROCESSED_SPLIT_NAMES[split]}.csv'), index=False)
            dataset[self.MAP_TO_PROCESSED_SPLIT_NAMES[split]] = df.copy()
        return dataset

    def get_arc_df_by_split(self, data_dir: str, split: str) -> pd.DataFrame:
        df_easy = pd.read_csv(os.path.join(data_dir, 'ARC-Easy', f'ARC-Easy-{split}.csv'))
        df_easy = self._get_arc_df(df_easy)
        df_challenge = pd.read_csv(os.path.join(data_dir, 'ARC-Challenge', f'ARC-Challenge-{split}.csv'))
        df_challenge = self._get_arc_df(df_challenge)
        df = pd.concat([df_easy, df_challenge])
        return df

    def _get_arc_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df[self.ARC_IS_MULTIPLE_CHOICE_QUESTION] == 1]  # to keep only the MCQs
        df = df[df[self.ARC_INCLUDES_DIAGRAM] == 0]  # to keep only the questions without diagram
        df = df[[self.ARC_QUESTION_ID, self.ARC_ANSWER_KEY, self.ARC_SCHOOL_GRADE, self.ARC_QUESTION, self.ARC_CATEGORY]]
        df = df.rename(
            columns={
                self.ARC_QUESTION_ID: Q_ID,  # analyze if there is any difference w.r.t. using originalQuestionID
                self.ARC_ANSWER_KEY: CORRECT_ANSWER,
                self.ARC_SCHOOL_GRADE: DIFFICULTY,
                # self.ARC_QUESTION: QUESTION,  # it already has the right name
                self.ARC_CATEGORY: SPLIT,
            }
        )
        df[CORRECT_ANSWER] = df.apply(
            lambda r:
            int(r[CORRECT_ANSWER])-1 if ord('1') <= ord(r[CORRECT_ANSWER]) <= ord('9') else ord(r[CORRECT_ANSWER])-ord('A'),
            axis=1
        )
        # this is to have integer indexes and 0 as first index of the correct choices (in the original dataset it is 1)
        df[SPLIT] = df.apply(lambda r: self.MAP_TO_PROCESSED_SPLIT_NAMES[r[SPLIT]], axis=1)

        df_num = df[(df[QUESTION].str.contains('\(1\) '))
                    & (df[QUESTION].str.contains('\(2\) '))
                    & (df[QUESTION].str.contains('\(3\) '))
                    & (df[QUESTION].str.contains('\(4\) '))
                    & ~(df[QUESTION].str.contains('\(5\) '))].copy()
        df_num[OPTION_0] = df_num.apply(lambda r: r[QUESTION].split('(1) ')[1].split('(2) ')[0], axis=1)
        df_num[OPTION_1] = df_num.apply(lambda r: r[QUESTION].split('(2) ')[1].split('(3) ')[0], axis=1)
        df_num[OPTION_2] = df_num.apply(lambda r: r[QUESTION].split('(3) ')[1].split('(4) ')[0], axis=1)
        df_num[OPTION_3] = df_num.apply(lambda r: r[QUESTION].split('(4) ')[1], axis=1)
        df_num[OPTIONS] = df_num.apply(lambda r: [r[OPTION_0], r[OPTION_1], r[OPTION_2], r[OPTION_3]], axis=1)
        df_num[QUESTION] = df_num.apply(lambda r: r[QUESTION].split(' (1) ')[0], axis=1)

        df_char = df[(df[QUESTION].str.contains('\(A\) '))
                     & (df[QUESTION].str.contains('\(B\) '))
                     & (df[QUESTION].str.contains('\(C\) '))
                     & (df[QUESTION].str.contains('\(D\) '))
                     & ~(df[QUESTION].str.contains('\(E\) '))].copy()
        df_char[OPTION_0] = df_char.apply(lambda r: r[QUESTION].split('(A) ')[1].split('(B) ')[0], axis=1)
        df_char[OPTION_1] = df_char.apply(lambda r: r[QUESTION].split('(B) ')[1].split('(C) ')[0], axis=1)
        df_char[OPTION_2] = df_char.apply(lambda r: r[QUESTION].split('(C) ')[1].split('(D) ')[0], axis=1)
        df_char[OPTION_3] = df_char.apply(lambda r: r[QUESTION].split('(D) ')[1], axis=1)
        df_char[OPTIONS] = df_char.apply(lambda r: [r[OPTION_0], r[OPTION_1], r[OPTION_2], r[OPTION_3]], axis=1)
        df_char[QUESTION] = df_char.apply(lambda r: r[QUESTION].split(' (A) ')[0], axis=1)

        df = pd.concat([df_num, df_char])

        # kept just for consistency with the race dataset
        df[CONTEXT] = ''
        df[CONTEXT_ID] = ''

        assert set(df.columns) == set(DF_COLS)

        return df

    def get_arc_balanced_dataset(self, dataset: Dict[str, pd.DataFrame], out_data_dir: str):
        logger.info("Starting preparation ARC Balanced")
        df_train = dataset[TRAIN]
        balanced_df_train = pd.DataFrame(columns=df_train.columns)
        # TODO add some params instead of hard-coding the number of samples to keep
        for diff in range(3, 10):
            if diff in {5, 8}:
                balanced_df_train = pd.concat(
                    [balanced_df_train, df_train[df_train[DIFFICULTY] == diff].sample(n=500)], ignore_index=True)
            else:
                balanced_df_train = pd.concat([balanced_df_train, df_train[df_train[DIFFICULTY] == diff]], ignore_index=True)
        balanced_df_train = balanced_df_train.sample(frac=1.0)

        balanced_dataset = {TRAIN: balanced_df_train.copy(), DEV: dataset[DEV].copy(), TEST: dataset[TEST].copy()}
        for split in [TRAIN, DEV, TEST]:
            balanced_dataset[split].to_csv(os.path.join(out_data_dir, f'arc_balanced_{split}.csv'), index=False)
        return balanced_dataset
