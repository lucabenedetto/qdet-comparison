from typing import Tuple

import pandas as pd
import os
from collections import defaultdict

from src.constants import (
    DF_COLS,
    CORRECT_ANSWER,
    OPTIONS,
    QUESTION,
    CONTEXT,
    CONTEXT_ID,
    Q_ID,
    CORRECT_ANSWERS_LIST,
)


def prepare_ca_dataset(ca_data_dir: str, output_data_dir: str):
    df_ca = prepare_and_return_ca_df(ca_data_dir)
    assert set(df_ca.columns) == set(DF_COLS)
    df_ca.to_csv(os.path.join(output_data_dir, f'ca.csv'), index=False)  # TODO split in train test dev ans store info


def prepare_and_return_ca_df(data_dir: str) -> pd.DataFrame:
    answers_df = pd.read_csv(os.path.join(data_dir, 'answers_texts.csv'))
    qid_to_choices_text_dict = defaultdict(list)
    correct_choices = defaultdict(list)
    previous_qid, idx = None, 0
    for correct, description, ans_id, question_id in answers_df.values:
        if previous_qid is None or previous_qid != question_id:
            idx = 0
        previous_qid = question_id
        qid_to_choices_text_dict[question_id].append(description)
        if correct:
            correct_choices[question_id].append(idx)
        idx += 1
        # with this approach I consider 0 as the first element in the list !

    questions_df = pd.read_csv(os.path.join(data_dir, 'questions_texts.csv'))
    questions_df = questions_df.rename(columns={'description': QUESTION, 'id': Q_ID})

    # kept just for consistency with the race dataset
    questions_df[CONTEXT] = ''
    questions_df[CONTEXT_ID] = ''

    questions_df[CORRECT_ANSWERS_LIST] = questions_df.apply(lambda r: tuple(correct_choices[r[Q_ID]]), axis=1)
    questions_df[CORRECT_ANSWER] = questions_df.apply(
        lambda r: r[CORRECT_ANSWERS_LIST][0] if len(r[CORRECT_ANSWERS_LIST]) == 1 else None,
        axis=1
    )
    questions_df[OPTIONS] = questions_df.apply(lambda r: qid_to_choices_text_dict[r[Q_ID]], axis=1)

    #     SPLIT,  # TODO
    #     DIFFICULTY,  # TODO

    return questions_df


def calibrate_questions(
        interactions_filepath: str,
        train_dev_test_split: Tuple[float, float, float],
        question_count_threshold: int = 0,
):
    # read df

    # remove duplicates
    # keep only completed sessions

    # calibrate items with IRT
    # split them
    # save calibrated items
    pass