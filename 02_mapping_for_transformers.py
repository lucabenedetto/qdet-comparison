import os

import numpy as np
import pandas as pd

from src.constants import DIFFICULTY, DATA_DIR, RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, AM, DF_COLS
CORRECT = 'correct'
DESCRIPTION = 'description'
QUESTION_ID = 'question_id'

# todo AM without the text of the answers


def main_tf():
    for dataset in [AM]:

        df_train = pd.read_csv(os.path.join(DATA_DIR, f'{dataset}_train.csv'))[DF_COLS]
        if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
            df_test = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_test.csv'))[DF_COLS]
            df_dev = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_dev.csv'))[DF_COLS]
        else:
            df_test = pd.read_csv(os.path.join(DATA_DIR, f'{dataset}_test.csv'))[DF_COLS]
            df_dev = pd.read_csv(os.path.join(DATA_DIR, f'{dataset}_dev.csv'))[DF_COLS]

        out_df_answers_text = pd.DataFrame(columns=[CORRECT, DESCRIPTION, 'id', QUESTION_ID])

        print(f"DATASET {dataset}")
        print("doing train")
        out_df_text_difficulty = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
        for correct_ans, _, option_0, option_1, option_2, option_3, question, context, _, q_id, _, difficulty in df_train.values:
            out_df_answers_text, out_df_text_difficulty = get_updated_out_df_answers_text_and_out_df_text_difficulty(
                out_df_answers_text,
                out_df_text_difficulty,
                q_id, question, context, difficulty,
                correct_ans, option_0, option_1, option_2, option_3,
            )
        out_df_text_difficulty.to_csv(f'data/processed_for_tf/text_difficulty_{dataset}_train.csv', index=False)

        print("doing test")
        out_df_text_difficulty = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
        for correct_ans, _, option_0, option_1, option_2, option_3, question, context, _, q_id, _, difficulty in df_test.values:
            out_df_answers_text, out_df_text_difficulty = get_updated_out_df_answers_text_and_out_df_text_difficulty(
                out_df_answers_text,
                out_df_text_difficulty,
                q_id, question, context, difficulty,
                correct_ans, option_0, option_1, option_2, option_3,
            )
        out_df_text_difficulty.to_csv(f'data/processed_for_tf/text_difficulty_{dataset}_test.csv', index=False)

        print("doing dev")
        out_df_text_difficulty = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
        for correct_ans, _, option_0, option_1, option_2, option_3, question, context, _, q_id, _, difficulty in df_dev.values:
            out_df_answers_text, out_df_text_difficulty = get_updated_out_df_answers_text_and_out_df_text_difficulty(
                out_df_answers_text,
                out_df_text_difficulty,
                q_id, question, context, difficulty,
                correct_ans, option_0, option_1, option_2, option_3,
            )
        out_df_text_difficulty.to_csv(f'data/processed_for_tf/text_difficulty_{dataset}_dev.csv', index=False)

        out_df_answers_text.to_csv(f'data/processed_for_tf/answers_texts_{dataset}.csv', index=False)


def get_new_rows_out_df_answers_text(
        correct_ans,
        q_id,
        option_0,
        option_1,
        option_2,
        option_3,
):
    out_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, 'id', QUESTION_ID])
    options = [option_0, option_1, option_2, option_3]
    for idx, option in enumerate(options):
        new_row_df_answers_text = pd.DataFrame([{
            CORRECT: idx == correct_ans,
            DESCRIPTION: option,
            'id': idx,
            QUESTION_ID: q_id,
        }])
        out_df = pd.concat(
            [
                out_df.astype({CORRECT: bool, DESCRIPTION: str, 'id': str, QUESTION_ID: str}),
                new_row_df_answers_text.astype({CORRECT: bool, DESCRIPTION: str, 'id': str, QUESTION_ID: str})
            ],
            ignore_index=True)
    return out_df


def get_new_row_df_text_difficulty(q_id, question, context, difficulty):
    if type(context) != str:  # i.e. if it is None
        context = ''
    else:
        context = context + ' '
    return pd.DataFrame([{
        DESCRIPTION: context + question,
        QUESTION_ID: q_id,
        DIFFICULTY: difficulty,
    }])


def get_updated_out_df_answers_text_and_out_df_text_difficulty(
        out_df_answers_text,
        out_df_text_difficulty,
        q_id, question, context, difficulty,
        correct_ans, option_0, option_1, option_2, option_3,
):
    new_rows_df = get_new_rows_out_df_answers_text(correct_ans, q_id, option_0, option_1, option_2, option_3)
    out_df_answers_text = pd.concat(
        [out_df_answers_text.astype({CORRECT: bool, DESCRIPTION: str, 'id': str, QUESTION_ID: str}),
         new_rows_df.astype({CORRECT: bool, DESCRIPTION: str, 'id': str, QUESTION_ID: str})], ignore_index=True)
    # get question text (and difficulty)
    new_row_df_text_difficulty = get_new_row_df_text_difficulty(q_id, question, context, difficulty)
    out_df_text_difficulty = pd.concat([out_df_text_difficulty, new_row_df_text_difficulty], ignore_index=True)
    return out_df_answers_text, out_df_text_difficulty


main_tf()
