import os

import pandas as pd

from src.constants import (
    DIFFICULTY,
    DATA_DIR,
    RACE_PP,
    RACE_PP_4K,
    RACE_PP_8K,
    RACE_PP_12K,
    ARC,
    ARC_BALANCED,
    AM,
    DF_COLS,
    CORRECT,
    DESCRIPTION,
    QUESTION_ID,
    ANS_ID,
    QUESTION, CONTEXT, Q_ID,
)
AS_TYPE_DICT = {CORRECT: bool, DESCRIPTION: str, ANS_ID: str, QUESTION_ID: str}


def main_tf():
    for dataset_name in [RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED, AM]:

        df_train = pd.read_csv(os.path.join(DATA_DIR, f'{dataset_name}_train.csv'))[DF_COLS]
        if dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
            df_test = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_test.csv'))[DF_COLS]
            df_dev = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_dev.csv'))[DF_COLS]
        else:
            df_test = pd.read_csv(os.path.join(DATA_DIR, f'{dataset_name}_test.csv'))[DF_COLS]
            df_dev = pd.read_csv(os.path.join(DATA_DIR, f'{dataset_name}_dev.csv'))[DF_COLS]

        print(f"Doing dataset {dataset_name}")
        skip_answer_texts = dataset_name in {AM}

        answer_texts_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, ANS_ID, QUESTION_ID])

        print("Doing train...")
        text_difficulty_df, answer_texts_df = get_text_difficulty_and_answer_texts(df_train, answer_texts_df, skip_answer_texts)
        text_difficulty_df.to_csv(os.path.join(DATA_DIR, 'for_tf', f'text_difficulty_{dataset_name}_train.csv'), index=False)

        print("Doing test...")
        text_difficulty_df, answer_texts_df = get_text_difficulty_and_answer_texts(df_test, answer_texts_df, skip_answer_texts)
        text_difficulty_df.to_csv(os.path.join(DATA_DIR, 'for_tf', f'text_difficulty_{dataset_name}_test.csv'), index=False)

        print("Doing dev...")
        text_difficulty_df, answer_texts_df = get_text_difficulty_and_answer_texts(df_dev, answer_texts_df, skip_answer_texts)
        text_difficulty_df.to_csv(os.path.join(DATA_DIR, 'for_tf', f'text_difficulty_{dataset_name}_dev.csv'), index=False)

        if not skip_answer_texts:
            answer_texts_df.to_csv(os.path.join(DATA_DIR, 'for_tf', f'answers_texts_{dataset_name}.csv'), index=False)


def get_text_difficulty_and_answer_texts(df, answers_text_df, skip_ans_texts):
    text_difficulty_df = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
    if skip_ans_texts:
        for question, context, q_id, difficulty in df[[QUESTION, CONTEXT, Q_ID, DIFFICULTY]].values:
            text_difficulty_df = pd.concat(
                [text_difficulty_df, get_new_row_text_difficulty_df(q_id, question, context, difficulty)],
                ignore_index=True
            )
    else:
        for correct_option, _, opt0, opt1, opt2, opt3, question, context, _, q_id, _, difficulty in df.values:
            answers_text_df = pd.concat(
                [
                    answers_text_df.astype(AS_TYPE_DICT),
                    get_new_rows_answers_text_df(correct_option, q_id, [opt0, opt1, opt2, opt3]).astype(AS_TYPE_DICT)
                ],
                ignore_index=True
            )
            text_difficulty_df = pd.concat(
                [text_difficulty_df, get_new_row_text_difficulty_df(q_id, question, context, difficulty)],
                ignore_index=True
            )
    return text_difficulty_df, answers_text_df


def get_new_rows_answers_text_df(correct_ans, q_id, options):
    out_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, ANS_ID, QUESTION_ID])
    for idx, option in enumerate(options):
        new_row = pd.DataFrame([{CORRECT: idx == correct_ans, DESCRIPTION: option, ANS_ID: idx, QUESTION_ID: q_id}])
        out_df = pd.concat([out_df.astype(AS_TYPE_DICT), new_row.astype(AS_TYPE_DICT)], ignore_index=True)
    return out_df


def get_new_row_text_difficulty_df(q_id, question, context, difficulty):
    context = '' if type(context) != str else context + ' '
    return pd.DataFrame([{DESCRIPTION: context + question, QUESTION_ID: q_id, DIFFICULTY: difficulty,}])


main_tf()
