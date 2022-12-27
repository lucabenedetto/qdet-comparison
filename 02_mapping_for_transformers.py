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


def main_tf():
    for dataset_name in [RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED, AM]:

        df_train = pd.read_csv(os.path.join(DATA_DIR, f'{dataset_name}_train.csv'))[DF_COLS]
        if dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
            df_test = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_test.csv'))[DF_COLS]
            df_dev = pd.read_csv(os.path.join(DATA_DIR, 'race_pp_dev.csv'))[DF_COLS]
        else:
            df_test = pd.read_csv(os.path.join(DATA_DIR, f'{dataset_name}_test.csv'))[DF_COLS]
            df_dev = pd.read_csv(os.path.join(DATA_DIR, f'{dataset_name}_dev.csv'))[DF_COLS]

        answers_text_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, ANS_ID, QUESTION_ID])

        print(f"DATASET {dataset_name}")
        print("Doing train...")
        answers_text_df = store_text_difficulty_df_and_return_updated_answers_text_df(dataset_name, df_train, 'train', answers_text_df)

        print("Doing test...")
        answers_text_df = store_text_difficulty_df_and_return_updated_answers_text_df(dataset_name, df_test, 'test', answers_text_df)

        print("Doing dev...")
        answers_text_df = store_text_difficulty_df_and_return_updated_answers_text_df(dataset_name, df_dev, 'dev', answers_text_df)

        if dataset_name not in {AM}:
            answers_text_df.to_csv(f'data/processed_for_tf/answers_texts_{dataset_name}.csv', index=False)


def store_text_difficulty_df_and_return_updated_answers_text_df(dataset, df, split, answers_text_df):
    text_difficulty_df = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
    if dataset in {AM}:
        for question, context, q_id, difficulty in df[[QUESTION, CONTEXT, Q_ID, DIFFICULTY]].values:
            text_difficulty_df = get_updated_text_difficulty_df(text_difficulty_df, q_id, question, context, difficulty)
    else:
        for correct_ans, _, opt_0, opt_1, opt_2, opt_3, question, context, _, q_id, _, difficulty in df.values:
            answers_text_df, text_difficulty_df = get_updated_answers_text_df_and_text_difficulty_df(
                answers_text_df, text_difficulty_df,
                q_id, question, context, difficulty, correct_ans, opt_0, opt_1, opt_2, opt_3,
            )
    text_difficulty_df.to_csv(f'data/processed_for_tf/text_difficulty_{dataset}_{split}.csv', index=False)
    return answers_text_df


def get_new_rows_answers_text_df(
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
        new_row = pd.DataFrame([{CORRECT: idx == correct_ans, DESCRIPTION: option, ANS_ID: idx, QUESTION_ID: q_id}])
        out_df = pd.concat([out_df.astype({CORRECT: bool, DESCRIPTION: str, ANS_ID: str, QUESTION_ID: str}),
                            new_row.astype({CORRECT: bool, DESCRIPTION: str, ANS_ID: str, QUESTION_ID: str})],
                           ignore_index=True)
    return out_df


def get_new_row_text_difficulty_df(q_id, question, context, difficulty):
    if type(context) != str:  # i.e. if it is None
        context = ''
    else:
        context = context + ' '
    return pd.DataFrame([{
        DESCRIPTION: context + question,
        QUESTION_ID: q_id,
        DIFFICULTY: difficulty,
    }])


def get_updated_answers_text_df_and_text_difficulty_df(
        out_df_answers_text,
        out_df_text_difficulty,
        q_id, question, context, difficulty,
        correct_ans, option_0, option_1, option_2, option_3,
):
    new_rows_df = get_new_rows_answers_text_df(correct_ans, q_id, option_0, option_1, option_2, option_3)
    out_df_answers_text = pd.concat(
        [out_df_answers_text.astype({CORRECT: bool, DESCRIPTION: str, 'id': str, QUESTION_ID: str}),
         new_rows_df.astype({CORRECT: bool, DESCRIPTION: str, 'id': str, QUESTION_ID: str})], ignore_index=True)
    # get question text (and difficulty)
    new_row_df_text_difficulty = get_new_row_text_difficulty_df(q_id, question, context, difficulty)
    out_df_text_difficulty = pd.concat([out_df_text_difficulty, new_row_df_text_difficulty], ignore_index=True)
    return out_df_answers_text, out_df_text_difficulty


def get_updated_text_difficulty_df(out_df_text_difficulty, q_id, question, context, difficulty):
    new_row_df_text_difficulty = get_new_row_text_difficulty_df(q_id, question, context, difficulty)
    out_df_text_difficulty = pd.concat([out_df_text_difficulty, new_row_df_text_difficulty], ignore_index=True)
    return out_df_text_difficulty


main_tf()
