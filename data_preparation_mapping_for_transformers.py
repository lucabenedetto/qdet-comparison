import os
import pandas as pd
from src.data_utils.mapping_tranformers import get_text_difficulty_and_answer_texts
from src.constants import (
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
)


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
