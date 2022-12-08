import os
import pandas as pd
from src.constants import DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, OPTION_

CORRECT_TEXTS = 'text_correct_choices'
WRONG_TEXTS = 'text_wrong_choices'
Q_TEXT = 'question_text'
Q_ID_R2DE = 'question_id'


def convert_to_r2de_format_and_store_data(
        df_train,
        df_dev,
        df_test,
        data_dir,
        dataset_name,
):
    get_df_for_r2de(df_train).to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_train.csv'), index=False)
    get_df_for_r2de(df_test).to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_test.csv'), index=False)
    get_df_for_r2de(df_dev).to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_dev.csv'), index=False)


def get_df_for_r2de(df: pd.DataFrame) -> pd.DataFrame:
    assert set(DF_COLS).issubset(set(df.columns))
    df[Q_ID_R2DE] = df[Q_ID]

    if len(df[df[CONTEXT].isnull()]) == 0:
        print("[INFO] All questions in the dataset have a context.")
        df[Q_TEXT] = df.apply(lambda r: r[CONTEXT] + ' ' + r[QUESTION], axis=1)
    else:
        print("[INFO] Context is not available.")
        df[Q_TEXT] = df[QUESTION]
    if len(df[df[CORRECT_ANSWER].isnull()]) == 0:
        print("[INFO] Information about the correct choice is available for all the questions in the dataset.")
        df[CORRECT_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(r[CORRECT_ANSWER])]], axis=1)
        df[WRONG_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(x)] for x in range(4) if x != r[CORRECT_ANSWER]], axis=1)
    else:
        print("[INFO] No information about the text of the answer choices.")
        df[CORRECT_TEXTS] = None
        df[WRONG_TEXTS] = None
    return df[[Q_ID_R2DE, Q_TEXT, CORRECT_TEXTS, WRONG_TEXTS]]
