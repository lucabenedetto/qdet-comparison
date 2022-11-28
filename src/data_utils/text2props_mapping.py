from typing import Dict
import pandas as pd
from text2props.constants import (
    QUESTION_DF_COLS, CORRECT_TEXTS, WRONG_TEXTS, Q_TEXT, Q_ID as Q_ID_T2P, DIFFICULTY as DIFFICULTY_T2P
)
from src.constants import DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, DIFFICULTY, OPTION_


def get_difficulty_dict_for_text2props(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    assert set(DF_COLS) == set(df.columns)
    out_dict = dict()
    out_dict[DIFFICULTY_T2P] = dict()
    for q_id, diff in df[[Q_ID, DIFFICULTY]].values:
        if q_id in out_dict[DIFFICULTY_T2P].keys():
            raise ValueError("Item already encountered.")
        out_dict[DIFFICULTY_T2P][str(q_id)] = float(diff)  # difficulty must be a float, here, even for categorical ones
    return out_dict


def get_df_for_text2props(df: pd.DataFrame) -> pd.DataFrame:
    assert set(DF_COLS) == set(df.columns)
    df[Q_ID_T2P] = df[Q_ID]

    if len(df[df[CONTEXT].isnull()]) == 0:
        print("[INFO] All questions in the dataset have a context.")
        df[Q_TEXT] = df.apply(lambda r: r[CONTEXT] + ' ' + r[QUESTION], axis=1)
    else:
        print("[INFO] Context is not available.")
        df[Q_TEXT] = df[QUESTION]
    if len(df[df[CORRECT_ANSWER].isnull()]) == 0:
        print("[INFO] Information about the correct choice is available for all the questions in the dataset.")
        df[CORRECT_TEXTS] = df.apply(lambda r: r[OPTION_ + str(r[CORRECT_ANSWER])], axis=1)
        df[WRONG_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(x)] for x in range(4) if x != r[CORRECT_ANSWER]], axis=1)
    else:
        print("[INFO] No information about the text of the answer choices.")
        df[CORRECT_TEXTS] = None
        df[WRONG_TEXTS] = None
    return df[QUESTION_DF_COLS]
