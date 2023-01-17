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

ARC_DEV = 'Dev'
ARC_TEST = 'Test'
ARC_TRAIN = 'Train'
MAP_TO_PROCESSED_SPLIT_NAMES = {ARC_DEV: DEV, ARC_TEST: TEST, ARC_TRAIN: TRAIN}


def prepare_arc_dataset(arc_data_dir: str, output_data_dir: str):
    out_dfs = []
    for split in [ARC_TRAIN, ARC_DEV, ARC_TEST]:
        df_arc = prepare_and_return_arc_df(arc_data_dir, split)
        assert set(df_arc.columns) == set(DF_COLS)
        df_arc.to_csv(os.path.join(output_data_dir, f'arc_{MAP_TO_PROCESSED_SPLIT_NAMES[split]}.csv'), index=False)
        out_dfs.append(df_arc.copy())
    return out_dfs


def prepare_and_return_arc_df(data_dir: str, split: str) -> pd.DataFrame:
    df_easy = pd.read_csv(os.path.join(data_dir, 'ARC-Easy', f'ARC-Easy-{split}.csv'))
    df_easy = _prepare_and_return_arc_df(df_easy)
    df_chall = pd.read_csv(os.path.join(data_dir, 'ARC-Challenge', f'ARC-Challenge-{split}.csv'))
    df_chall = _prepare_and_return_arc_df(df_chall)
    df = pd.concat([df_easy, df_chall])
    return df


def _prepare_and_return_arc_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['isMultipleChoiceQuestion'] == 1]  # to keep only the MCQs
    df = df[df['includesDiagram'] == 0]  # to keep only the questions without diagram
    df = df[['questionID', 'AnswerKey', 'schoolGrade', 'question', 'category']]
    df = df.rename(
        columns={
            'questionID': Q_ID,  # analyze if there is any difference w.r.t. using originalQuestionID
            'AnswerKey': CORRECT_ANSWER,
            'schoolGrade': DIFFICULTY,
            # 'question': QUESTION,  # it already has the right name
            'category': SPLIT,
        }
    )
    df[CORRECT_ANSWER] = df.apply(
        lambda r:
        int(r[CORRECT_ANSWER])-1 if ord('1') <= ord(r[CORRECT_ANSWER]) <= ord('9') else ord(r[CORRECT_ANSWER])-ord('A'),
        axis=1
    )
    # this is to have integer indexes and 0 as first index of the correct choices (in the original dataset it is 1)
    df[SPLIT] = df.apply(lambda r: MAP_TO_PROCESSED_SPLIT_NAMES[r[SPLIT]], axis=1)

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

    return df
