import os
import pandas as pd
import pickle

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


def prepare_assistments_dataset(
        data_dir: str,
        output_data_dir: str,
        random_state: int = 42,
        train_size: float = 0.6,
        test_size: float = 0.2,
):
    df1 = pd.read_csv(os.path.join(data_dir, 'dataset_am_train.csv')).drop_duplicates('question_id')
    df2 = pd.read_csv(os.path.join(data_dir, 'dataset_am_test.csv')).drop_duplicates('question_id')
    # there are some duplicates. Let's remove them
    df1_items = set(df1['question_id'].unique())
    df2 = df2[~df2['question_id'].isin(df1_items)]

    in_df = pd.concat([df1, df2], ignore_index=True)
    print("[INFO] input_df len = %d (df1=%d, df2=%d)" % (len(in_df), len(df1), len(df2)))

    difficulty_dict = pickle.load(open(os.path.join(data_dir, 'irt_difficulty_am.p'), 'rb'))[DIFFICULTY]
    print("[INFO] Num items in dictionary = %d" % len(difficulty_dict.keys()))
    in_df = in_df[in_df['question_id'].isin(difficulty_dict.keys())]

    in_df = in_df.sample(frac=1.0, random_state=random_state)
    train_size = int(train_size * len(in_df))
    test_size = int(test_size * len(in_df))

    train_df = in_df[:train_size]
    test_df = in_df[train_size:train_size+test_size]
    dev_df = in_df[train_size+test_size:]

    df_train = _get_df_single_split(train_df, difficulty_dict, output_data_dir, TRAIN)
    df_test = _get_df_single_split(test_df, difficulty_dict, output_data_dir, TEST)
    df_dev = _get_df_single_split(dev_df, difficulty_dict, output_data_dir, DEV)
    return df_train, df_dev, df_test


def _get_df_single_split(df, difficulty_dict, output_data_dir, split):
    out_df = pd.DataFrame(columns=DF_COLS)
    for q_id, q_text in df[['question_id', 'question_text']].values:
        assert q_id in difficulty_dict.keys()
        new_row_df = pd.DataFrame([{
            Q_ID: str(q_id),
            CORRECT_ANSWER: None,
            DIFFICULTY: difficulty_dict[q_id],
            QUESTION: q_text,
            SPLIT: split,
            OPTIONS: None,
            OPTION_0: None,
            OPTION_1: None,
            OPTION_2: None,
            OPTION_3: None,
            CONTEXT: None,
            CONTEXT_ID: None,
        }])
        out_df = pd.concat([out_df, new_row_df], ignore_index=True)

    assert set(out_df.columns) == set(DF_COLS)
    out_df.to_csv(os.path.join(output_data_dir, f'am_{split}.csv'), index=False)
    return out_df
