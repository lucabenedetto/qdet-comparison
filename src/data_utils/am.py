import os
import pandas as pd
import pickle

from src.data_utils.constants import DF_COLS


def prepare_assistments_dataset(data_dir: str, output_data_dir: str):
    df1 = pd.read_csv(os.path.join(data_dir, 'dataset_am_train.csv'))
    df2 = pd.read_csv(os.path.join(data_dir, 'dataset_am_test.csv'))
    # there are some duplicates. Let's remove them
    df1_items = set(df1['question_id'].unique())
    df2 = df2[~df2['question_id'].isin(df1_items)]

    in_df = pd.concat([df1, df2], ignore_index=True)
    print("[INFO] input_df len = %d (df1=%d, df2=%d)" % (len(in_df), len(df1), len(df2)))

    difficulty_dict = pickle.load(open(os.path.join(data_dir, 'irt_difficulty_am.p'), 'rb'))['difficulty']
    print("[INFO] Num items in dictionary = %d" % len(difficulty_dict.keys()))
    in_df = in_df[in_df['question_id'].isin(difficulty_dict.keys())]

    in_df = in_df.sample(frac=1.0, random_state=42)  # TODO make constant the random state
    train_size = int(0.6 * len(in_df))  # TODO make these coefficients constant
    test_size = int(0.2 * len(in_df))

    train_df = in_df[:train_size]
    test_df = in_df[train_size:train_size+test_size]
    dev_df = in_df[train_size+test_size:]

    _get_df_single_split(train_df, difficulty_dict, output_data_dir, 'train')
    _get_df_single_split(test_df, difficulty_dict, output_data_dir, 'test')
    _get_df_single_split(dev_df, difficulty_dict, output_data_dir, 'dev')


def _get_df_single_split(df, difficulty_dict, output_data_dir, split):
    out_df = pd.DataFrame(columns=DF_COLS)
    for q_id, q_text in df[['question_id', 'question_text']].values:
        assert q_id in difficulty_dict.keys()
        new_row_df = pd.DataFrame([{
            'q_id': q_id,
            'correct_answer': None,
            'difficulty': difficulty_dict[q_id],
            'question': q_text,
            'split': split,
            'options': None,
            'context': None,
            'context_id': None,
        }])
        out_df = pd.concat([out_df, new_row_df], ignore_index=True)

    assert set(out_df.columns) == set(DF_COLS)
    out_df.to_csv(os.path.join(output_data_dir, f'am_{split}.csv'), index=False)
