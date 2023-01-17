import os
import pandas as pd
import pickle
from text2props.constants import (
    QUESTION_DF_COLS, CORRECT_TEXTS, WRONG_TEXTS, Q_TEXT, Q_ID as Q_ID_T2P, DIFFICULTY as DIFFICULTY_T2P
)
from qdet_utils.constants import DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, DIFFICULTY, OPTION_, AM


def convert_to_text2props_format_and_store_data(
        df_train,
        df_dev,
        df_test,
        data_dir,
        dataset_name,
):
    df = pd.concat([df_train, df_dev, df_test])
    assert set(DF_COLS).issubset(set(df.columns))
    out_dict = dict()
    out_dict[DIFFICULTY_T2P] = dict()
    for q_id, diff in df[[Q_ID, DIFFICULTY]].values:
        if q_id in out_dict[DIFFICULTY_T2P].keys():
            print(f"[WARNING] Item {str(q_id)} already seen. It was b={out_dict[DIFFICULTY_T2P][q_id]}, now b={diff}.")
            continue
        out_dict[DIFFICULTY_T2P][str(q_id)] = float(diff)  # difficulty must be a float, here, even for categorical ones
    pickle.dump(out_dict, open(os.path.join(data_dir, f't2p_{dataset_name}_difficulty_dict.p'), 'wb'))

    pickle.dump([out_dict[DIFFICULTY][x] for x in df_train[Q_ID].values], open(os.path.join(data_dir, f'y_true_train_{dataset_name}.p'), 'wb'))
    pickle.dump([out_dict[DIFFICULTY][x] for x in df_dev[Q_ID].values], open(os.path.join(data_dir, f'y_true_dev_{dataset_name}.p'), 'wb'))
    pickle.dump([out_dict[DIFFICULTY][x] for x in df_test[Q_ID].values], open(os.path.join(data_dir, f'y_true_test_{dataset_name}.p'), 'wb'))

    get_df_for_text2props(df_train).to_csv(os.path.join(data_dir, f't2p_{dataset_name}_train.csv'), index=False)
    get_df_for_text2props(df_test).to_csv(os.path.join(data_dir, f't2p_{dataset_name}_test.csv'), index=False)
    get_df_for_text2props(df_dev).to_csv(os.path.join(data_dir, f't2p_{dataset_name}_dev.csv'), index=False)


def get_df_for_text2props(df: pd.DataFrame) -> pd.DataFrame:
    assert set(DF_COLS).issubset(set(df.columns))
    df[Q_ID_T2P] = df[Q_ID].astype(str)

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
