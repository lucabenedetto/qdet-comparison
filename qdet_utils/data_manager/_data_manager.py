import os
import pandas as pd
from qdet_utils.constants import (
    DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, OPTION_, DEV, TEST, TRAIN,
)
import pickle
from text2props.constants import (
    QUESTION_DF_COLS, CORRECT_TEXTS, WRONG_TEXTS, Q_TEXT, Q_ID as Q_ID_T2P, DIFFICULTY as DIFFICULTY_T2P
)
from qdet_utils.constants import DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, DIFFICULTY, OPTION_, AM


class DataManager:
    def __init__(self):
        pass

    def get_df_converted_to_r2de_format(self):
        pass

    def get_df_converted_to_text2props_format(self):
        pass

# convert_to_r2de_format_and_store_data
# convert_to_text2props_format_and_store_data

    CORRECT_TEXTS = 'text_correct_choices'
    WRONG_TEXTS = 'text_wrong_choices'
    Q_TEXT = 'question_text'
    Q_ID_R2DE = 'question_id'

    def convert_to_r2de_format_and_store_data(
            self,
            dataset,
            data_dir,
            dataset_name,
    ):
        self.get_df_for_r2de(dataset[TRAIN]).to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_train.csv'), index=False)
        self.get_df_for_r2de(dataset[TEST]).to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_test.csv'), index=False)
        self.get_df_for_r2de(dataset[DEV]).to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_dev.csv'), index=False)

    def get_df_for_r2de(self, df: pd.DataFrame) -> pd.DataFrame:
        assert set(DF_COLS).issubset(set(df.columns))
        df[self.Q_ID_R2DE] = df[Q_ID].astype(str)

        if len(df[df[CONTEXT].isnull()]) == 0:
            print("[INFO] All questions in the dataset have a context.")
            df[self.Q_TEXT] = df.apply(lambda r: r[CONTEXT] + ' ' + r[QUESTION], axis=1)
        else:
            print("[INFO] Context is not available.")
            df[self.Q_TEXT] = df[QUESTION]
        if len(df[df[CORRECT_ANSWER].isnull()]) == 0:
            print("[INFO] Information about the correct choice is available for all the questions in the dataset.")
            df[self.CORRECT_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(r[CORRECT_ANSWER])]], axis=1)
            df[self.WRONG_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(x)] for x in range(4) if x != r[CORRECT_ANSWER]], axis=1)
        else:
            print("[INFO] No information about the text of the answer choices.")
            df[self.CORRECT_TEXTS] = None
            df[self.WRONG_TEXTS] = None
        return df[[self.Q_ID_R2DE, self.Q_TEXT, self.CORRECT_TEXTS, self.WRONG_TEXTS]]


    ############################### TODO BELOW T2P STUFF


    def convert_to_text2props_format_and_store_data(
            self,
            dataset,
            data_dir,
            dataset_name,
    ):
        df = pd.concat([dataset[TRAIN], dataset[DEV], dataset[TEST]])
        assert set(DF_COLS).issubset(set(df.columns))
        out_dict = dict()
        out_dict[DIFFICULTY_T2P] = dict()
        for q_id, diff in df[[Q_ID, DIFFICULTY]].values:
            if q_id in out_dict[DIFFICULTY_T2P].keys():
                print(f"[WARNING] Item {str(q_id)} already seen. It was b={out_dict[DIFFICULTY_T2P][q_id]}, now b={diff}.")
                continue
            out_dict[DIFFICULTY_T2P][str(q_id)] = float(diff)  # difficulty must be a float, here, even for categorical ones
        pickle.dump(out_dict, open(os.path.join(data_dir, f't2p_{dataset_name}_difficulty_dict.p'), 'wb'))

        pickle.dump([out_dict[DIFFICULTY][x] for x in dataset[TRAIN][Q_ID].values], open(os.path.join(data_dir, f'y_true_train_{dataset_name}.p'), 'wb'))
        pickle.dump([out_dict[DIFFICULTY][x] for x in dataset[DEV][Q_ID].values], open(os.path.join(data_dir, f'y_true_dev_{dataset_name}.p'), 'wb'))
        pickle.dump([out_dict[DIFFICULTY][x] for x in dataset[TEST][Q_ID].values], open(os.path.join(data_dir, f'y_true_test_{dataset_name}.p'), 'wb'))

        self.get_df_for_text2props(dataset[TRAIN]).to_csv(os.path.join(data_dir, f't2p_{dataset_name}_train.csv'), index=False)
        self.get_df_for_text2props(dataset[TEST]).to_csv(os.path.join(data_dir, f't2p_{dataset_name}_test.csv'), index=False)
        self.get_df_for_text2props(dataset[DEV]).to_csv(os.path.join(data_dir, f't2p_{dataset_name}_dev.csv'), index=False)

    def get_df_for_text2props(self, df: pd.DataFrame) -> pd.DataFrame:
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
