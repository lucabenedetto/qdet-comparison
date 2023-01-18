from typing import Dict
import logging
import os
import pandas as pd
from qdet_utils.constants import (
    DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, OPTION_, DEV, TEST, TRAIN,
)
from qdet_utils.constants import (
    CORRECT,
    DESCRIPTION,
    QUESTION_ID,
    ANS_ID,
    # TODO these are used only for the transformers, I should rename them!
)
import pickle
from text2props.constants import (
    QUESTION_DF_COLS, CORRECT_TEXTS, WRONG_TEXTS, Q_TEXT, Q_ID as Q_ID_T2P, DIFFICULTY as DIFFICULTY_T2P
)
from qdet_utils.constants import DF_COLS, CORRECT_ANSWER, CONTEXT, QUESTION, Q_ID, DIFFICULTY, OPTION_, AM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self):
        pass

    CORRECT_TEXTS = 'text_correct_choices'
    WRONG_TEXTS = 'text_wrong_choices'
    Q_TEXT = 'question_text'
    R2DE_Q_ID = 'question_id'

    def convert_to_r2de_format_and_store_dataset(
            self,
            dataset: Dict[str, pd.DataFrame],
            data_dir: str,
            dataset_name: str,
    ) -> None:
        for split in [TRAIN, DEV, TEST]:
            converted_df = self.convert_df_to_r2de_format(dataset[split])
            converted_df.to_csv(os.path.join(data_dir, f'r2de_{dataset_name}_{split}.csv'), index=False)

    def convert_df_to_r2de_format(self, df: pd.DataFrame) -> pd.DataFrame:
        assert set(DF_COLS).issubset(set(df.columns))
        df[self.R2DE_Q_ID] = df[Q_ID].astype(str)

        if len(df[df[CONTEXT].isnull()]) == 0:
            logger.info("All questions in the dataset have a context.")
            df[self.Q_TEXT] = df.apply(lambda r: r[CONTEXT] + ' ' + r[QUESTION], axis=1)
        else:
            logger.info("Context is not available.")
            df[self.Q_TEXT] = df[QUESTION]
        if len(df[df[CORRECT_ANSWER].isnull()]) == 0:
            logger.info("Information about the correct choice is available for all the questions in the dataset.")
            df[self.CORRECT_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(r[CORRECT_ANSWER])]], axis=1)
            df[self.WRONG_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(x)] for x in range(4) if x != r[CORRECT_ANSWER]], axis=1)
        else:
            logger.info("No information about the text of the answer choices.")
            df[self.CORRECT_TEXTS] = None
            df[self.WRONG_TEXTS] = None
        return df[[self.R2DE_Q_ID, self.Q_TEXT, self.CORRECT_TEXTS, self.WRONG_TEXTS]]

    def convert_to_text2props_format_and_store_dataset(
            self,
            dataset: Dict[str, pd.DataFrame],
            data_dir: str,
            dataset_name: str,
    ) -> None:
        df = pd.concat([dataset[TRAIN], dataset[DEV], dataset[TEST]])
        assert set(DF_COLS).issubset(set(df.columns))
        out_dict = dict()
        out_dict[DIFFICULTY_T2P] = dict()
        for q_id, diff in df[[Q_ID, DIFFICULTY]].values:
            if q_id in out_dict[DIFFICULTY_T2P].keys():
                logger.warning(f"Item {str(q_id)} already seen. It was b={out_dict[DIFFICULTY_T2P][q_id]}, now b={diff}.")
                continue
            out_dict[DIFFICULTY_T2P][str(q_id)] = float(diff)
        pickle.dump(out_dict, open(os.path.join(data_dir, f't2p_{dataset_name}_difficulty_dict.p'), 'wb'))

        pickle.dump([out_dict[DIFFICULTY][x] for x in dataset[TRAIN][Q_ID].values], open(os.path.join(data_dir, f'y_true_train_{dataset_name}.p'), 'wb'))
        pickle.dump([out_dict[DIFFICULTY][x] for x in dataset[DEV][Q_ID].values], open(os.path.join(data_dir, f'y_true_dev_{dataset_name}.p'), 'wb'))
        pickle.dump([out_dict[DIFFICULTY][x] for x in dataset[TEST][Q_ID].values], open(os.path.join(data_dir, f'y_true_test_{dataset_name}.p'), 'wb'))

        for split in [TRAIN, DEV, TEST]:
            converted_df = self.convert_df_to_text2props_format(dataset[split])
            converted_df.to_csv(os.path.join(data_dir, f't2p_{dataset_name}_{split}.csv'), index=False)

    def convert_df_to_text2props_format(self, df: pd.DataFrame) -> pd.DataFrame:
        assert set(DF_COLS).issubset(set(df.columns))
        df[Q_ID_T2P] = df[Q_ID].astype(str)

        if len(df[df[CONTEXT].isnull()]) == 0:
            logger.info("All questions in the dataset have a context.")
            df[Q_TEXT] = df.apply(lambda r: r[CONTEXT] + ' ' + r[QUESTION], axis=1)
        else:
            logger.info("Context is not available.")
            df[Q_TEXT] = df[QUESTION]
        if len(df[df[CORRECT_ANSWER].isnull()]) == 0:
            logger.info("Information about the correct choice is available for all the questions in the dataset.")
            df[CORRECT_TEXTS] = df.apply(lambda r: r[OPTION_ + str(r[CORRECT_ANSWER])], axis=1)
            df[WRONG_TEXTS] = df.apply(lambda r: [r[OPTION_ + str(x)] for x in range(4) if x != r[CORRECT_ANSWER]], axis=1)
        else:
            logger.info("No information about the text of the answer choices.")
            df[CORRECT_TEXTS] = None
            df[WRONG_TEXTS] = None
        return df[QUESTION_DF_COLS]

    AS_TYPE_DICT = {CORRECT: bool, DESCRIPTION: str, ANS_ID: str, QUESTION_ID: str}
    # TODO fix this

    def convert_to_transformers_format_and_store_dataset(
            self,
            dataset: Dict[str, pd.DataFrame],
            data_dir: str,
            dataset_name: str,
            skip_answers_texts: bool,
    ) -> None:
        answer_texts_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, ANS_ID, QUESTION_ID])

        text_difficulty_df, answer_texts_df = self.get_text_difficulty_and_answer_texts(dataset[TRAIN], answer_texts_df, skip_answers_texts)
        # TODO this line below is the old version, I have to update the notebook as well! All the filenames!
        # text_difficulty_df.to_csv(os.path.join(data_dir, 'for_tf', f'text_difficulty_{dataset_name}_train.csv'), index=False)
        text_difficulty_df.to_csv(os.path.join(data_dir, f'tf_{dataset_name}_text_difficulty_train.csv'), index=False)

        text_difficulty_df, answer_texts_df = self.get_text_difficulty_and_answer_texts(dataset[TEST], answer_texts_df, skip_answers_texts)
        text_difficulty_df.to_csv(os.path.join(data_dir, f'tf_{dataset_name}_text_difficulty_test.csv'), index=False)

        text_difficulty_df, answer_texts_df = self.get_text_difficulty_and_answer_texts(dataset[DEV], answer_texts_df, skip_answers_texts)
        text_difficulty_df.to_csv(os.path.join(data_dir, f'tf_{dataset_name}_text_difficulty_dev.csv'), index=False)

        if not skip_answers_texts:
            answer_texts_df.to_csv(os.path.join(data_dir, f'tf_{dataset_name}_answers_texts.csv'), index=False)

    def get_text_difficulty_and_answer_texts(self, df, answers_text_df, skip_ans_texts):
        text_difficulty_df = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
        if skip_ans_texts:
            for question, context, q_id, difficulty in df[[QUESTION, CONTEXT, Q_ID, DIFFICULTY]].values:
                text_difficulty_df = pd.concat(
                    [text_difficulty_df, self._get_new_row_text_difficulty_df(q_id, question, context, difficulty)],
                    ignore_index=True
                )
        else:
            for correct_option, _, opt0, opt1, opt2, opt3, question, context, _, q_id, _, difficulty in df.values:
                answers_text_df = pd.concat(
                    [
                        answers_text_df.astype(self.AS_TYPE_DICT),
                        self._get_new_rows_answers_text_df(correct_option, q_id, [opt0, opt1, opt2, opt3]).astype(self.AS_TYPE_DICT)
                    ],
                    ignore_index=True
                )
                text_difficulty_df = pd.concat(
                    [text_difficulty_df, self._get_new_row_text_difficulty_df(q_id, question, context, difficulty)],
                    ignore_index=True
                )
        return text_difficulty_df, answers_text_df

    def _get_new_rows_answers_text_df(self, correct_ans, q_id, options):
        out_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, ANS_ID, QUESTION_ID])
        for idx, option in enumerate(options):
            new_row = pd.DataFrame([{CORRECT: idx == correct_ans, DESCRIPTION: option, ANS_ID: idx, QUESTION_ID: q_id}])
            out_df = pd.concat([out_df.astype(self.AS_TYPE_DICT), new_row.astype(self.AS_TYPE_DICT)], ignore_index=True)
        return out_df

    def _get_new_row_text_difficulty_df(self, q_id, question, context, difficulty):
        context = '' if type(context) != str else context + ' '
        return pd.DataFrame([{DESCRIPTION: context + question, QUESTION_ID: q_id, DIFFICULTY: difficulty,}])
