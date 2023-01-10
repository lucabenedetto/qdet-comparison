import pandas as pd
from src.constants import (
    DIFFICULTY,
    CORRECT,
    DESCRIPTION,
    QUESTION_ID,
    ANS_ID,
    QUESTION, CONTEXT, Q_ID,
)
AS_TYPE_DICT = {CORRECT: bool, DESCRIPTION: str, ANS_ID: str, QUESTION_ID: str}


def get_text_difficulty_and_answer_texts(df, answers_text_df, skip_ans_texts):
    text_difficulty_df = pd.DataFrame(columns=[DESCRIPTION, QUESTION_ID, DIFFICULTY])
    if skip_ans_texts:
        for question, context, q_id, difficulty in df[[QUESTION, CONTEXT, Q_ID, DIFFICULTY]].values:
            text_difficulty_df = pd.concat(
                [text_difficulty_df, _get_new_row_text_difficulty_df(q_id, question, context, difficulty)],
                ignore_index=True
            )
    else:
        for correct_option, _, opt0, opt1, opt2, opt3, question, context, _, q_id, _, difficulty in df.values:
            answers_text_df = pd.concat(
                [
                    answers_text_df.astype(AS_TYPE_DICT),
                    _get_new_rows_answers_text_df(correct_option, q_id, [opt0, opt1, opt2, opt3]).astype(AS_TYPE_DICT)
                ],
                ignore_index=True
            )
            text_difficulty_df = pd.concat(
                [text_difficulty_df, _get_new_row_text_difficulty_df(q_id, question, context, difficulty)],
                ignore_index=True
            )
    return text_difficulty_df, answers_text_df


def _get_new_rows_answers_text_df(correct_ans, q_id, options):
    out_df = pd.DataFrame(columns=[CORRECT, DESCRIPTION, ANS_ID, QUESTION_ID])
    for idx, option in enumerate(options):
        new_row = pd.DataFrame([{CORRECT: idx == correct_ans, DESCRIPTION: option, ANS_ID: idx, QUESTION_ID: q_id}])
        out_df = pd.concat([out_df.astype(AS_TYPE_DICT), new_row.astype(AS_TYPE_DICT)], ignore_index=True)
    return out_df


def _get_new_row_text_difficulty_df(q_id, question, context, difficulty):
    context = '' if type(context) != str else context + ' '
    return pd.DataFrame([{DESCRIPTION: context + question, QUESTION_ID: q_id, DIFFICULTY: difficulty,}])
