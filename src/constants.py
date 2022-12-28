DATA_DIR = 'data/processed'
OUTPUT_DIR = 'output'

# names of datasets
RACE_PP = 'race_pp'
RACE_PP_4K = 'race_pp_4k'
RACE_PP_8K = 'race_pp_8k'
RACE_PP_12K = 'race_pp_12k'
ARC = 'arc'
ARC_BALANCED = 'arc_balanced'
AM = 'am'

# name of splits (used in filenames, etc.)
DEV = 'dev'
TEST = 'test'
TRAIN = 'train'

ANS_ID = 'id'
CORRECT = 'correct'
CORRECT_ANSWER = 'correct_answer'
CORRECT_ANSWERS_LIST = 'correct_answers_list'
DESCRIPTION = 'description'
OPTIONS = 'options'
OPTION_ = 'option_'
OPTION_0 = 'option_0'
OPTION_1 = 'option_1'
OPTION_2 = 'option_2'
OPTION_3 = 'option_3'
QUESTION = 'question'
CONTEXT = 'context'
CONTEXT_ID = 'context_id'
Q_ID = 'q_id'
QUESTION_ID = 'question_id'
SPLIT = 'split'
DIFFICULTY = 'difficulty'

DF_COLS = [
    CORRECT_ANSWER, OPTIONS, OPTION_0, OPTION_1, OPTION_2, OPTION_3,
    QUESTION, CONTEXT, CONTEXT_ID, Q_ID, SPLIT, DIFFICULTY
]

TF_Q_ONLY = 'question_only'
TF_Q_CORRECT = 'question_correct'
TF_Q_ALL = 'question_all'

LIST_TF_ENCODINGS = [TF_Q_ONLY, TF_Q_CORRECT, TF_Q_ALL]

BERT = 'BERT'
DISTILBERT = 'DistilBERT'
TF_MODELS = [BERT, DISTILBERT]
