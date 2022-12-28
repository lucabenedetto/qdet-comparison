import pandas as pd
import pickle

from src.plot_utils import plot_violinplot_race


LING_RF = 'Ling.'
# READ_RF = 'Read.'
# W2V_QA = r'W2V $Q_A$'
# R2DE_QC = r'R2DE $Q_C$'
DISTILBERT_QA = 'DistilBERT $Q_A$'
BERT_QA = 'BERT $Q_A$'
W2V_QO_N_LING = r'W2V $Q_{Only}$ & Ling.'

# BERT_QA
df = pd.read_csv('data/transformers_predictions/race_pp/predictions_test_BERT_3_question_all_0.csv')
plot_violinplot_race(df, title=BERT_QA, output_filename='bert')

# DISTILBERT_QA
df = pd.read_csv('data/transformers_predictions/race_pp/predictions_test_DistilBERT_question_all_0.csv')
plot_violinplot_race(df, title=DISTILBERT_QA, output_filename='distilbert')

# LING_RF
df = pd.read_csv('data/processed/race_pp_test.csv')
df['predicted_difficulty'] = pickle.load(open('output/race_pp/seed_0/predictions_test_ling__RF.p', 'rb'))
plot_violinplot_race(df, title=LING_RF, output_filename='ling')

# READ_RF
# df = pd.read_csv('data/processed/race_pp_test.csv')
# df['predicted_difficulty'] = pickle.load(open('output/race_pp/seed_0/predictions_test_read__RF.p', 'rb'))
# plot_violinplot_race(df, title=READ_RF)

# W2V_QA
# df = pd.read_csv('data/processed/race_pp_test.csv')
# df['predicted_difficulty'] = pickle.load(open('output/race_pp/seed_0/predictions_test_w2v_q_all__RF.p', 'rb'))
# plot_violinplot_race(df, title=W2V_QA)

# R2DE_QC
# df = pd.read_csv('data/processed/race_pp_test.csv')
# df['predicted_difficulty'] = pickle.load(open('output/race_pp/seed_0/predictions_test_r2de_encoding_1.p', 'rb'))
# plot_violinplot_race(df, title=R2DE_QC)

# W2V_QO_N_LING
df = pd.read_csv('data/processed/race_pp_test.csv')
df['predicted_difficulty'] = pickle.load(open('output/race_pp/seed_0/predictions_test_w2v_q_only_and_ling__RF.p', 'rb'))
plot_violinplot_race(df, title=W2V_QO_N_LING, output_filename='w2v_q_only_n_ling')
