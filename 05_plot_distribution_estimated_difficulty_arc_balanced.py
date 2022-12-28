import pandas as pd
import pickle

from src.plot_utils import plot_violinplot_arc

R2DE_QA = r'R2DE $Q_A$'
DISTILBERT_QA = 'DistilBERT $Q_A$'
BERT_QA = 'BERT $Q_A$'
LING_N_READ = 'Ling. & Read.'

# BERT_QA
df = pd.read_csv('data/transformers_predictions/arc_balanced/predictions_test_BERT_question_all_0.csv')
plot_violinplot_arc(df, BERT_QA, output_filename='bert_qa')

# DISTILBERT_QA
df = pd.read_csv('data/transformers_predictions/arc_balanced/predictions_test_DistilBERT_question_all_0.csv')
plot_violinplot_arc(df, DISTILBERT_QA, output_filename='distilbert_qa')

# R2DE_QA
df = pd.read_csv('data/processed/arc_test.csv')
df['predicted_difficulty'] = pickle.load(open('output/arc_balanced/seed_0/predictions_test_r2de_encoding_2.p', 'rb'))
plot_violinplot_arc(df, R2DE_QA, output_filename='r2de_qa')

# LING_N_READ
df = pd.read_csv('data/processed/arc_test.csv')
df['predicted_difficulty'] = pickle.load(open('output/arc_balanced/seed_0/predictions_test_ling_and_read__RF.p', 'rb'))
plot_violinplot_arc(df, LING_N_READ, output_filename='ling_n_read')
