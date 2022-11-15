import pandas as pd

from src.data_utils.ca import prepare_and_return_ca_df

ca_data_dir = 'data/raw/cloudacademy/content-text-data'

df = prepare_and_return_ca_df(ca_data_dir)
print(df)
print(df.shape)

# print(df[(df['correct_answer'] >= 5) & (df['correct_answer'] < 100)])
print(df.groupby('correct_answer').size())

# In ARC Challenge Dev
# 4 items with (1), (2), (3), (4)
# 295-292-1 = 2 with (A), (B), (C)
# 292-1 = 291 with (A), (B), (C), (D)
# 1 with (A), (B), (C), (D), (E)
