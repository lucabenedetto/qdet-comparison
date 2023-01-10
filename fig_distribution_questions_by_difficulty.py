import os
import pandas as pd

from src.constants import DATA_DIR, DIFFICULTY
from src.plot_utils import bar_plot_question_distribution_by_difficulty, hist_plot_question_distribution_by_difficulty


def main():
    # RACE
    df = pd.read_csv(os.path.join(DATA_DIR, f'race_pp_train.csv'))
    df = df.groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot_question_distribution_by_difficulty(df, '#c41331', 'difficulty_distribution_race_train.pdf')
    df = pd.read_csv(os.path.join(DATA_DIR, f'race_pp_test.csv'))
    df = df.groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot_question_distribution_by_difficulty(df, '#c41331', 'difficulty_distribution_race_test.pdf')

    # ARC
    df = pd.read_csv(os.path.join(DATA_DIR, f'arc_train.csv'))
    df = df.groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot_question_distribution_by_difficulty(df, '#223266', 'difficulty_distribution_arc_train.pdf')
    df = pd.read_csv(os.path.join(DATA_DIR, f'arc_test.csv'))
    df = df.groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot_question_distribution_by_difficulty(df, '#223266', 'difficulty_distribution_arc_test.pdf')

    # ARC Balanced
    df = pd.read_csv(os.path.join(DATA_DIR, f'arc_balanced_train.csv'))
    df = df.groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot_question_distribution_by_difficulty(df, '#223266', 'difficulty_distribution_arc_balanced_train.pdf')

    # AM
    df = pd.read_csv(os.path.join(DATA_DIR, f'am_train.csv'))
    hist_plot_question_distribution_by_difficulty(df, '#088c54', 'difficulty_distribution_am_train.pdf')
    df = pd.read_csv(os.path.join(DATA_DIR, f'am_test.csv'))
    hist_plot_question_distribution_by_difficulty(df, '#088c54', 'difficulty_distribution_am_test.pdf')


if __name__ == '__main__':
    main()
