import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

from src.constants import DATA_DIR, DIFFICULTY

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 24


def bar_plot(df, color, output_filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(df[DIFFICULTY], df[0], color=color)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("N. of questions")
    ax.set_xticks(df[DIFFICULTY])
    # plt.show()
    plt.savefig(output_filename)
    plt.close(fig)


def hist_plot(df, color, output_filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(df[DIFFICULTY], bins=25, color=color)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("N. of questions")
    # plt.show()
    plt.savefig(output_filename)
    plt.close(fig)


def main():
    # RACE
    df = pd.read_csv(os.path.join(DATA_DIR, f'race_pp_train.csv')).groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot(df, '#c41331', 'output_figures/difficulty_distribution_race_train.pdf')
    df = pd.read_csv(os.path.join(DATA_DIR, f'race_pp_test.csv')).groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot(df, '#c41331', 'output_figures/difficulty_distribution_race_test.pdf')

    # ARC
    df = pd.read_csv(os.path.join(DATA_DIR, f'arc_train.csv')).groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot(df, '#223266', 'output_figures/difficulty_distribution_arc_train.pdf')
    df = pd.read_csv(os.path.join(DATA_DIR, f'arc_test.csv')).groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot(df, '#223266', 'output_figures/difficulty_distribution_arc_test.pdf')

    # ARC Balanced
    df = pd.read_csv(os.path.join(DATA_DIR, f'arc_balanced_train.csv')).groupby(DIFFICULTY).size().reset_index().sort_values(DIFFICULTY, ascending=True)
    bar_plot(df, '#223266', 'output_figures/difficulty_distribution_arc_balanced_train.pdf')

    # AM
    df = pd.read_csv(os.path.join(DATA_DIR, f'am_train.csv'))
    hist_plot(df, '#088c54', 'output_figures/difficulty_distribution_am_train.pdf')
    df = pd.read_csv(os.path.join(DATA_DIR, f'am_test.csv'))
    hist_plot(df, '#088c54', 'output_figures/difficulty_distribution_am_test.pdf')


if __name__ == '__main__':
    main()
