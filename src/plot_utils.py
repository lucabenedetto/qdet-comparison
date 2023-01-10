import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.constants import DIFFICULTY, PRED_DIFFICULTY

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 22


def bar_plot_question_distribution_by_difficulty(df: pd.DataFrame, color: str, output_filename: str = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(df[DIFFICULTY], df[0], color=color)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("N. of questions")
    ax.set_xticks(df[DIFFICULTY])
    if output_filename is None:
        plt.show()
    else:
        plt.savefig(os.path.join('output_figures', output_filename))
    plt.close(fig)


def hist_plot_question_distribution_by_difficulty(df: pd.DataFrame, color: str, output_filename: str = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(df[DIFFICULTY], bins=25, color=color)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("N. of questions")
    if output_filename is None:
        plt.show()
    else:
        plt.savefig(os.path.join('output_figures', output_filename))
    plt.close(fig)


def plot_violinplot_race(df: pd.DataFrame, title: str, output_filename: str = None):
    data = [df[df[DIFFICULTY] == 0][PRED_DIFFICULTY],
            df[df[DIFFICULTY] == 1][PRED_DIFFICULTY],
            df[df[DIFFICULTY] == 2][PRED_DIFFICULTY]]
    m, b = np.polyfit(df[DIFFICULTY], df[PRED_DIFFICULTY], 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(data, color='#c41331', alpha=0.25)
    ax.plot([-0.5, 2.5], [0.5, 0.5], c='k', alpha=0.25)
    ax.plot([-0.5, 2.5], [1.5, 1.5], c='k', alpha=0.25)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Predicted difficulty')
    ax.set_xticks([0, 1, 2])
    ax.set_title(title)
    if m and b:
        x0, x1 = -0.5, 2.5
        ax.plot([x0, x1], [x0*m + b, x1*m + b], c='#c41331', label='linear fit')
        ax.plot([x0, x1], [x0, x1], '--', c='darkred', label='ideal')
    ax.legend()
    if output_filename is None:
        plt.show()
    else:
        plt.savefig(os.path.join('output_figures', f'distribution_estimated_difficulty_race_pp_{output_filename}.pdf'))
    plt.close(fig)


def plot_violinplot_arc(df: pd.DataFrame, title: str, output_filename: str = None):
    data = [df[df[DIFFICULTY] == int(idx)][PRED_DIFFICULTY] for idx in range(3, 10)]
    m, b = np.polyfit(df[DIFFICULTY], df[PRED_DIFFICULTY], 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(data, color='#223266')
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Predicted difficulty')
    ax.set_title(title)
    ax.set_xticklabels([3, 4, 5, 6, 7, 8, 9])
    if m and b:
        x0, x1 = -0.5, 6.5
        m_i, b_i = 1, 3
        ax.plot([x0, x1], [x0*m + b, x1*m + b], c='#223266', label='linear fit')
        ax.plot([x0, x1], [x0*m_i + b_i,  x1*m_i + b_i], '--', c='tab:blue', label='ideal')
    ax.legend()
    if output_filename is None:
        plt.show()
    else:
        plt.savefig(os.path.join('output_figures', f'distribution_estimated_difficulty_arc_{output_filename}.pdf'))
    plt.close(fig)


def plot_hexbin_am(df: pd.DataFrame, title: str, output_filename: str = None):
    x = df[DIFFICULTY].values
    y = df[PRED_DIFFICULTY].values
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hexbin(x, y, gridsize=(50, 50), cmap='Greens')

    x0, x1 = -3, 3
    ax.plot([x0, x1], [x0*m + b, x1*m + b], c='#088c54', label='linear fit')
    ax.plot([x0, x1], [x0, x1], '--', c='#088c54', label='ideal')

    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Predicted difficulty')

    ax.legend()
    ax.set_title(title)
    if output_filename is None:
        plt.show()
    else:
        plt.savefig(os.path.join('output_figures', f'distribution_estimated_difficulty_am_{output_filename}.pdf'))
    plt.close(fig)
