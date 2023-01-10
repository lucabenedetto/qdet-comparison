import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.constants import DIFFICULTY

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


def plot_violinplot_race(df, title, output_filename=None):
    data = [
        df[df['difficulty'] == 0]['predicted_difficulty'],
        df[df['difficulty'] == 1]['predicted_difficulty'],
        df[df['difficulty'] == 2]['predicted_difficulty']
    ]
    m, b = np.polyfit(df['difficulty'], df['predicted_difficulty'], 1)

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
    plt.show()
    # plt.savefig(f'output_figures/distribution_estimated_difficulty_race_pp_{output_filename}.pdf')
    # plt.close(fig)


def plot_violinplot_arc(df: pd.DataFrame, title: str, output_filename=None):
    data = [df[df['difficulty'] == int(idx)]['predicted_difficulty'] for idx in range(3, 10)]
    m, b = np.polyfit(df['difficulty'], df['predicted_difficulty'], 1)

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
    plt.show()
    # plt.savefig(f'output_figures/distribution_estimated_difficulty_arc_{output_filename}.pdf')
    # plt.close(fig)


def plot_hexbin_am(df, title, output_filename=None):
    x = df['difficulty'].values
    y = df['predicted_difficulty'].values
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.scatter(x, y, alpha=0.2)
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
    plt.show()
    # plt.savefig(f'output_figures/distribution_estimated_difficulty_am_{output_filename}.pdf')
    # plt.close(fig)
