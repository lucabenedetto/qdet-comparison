import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_violinplot_race(data, title, m=None, b=None):
    fig, ax = plt.subplots()
    sns.violinplot(data, color='#c41331')
    ax.plot([-0.5, 2.5], [0.5, 0.5], c='k', alpha=0.25)
    ax.plot([-0.5, 2.5], [1.5, 1.5], c='k', alpha=0.25)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Predicted difficulty')
    ax.set_xticks([0, 1, 2])
    ax.set_title(title)
    if m and b:
        ax.plot([-0.5, 2.5], [-0.5*m + b, 2.5*m + b], c='darkblue', alpha=0.5)
    plt.show()


def plot_violinplot_arc(data, title, m=None, b=None):
    fig, ax = plt.subplots()
    sns.violinplot(data, color='#223266')
    # ax.plot([-0.5, 2.5], [0.5, 0.5], c='k', alpha=0.25)
    # ax.plot([-0.5, 2.5], [1.5, 1.5], c='k', alpha=0.25)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Predicted difficulty')
    ax.set_title(title)
    # ax.set_xticks([3, 4, 5, 6, 7, 8, 9])
    if m and b:
        ax.plot([-0.5, 6.5], [-0.5*m + b, 6.5*m + b], c='green', alpha=0.5)
    plt.show()


def plot_hexbin_am(df, title):
    x = df['difficulty'].values
    y = df['predicted_difficulty'].values
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots()
    # ax.scatter(x, y, alpha=0.2)
    ax.hexbin(x, y, gridsize=(50, 50), cmap='Greens')
    plt.plot(x, m * x + b, c='red', alpha=0.5)
    # ax.set_ylim([-5, 5])
    # ax.set_xlim([-5, 5])
    ax.set_title(title)
    plt.show()