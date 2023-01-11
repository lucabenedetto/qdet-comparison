from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 24

cs = ['tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']

LING_RF = 'Ling.'
READ_RF = 'Read.'
W2V_QA = r'W2V $Q_A$'
R2DE_QC = r'TF-IDF $Q_C$'
DISTILBERT_QA = 'DistilBERT $Q_A$'
BERT_QA = 'BERT $Q_A$'

X = [0, 1, 2, 3]


def plot_metrics_per_different_sizes(
        dict_metrics_mean: Dict[str, List[float]],
        dict_metrics_std: Dict[str, List[float]],
        metric_name: str,
        output_filename: str = None,
):
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, key in enumerate(dict_metrics_mean.keys()):
        ax.plot(X[:3], dict_metrics_mean[key][:3], label=key, c=cs[idx])
        ax.plot(X[2:], dict_metrics_mean[key][2:], '--', c=cs[idx])
        ax.fill_between(
            X,
            [dict_metrics_mean[key][i] + dict_metrics_std[key][i] for i in range(4)],
            [dict_metrics_mean[key][i] - dict_metrics_std[key][i] for i in range(4)],
            color=cs[idx],
            alpha=0.2
        )
        ax.scatter(X, dict_metrics_mean[key], c=cs[idx])
    ax.legend()
    ax.grid(c='k', alpha=0.25)
    ax.set_ylabel(f"{metric_name} ($\mu \pm \sigma$)")
    ax.set_xlabel("Dataset size")
    ax.set_xticks(X)
    # ax.set_xticklabels(['RACE++ 4k', 'RACE++ 8k', 'RACE++ 12k', 'RACE++'])
    ax.set_xticklabels(['R.4k', 'R.8k', 'R.12k', 'RACE++'])
    if output_filename is not None:
        plt.savefig(os.path.join('output_figures', output_filename))
        plt.close(fig)
    else:
        plt.show()


def main():
    # these results are copied from the file created by the script "get metrics dataframe"
    results_rmse = {LING_RF:        [0.541, 0.539, 0.531, 0.471],
                    W2V_QA:         [0.545, 0.532, 0.528, 0.507],
                    R2DE_QC:        [0.547, 0.555, 0.547, 0.508],
                    DISTILBERT_QA:  [0.480, 0.474, 0.468, 0.381],
                    BERT_QA:        [0.447, 0.435, 0.407, 0.372]}
    results_rmse_std = {LING_RF:        [0.006, 0.003, 0.006, 0.004],
                        W2V_QA:         [0.003, 0.005, 0.003, 0.005],
                        R2DE_QC:        [0.002, 0.006, 0.007, 0.005],
                        DISTILBERT_QA:  [0.020, 0.013, 0.005, 0.009],
                        BERT_QA:        [0.003, 0.003, 0.002, 0.011]}
    plot_metrics_per_different_sizes(results_rmse, results_rmse_std, 'RMSE', 'metrics_per_different_train_size_rmse.pdf')

    results_r2 = {LING_RF:        [0.193, 0.200, 0.224, 0.388],
                  W2V_QA:         [0.183, 0.221, 0.232, 0.291],
                  R2DE_QC:        [0.175, 0.151, 0.174, 0.290],
                  DISTILBERT_QA:  [0.363, 0.381, 0.398, 0.600],
                  BERT_QA:        [0.479, 0.450, 0.543, 0.619]}
    results_r2_std = {LING_RF:        [0.018, 0.009, 0.017, 0.011],
                      W2V_QA:         [0.010, 0.015, 0.009, 0.013],
                      R2DE_QC:        [0.007, 0.006, 0.020, 0.015],
                      DISTILBERT_QA:  [0.056, 0.034, 0.012, 0.019],
                      BERT_QA:        [0.008, 0.010, 0.004, 0.028]}
    plot_metrics_per_different_sizes(results_r2, results_r2_std, 'R2', 'metrics_per_different_train_size_r2.pdf')

    results_rho = {LING_RF:        [0.626, 0.629, 0.642, 0.653],
                   W2V_QA:         [0.596, 0.611, 0.613, 0.580],
                   R2DE_QC:        [0.576, 0.560, 0.574, 0.585],
                   DISTILBERT_QA:  [0.707, 0.717, 0.723, 0.790],
                   BERT_QA:        [0.737, 0.757, 0.770, 0.789]}
    results_rho_std = {LING_RF:        [0.005, 0.003, 0.006, 0.005],
                       W2V_QA:         [0.005, 0.005, 0.003, 0.008],
                       R2DE_QC:        [0.002, 0.014, 0.009, 0.010],
                       DISTILBERT_QA:  [0.018, 0.016, 0.004, 0.007],
                       BERT_QA:        [0.005, 0.005, 0.006, 0.014]}
    plot_metrics_per_different_sizes(results_rho, results_rho_std, r"Spearman's $\rho$", 'metrics_per_different_train_size_rho.pdf')


if __name__ == '__main__':
    main()
