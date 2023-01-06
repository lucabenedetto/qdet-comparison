import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 24

# the lists of metrics below do not include any of the hybrid models

cs = [
    # 'tab:olive',
    'tab:cyan',
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
]

LING_RF = 'Ling.'
READ_RF = 'Read.'
W2V_QA = r'W2V $Q_A$'
R2DE_QC = r'TF-IDF $Q_C$'
DISTILBERT_QA = 'DistilBERT $Q_A$'
BERT_QA = 'BERT $Q_A$'

x = [0, 1, 2, 3]

models = [LING_RF, W2V_QA, R2DE_QC, DISTILBERT_QA, BERT_QA]

# RMSE
results_rmse = {
    LING_RF:        [0.541, 0.539, 0.531, 0.471],
    LING_RF+'_unc': [0.006, 0.003, 0.006, 0.004],
    W2V_QA:         [0.545, 0.532, 0.528, 0.507],
    W2V_QA+'_unc':  [0.003, 0.005, 0.003, 0.005],
    R2DE_QC:        [0.547, 0.555, 0.547, 0.508],
    R2DE_QC+'_unc': [0.002, 0.006, 0.007, 0.005],
    DISTILBERT_QA:  [0.480, 0.474, 0.468, 0.381],
    DISTILBERT_QA+'_unc':   [0.020, 0.013, 0.005, 0.009],
    BERT_QA:        [0.447, 0.435, 0.407, 0.372],
    BERT_QA+'_unc': [0.003, 0.003, 0.002, 0.011],  # stddev of BERT to check
}

fig, ax = plt.subplots(figsize=(12, 8))
for idx, key in enumerate(models):
    ax.plot(x[:3], results_rmse[key][:3], label=key, c=cs[idx])
    ax.plot(x[2:], results_rmse[key][2:], '--', c=cs[idx])
    ax.fill_between(
        x,
        [results_rmse[key][i]+results_rmse[key+'_unc'][i] for i in range(4)],
        [results_rmse[key][i]-results_rmse[key+'_unc'][i] for i in range(4)],
        color=cs[idx],
        alpha=0.2
    )
    ax.scatter(x, results_rmse[key], c=cs[idx])
ax.legend()
ax.grid(c='k', alpha=0.25)
ax.set_ylabel("RMSE ($\mu \pm \sigma$)")
ax.set_xlabel("Dataset size")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['RACE++ 4k', 'RACE++ 8k', 'RACE++ 12k', 'RACE++'])
# plt.show()
plt.savefig('output_figures/metrics_per_different_train_size_rmse.pdf')
plt.close(fig)

#################################################################################################

# R2
results_r2 = {
    LING_RF:        [0.193, 0.200, 0.224, 0.388],
    LING_RF+'_unc': [0.018, 0.009, 0.017, 0.011],
    W2V_QA:         [0.183, 0.221, 0.232, 0.291],
    W2V_QA+'_unc':  [0.010, 0.015, 0.009, 0.013],
    R2DE_QC:        [0.175, 0.151, 0.174, 0.290],
    R2DE_QC+'_unc': [0.007, 0.006, 0.020, 0.015],
    DISTILBERT_QA:  [0.363, 0.381, 0.398, 0.600],
    DISTILBERT_QA+'_unc':   [0.056, 0.034, 0.012, 0.019],
    BERT_QA:        [0.479, 0.450, 0.543, 0.619],
    BERT_QA+'_unc': [0.008, 0.010, 0.004, 0.028],  # stddev of BERT to check
}

fig, ax = plt.subplots(figsize=(12, 8))
for idx, key in enumerate(models):
    ax.plot(x[:3], results_r2[key][:3], label=key, c=cs[idx])
    ax.plot(x[2:], results_r2[key][2:], '--', c=cs[idx])
    ax.fill_between(
        x,
        [results_r2[key][i]+results_r2[key+'_unc'][i] for i in range(4)],
        [results_r2[key][i]-results_r2[key+'_unc'][i] for i in range(4)],
        color=cs[idx],
        alpha=0.2
    )
    ax.scatter(x, results_r2[key], c=cs[idx])
ax.legend()
ax.grid(c='k', alpha=0.25)
ax.set_ylabel("R2 ($\mu \pm \sigma$)")
ax.set_xlabel("Dataset size")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['RACE++ 4k', 'RACE++ 8k', 'RACE++ 12k', 'RACE++'])
# plt.show()
plt.savefig('output_figures/metrics_per_different_train_size_r2.pdf')
plt.close(fig)

#################################################################################################

# spearman's rho
results_rho = {
    LING_RF:        [0.626, 0.629, 0.642, 0.653],
    LING_RF+'_unc': [0.005, 0.003, 0.006, 0.005],
    W2V_QA:         [0.596, 0.611, 0.613, 0.580],
    W2V_QA+'_unc':  [0.005, 0.005, 0.003, 0.008],
    R2DE_QC:        [0.576, 0.560, 0.574, 0.585],
    R2DE_QC+'_unc': [0.002, 0.014, 0.009, 0.010],
    DISTILBERT_QA:  [0.707, 0.717, 0.723, 0.790],
    DISTILBERT_QA+'_unc':   [0.018, 0.016, 0.004, 0.007],
    BERT_QA:        [0.737, 0.757, 0.770, 0.789],
    BERT_QA+'_unc': [0.005, 0.005, 0.006, 0.014],  # stddev of BERT to check
}

fig, ax = plt.subplots(figsize=(12, 8))
for idx, key in enumerate(models):
    ax.plot(x[:3], results_rho[key][:3], label=key, c=cs[idx])
    ax.plot(x[2:], results_rho[key][2:], '--', c=cs[idx])
    ax.fill_between(
        x,
        [results_rho[key][i]+results_rho[key+'_unc'][i] for i in range(4)],
        [results_rho[key][i]-results_rho[key+'_unc'][i] for i in range(4)],
        color=cs[idx],
        alpha=0.2
    )
    ax.scatter(x, results_rho[key], c=cs[idx])
ax.legend()
ax.grid(c='k', alpha=0.25)
ax.set_ylabel(r"Spearman's $\rho$ ($\mu \pm \sigma$)")
ax.set_xlabel("Dataset size")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['RACE++ 4k', 'RACE++ 8k', 'RACE++ 12k', 'RACE++'])
# plt.show()
plt.savefig('output_figures/metrics_per_different_train_size_rho.pdf')
plt.close(fig)
