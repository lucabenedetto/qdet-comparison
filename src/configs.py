from scipy.stats import randint
from text2props.constants import DIFFICULTY


def get_dict_params_by_config(config):
    regressor_config = config.split('__')[1]
    if regressor_config == LR:
        return {DIFFICULTY: [{}]}
    elif regressor_config == RF:
        return {DIFFICULTY: [{'regressor__n_estimators': randint(20, 200), 'regressor__max_depth': randint(2, 50)}]}
    else:
        raise NotImplementedError


def get_config(feature_engineering_config, regression_config):
    assert feature_engineering_config in FEATURE_ENGINEERING_CONFIGS
    assert regression_config in REGRESSION_CONFIGS
    return feature_engineering_config + '__' + regression_config


# FEAT ENG. PARTS
LING = 'ling'
READ = 'read'
W2V_Q_ONLY = 'w2v_q_only'
W2V_Q_ALL = 'w2v_q_all'
W2V_Q_CORRECT = 'w2v_q_correct'
# below the hybrid approaches
LING_AND_READ = 'ling_and_read'
LING_AND_READ_AND_R2DE = 'ling_and_read_and_r2de'  # todo this needs the R2DE params from the run with only that model
W2V_Q_ONLY_AND_LING = 'w2v_q_only_and_ling'
W2V_Q_ALL_AND_LING = 'w2v_q_all_and_ling'
W2V_Q_CORRECT_AND_LING = 'w2v_q_correct_and_ling'
W2V_Q_ONLY_AND_LING_AND_R2DE = 'w2v_q_only_and_ling_and_r2de'    # todo this needs the trained R2DE params
W2V_Q_ALL_AND_LING_AND_R2DE = 'w2v_q_all_and_ling_and_r2de'    # todo this needs the trained R2DE params
W2V_Q_CORRECT_AND_LING_AND_R2DE = 'w2v_q_correct_and_ling_and_r2de'  # todo this needs the trained R2DE params
FEATURE_ENGINEERING_CONFIGS = {
    LING, READ, W2V_Q_ONLY, W2V_Q_ALL, W2V_Q_CORRECT,
    LING_AND_READ, LING_AND_READ_AND_R2DE,
    W2V_Q_ONLY_AND_LING, W2V_Q_ALL_AND_LING, W2V_Q_CORRECT_AND_LING,
    W2V_Q_ONLY_AND_LING_AND_R2DE, W2V_Q_ALL_AND_LING_AND_R2DE, W2V_Q_CORRECT_AND_LING_AND_R2DE
}
# below not implemented hybrid approaches
# LING_AND_R2DE = 'ling_and_r2de'
# READ_AND_R2DE = 'read_and_r2de'

# BASELINES
# # MAJORITY = 'majority'  # TODO with different code
# # R2DE_Q_ALL = 'r2de_q_all'    # TODO with different code
# # R2DE_Q_CORRECT = 'r2de_q_correct'    # TODO with different code
# # R2DE_Q_ONLY = 'r2de_q_only'    # TODO with different code

# Regression elements
LR = 'LR'
RF = 'RF'
REGRESSION_CONFIGS = {LR, RF}
