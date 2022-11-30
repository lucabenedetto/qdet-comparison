import os
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def evaluate_model(root_string, converted_y_pred_test, converted_y_pred_train, y_true_test, y_true_train, output_dir):

    f1_test = f1_score(y_true_test, converted_y_pred_test, average='weighted')
    bal_acc_test = balanced_accuracy_score(y_true_test, converted_y_pred_test)
    acc_test = accuracy_score(y_true_test, converted_y_pred_test)

    f1_train = f1_score(y_true_train, converted_y_pred_train, average='weighted')
    bal_acc_train = balanced_accuracy_score(y_true_train, converted_y_pred_train)
    acc_train = accuracy_score(y_true_train, converted_y_pred_train)

    out_df = pd.DataFrame([{
        'train_f1': acc_train, 'train_balanced_acc': acc_train, 'train_acc': acc_train,
        'test_f1': acc_test, 'test_balanced_acc': acc_test, 'test_acc': acc_test,
    }])
    out_df.to_csv(os.path.join(output_dir, 'eval_metrics_' + root_string + '.csv'), index=False)

    print("f1 test", f1_test)
    print("balanced accuracy test", bal_acc_test)
    print("accuracy test", acc_test)

    print("f1 train", f1_train)
    print("balanced accuracy train", bal_acc_train)
    print("accuracy train", acc_train)
