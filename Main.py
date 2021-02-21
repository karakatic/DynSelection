import json
import time
import timeit
import warnings

import numpy as np
from DynFeatureSelection import DynFeatureSelection
from EvoFeatureSelection import EvoFeatureSelection
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolution
from imblearn.datasets import fetch_datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action='ignore')

# This is for experimenting with setting options

if __name__ == '__main__':
    random_seed = 1000
    dataset_name = 'arrhythmia'  # 'libras_move' #'spectrometer' #'optical_digits'  # 'arrhythmia'
    datasets = fetch_datasets(verbose=True)
    dataset = datasets[dataset_name]
    scaler = MinMaxScaler()
    dataset.data = scaler.fit_transform(dataset.data)

    skf = StratifiedKFold(n_splits=10, random_state=random_seed)
    X, y = dataset.data, dataset.target

    with open('./results/results_all.csv', 'w') as f:
        print('Algorithm,Accuracy,Fscore,TrainingTime,NoFeatures,Solution')
        print('Algorithm,Accuracy,Fscore,TrainingTime,NoFeatures,Solution', file=f)

        fold_i = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            # === Regular ===
            # Train
            tic = timeit.default_timer()
            cls = DecisionTreeClassifier(random_state=random_seed)
            cls.fit(X_train, y_train)
            toc = timeit.default_timer()

            # Test
            predicted = cls.predict(X_test)
            print(
                f'CART,{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test.shape[1]}')
            print(
                f'CART,{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test.shape[1]}',
                file=f)

            # === FS ===
            # Train
            fs = EvoFeatureSelection(n_folds=2,
                                     random_seed=random_seed,
                                     n_runs=10,
                                     optimizer=SelfAdaptiveDifferentialEvolution,
                                     n_jobs=None,
                                     optimizer_settings={'nGEN': 960},
                                     evaluator=DecisionTreeClassifier(random_state=random_seed))

            tic = timeit.default_timer()
            X_train_fs = fs.fit_transform(X_train, y_train)
            cls = DecisionTreeClassifier(random_state=random_seed)
            cls.fit(X_train_fs, y_train)
            toc = timeit.default_timer()

            # Test
            X_test_fs = fs.transform(X_test)
            predicted = cls.predict(X_test_fs)
            print(
                f'EvoFS,{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}')
            print(
                f'EvoFS,{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}',
                file=f)

            # === DynFS ===
            setting_options = ['cut_type', 'cut_perc', 'cut_interval', 'continuation']
            for setting_opt in setting_options:
                if setting_opt == 'cut_type':
                    setting_values = ['diff', 'vote_all', 'best_vote_worst', 'worst_vote_best']
                elif setting_opt == 'cut_perc':
                    setting_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
                elif setting_opt == 'cut_interval':
                    setting_values = [(512, 256, 128, 64), (320, 106, 35, 64), [240] * 4, [120] * 8, [60] * 16]
                elif setting_opt == 'continuation':
                    setting_values = [True, False]

                for setting_val in setting_values:
                    # Settings
                    cut_type = setting_val if setting_opt == 'cut_type' else 'diff'
                    cut_perc = setting_val if setting_opt == 'cut_perc' else 0.1
                    cut_intervals = setting_val if setting_opt == 'cut_interval' else (512, 256, 128, 64)
                    continuation = setting_val if setting_opt == 'continuation' else True

                    # Train
                    fs = DynFeatureSelection(n_folds=2,
                                             random_seed=random_seed,
                                             n_runs=10,
                                             optimizer=SelfAdaptiveDifferentialEvolution,
                                             n_jobs=None,
                                             optimizer_settings={},
                                             evaluator=DecisionTreeClassifier(random_state=random_seed),
                                             continue_opt=continuation,
                                             cutting_perc=cut_perc,
                                             nGENs=cut_intervals,
                                             cut_type=cut_type)
                    tic = timeit.default_timer()
                    X_train_fs = fs.fit_transform(X_train, y_train)
                    cls = DecisionTreeClassifier(random_state=random_seed)
                    cls.fit(X_train_fs, y_train)
                    toc = timeit.default_timer()

                    # Test
                    X_test_fs = fs.transform(X_test)
                    predicted = cls.predict(X_test_fs)
                    print(
                        f'DynFS({setting_opt}={setting_val}),{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}')
                    print(
                        f'DynFS({setting_opt}={setting_val}),{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}',
                        file=f)
    print('Finished')
