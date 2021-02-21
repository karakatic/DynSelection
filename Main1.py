import json
import time
import timeit
import warnings

import numpy as np
from DynFeatureSelection import DynFeatureSelection
from EvoFeatureSelection import EvoFeatureSelection
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.basic import *
from imblearn.datasets import fetch_datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action='ignore')

# This is for comparison between EvoFS and DynFS on multiple datasets and multiple nia

if __name__ == '__main__':
    random_seed = 1000

    # Default settings
    cut_type = 'diff'
    cut_perc = 0.1
    cut_intervals = (512, 256, 128, 64)
    continuation = True

    # Nia to be used in the experiment
    evos = [GreyWolfOptimizer, SelfAdaptiveDifferentialEvolution, GeneticAlgorithm, EvolutionStrategyMpL,
            ParticleSwarmAlgorithm]

    # Datasets
    dataset_names = ['libras_move', 'spectrometer', 'optical_digits', 'oil', 'ozone_level', 'arrhythmia',
                     'us_crime', 'yeast_ml8']
    datasets = fetch_datasets(verbose=True)

    with open('./results/results_all5.csv', 'w') as f:
        print('Algorithm,Dataset,Fold,Accuracy,Fscore,TrainingTime,NoFeatures,Solution')
        print('Algorithm,Dataset,Fold,Accuracy,Fscore,TrainingTime,NoFeatures,Solution', file=f)

        # For each dataset
        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]
            scaler = MinMaxScaler()  # Scale it
            dataset.data = scaler.fit_transform(dataset.data)
            skf = StratifiedKFold(n_splits=10, random_state=random_seed)  # Make CV split
            X, y = dataset.data, dataset.target

            fold_i = 0
            for train_index, test_index in skf.split(X, y):  # For every split
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
                    f'CART,{dataset_name},{fold_i},{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test.shape[1]}')
                print(
                    f'CART,{dataset_name},{fold_i},{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test.shape[1]}',
                    file=f)

                # For every nia
                for evo in evos:
                    # === EvoFS ===
                    # Train
                    fs = EvoFeatureSelection(n_folds=2,
                                             random_seed=random_seed,
                                             n_runs=10,
                                             optimizer=SelfAdaptiveDifferentialEvolution,
                                             n_jobs=None,
                                             optimizer_settings={'nGEN': 960},
                                             evaluator=DecisionTreeClassifier(random_state=random_seed))

                    tic = timeit.default_timer()  # Time the optimization process
                    X_train_fs = fs.fit_transform(X_train, y_train)  # Select features
                    cls = DecisionTreeClassifier(random_state=random_seed)
                    cls.fit(X_train_fs, y_train)  # Train the final classifier
                    toc = timeit.default_timer()

                    # Test
                    X_test_fs = fs.transform(X_test)  # Select features
                    predicted = cls.predict(X_test_fs)  # Classify testing instances
                    print(
                        f'EvoFS({evo}),{dataset_name},{fold_i},{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}')
                    print(
                        f'EvoFS({evo}),{dataset_name},{fold_i},{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}',
                        file=f)

                    # === DynFS ===
                    # Train
                    fs = DynFeatureSelection(n_folds=2,
                                             random_seed=random_seed,
                                             n_runs=10,
                                             optimizer=evo,
                                             n_jobs=None,
                                             optimizer_settings={},
                                             evaluator=DecisionTreeClassifier(random_state=random_seed),
                                             continue_opt=continuation,
                                             cutting_perc=cut_perc,
                                             nGENs=cut_intervals,
                                             cut_type=cut_type)
                    tic = timeit.default_timer()  # Time the optimization process
                    X_train_fs = fs.fit_transform(X_train, y_train)  # Select features
                    cls = DecisionTreeClassifier(random_state=random_seed)
                    cls.fit(X_train_fs, y_train)  # Train the final classifier
                    toc = timeit.default_timer()

                    # Test
                    X_test_fs = fs.transform(X_test)  # Select features
                    predicted = cls.predict(X_test_fs)  # Classify testing instances
                    print(
                        f'DynFS({evo}),{dataset_name},{fold_i},{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]}')
                    print(
                        f'DynFS({evo}),{dataset_name},{fold_i},{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)},{toc - tic},{X_test_fs.shape[1]},{np.array2string(fs.scores_.astype(int), precision=0, separator=";", suppress_small=True, max_line_width=9999)}',
                        file=f)

                fold_i = fold_i + 1

    print('Finished')
