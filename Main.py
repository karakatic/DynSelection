import warnings

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from DynFeatureSelection import DynFeatureSelection

warnings.simplefilter(action='ignore')
from sklearn.tree import DecisionTreeClassifier
from imblearn.datasets import fetch_datasets

if __name__ == '__main__':
    random_seed = 1234
    dataset_name = 'arrhythmia'  # 'libras_move' #'spectrometer' #'optical_digits'  # 'arrhythmia'
    datasets = fetch_datasets(verbose=True)
    dataset = datasets[dataset_name]
    scaler = MinMaxScaler()
    dataset.data = scaler.fit_transform(dataset.data)

    skf = StratifiedKFold(n_splits=5, random_state=random_seed)
    X, y = dataset.data, dataset.target

    fold_i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        cls = DecisionTreeClassifier(random_state=random_seed)
        cls.fit(X_train, y_train)
        predicted = cls.predict(X_test)
        print('Algoritem,Accuracy,Fscore')
        print(f'CART,{accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)}')

        for cut_type in ['diff', 'vote_all', 'best_vote_worst', 'worst_vote_best']:
            fs = DynFeatureSelection(n_folds=2,
                                     random_seed=random_seed,
                                     n_runs=10,
                                     n_jobs=3,
                                     optimizer_settings={'NP': 150, 'Cr': 0.8, 'Mr': 0.5},
                                     evaluator=DecisionTreeClassifier(random_state=random_seed),
                                     continue_opt=True,
                                     nFESs=(512, 256, 128, 64),
                                     cut_type=cut_type)
            X_train_fs = fs.fit_transform(X_train, y_train)

            cls = DecisionTreeClassifier(random_state=random_seed)
            cls.fit(X_train_fs, y_train)

            X_test_fs = fs.transform(X_test)
            predicted = cls.predict(X_test_fs)
            print(f'Feature Selection ({cut_type}), {accuracy_score(y_test, predicted)},{f1_score(y_test, predicted)}')

    print('Konec')
