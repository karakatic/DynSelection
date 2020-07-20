"""
Class to perform feature selection with evolutionary and nature inspired algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0


import logging
import sys
import time
from multiprocessing import Pool

import numpy as np
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.task import StoppingTask, OptimizationType
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

import EvoPreprocess.utils.EvoSettings as es
from DynSelectionBenchmark import DynSelectionBenchmark

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class DynFeatureSelection(_BaseFilter):
    """
    Select features from the dataset with evolutionary and nature-inspired methods.

    Parameters
    ----------
    random_seed : int or None, optional (default=None)
        It used as seed by the random number generator.
        If None, the current system time is used for the seed.

    evaluator : classifier or regressor, optional (default=None)
        The classification or regression object from scikit-learn framework.
        If None, the GausianNB for classification is used.

    optimizer : evolutionary or nature-inspired optimization method, optional (default=GeneticAlgorithm)
        The evolutionary or or nature-inspired optimization method from NiaPy framework.

    n_runs : int, optional (default=3)
        The number of runs on each fold. Only the best performing result of all runs is used.

    n_folds : int, optional (default=3)
        The number of folds for cross-validation split into the training and validation sets.

    benchmark : object, optional (default=FeatureSelectionBenchmark)
        The benchmark object with mapping and fitness value calculation.

    n_jobs : int, optional (default=None)
        The number of jobs to run in parallel.
        If None, then the number of jobs is set to the number of cores.

    optimizer_settings : dict, optional (default={})
        Custom settings for the optimizer.
    """

    def __init__(self,
                 random_seed=None,
                 evaluator=None,
                 optimizer=GeneticAlgorithm,
                 n_runs=10,
                 n_folds=2,
                 benchmark=DynSelectionBenchmark,
                 n_jobs=None,
                 optimizer_settings={},
                 nFESs=(512, 256, 128, 64, 32, 16, 8),
                 continue_opt=False,
                 cut_type='diff'):
        super(DynFeatureSelection, self).__init__(self.select)

        self.evaluator = GaussianNB() if evaluator is None else evaluator
        self.random_seed = int(time.time()) if random_seed is None else random_seed
        self.random_state = check_random_state(self.random_seed)
        self.optimizer = optimizer
        self.n_runs = n_runs
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.benchmark = benchmark
        self.optimizer_settings = optimizer_settings
        self.continue_opt = continue_opt
        self.nFESs = nFESs
        self.cut_type = cut_type

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        return self.scores_ > 0

    def select(self, X, y):
        """Selects features from the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be processed with feature selection.

        y : array-like, shape (n_samples)
            Corresponding label for each instance in X.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples, n_features_new)
                The array containing the data with selected features.
        """

        if self.evaluator is ClassifierMixin:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        else:
            skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        evos = []  # Parameters for parallel threaded evolution run

        for train_index, val_index in skf.split(X, y):
            for j in range(self.n_runs):
                evos.append(
                    (X, y, train_index, val_index, self.random_seed + j + 1, self.optimizer, self.evaluator,
                     self.benchmark, self.optimizer_settings, self.nFESs, self.continue_opt, self.cut_type))

        with Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(DynFeatureSelection._run, evos)

        return DynFeatureSelection._reduce(results, self.n_runs, self.n_folds, self.benchmark, X.shape[1])

    @staticmethod
    def _run(X, y, train_index, val_index, random_seed, optimizer, evaluator, benchmark, optimizer_settings, nFESs,
             continue_opt, cut_type):
        opt_settings = es.get_args(optimizer)
        opt_settings.update(optimizer_settings)
        X1 = X
        cuted = []
        xb, fxb, benchm = None, -1, None
        pop, fpop = None, None
        for nFESp in nFESs:
            benchm = benchmark(X=X1, y=y,
                               train_indices=train_index, valid_indices=val_index,
                               random_seed=random_seed,
                               evaluator=evaluator)
            task = StoppingTask(D=X1.shape[1],
                                nFES=nFESp,
                                optType=OptimizationType.MINIMIZATION,
                                benchmark=benchm)

            evo = optimizer(seed=random_seed, **opt_settings)
            if continue_opt:
                pop, fpop, xb, fxb = DynFeatureSelection.runTask(evo, task, starting_pop=pop, starting_fpop=fpop)
            else:
                pop, fpop, xb, fxb = DynFeatureSelection.runTask(evo, task)

            if not isinstance(xb, np.ndarray):
                xb = xb.x
            xb = np.copy(xb)
            if cut_type == 'diff':
                idx = DynFeatureSelection.cut_n_vote_diff(pop, fpop, 0.1, benchm, n=25)
            elif cut_type == 'vote_all':
                idx = DynFeatureSelection.cut_all_vote_for_worst(pop, fpop, 0.1, benchm)
            elif cut_type == 'best_vote_worst':
                idx = DynFeatureSelection.cut_n_vote(pop, fpop, 0.1, benchm, n=50)
            elif cut_type == 'worst_vote_best':
                idx = DynFeatureSelection.cut_n_vote(pop, fpop, 0.1, benchm, n=-50)
            cuted.append(idx)
            X1 = np.delete(X1, idx, axis=1)
            if continue_opt:
                for ind in pop:
                    ind.x = np.delete(ind.x, idx)

        xb = benchmark.to_phenotype(xb, benchm.split)

        for i in range(len(cuted) - 2, -1, -1):
            cut = np.sort(cuted[i])
            for c in cut:
                xb = np.insert(xb, c, False)

        return xb, fxb

    @staticmethod
    def _reduce(results, runs, cv, benchmark, len_y=10):
        features = np.full((len_y, cv), np.nan)  # Columns are number of occurrences in one run

        result_list = [results[x:x + runs] for x in range(0, cv * runs, runs)]
        i = 0
        for cv_one in result_list:
            best_fitness = sys.float_info.max
            best_solution = None
            for result_one in cv_one:
                if (best_solution is None) or (best_fitness > result_one[1]):
                    best_solution, best_fitness = result_one[0], result_one[1]

            features[:, i] = best_solution.astype(int)
            i = i + 1

        features = stats.mode(features, axis=1, nan_policy='omit')[0].flatten()

        return features

    @staticmethod
    def runTask(nia, task, starting_pop=None, starting_fpop=None):
        pop, fpop, dparams = nia.initPopulation(task)
        if starting_pop is not None:
            pop = starting_pop
            fpop = starting_fpop

        xb, fxb = nia.getBest(pop, fpop)

        while not task.stopCond():
            pop, fpop, dparams = nia.runIteration(task, pop, fpop, xb, fxb, **dparams)
            task.nextIter()
            xb1, fxb1 = nia.getBest(pop, fpop)
            if fxb1 < fxb:
                fxb = fxb1
                xb = xb1
        return pop, fpop, xb, fxb

    @staticmethod
    def cut_all_vote_for_worst(pop, fpop, perc, benchm):
        pop_b, fpop = DynFeatureSelection.get_population(pop, fpop, benchm)

        feature_fitnesses = np.matmul(fpop.reshape(1, -1), pop_b)
        k = int(pop.shape[0] * perc)
        fitness_reversed = np.amax(feature_fitnesses) - feature_fitnesses
        idx = np.argpartition(fitness_reversed, range(k))[:, :k]

        return idx[0]

    # If n > 0, n best vote
    # If n < 0, n worst vote
    @staticmethod
    def cut_n_vote(pop, fpop, perc, benchm, n=50):
        pop_b, fpop = DynFeatureSelection.get_population(pop, fpop, benchm)

        if n >= 0:  # n best
            fpop_sorted_idx = np.argsort(fpop)[:n]
        else:  # n worst
            fpop_sorted_idx = np.argsort(fpop)[n:]

        feature_fitnesses = np.sum(pop_b[fpop_sorted_idx], axis=0)
        k = int(pop.shape[0] * perc)

        if n < 0:  # get worst features
            feature_fitnesses = np.amax(feature_fitnesses) - feature_fitnesses

        idx = np.argpartition(feature_fitnesses, range(k))[:k]

        return idx

    @staticmethod
    def cut_n_vote_diff(pop, fpop, perc, benchm, n=25):
        pop_b, fpop = DynFeatureSelection.get_population(pop, fpop, benchm)
        # n best
        fpop_best_idx = np.argsort(fpop)[:n]
        # n worst
        fpop_worst_idx = np.argsort(fpop)[n:]

        feature_fitnesses_best = np.sum(pop_b[fpop_best_idx], axis=0)  # Which features are common in best
        feature_fitnesses_worst = np.sum(pop_b[fpop_worst_idx], axis=0)  # Which features are common in worst

        feature_fitnesses = feature_fitnesses_best - feature_fitnesses_worst  # Diff between common in best and worst
        # The lowest numbers will be the ones, which are not common in best, but are common in worst
        k = int(pop.shape[0] * perc)
        idx = np.argpartition(feature_fitnesses, range(k))[:k]

        return idx

    @staticmethod
    def get_population(pop, fpop, benchm):
        pop_b = [DynSelectionBenchmark.to_phenotype(ind, benchm.split) for ind in pop]
        pop_b = np.stack(pop_b, axis=0)
        pop_b = pop_b.astype(int)
        fpop[fpop == np.inf] = 1

        return pop_b, fpop
