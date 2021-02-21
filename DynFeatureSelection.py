"""
Class to perform feature selection with evolutionary and nature inspired algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0


import logging
import sys
import time
import time
from multiprocessing import Pool

import EvoSettings as es
import numpy as np
from DynSelectionBenchmark import DynSelectionBenchmark
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.task import StoppingTask, OptimizationType
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

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
                 nGENs=(512, 256, 128, 64, 32, 16, 8),
                 continue_opt=False,
                 cut_type='diff',
                 cutting_perc=0.1):
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

        # DynFS specific settings
        self.continue_opt = continue_opt  # If optimization continues after cutting the genotype
        self.nGENs = nGENs  # List (or tuple) of nGEN before cutting the genotype
        self.cut_type = cut_type  # How to choose which genes to cut
        self.cutting_perc = cutting_perc  # How many genes to cut after every interval (from nGENs)

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
                     self.benchmark, self.optimizer_settings, self.nGENs, self.continue_opt, self.cut_type,
                     self.cutting_perc, j))

        with Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(DynFeatureSelection._run, evos)

        return DynFeatureSelection._reduce(results, self.n_runs, self.n_folds, self.benchmark, X.shape[1])

    @staticmethod
    def _run(X, y, train_index, val_index, random_seed, optimizer, evaluator, benchmark, optimizer_settings, nGENs,
             continue_opt, cut_type, cutting_perc, j):
        opt_settings = es.get_args(optimizer)
        opt_settings.update(optimizer_settings)
        X1 = X
        cuted = []  # Which genes (features) are cutted
        fitnesses = []   # Fitness values after every cutting
        xb, fxb, benchm = None, -1, None
        pop, fpop = None, None
        for nGENp in nGENs:  # Every interval before cutting
            benchm = benchmark(X=X1, y=y,
                               train_indices=train_index, valid_indices=val_index,
                               random_seed=random_seed,
                               evaluator=evaluator)
            task = StoppingTask(D=X1.shape[1],
                                nGEN=nGENp,
                                optType=OptimizationType.MINIMIZATION,
                                benchmark=benchm)

            evo = optimizer(seed=random_seed, **opt_settings)

            # Start new optimization. Continue on pop and give fitness of pop.
            pop, fpop, xb, fxb = DynFeatureSelection.runTask(evo, task, starting_pop=pop, starting_fpop=fpop)

            if not isinstance(xb, np.ndarray):
                xb = xb.x
            xb = np.copy(xb)

            # Cut genotype. Four different strategies
            if cut_type == 'diff':
                # Best solutions say which features are most common, and worst solutions vote which features are most common.
                # Difference between votes say which features will be cutted - those that are more coomon in worst solutions and
                # least common in best features.
                idx = DynFeatureSelection.cut_n_vote_diff(pop, fpop, cutting_perc, benchm, n=25)
            elif cut_type == 'vote_all':
                # Every solution votes which features are most common. Similar than best_vote worst, but every solution votes.
                idx = DynFeatureSelection.cut_all_vote_for_worst(pop, fpop, cutting_perc, benchm)
            elif cut_type == 'best_vote_worst':
                # Best solutions say which features are most common. Least common features are cutted.
                idx = DynFeatureSelection.cut_n_vote(pop, fpop, cutting_perc, benchm, n=50)
            elif cut_type == 'worst_vote_best':
                # Worst solutions say which features are most common. Most common features are cutted.
                idx = DynFeatureSelection.cut_n_vote(pop, fpop, cutting_perc, benchm, n=-50)
            cuted.append(idx)  # Log which genes are cutted
            fitnesses.append(fxb)  # Log fitenss value after the cutting

            X1 = np.delete(X1, idx, axis=1)  # Delete columns (feature) from genotypes

            # If we want the optimization to continue on genes not cutted
            if continue_opt:
                if isinstance(pop[0], np.ndarray):  # If population is in shape of ndarray
                    pop = np.delete(pop, idx, 1)
                else:  # If population is in shape of individuals
                    for ind in pop:
                        ind.x = np.delete(ind.x, idx)
            else:  # If we want that after cutting, the solutions are reseted (random reinitialization)
                pop = None
                fpop = None

        # Transform solutions to datasets with selected features
        xb = benchmark.to_phenotype(xb, benchm.split)

        # Remove not selected features
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
        if np.sum(features) == 0:
            features[0] = 1

        return features

    @staticmethod
    def runTask(nia, task, starting_pop=None, starting_fpop=None):
        if starting_pop is not None:
            # IF we give starting population, we want to optimization to continue on given population
            pop = starting_pop
            fpop = starting_fpop
            _, _, dparams = nia.initPopulation(task)
        else:
            # If we do not give starting_pop, make new random population
            pop, fpop, dparams = nia.initPopulation(task)

        xb, fxb = nia.getBest(pop, fpop)

        while not task.stopCond():
            pop, fpop, xb, fxb, dparams = nia.runIteration(task, pop, fpop, xb, fxb, **dparams)
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
        k = int(len(feature_fitnesses[0]) * perc)
        fitness_reversed = np.amax(feature_fitnesses) - feature_fitnesses
        try:
            idx = np.argpartition(fitness_reversed, range(k))[:, :k]
        except:
            print('Napaka')

        return idx[0]

    # If n > 0, n best vote
    # If n < 0, n worst vote
    @staticmethod
    def cut_n_vote(pop, fpop, perc, benchm, n=50):
        pop_b, fpop = DynFeatureSelection.get_population(pop, fpop, benchm)

        if n >= 0:  # n best
            fpop_sorted_idx = np.argsort(fpop)[:n]
        else:  # n worst
            fpop_sorted_idx = np.argsort(fpop)[len(fpop)-n:]

        feature_fitnesses = np.sum(pop_b[fpop_sorted_idx], axis=0)
        k = int(len(feature_fitnesses) * perc)

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
        fpop_worst_idx = np.argsort(fpop)[len(fpop)-n:]

        feature_fitnesses_best = np.sum(pop_b[fpop_best_idx], axis=0)  # Which features are common in best
        feature_fitnesses_worst = np.sum(pop_b[fpop_worst_idx], axis=0)  # Which features are common in worst

        feature_fitnesses = feature_fitnesses_best - feature_fitnesses_worst  # Diff between common in best and worst
        # The lowest numbers will be the ones, which are not common in best, but are common in worst
        k = int(len(feature_fitnesses) * perc)
        try:
            idx = np.argpartition(feature_fitnesses, range(k))[:k]
        except:
            idx = np.argpartition(feature_fitnesses, range(len(feature_fitnesses)))

        if len(idx) == 0:
            print('nestima')
        #elif len(idx) == pop.shape[1]:
        #    print('nestima1')

        return idx

    @staticmethod
    def get_population(pop, fpop, benchm):
        pop_b = [DynSelectionBenchmark.to_phenotype(ind, benchm.split) for ind in pop]
        pop_b = np.stack(pop_b, axis=0)
        pop_b = pop_b.astype(int)
        fpop[fpop == np.inf] = 1

        return pop_b, fpop
