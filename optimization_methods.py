from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import cma
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred, squared=False))  # RMSE
    return np.mean(scores)

def random_search(model, X, y, n_iter=10, n_splits=5):
    param_dist = {
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 50),
        'min_samples_split': uniform(0.01, 0.5),
        'min_samples_leaf': uniform(0.01, 0.5)
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
        scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def random_search(model, X, y, n_iter=10, n_splits=5):
    param_dist = {
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 50),
        'min_samples_split': uniform(0.01, 0.5),
        'min_samples_leaf': uniform(0.01, 0.5)
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
        scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_

def bayesian_optimization_gp(model, X, y, n_iter=10, n_splits=5):
    param_space = {
        'n_estimators': Integer(10, 200),
        'max_depth': Integer(1, 50),
        'min_samples_split': Real(0.01, 0.5),
        'min_samples_leaf': Real(0.01, 0.5)
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = BayesSearchCV(
        model, search_spaces=param_space, n_iter=n_iter, cv=tscv,
        scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_

class SklearnWorker(Worker):
    def __init__(self, model, X, y, n_splits=5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.X = X
        self.y = y
        self.n_splits = n_splits

    def compute(self, config, budget, **kwargs):
        model = self.model.__class__(**config)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(mean_squared_error(y_test, y_pred, squared=False))
        return {'loss': np.mean(scores), 'info': {}}

def bohb_kde(model, X, y, n_iter=10, n_splits=5):
    config_space = CS.ConfigurationSpace()
    config_space.add([
        CS.UniformIntegerHyperparameter('n_estimators', lower=10, upper=200),
        CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=50),
        CS.UniformFloatHyperparameter('min_samples_split', lower=0.01, upper=0.5),
        CS.UniformFloatHyperparameter('min_samples_leaf', lower=0.01, upper=0.5)
    ])
    worker = SklearnWorker(model=model, X=X, y=y, n_splits=n_splits)
    ns = hpns.NameServer(run_id='bohb', host='localhost', port=0)
    ns.start()
    bohb = BOHB(configspace=config_space, run_id='bohb', min_budget=1, max_budget=10, eta=3)
    result = bohb.run(n_iterations=n_iter, min_n_workers=1)
    ns.shutdown()
    idx = result.get_incumbent_id()
    best_config = result.get_incumbent_trajectory().configs[-1]
    best_model = model.__class__(**best_config)
    best_model.fit(X, y)
    best_score = evaluate_model(best_model, X, y, n_splits)
    return best_model, best_config, best_score


def cma_es_optimization(model, X, y, n_iter=10, n_splits=5):
    def objective(params):
        params_dict = {
            'n_estimators': int(params[0]),
            'max_depth': int(params[1]),
            'min_samples_split': params[2],
            'min_samples_leaf': params[3]
        }
        model.set_params(**params_dict)
        return evaluate_model(model, X, y, n_splits)

    # Bounds: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    bounds = [(10, 200), (1, 50), (0.01, 0.5), (0.01, 0.5)]
    x0 = [100, 25, 0.1, 0.1]  # Initial guess
    best_params, _ = cma.fmin(
        objective, x0, sigma0=0.5, options={'bounds': list(zip(*bounds)), 'maxiter': n_iter}
    )
    best_params_dict = {
        'n_estimators': int(best_params[0]),
        'max_depth': int(best_params[1]),
        'min_samples_split': best_params[2],
        'min_samples_leaf': best_params[3]
    }
    best_model = model.__class__(**best_params_dict)
    best_model.fit(X, y)
    best_score = evaluate_model(best_model, X, y, n_splits)
    return best_model, best_params_dict, best_score