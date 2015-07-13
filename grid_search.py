__author__ = 'Martin'

from sklearn import svm, linear_model, naive_bayes, tree, grid_search, metrics, preprocessing, cross_validation
import pandas as pd
import ml_metrics as mm
import numpy as np

def eval_model(params, features, targets, m_id):

    model = svm.SVC()

    scorer = metrics.make_scorer(mm.quadratic_weighted_kappa)
    #scorer = metrics.make_scorer(metrics.accuracy_score)
    model.set_params(**params)
    scores = cross_validation.cross_val_score(model, features, targets, cv=5, scoring=scorer)
    return np.mean(scores), np.std(scores), params, m_id

from scoop import futures
import sys

if __name__ == '__main__':

    data = pd.read_csv('data/magic.csv', sep=';')
    features = data[data.columns[:-1]]
    targets = data[data.columns[-1]]

    print(np.unique(targets))
    targets = preprocessing.LabelEncoder().fit_transform(targets)

    print(np.bincount(targets))
    param_grid = [
        {
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'gamma': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5],
            'tol': [0.0001, 0.001, 0.01]
        }
    ]

    p_grid = list(grid_search.ParameterGrid(param_grid))

    print("Starting...", len(p_grid))

    sys.stdout.flush()

    from pprint import pprint

    results = []
    for r in futures.map_as_completed(lambda x: eval_model(x[1], features, targets, x[0]), enumerate(p_grid)):
        pprint(r, width=120)
        sys.stdout.flush()
        results.append(r)

    sor_res = sorted(results, key=lambda res: res[0], reverse=True)

    print('Best Score:', sor_res[0][0])
    print('Best Estimator', sor_res[0][2])

