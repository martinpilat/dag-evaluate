__author__ = 'Martin'

from sklearn import svm, linear_model, naive_bayes, tree, grid_search, metrics, preprocessing, cross_validation
import pandas as pd
import ml_metrics as mm
import numpy as np

def eval_model(params, features, targets, m_id):

    model = tree.DecisionTreeClassifier()

    scorer = metrics.make_scorer(mm.quadratic_weighted_kappa)

    model.set_params(**params)
    scores = cross_validation.cross_val_score(model, features, targets, cv=5, scoring=scorer)
    return np.mean(scores), np.std(scores), params, m_id

from scoop import futures
import sys

if __name__ == '__main__':

    data = pd.read_csv('wilt.csv', sep=';')
    features = data[data.columns[:-1]]
    targets = data[data.columns[-1]]

    targets = preprocessing.LabelEncoder().fit_transform(targets)

    param_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'auto'],
            'max_depth': [6, 7, 8, 9, 10],
            'min_samples_leaf': [2, 4, 5, 6, 7, 10],
            'min_samples_split': [10, 15, 20, 25, 30]
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

