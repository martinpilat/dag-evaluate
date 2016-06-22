__author__ = 'Martin'

import custom_models
from sklearn import decomposition, feature_selection, svm, linear_model, naive_bayes, tree
import json

model_names = {
    "PCA":          custom_models.make_transformer(decomposition.PCA),
    "kBest":        custom_models.make_transformer(feature_selection.SelectKBest),
    "kMeans":       custom_models.make_transformer(custom_models.KMeansSplitter),
    "copy":         [],
    "SVC":          custom_models.make_predictor(svm.SVC),
    "logR":         custom_models.make_predictor(linear_model.LogisticRegression),
    "gaussianNB":   custom_models.make_predictor(naive_bayes.GaussianNB),
    "DT":           custom_models.make_predictor(tree.DecisionTreeClassifier),
    "union":        custom_models.Voter,
    "vote":         custom_models.Voter,
    "stacker":      custom_models.Stacker,
    "booster":      custom_models.Booster
}


def create_param_set(num_features, num_instances):
    """
    Creates the set of parameters for all the methods supported by the dag evaluator.
    :param num_features: The number of features in the dataset.
    :param num_instances: The number of instances in the dataset.
    :return: Dictionary containing dictionaries with lists of values of parameters for each method.
    """

    feat_frac = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    params = {
        'PCA': {
            'feat_frac': feat_frac,
            'whiten': [False, True],
        },
        'kBest': {
            'feat_frac': feat_frac,
        },
        'SVC': {
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'gamma': ['auto', 0.0001, 0.001, 0.01, 0.1, 0.5],
            'tol': [0.0001, 0.001, 0.01]
        },
        'logR': {
            'penalty': ['l2'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'tol': [0.0001, 0.001, 0.01],
            'solver': ['sag']

        },
        'DT': {
            'criterion': ['gini', 'entropy'],
            'max_features': [0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'max_depth': [1, 2, 5, 10, 15, 25, 50, 100],
            'min_samples_split': [1, 2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10, 20]
        },
        'gaussianNB': {},
        'copy': {},
        'kMeans': {},
        'union': {},
        'vote': {}
    }

    return json.dumps(params)

if __name__ == '__main__':
    print(create_param_set(3, 200))
