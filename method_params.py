__author__ = 'Martin'

import custom_models
from sklearn import decomposition, feature_selection, svm, linear_model, naive_bayes, tree
import numpy as np
import json


def make_transformer(cls):
    """
    Adds Transformer to the bases of the cls class, useful in order to distinguish between transformers and predictors.
    :param cls: The class to turn into a Transformer
    :return: A class equivalent to cls, but with Transformer among its bases
    """
    class Tr(cls, custom_models.Transformer):
        def __repr__(self):
            return cls.__name__ + ":" + cls.__repr__(self)
    return Tr


def make_predictor(cls):
    """
    Adds Predictor to the bases of the cls class, useful in order to distinguish between transformers and predictors.
    :param cls: The class to turn into a Predictor
    :return: A class equivalent to cls, but with Predictor among its bases
    """
    class Pr(cls, custom_models.Predictor):
        def __repr__(self):
            return cls.__name__ + ":" + cls.__repr__(self)
    return Pr

model_names = {
    "PCA":          make_transformer(decomposition.PCA),
    "kBest":        make_transformer(feature_selection.SelectKBest),
    "kMeans":       make_transformer(custom_models.KMeansSplitter),
    "copy":         [],
    "SVC":          make_predictor(svm.SVC),
    "logR":         make_predictor(linear_model.LogisticRegression),
    "gaussianNB":   make_predictor(naive_bayes.GaussianNB),
    "DT":           make_predictor(tree.DecisionTreeClassifier),
    "union":        custom_models.Voter,
    "vote":         custom_models.Voter,
}


def create_param_set(num_features, num_instances):
    """
    Creates the set of parameters for all the methods supported by the dag evaluator.
    :param num_features: The number of features in the dataset.
    :param num_instances: The number of instances in the dataset.
    :return: Dictionary containing dictionaries with lists of values of parameters for each method.
    """
    column_counts = [1, 0.05*num_features, 0.1*num_features, 0.25*num_features,
                     0.5*num_features, 0.75*num_features, num_features]
    column_counts = list(map(int, column_counts))
    column_counts = np.unique(column_counts)
    column_counts = list(column_counts[column_counts > 0])
    column_counts = list(map(int, column_counts))

    params = {
        'PCA': {
            'n_components': column_counts,
            'whiten': [False, True],
        },
        'kBest': {
            'k': column_counts,
        },
        'SVC': {
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'gamma': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5],
            'tol': [0.0001, 0.001, 0.01]
        },
        'logR': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'tol': [0.0001, 0.001, 0.01]
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
