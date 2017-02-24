__author__ = 'Martin'

import custom_models
from sklearn import decomposition, feature_selection, svm, linear_model, naive_bayes, tree, discriminant_analysis, neural_network
import json

model_names = {
    "PCA":          custom_models.make_transformer(decomposition.PCA),
    "kBest":        custom_models.make_transformer(feature_selection.SelectKBest),
    "kMeans":       custom_models.make_transformer(custom_models.KMeansSplitter),
    "copy":         [],
    "SVC":          custom_models.make_predictor(svm.SVC),
    "logR":         custom_models.make_predictor(linear_model.LogisticRegression),
    "Perceptron":   custom_models.make_predictor(linear_model.Perceptron),
    "SGD":          custom_models.make_predictor(linear_model.SGDClassifier),
    "PAC":          custom_models.make_predictor(linear_model.PassiveAggressiveClassifier),
    "LDA":          custom_models.make_predictor(discriminant_analysis.LinearDiscriminantAnalysis),
    "QDA":          custom_models.make_predictor(discriminant_analysis.QuadraticDiscriminantAnalysis),
    "MLP":          custom_models.make_predictor(neural_network.MLPClassifier),
    "gaussianNB":   custom_models.make_predictor(naive_bayes.GaussianNB),
    "DT":           custom_models.make_predictor(tree.DecisionTreeClassifier),
    "union":        custom_models.Voter,
    "vote":         custom_models.Voter,
    "stacker":      custom_models.Stacker,
    "booster":      custom_models.Booster
}


def create_param_set():
    """
    Creates the set of parameters for all the methods supported by the dag evaluator.
    :param num_features: The number of features in the dataset.
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
        'Perceptron': {
            'penalty': ['None', 'l2', 'l1', 'elasticnet'], # nebo none? u perc je del doc None, u SGD str 'none'
            'n_iter': [1, 2, 5, 10, 100],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'SGD': {
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'n_iter': [5, 10, 100],
            'alpha': [0.0001, 0.001, 0.01],
            'l1_ratio': [0, 0.15, 0.5, 1],
            'epsilon': [0.01, 0.05, 0.1, 0.5],
            'learning_rate': ['constant', 'optimal'],
            'eta0': [0.01, 0.1, 0.5], # wild guess
            'power_t': [0.1, 0.5, 1, 2] # dtto
        },
        'PAC': {
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15]
        },
        'LDA': {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5, 1.0] # zas to None
        },
        'QDA': {
            'reg_param': [0.0, 0.1, 0.5, 1], # nevim
            'tol': [0.0001, 0.001, 0.01]
        },
        'MLP': {
            'activation': ['identity', 'logistic', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'tol': [0.0001, 0.001, 0.01],
            'max_iter': [10, 100, 200], #200 je default, snad 500 neni moc. ale iteruje to dle tol i max_iter
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'power_t': [0.1, 0.5, 1, 2], # dtto
            'momentum': [0.1, 0.5, 0.9],
            'hidden_layer_sizes': [(100,), (50,), (20,), (10,) ] # co???
        },
        'DT': {
            'criterion': ['gini', 'entropy'],
            'max_features': [0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'max_depth': [1, 2, 5, 10, 15, 25, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
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