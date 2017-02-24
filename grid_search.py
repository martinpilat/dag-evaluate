import method_params
import json
import custom_models
import pandas as pd
import ml_metrics as mm
import numpy as np

from sklearn import cross_validation, metrics, preprocessing
from multiprocessing import Pool

def make_grid(param_set, partial_dict):
    if not param_set:
        return [{}]

    if len(param_set) == len(partial_dict):
        return [partial_dict.copy()]

    result = []
    for param_name, param_values in param_set.items():
        if param_name not in partial_dict:
            for value in param_values:
                partial_dict[param_name] = value
                result += make_grid(param_set, partial_dict.copy())
            break
    return result

def test_classifier(clsClass, parameters):
    errors = []
    for train_idx, test_idx in cross_validation.StratifiedKFold(targets, n_folds=5):
        cls = clsClass(**parameters)
        train_data = (feats.iloc[train_idx], targets.iloc[train_idx])
        test_data = (feats.iloc[test_idx], targets.iloc[test_idx])

        cls.fit(train_data[0], train_data[1])
        preds = cls.predict(test_data[0])

        acc = mm.quadratic_weighted_kappa(test_data[1], preds)
        if filename == 'ml-prove.csv':
            acc = metrics.accuracy_score(test_data[1], preds)
        errors.append(acc)

    return errors, parameters

param_set = json.loads(method_params.create_param_set())
filename = 'wilt.csv'

data = pd.read_csv('data/' + filename, sep=';')
feats = data[data.columns[:-1]]
targets = data[data.columns[-1]]
le = preprocessing.LabelEncoder()

ix = targets.index
targets = pd.Series(le.fit_transform(targets), index=ix)

best_results = {}

for model_name in param_set:
    if model_name == 'SGD' or model_name == 'MLP':
        continue
    clsClass = method_params.model_names[model_name]
    if custom_models.is_predictor(clsClass):
        best_results[model_name] = (-100, [], [])
        grid = make_grid(param_set[model_name], {})
        p = Pool(4)
        error_list = p.map(lambda params: test_classifier(clsClass, params), grid)

        for errors, params in error_list:
            if best_results[model_name][0] < np.mean(errors):
                best_results[model_name] = (np.mean(errors), params, errors)
                print(best_results[model_name])

print(best_results)