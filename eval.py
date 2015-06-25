__author__ = "Martin Pilat"

import pandas as pd
import utils
from scoop import futures
import sys
from sklearn import cross_validation
import numpy as np

def data_ready(req, cache):
    """
    Checks that all required data are in the data_cache

    :param req: string or list of string containing the keys of required data in cache
    :param cache: dictionary with the computed data
    :return: Boolean indicating whether all required data are in cache
    """
    if not isinstance(req, list):
        req = [req]
    return all([r in cache for r in req])

def get_data(data_list, data_cache):
    """
    Gets the data specified by the keys in the data_list from the data_cache

    :param data_list: string or list of strings
    :param data_cache: dictionary containing the stored data
    :return: a single pandas.DataFrame if input is a string, a list of DataFrames if the input is a list of strings
    """
    if not isinstance(data_list, list):
        data_list = [data_list]
    tmp = [data_cache[d] for d in data_list]
    if len(tmp) == 1:
        return tmp[0]
    res = ([t[0] for t in tmp], [t[1] for t in tmp])
    return res

def append_all(data_frames):
    if not isinstance(data_frames, list):
        return data_frames
    res = data_frames[0]
    for i in range(1, len(data_frames)):
        res.append(data_frames[i])
    return res

from sklearn import preprocessing, feature_selection

def train_dag(dag, train_data):
    models = dict()
    data_cache = dict()

    data_cache[dag['input'][2]] = train_data
    models['input'] = True

    num_features = train_data[0].shape[1]

    unfinished_models = lambda: [m for m in dag if m not in models]
    data_available = lambda: [m for m in dag if data_ready(dag[m][0], data_cache)]
    next_methods = lambda: [m for m in unfinished_models() if m in data_available()]

    # print("UNDONE:", unfinished_models())
    # print("AVAIL:", data_available())
    # print("NEXT:", next_methods())

    while next_methods():

        for m in next_methods():
            # print("Processing:", m)

            # obtain the data
            features, targets = get_data(dag[m][0], data_cache)
            ModelClass, model_params = utils.get_model_by_name(dag[m][1])
            out_name = dag[m][2]
            if isinstance(out_name, list):
                model = ModelClass(len(out_name), **model_params)
            else:
                model = ModelClass(**model_params)

            if isinstance(model, feature_selection.SelectKBest):
                model = ModelClass(k=(num_features*3)//4, **model_params)

            # build the model
            # some models cannot handle cases with only one class, we also need to check we are not working with a list
            # of inputs for an aggregator
            if isinstance(model, utils.Predictor) and isinstance(targets, pd.Series) and len(targets.unique()) == 1:
                model = utils.ConstantModel(targets.iloc[0])
            models[m] = model.fit(features, targets)

            # use the model to process the data
            if isinstance(model, utils.Aggregator):
                data_cache[out_name] = model.aggregate(features, targets)
                continue
            if isinstance(model, utils.Transformer):
                trans = model.transform(features)
            else:              # this is a classifier not a preprocessor
                trans = features                # the data do not change
                targets = pd.Series(list(model.predict(features)), index=features.index)

            # save the outputs
            if isinstance(trans, list):         # the previous model divided the data into several data-sets
                trans = [(x, targets.loc[x.index]) for x in trans]     # need to divide the targets
                for i in range(len(trans)):
                    data_cache[out_name[i]] = trans[i]          # save all the data to the cache
            else:
                trans = pd.DataFrame(trans, index=features.index)       # we have only one output, can be numpy array
                data_cache[out_name] = (trans, targets)                 # save it

    #     print("models:", models)
    #
    #     print("UNDONE:", unfinished_models())
    #     print("AVAIL:", data_available())
    #     print("NEXT:", next_methods())
    #
    # print("DATA:", data_cache.keys())

    return models


def test_dag(dag, models, test_data):
    data_cache = dict()
    finished = dict()

    data_cache[dag['input'][2]] = test_data
    finished['input'] = True

    unfinished_models = lambda: [m for m in dag if m not in finished]
    data_available = lambda: [m for m in dag if data_ready(dag[m][0], data_cache)]
    next_methods = lambda: [m for m in unfinished_models() if m in data_available()]

    # print("UNDONE:", unfinished_models())
    # print("AVAIL:", data_available())
    # print("NEXT:", next_methods())

    while next_methods():

        for m in next_methods():
            # print("Processing:", m)

            # obtain the data
            features, targets = get_data(dag[m][0], data_cache)
            model = models[m]
            out_name = dag[m][2]

            if isinstance(features, pd.DataFrame) and features.empty:   # we got empty dataset (after same division)
                if isinstance(out_name, list):                          # and we should divide it further
                    for o in out_name:
                        data_cache[o] = (features, targets)
                else:
                    data_cache[out_name] = (features, targets)
                finished[m] = True
                continue
            # use the model to process the data
            if isinstance(model, utils.Aggregator):
                data_cache[out_name] = model.aggregate(features, targets)
                finished[m] = True
                continue
            elif isinstance(model, utils.Transformer):
                trans = model.transform(features)
                targets = pd.Series(targets, index=features.index)
            else:              # this is a classifier not a preprocessor
                trans = features                # the data do not change
                targets = pd.Series(list(model.predict(features)), index=features.index)

            # save the outputs
            if isinstance(trans, list):         # the previous model divided the data into several data-sets
                trans = [(x, targets.loc[x.index]) for x in trans]     # need to divide the targets
                for i in range(len(trans)):
                    data_cache[out_name[i]] = trans[i]          # save all the data to the cache
            else:
                trans = pd.DataFrame(trans, index=features.index)       # we have only one output, can be numpy array
                data_cache[out_name] = (trans, targets)                 # save it

            finished[m] = True

        # print("models:", models)
        #
        # print("UNDONE:", unfinished_models())
        # print("AVAIL:", data_available())
        # print("NEXT:", next_methods())
        #
        # print("DATA:", data_cache.keys())

    return data_cache['output'][1]

def normalize_spec(spec):
    ins, mod, outs = spec
    if len(ins) == 1:
        ins = ins[0]
    if len(outs) == 1:
        outs = outs[0]
    if len(outs) == 0:
        outs = 'output'
    return ins, mod, outs

def normalize_dag(dag):
    normalized_dag = {k: normalize_spec(v) for (k, v) in dag.items()}

    original_len = len(normalized_dag)

    aliases = {normalized_dag[k][0]: normalized_dag[k][2] for k in normalized_dag if normalized_dag[k][1] == "copy"}
    normalized_dag = {k: v for (k, v) in normalized_dag.items() if v[1] != 'copy'}

    new_len = len(normalized_dag)

    rev_aliases = {v: k for k in aliases for v in aliases[k]}
    for i in range(original_len-new_len):
        normalized_dag = {k: ((rev_aliases[ins] if not isinstance(ins, list) and ins in rev_aliases else ins), mod, out)
                        for (k, (ins, mod, out)) in normalized_dag.items()}

    return normalized_dag

import ml_metrics as mm

def eval_dag(dag, filename, dag_id=None):

    dag = normalize_dag(dag)

    data = pd.read_csv('data/'+filename, sep=';')

    feats = data[data.columns[:-1]]
    targets = data[data.columns[-1]]

    le = preprocessing.LabelEncoder()

    ix = targets.index
    targets = pd.Series(le.fit_transform(targets), index=ix)

    errors = []

    for train_idx, test_idx in cross_validation.StratifiedKFold(targets, n_folds=5):
        train_data = (feats.iloc[train_idx], targets.iloc[train_idx])
        test_data = (feats.iloc[test_idx], targets.iloc[test_idx])

        ms = train_dag(dag, train_data)
        # print("-"*80)
        # print("Training Finished")
        # print("-"*80)
        # print("MODELS:", ms)
        preds = test_dag(dag, ms, test_data)
        # print("-"*80)
        # print("Evaluation finished")
        # print("-"*80)
        acc = mm.quadratic_weighted_kappa(test_data[1], preds)
        errors.append(acc)
        # print("Test error:", acc)

    m_errors = float(np.mean(errors))
    s_errors = float(np.std(errors))

    # print("Model %s: Cross-validation error: %.5f (+-%.5f)" % (dag_id, m_errors, s_errors))

    return m_errors, s_errors

def safe_dag_eval(dag, filename, dag_id=None):
    try:
        return eval_dag(dag, filename, dag_id), dag_id
    except Exception as e:
        with open('error.'+str(dag_id), 'w') as err:
            err.write(str(e)+'\n')
            err.write(str(dag))
    return (), dag_id

class Evaluator:

    def __init__(self, filename):
        self.filename = filename

    def __call__(self, x):
        return safe_dag_eval(x[1], self.filename, x[0])

def eval_all(dags, filename):

    results = {}

    inner_eval = Evaluator(filename)

    for e in futures.map_as_completed(inner_eval, enumerate(dags)):
        results[str(e[1])] = e

    res = [results[str(d[0])][0] for d in enumerate(dags)]

    return res

if __name__ == '__main__':

    import shelve
    import pickle

    results = shelve.open("results_wilt_tuned", protocol=pickle.HIGHEST_PROTOCOL)

    datafile = "wilt.csv"
    dags = utils.read_json('population_65536.json')
    # dags = dags[36076:36077]

    remaining_dags = [d for d in enumerate(dags) if str(d[0]) not in results]
    print("Starting...", len(remaining_dags))

    for e in futures.map_as_completed(lambda x: safe_dag_eval(x[1], datafile, x[0]), remaining_dags):
        results[str(e[1])] = e
        print("Model %4d: Cross-validation error: %.5f (+-%.5f)" % (e[1], e[0][0], e[0][1]))
        sys.stdout.flush()

    print("-"*80)
    best_error = sorted(results.values(), key=lambda x: x[0][0]-2*x[0][1], reverse=True)[0]
    print("Best model CV error: %.5f (+-%.5f)" % (best_error[0][0], best_error[0][1]))

    import pprint
    print("Model: ", end='')
    pprint.pprint(dags[best_error[1]])
