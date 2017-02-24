__author__ = 'Martin'

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import cross_validation
from sklearn import ensemble

def is_transformer(cls):
    return hasattr(cls, '__dageva_type') and cls.__dageva_type == 'transformer'

def is_predictor(cls):
    return hasattr(cls, '__dageva_type') and cls.__dageva_type == 'predictor'

def make_transformer(cls):
    """
    Adds Transformer to the bases of the cls class, useful in order to distinguish between transformers and predictors.
    :param cls: The class to turn into a Transformer
    :return: A class equivalent to cls, but with Transformer among its bases
    """
    cls.__dageva_type = 'transformer'
    return cls


def make_predictor(cls):
    """
    Adds Predictor to the bases of the cls class, useful in order to distinguish between transformers and predictors.
    :param cls: The class to turn into a Predictor
    :return: A class equivalent to cls, but with Predictor among its bases
    """
    cls.__dageva_type = 'predictor'
    return cls


class KMeansSplitter:

    def __init__(self, k):
        from sklearn import cluster
        self.kmeans = cluster.KMeans(n_clusters=k)
        self.sorted_outputs = None
        self.weight_idx = []

    def fit(self, x, y, sample_weight=None):
        self.kmeans.fit(x, y)
        preds = self.kmeans.predict(x)
        out = []
        for i in range(self.kmeans.n_clusters):
            idx = [n for n in range(len(preds)) if preds[n] == i]
            self.weight_idx.append(idx)
            if isinstance(x, pd.DataFrame):
                out.append(x.iloc[idx])
            else:
                out.append(x[idx])
        mins = [len(x.index) for x in out]
        self.sorted_outputs = list(np.argsort(mins))
        self.weight_idx = [self.weight_idx[i] for i in self.sorted_outputs]
        return self

    def transform(self, x):
        preds = self.kmeans.predict(x)
        out = []
        for i in range(self.kmeans.n_clusters):
            idx = [n for n in range(len(preds)) if preds[n] == i]
            if isinstance(x, pd.DataFrame):
                out.append(x.iloc[idx])
            else:
                out.append(x[idx])
        return [out[i] for i in self.sorted_outputs]


class ConstantModel:

    def __init__(self, cls):
        self.cls = cls

    def fit(self, x, y):
        return self

    def predict(self, x):
        return pd.Series(np.array([self.cls]*len(x)), index=x.index)


class Aggregator:

    def aggregate(self, x, y):
        pass

class Voter(Aggregator):

    def fit(self, x, y):
        return self

    def union_aggregate(self, x, y):
        f_list, t_list = x, y
        f_frame, t_frame = pd.DataFrame(), pd.Series()
        for i in range(len(t_list)):
            fl = f_list[i]
            assert isinstance(fl, pd.DataFrame)
            if fl.columns.dtype == np.dtype('int64'):
                cols = map(lambda z: str(id(fl)) + '_' + str(z), fl.columns)
                fl.columns = cols
            t_frame = t_frame.append(t_list[i])
            f_frame = f_frame.append(f_list[i])
        f_frame.sort_index(inplace=True)
        t_frame = t_frame.sort_index()
        return f_frame, t_frame

    def aggregate(self, x, y):
        if not all([x[0].index.equals(xi.index) for xi in x]):
            return self.union_aggregate(x, y)
        res = pd.DataFrame(index=y[0].index)
        for i in range(len(y)):
            res["p"+str(i)] = y[i]
        modes = res.apply(lambda row: stats.mode(row, axis=None)[0][0], axis=1)
        if modes.empty:
            return x[0], pd.Series()
        return x[0], pd.Series(modes, index=y[0].index)


class Workflow:

    def __init__(self, dag=None):
        self.dag = dag
        self.sample_weight = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        import eval  #TODO: Refactor to remove circular imports
        self.models = eval.train_dag(self.dag, train_data=(X, y), sample_weight=sample_weight)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        import eval  #TODO: Refactor to remove circular imports
        return np.array(eval.test_dag(self.dag, self.models, test_data=(X, None)))

    def transform(self, X):
        import eval
        return eval.test_dag(self.dag, self.models, test_data=(X, None), output='feats_only')

    def get_params(self, deep=False):
        return {'dag': self.dag}

    def set_params(self, **params):
        if 'sample_weight' in params:
            self.sample_weight = params['sample_weight']


class Stacker(Aggregator):

    def __init__(self, sub_dags=None, initial_dag=None):
        self.sub_dags = sub_dags
        self.initial_dag = initial_dag

    def fit(self, X, y, sample_weight=None):
        import eval
        preds = [[] for _ in self.sub_dags]

        for train_idx, test_idx in cross_validation.StratifiedKFold(y, n_folds=5):
            tr_X, tr_y = X.iloc[train_idx], y.iloc[train_idx]
            tst_X, tst_y = X.iloc[test_idx], y.iloc[test_idx]
            wf_init = Workflow(self.initial_dag)
            wf_init.fit(tr_X, tr_y, sample_weight=sample_weight)
            preproc_X, preproc_y = eval.test_dag(self.initial_dag, wf_init.models, test_data=(tr_X, tr_y), output='all')
            pp_tst_X = wf_init.transform(tst_X)
            if pp_tst_X.empty:
                continue
            for i, dag in enumerate(self.sub_dags):
                wf = Workflow(dag)
                wf.fit(preproc_X, preproc_y)
                res = wf.predict(pp_tst_X)
                preds[i].append(pd.DataFrame(res, index=pp_tst_X.index))

        preds = [pd.concat(ps) for ps in preds]

        self.train = pd.concat(preds, axis=1)
        self.train.columns = ['p' + str(x) for x in range(len(preds))]

        return self

    def aggregate(self, X, y):
        res = pd.DataFrame(index=y[0].index)
        for i in range(len(X)):
            res["p" + str(i)] = y[i]
        return res, y[0]


class Booster(ensemble.AdaBoostClassifier):

    def __init__(self, sub_dags=()):
        self.sub_dags = sub_dags
        self.current_sub_dag = 0
        super(Booster, self).__init__(base_estimator=Workflow(), n_estimators=len(sub_dags), algorithm='SAMME')

    def _make_estimator(self, append=True, random_state=0):

        estimator = Workflow(self.sub_dags[self.current_sub_dag])
        self.current_sub_dag += 1

        if append:
            self.estimators_.append(estimator)

        return estimator