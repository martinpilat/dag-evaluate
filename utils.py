__author__ = 'Martin'

import pandas as pd
import numpy as np
import json


from sklearn import decomposition, feature_selection, svm, tree, linear_model, naive_bayes


def read_json(file_name):
    return json.load(open(file_name, 'r'))


class KMeansSplitter:

    def __init__(self, k):
        from sklearn import cluster
        self.kmeans = cluster.KMeans(n_clusters=k)
        self.sorted_outputs = None

    def fit(self, x, y):
        self.kmeans.fit(x, y)
        preds = self.kmeans.predict(x)
        out = []
        for i in range(self.kmeans.n_clusters):
            idx = [n for n in range(len(preds)) if preds[n] == i]
            out.append(x.iloc[idx])
        mins = [min(x.index) for x in out]
        self.sorted_outputs = list(np.argsort(mins))
        return self

    def transform(self, x):
        preds = self.kmeans.predict(x)
        out = []
        for i in range(self.kmeans.n_clusters):
            idx = [n for n in range(len(preds)) if preds[n] == i]
            out.append(x.iloc[idx])
        return [out[i] for i in self.sorted_outputs]


class ConstantModel:

    def __init__(self, cls):
        self.cls = cls

    def fit(self, x, y):
        return self

    def predict(self, x):
        return pd.DataFrame(np.array([self.cls]*len(x)), index=x.index)


class Aggregator:

    def aggregate(self, x, y):
        pass


class Transformer:
    pass


class Predictor:
    pass


def make_transformer(cls):
    class Tr(cls, Transformer):
        def __repr__(self):
            return cls.__name__ + ":" + cls.__repr__(self)
    return Tr


def make_predictor(cls):
    class Pr(cls, Predictor):
        def __repr__(self):
            return cls.__name__ + ":" + cls.__repr__(self)
    return Pr

from scipy import stats

class Voter(Aggregator):

    def fit(self, x, y):
        return self

    def union_aggregate(self, x, y):
        f_list, t_list = x, y
        f_frame, t_frame = pd.DataFrame(), pd.Series()
        for i in range(len(t_list)):
            t_frame = t_frame.append(t_list[i])
            f_frame = f_frame.append(f_list[i])
        f_frame.sort_index(inplace=True)
        t_frame = t_frame.sort_index()
        return f_frame, t_frame

    def aggregate(self, x, y):
        if not x[0].index.equals(x[1].index):
            return self.union_aggregate(x, y)
        res = pd.DataFrame(index=y[0].index)
        for i in range(len(y)):
            res["p"+str(i)] = y[i]
        modes = res.apply(lambda row: stats.mode(row, axis=None)[0][0], axis=1)
        if modes.empty:
            return x[0], pd.Series()
        return x[0], pd.Series(modes, index=y[0].index)


model_names = {
    "PCA":          (make_transformer(decomposition.PCA), {}),
    "kBest":        (make_transformer(feature_selection.SelectKBest), {}),
    "kMeans":       (make_transformer(KMeansSplitter), {}),
    "copy":         ([], {}),
    "SVC":          (make_predictor(svm.SVC), {}),  # {'C': 8, 'gamma': 0.0001}),
    "logR":         (make_predictor(linear_model.LogisticRegression), {}),  #{'solver': 'lbfgs', 'C': 0.2, 'penalty': 'l1'}),
    "gaussianNB":   (make_predictor(naive_bayes.GaussianNB), {}),
    "DT":           (make_predictor(tree.DecisionTreeClassifier), {}),  # {'max_depth': 8, 'criterion': 'entropy', 'min_samples_leaf': 7, 'min_samples_split': 20}),
    "union":        (Voter, {}),
    "vote":         (Voter, {})
}


def get_model_by_name(model_name):
    return model_names[model_name]
