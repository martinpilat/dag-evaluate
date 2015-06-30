__author__ = 'Martin'

import pandas as pd
import numpy as np
from scipy import stats


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
