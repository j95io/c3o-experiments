from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import sklearn
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

class SSModel(BaseEstimator, RegressorMixin):  # Scaleout-Speedup model

    def __init__(self, instance_count_index=0):
        degree=3
        self.polyreg = make_pipeline(PolynomialFeatures(degree) ,LinearRegression())
        self.scales = []
        self.instance_count_index = instance_count_index  # index within features

    def preprocess(self, X, y):
        # Find biggest group of same features besides instance_count to learn from
        Xy = np.concatenate((X, y.reshape(-1,1)), axis=1)
        features = pd.DataFrame(Xy)
        indices = list(range(len(X[0])))
        indices.remove(self.instance_count_index)
        groups = features.groupby(by=indices)
        max_group = sorted(groups, key=lambda x:len(x[1]))[-1][1]
        X = max_group.iloc[:,0].to_numpy().reshape((-1,1))
        y = max_group.iloc[:, -1]
        return X, y

    def fit(self, X, y):
        if X.shape[1] > 1:
            X, y = self.preprocess(X, y)

        self.min, self.max = X.min(), X.max()
        self.polyreg.fit(X,y)

    def predict(self, X):
        rt_for_min_scaleout = self.polyreg.predict([[self.min]])
        rt = self.polyreg.predict(X)
        # Replace scale-outs of more than self.max with pred for self.max
        # (poly3 curve does not continue as desired)
        rt[X.flatten() > self.max] = self.polyreg.predict(np.array([[self.max]]))
        return (rt/rt_for_min_scaleout)  # return expected speedup

class OptimisticModel(BaseEstimator, RegressorMixin):

    def fit(self, X, y):#, ssm=SSModel, ibm=LinearRegression, instance_count_index=0):
        self.instance_count_index = 0
        self.ssm= SSModel()
        self.ibm= LinearRegression()
        # Train scale-out speed-up model
        self.ssm.fit(X, y)
        scales = self.ssm.predict(X[:,[self.instance_count_index]])
        # Project all runtimes to expected runtimes at scaleout = min_scaleout
        y_projection = y/scales
        # Train the inputs-behavior model on all inputs (not the instance_count)
        inputs = [i for i in range(X.shape[1]) if i != self.instance_count_index] or [0]
        self.ibm.fit(X[:,inputs], y_projection)

    def predict(self, X):
        instance_count = X[:, [self.instance_count_index]]
        inputs = [i for i in range(X.shape[1]) if i != self.instance_count_index]
        return self.ssm.predict(instance_count) * self.ibm.predict(X[:,inputs])


class SimilarityModel(BaseEstimator, RegressorMixin):

    def fit(self, X, y):

        # Check for columns with all-same values -> leave out (messes up corr())
        self.relevant_features = list(range(len(X[0])))
        for i in range(len(X[0])):
            if all(X[0, i] == X[:,i]):
                self.relevant_features.remove(i)
        X = X[:, self.relevant_features]

        Z = np.concatenate((X, y.reshape(-1,1)), axis=1)
        self.correlations = pd.DataFrame(Z).corr().to_numpy()[:-1, -1]
        self.X_train = X
        self.y_train = y

    def predict(self, X, get_similarity_score=False):

        X = X[:, self.relevant_features]

        def diff(a,b): return abs(a-b) / a

        y = []
        for x_i in X:
            diffs = []
            for x_j in self.X_train:
                feature_diffs = diff(x_i, x_j)
                diffs.append(sum(feature_diffs * abs(self.correlations)))

            if get_similarity_score:
                y.append(min(diffs))
            else:
                most_similar = self.y_train[diffs.index(min(diffs))]
                y.append(most_similar)

        return np.array(y)

