import sklearn
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin


class CustomModel(RegressorMixin):

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class NullRegressor(BaseEstimator, RegressorMixin):

    def fit(self, X, y):
        # The prediction will always just be the mean of y
        self.y_mean = np.mean(y)

    def predict(self, X):
        # Return mean of y, in the same length as the number of X observations
        return np.ones(X.shape[0]) * self.y_mean

