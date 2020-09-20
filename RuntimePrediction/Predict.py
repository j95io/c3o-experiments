import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import explained_variance_score as evs
from .DefaultModels import (SimilarityModel, OptimisticModel)
from .CustomModels import NullRegressor
from sklearn.linear_model import LinearRegression

models = {
    'Optimistic Model v1': OptimisticModel,
    'Pessimistic Model v1': SimilarityModel,
    'Reference Model (LR)': LinearRegression,
    'Null Regressor': NullRegressor,
         }

def mre(y_true, y_pred):  # Mean relative error
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

class Predictor:

    def __init__(self):
        self.all_models = {name: model() for name, model in models.items()}
        self.scores = {}
        self.model = None
        self.model_name = None
        #self.sc = 'neg_mean_squared_error'
        self.sc = 'neg_root_mean_squared_error'

    def fit(self, X, y, verbose=False):
        # Evaluate available prediction models
        score_info = []
        for name, model in self.all_models.items():
            cv = LeaveOneOut().split(X)
            score =  cross_val_score(model, X, y, cv=cv, scoring=self.sc)
            self.scores[name] = (-score/y).mean()
            score_info.append((name, self.scores[name]))
            self.all_models[name].fit(X, y)

        score_info.sort(key=lambda x:x[1])

        if verbose: Predictor.print_eval_table(score_info, len(y))

        return score_info

    def predict(self, X):
        best_model_name = sorted((s,n) for n,s in self.scores.items())[0][1]
        return self.all_models[best_model_name].predict(X)

    def eval(self, X, y, train_size=0.8, duplicate_ratio=0.5, iterations=100, use_mre=False, eval_switcher=False):
        scores = {}
        errors = {}

        if eval_switcher:
            self.all_models['C3O Predictor'] = Predictor()
        for name, model in self.all_models.items():
            s = []
            e = []
            for _ in range(iterations):
                if duplicate_ratio > 0.05: # add duplicates
                    X_dup, _a, y_dup, _b = tts(X, y, train_size=duplicate_ratio)
                    X_a, y_a = np.append(X, X_dup, axis=0), np.append(y, y_dup, axis=0)
                else:  # add no duplicates
                    X_a, y_a = X[:,:], y[:]

                X_train, X_test, y_train, y_test = tts(X_a, y_a, train_size=train_size)
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                if use_mre:
                    e.append(mre(y_test, y_hat))
                else:
                    e.append(mae(y_test, y_hat))
                s.append(evs(y_test, y_hat))
            errors[name] = sum(e)/len(e)
            scores[name] = sum(s)/len(s)

        return errors, scores

    @staticmethod
    def print_eval_table(scores, num_observations):
        print(f"{'Model':30s}{'Mean Abs Err':10s}\n")
        for name, err in scores:
            print(f"{name:30s}{err:.5f}")

        print(f"\nFrom {num_observations} training data points")
        print("Using 'Leave One Out Cross Validation'")
