import config
import numpy as np
import os

from sklearn.model_selection import KFold

class FeatureStacking():
    def model(self):
        raise NotImplementedError()

    def feature_name(self):
        raise NotImplementedError()

    def fit(self, X, y):
        self.model().fit(X, y)

    def transform(self, X):
        return self.model().predict_proba(X)[:, 1]

    def fit_transform(self, X, y):
        features = np.array([0.0] * len(y))
        kf = KFold(n_splits=5, random_state=config.RANDOM_SEED)
        for train_index, test_index in kf.split(X):
            self.model().fit(X[train_index], y[train_index])
            features[test_index] = self.model().predict_proba(X[test_index])[:,1]
        return features
