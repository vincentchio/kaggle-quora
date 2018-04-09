import config

from feature_stacking import FeatureStacking
from sklearn.ensemble import RandomForestClassifier

class RandomForestStacking(FeatureStacking):
    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)

    def feature_name(self):
        return 'stacking_random_forest'

    def model(self):
        return self._model
