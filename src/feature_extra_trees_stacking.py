import config

from feature_stacking import FeatureStacking
from sklearn.ensemble import ExtraTreesClassifier

class ExtraTreesStacking(FeatureStacking):
    def __init__(self):
        self._model = ExtraTreesClassifier(n_estimators=100, random_state=config.RANDOM_SEED)

    def feature_name(self):
        return 'stacking_extra_trees'

    def model(self):
        return self._model
