import config

from feature_stacking import FeatureStacking
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingStacking(FeatureStacking):
    def __init__(self):
        self._model = GradientBoostingClassifier(n_estimators=500, random_state=config.RANDOM_SEED)

    def feature_name(self):
        return 'stacking_gradient_boosting'

    def model(self):
        return self._model
