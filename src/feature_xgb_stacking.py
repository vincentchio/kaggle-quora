import config
import xgboost as xgb

from feature_stacking import FeatureStacking

class XGBStacking(FeatureStacking):
    def __init__(self):
        self._model = xgb.XGBClassifier(
            max_depth=9, 
            learning_rate=0.1,
            n_estimators=500, 
            objective='binary:logistic',
            nthread=16, 
            gamma=0, 
            subsample=0.75, 
            colsample_bytree=0.75, 
            colsample_bylevel=1,
            reg_alpha=0, 
            reg_lambda=1, 
            scale_pos_weight=1,
            seed=config.RANDOM_SEED
        )

    def feature_name(self):
        return 'stacking_xgb'

    def model(self):
        return self._model
