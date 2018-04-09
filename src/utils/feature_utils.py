import config
import os
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import log_loss
from utils.file_utils import save_to_csv

__all__ = ['load_features', 'load_target', 'calibrate', 'log_loss_dup']

def load_features(data_format='train', feature_type='feature'):
    feature_df = None
    id_field = 'id' if data_format == 'train' else 'test_id'
    for root, dirs, filenames in os.walk(os.path.join(config.FEATURE_DIR, data_format)):
        for filename in filenames:
            if filename.endswith('.csv'):
                if (feature_type == 'feature' and not filename.startswith('stacking_')) or \
                    (feature_type == 'stacking' and filename.startswith('stacking_')) or \
                    (feature_type == 'all'):
                    df = pd.read_csv(os.path.join(root, filename))
                    if feature_df is None:
                        feature_df = df
                    else:
                        feature_df = feature_df.merge(df, on=id_field)
    return feature_df

def load_target():
    target_df = pd.read_csv(os.path.join(config.DATA_DIR, 'train.csv'))[['id', 'is_duplicate']]
    return target_df

def calibrate(x):
    A = 0.17426 / 0.37
    B = (1 - 0.17426) / (1 - 0.17426)
    return (A*x) / ( (A*x) + B*(1-x))

def log_loss_dup(y_true, y_pred):
    return log_loss(y_true, y_pred[:, 1])

def generate_stacking_feature(stacking, train_features, train_target, test_features=None):
    train_X = train_features.drop('id', 1).as_matrix()
    train_y = train_target['is_duplicate'].as_matrix()
    train_stacking_features = train_features[['id']]

    train_stacking_features[stacking.feature_name()] = stacking.fit_transform(train_X, train_y)
    save_to_csv(train_stacking_features, os.path.join(config.FEATURE_DIR, 'train', '%s.csv' % stacking.feature_name()))

    if test_features is not None:
        test_X = test_features.drop('test_id', 1).as_matrix()
        test_stacking_features = test_features[['test_id']]

        stacking.fit(train_X, train_y)
        test_stacking_features[stacking.feature_name()] = stacking.transform(test_X)
        save_to_csv(test_stacking_features, os.path.join(config.FEATURE_DIR, 'test', '%s.csv' % stacking.feature_name()))
