class FeatureTransform():
    def feature_name(self):
        raise NotImplementedError()

    def transform(self, s1, s2=None):
        raise NotImplementedError()

    def transform_all(self, s1, s2=None):
        raise NotImplementedError()
