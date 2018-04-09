from feature_transform import FeatureTransform
from utils.nlp_utils import ngrams, ngrams_name
from utils.distance_utils import jaccard_similarity

class JaccardDistanceTransform(FeatureTransform):
    def __init__(self, ngram):
        self.ngram = ngram

    def feature_name(self):
        return 'jaccard_distance_%s' % (ngrams_name(self.ngram))

    def transform(self, s1, s2=None):
        return jaccard_similarity(ngrams(s1, self.ngram), ngrams(s2, self.ngram))

if __name__ == "__main__":
    jaccard_distance_transform = JaccardDistanceTransform(1)
    assert jaccard_distance_transform.transform('what do you think china food', 'how do you think of chinese food') == 4.0/9
    assert jaccard_distance_transform.feature_name() == 'jaccard_distance_unigram'
