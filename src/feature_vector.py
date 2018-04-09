import config
import numpy as np

from feature_transform import FeatureTransform
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import read_corpus, transform_all_feature, save_feature_to_csv, read_clean_train, read_clean_test, corr
from utils.nlp_utils import ngrams, ngrams_name
from utils.distance_utils import jaccard_similarity, cosine_similarity


class TfidfCosineSimilarityTransform(FeatureTransform):
    def __init__(self, corpus, ngram):
        self.ngram = ngram
        self.vectorizer = TfidfVectorizer(min_df=3, max_df=0.75, sublinear_tf=True, ngram_range=(ngram, ngram), max_features=100000)
        self.vectorizer.fit(corpus)

    def feature_name(self):
        return 'tfidf_cosine_similarity_%s' % (ngrams_name(self.ngram))

    def transform(self, s1, s2=None):
        return cosine_similarity(self.vectorizer.transform([s1]).toarray()[0], self.vectorizer.transform([s2]).toarray()[0])

    def transform_all(self, s1, s2=None):
        return np.array(map(
            cosine_similarity,
            self.vectorizer.transform(s1).toarray(),
            self.vectorizer.transform(s2).toarray()))


class TfidfCharCosineSimilarityTransform(FeatureTransform):
    def __init__(self, corpus, ngram):
        self.ngram = ngram
        self.vectorizer = TfidfVectorizer(
            min_df=3, max_df=0.75, sublinear_tf=True, ngram_range=(ngram, ngram), max_features=200000, analyzer='char')
        self.vectorizer.fit(corpus)

    def feature_name(self):
        return 'tfidf_char_cosine_similarity_%s' % (ngrams_name(self.ngram))

    def transform(self, s1, s2=None):
        return cosine_similarity(self.vectorizer.transform([s1]).toarray()[0], self.vectorizer.transform([s2]).toarray()[0])

    def transform_all(self, s1, s2=None):
        return np.array(map(
            cosine_similarity,
            self.vectorizer.transform(s1).toarray(),
            self.vectorizer.transform(s2).toarray()))


class LSACosineSimilarityTransform(FeatureTransform):
    def __init__(self, corpus, ngram):
        self.ngram = ngram
        self.vectorizer = TfidfVectorizer(min_df=3, max_df=0.75, sublinear_tf=True, ngram_range=(ngram, ngram), max_features=100000)
        corpus = self.vectorizer.fit_transform(corpus)

        self.svd = TruncatedSVD(n_components=100, random_state=config.RANDOM_SEED, algorithm='arpack')
        self.svd.fit(corpus)

    def feature_name(self):
        return 'lsa_cosine_similarity_%s' % (ngrams_name(self.ngram))

    def transform(self, s1, s2=None):
        # This is way too inefficient

        # return cosine_similarity(
        #     self.svd.transform(self.vectorizer.transform([s1]))[0],
        #     self.svd.transform(self.vectorizer.transform([s2]))[0])
        return FeatureTransform.transform(self, s1, s2)

    def transform_all(self, s1, s2=None):
        return np.array(map(
            cosine_similarity,
            self.svd.transform(self.vectorizer.transform(s1)),
            self.svd.transform(self.vectorizer.transform(s2))))


class LSACharCosineSimilarityTransform(FeatureTransform):
    def __init__(self, corpus, ngram):
        self.ngram = ngram
        self.vectorizer = TfidfVectorizer(
            min_df=3, max_df=0.75, sublinear_tf=True, ngram_range=(ngram, ngram), max_features=200000, analyzer='char')
        corpus = self.vectorizer.fit_transform(corpus)

        self.svd = TruncatedSVD(n_components=100, random_state=config.RANDOM_SEED, algorithm='arpack')
        self.svd.fit(corpus)

    def feature_name(self):
        return 'lsa_char_cosine_similarity_%s' % (ngrams_name(self.ngram))

    def transform_all(self, s1, s2=None):
        return np.array(map(
            cosine_similarity,
            self.svd.transform(self.vectorizer.transform(s1)),
            self.svd.transform(self.vectorizer.transform(s2))))


if __name__ == "__main__":
    corpus = [
        'quora kaggle competition is so fun',
        'text mining is so fun',
        'this is very cool',
        'fun fun fun'
    ]
    tfidf_cosine_similarity_transform = TfidfCosineSimilarityTransform(corpus, 1)
    assert tfidf_cosine_similarity_transform.transform('quora kaggle competition is so fun', 'text mining is so fun') == 1
    assert tfidf_cosine_similarity_transform.transform_all(['quora kaggle competition is so fun'], ['text mining is so fun']) == [1]
    assert np.isclose(tfidf_cosine_similarity_transform.transform('fun fun fun', 'text mining is so fun'), 0.70711)

    tfidf_char_cosine_similarity_transform = TfidfCharCosineSimilarityTransform(corpus, 2)
    assert np.isclose(tfidf_char_cosine_similarity_transform.transform('quora kaggle competition is so fun', 'text mining is so fun'), 1)
    assert np.isclose(tfidf_char_cosine_similarity_transform.transform_all(['quora kaggle competition is so fun'], ['text mining is so fun']), [1])
