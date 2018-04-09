import numpy as np

from scipy.spatial.distance import cosine
from utils.math_utils import safe_divide

def jaccard_similarity(s1, s2):
    if not isinstance(s1, set):
        s1 = set(s1)
    if not isinstance(s2, set):
        s2 = set(s2)

    return safe_divide(float(len(s1 & s2)), len(s1 | s2))

def cosine_similarity(v1, v2):
    cos_sim = 1 - cosine(v1, v2)
    return cos_sim if not np.isnan(cos_sim) else -1

if __name__ == "__main__":
    assert jaccard_similarity(['a', 'b'], ['b', 'c']) == 1.0/3
    assert jaccard_similarity(['a', 'b'], ['c', 'd']) == 0
    assert jaccard_similarity(['a', 'b'], ['b', 'b']) == 0.5
    assert jaccard_similarity(['a', 'a'], ['a', 'b']) == 0.5

    assert cosine_similarity([1,2,3], [1,2,3]) == 1
    assert np.isclose(cosine_similarity([1,2,3], [2,3,4]), 0.99258)
    assert cosine_similarity([1,2,3], [-1,-2,-3]) == -1
    assert cosine_similarity([0,0,0], [1,2,3]) == -1
