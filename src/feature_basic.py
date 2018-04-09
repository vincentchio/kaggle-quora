from feature_transform import FeatureTransform
from utils.nlp_utils import tokenize

class QuestionLengthTransform(FeatureTransform):
    def __init__(self, field='question1'):
        assert field == 'question1' or field == 'question2'
        self.field = field

    def feature_name(self):
        return '%s_length' % (self.field)

    def transform(self, s1, s2=None):
        if self.field == 'question1':
            return len(tokenize(s1))
        else:
            return len(tokenize(s2))


if __name__ == "__main__":
    question1_length_transform = QuestionLengthTransform()
    assert question1_length_transform.transform('what do you think china food', 'how do you think of chinese food') == 6
    assert question1_length_transform.feature_name() == 'question1_length'

    question2_length_transform = QuestionLengthTransform('question2')
    assert question2_length_transform.transform('what do you think china food', 'how do you think of chinese food') == 7
    assert question2_length_transform.feature_name() == 'question2_length'
