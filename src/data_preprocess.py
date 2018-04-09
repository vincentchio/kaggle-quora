# -*- coding: utf-8 -*-
import re

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')

DIGIT_LETTER_RE = re.compile(r"([a-zA-Z]+)[\.\-]*(\d+)")
MATH_RE = re.compile(r'\[math\].*?\[\/math\]')
DIGIT_COMMA_RE = re.compile(r'(\d+),(?=000)')
LETTER_LETTER_RE = re.compile(r"([a-zA-Z]+)[/\-](?=[a-zA-Z]+)")
REPLACE_WORDS_RE = [
    (re.compile(ur'köln'), 'cologne'),
    (re.compile(ur'gülen movement'), 'hizmet movement'),
    (re.compile(ur'₹'), ' rupee '),
    (re.compile(ur'’'), "'"),
    (re.compile(ur'∧'), '^')
]

def tokenize(text):
    return " ".join([stemmer.stem(lemmatizer.lemmatize(token)) for token in tokenizer.tokenize(text)])

def split_tokenize(text):
    return " ".join([stemmer.stem(lemmatizer.lemmatize(token)) for token in text.split()])

def clean_unicode(text):
    return ''.join([c for c in text if ord(c) <= 127])

def split_digit_letter(text):
    """ ABC123 -> ABC 123
    """
    return DIGIT_LETTER_RE.sub(r'\1 \2', text)

def replace_math_syntax(text):
    """ [math]a=b+c[/math] -> _math_
    """
    return MATH_RE.sub('_math_', text)

def merge_digit_comma(text):
    """ 15,000 > 15000
    """
    return DIGIT_COMMA_RE.sub(r'\1', text)

def split_letter_letter(text):
    """ mind-blowing -> mind blowing
    improvement/clarification -> improvement clarification
    """
    return LETTER_LETTER_RE.sub(r'\1 ', text)

def replace_words(text):
    for regex, replaced_word in REPLACE_WORDS_RE:
        text = regex.sub(replaced_word, text)
    return text

def preprocess_text(text):
    text = text.lower()
    text = replace_math_syntax(text)
    text = replace_words(text)
    text = merge_digit_comma(text)
    text = split_digit_letter(text)
    text = split_letter_letter(text)
    text = clean_unicode(text)
    # return tokenize(text)
    return split_tokenize(text)

if __name__ == "__main__":
    assert preprocess_text('this is unit test for natural language processing') == 'this is unit test for natur languag process'
    assert preprocess_text(u'₹100') == 'rupe 100'
    assert preprocess_text(u'What is it like to live in Köln, Germany') == 'what is it like to live in cologne, germani'
    assert preprocess_text(u'What is the Gülen movement') == 'what is the hizmet movement'
    assert preprocess_text('150,000') == '150000'
    assert preprocess_text('nice-to-have') == 'nice to have'
    assert preprocess_text('complex/center/forum/hub') == 'complex center forum hub'
    assert preprocess_text('30/12/2016') == '30/12/2016'
    assert preprocess_text(' NVidia GeForce GT 610 3gb DDR3') == 'nvidia geforc gt 610 3gb ddr 3'
    assert preprocess_text('[math]f(x)=x^2[/math]') == '_math_'
    assert preprocess_text(u'When do you use シ instead of し') == 'when do you use instead of'
