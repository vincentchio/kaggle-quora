def tokenize(s):
    return s.split(' ')

def ngrams(s, n):
    tokens = tokenize(s)
    size = len(tokens)

    if size == 0:
        return ''
    elif size <= n:
        return [' '.join(tokens)]
    else:
        return [' '.join(tokens[i:(i+n)]) for i in xrange(size - n + 1)]

def ngrams_name(n):
    if n == 1:
        return 'unigram'
    elif n == 2:
        return 'bigram'
    elif n == 3:
        return 'trigram'
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    text = 'this is unit test for natural language processing'

    assert tokenize(text) == ['this', 'is', 'unit', 'test', 'for', 'natural', 'language', 'processing']

    assert ngrams(text, 1) == ['this', 'is', 'unit', 'test', 'for', 'natural', 'language', 'processing']
    assert ngrams(text, 2) == ['this is', 'is unit', 'unit test', 'test for', 'for natural', 'natural language', 'language processing']
    assert ngrams(text, 3) == ['this is unit', 'is unit test', 'unit test for', 'test for natural', 'for natural language', 'natural language processing']
    assert ngrams('this', 2) == ['this']
    assert ngrams('this', 3) == ['this']
    assert ngrams('this is', 3) == ['this is']
