import Word2Vec
from collections import Counter


def load_wordvectors(filename):
    return Word2Vec.load_word2vec_format(filename, binary=False)


def read_wordlist(filename):
    """
    Returns a dict of words and their corresponding counts.
    """
    d = Counter()
    with open(filename) as f:
        for line in f:
            word, count = line.split()
            d[word] = int(count)
    return d


def read_segmentations(filename):
    """
    Return a dict of words to a list of segments.
    """
    d = {}
    with open(filename) as f:
        for line in f:
            word, segments = line.split(':')
            segments = segments.split('-')
            d[word] = segments
    return d
