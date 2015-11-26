from gensim.models.word2vec import Word2Vec
from collections import Counter


def load_wordvectors(filename, fvocab=None, binary=False):
    return Word2Vec.load_word2vec_format(filename, fvocab=fvocab, binary=binary)


def read_wordlist(filename):
    """
    Returns a dict of words and their corresponding counts.
    """
    d = Counter()
    with open(filename) as f:
        for line in f:
            count, word = line.split()
            d[word] = int(count)
    return d


# Reads from morpho challege
def readCorpus(filename):
    f = open(filename, 'r')
    d = {}  # maps word to a tuple containing segmentation and a list of tags
    for lines in f.readlines():
        lines = lines.split()
        seg = [g.split(':')[0] for g in lines[1:]]
        tags = [g.split(':')[1] for g in lines[1:]]
        d[lines[0]] = (seg, tags)
    return d

#read_wordlist('../data/wordlist-2010.eng.txt')
