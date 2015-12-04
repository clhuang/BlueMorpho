from gensim.models.word2vec import Word2Vec
from collections import Counter
import pickle

def load_wordvectors(filename, fvocab=None, binary=False):
    return Word2Vec.load_word2vec_format(filename, fvocab=fvocab, binary=binary)


def read_wordcounts(filename, dump = False):
    """
    Returns a dict of words and their corresponding counts.
    """
    d = Counter()
    with open(filename) as f:
        for line in f:
            count, word = line.split()
            d[word] = int(count)
    if dump:
        pickle.dump(d, open("data/wordcounts.p", "wb"))
    return d


# Reads from morpho challege
def readCorpus(filename, dump=False):
    f = open(filename, 'r')
    d = {}  # maps word to a tuple containing segmentation and a list of tags
    for line in f:
        line = line.split('\t')
        word = line[0]
        line = line[1].split(',')
        segs = []
        tags = []
        for segmentation in line:
            segment = segmentation.split()
            segs.append([g.split(':')[0] for g in segment])
            tags.append([g.split(':')[1] for g in segment])
        d[word] = (segs, tags)
    if dump:
        pickle.dump(d, open("data/corpus.p", "wb"))
    return d

#read_wordlist('data/wordlist-2010.eng.txt')


def read_words(filename):
    """Read words into a set without counts."""
    s = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            s.add(line.strip())
    return s

# readCorpus("data/goldstd_trainset.segmentation.eng.txt", True)
