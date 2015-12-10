# -*- coding: utf-8 -*-
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
import re
import codecs
try:
    import cPickle as pickle
except:
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


def read_somewords(filename, dump = False):
    """
    Returns a dict of words and their corresponding counts.
    """
    d = Counter()
    with open(filename) as f:
        for line in f:
            word = line.split()
            word = word[0]
            d[word] = 200
    if dump:
        pickle.dump(d, open("data/somewords.p", "wb"))
    return d


# Reads from morpho challege
def readCorpus(filename, dump=None):
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
        pickle.dump(d, open(dump, "wb"))
    return d

#read_wordlist('data/wordlist-2010.eng.txt')


def read_words(filename):
    """Read words into a set without counts."""
    s = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            s.add(line.strip())
    return s


ROMAN_TT = {ord(c): ord(t) for c, t in zip('öüçýiþðâ', 'OUCiISGA')}
def read_dictionary(filename):
    eng_to_tur = defaultdict(list)
    tur_to_eng = defaultdict(list)

    WWDR = re.compile(r'^([^ ]+)\t([^ ]+)\t')
    with codecs.open(filename, 'r', 'latin-1') as f:
        for line in f:
            match = WWDR.match(line)
            if match:
                eng, tur = match.groups()
                eng = eng.lower()
                tur = tur.lower().translate(ROMAN_TT)
                eng_to_tur[eng].append(tur)
                tur_to_eng[tur].append(eng)
    return eng_to_tur, tur_to_eng


#read_somewords('data/somewords.txt', True)

if __name__=="__main__":
    for lang in ('eng', 'tur'):
        readCorpus('data/goldstd_trainset.segmentation.%s.txt' % lang, 'data/%s_traincorpus.p' % lang)
        readCorpus('data/goldstd_develset.segmentation.%s.txt' % lang, 'data/%s_devcorpus.p' % lang)
