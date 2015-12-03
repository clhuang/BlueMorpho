from gensim.models.word2vec import Word2Vec


def load_wordvectors(filename, fvocab=None, binary=False):
    return Word2Vec.load_word2vec_format(filename, fvocab=fvocab, binary=binary)


def read_wordcounts(filename):
    """
    Returns a dict of words and their corresponding counts.
    """
    d = {}
    with open(filename) as f:
        for line in f:
            count, word = line.split()
            d[word] = int(count)
    return d


# Reads from morpho challege
def readCorpus(filename):
    f = open(filename, 'r')
    d = {}  # maps word to a tuple containing segmentation and a list of tags
    for lines in f:
        lines = lines.split()
        lines = lines.split(',')
#TODO deal with multiple possible segmentations for each word - along
        seg = [g.split(':')[0] for g in lines[1:]]
        tags = [g.split(':')[1] for g in lines[1:]]
        d[lines[0]] = (seg, tags)
    return d

#read_wordlist('../data/wordlist-2010.eng.txt')


def read_words(filename):
    """Read words into a set without counts."""
    s = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            s.add(line.strip())
    return s
