import fileio
from collections import Counter
import math

MIN_WORD_FREQ = 1
MAX_AFFIX_LEN = 6


def genAffixesList(filename):
    suffixes = Counter()
    prefixes = Counter()
    d = fileio.read_wordcounts(filename)
    for word, count in d.iteritems():
        if count < MIN_WORD_FREQ:
            continue
        for x in xrange(1, len(word)):
            left = word[:x]
            right = word[x:]
            if len(right) <= MAX_AFFIX_LEN and d.get(left, 0) >= MIN_WORD_FREQ:
                suffixes[right] += 1
            if len(left) <= MAX_AFFIX_LEN and d.get(right, 0) >= MIN_WORD_FREQ:
                prefixes[left] += 1

    suffixes = [s[0] for s in suffixes.most_common(100)]
    prefixes = [p[0] for p in prefixes.most_common(100)]
    return suffixes, prefixes


def genAffixesListOpt(wordlist, wordvectors):
    suffixes = Counter()
    prefixes = Counter()
    for word, count in d.iteritems():
        if count < MIN_WORD_FREQ:
            continue
        for x in xrange(1, len(word)):
            prefix = word[:x]
            suffix = word[x:]
            if len(suffix) <= MAX_AFFIX_LEN and d.get(prefix, 0) >= 30 and '-' not in suffix and '\'' not in suffix[1:]:
                if word in wordvectors and prefix in wordvectors:
                    sim = wordvectors.similarity(word, prefix)
                    suffixes[suffix] += sim * (math.log(count) + math.log(d[prefix]))
                else:
                    sim = 0.2
                    suffixes[suffix] += sim * (math.log(count) + math.log(d[prefix]))
    print [s[0] for s in suffixes.most_common(100)]


def genAffixCorrelation(affixes, wordlist):
    d = {}
    for affix in affixes:
        d[affix] = []




filename = '../data/wordlist-2010.eng.txt'
#filename = '../data/somewords.txt'
#genAffixesList(filename)
wordvectors = fileio.load_wordvectors('../data/GoogleNews-vectors-negative300.bin', binary=True)
wordlist = fileio.read_wordcounts(filename)
genAffixesListOpt(wordlist, wordvectors)
