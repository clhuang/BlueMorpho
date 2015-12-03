import fileio
from collections import Counter
import math
import pickle

MIN_WORD_FREQ = 1
MAX_AFFIX_LEN = 6


def genAffixesList(filename):
    suffixes = Counter()
    prefixes = Counter()
    d = fileio.read_wordcounts(filename)
    for word, count in d.items():
        if count < MIN_WORD_FREQ:
            continue
        for x in range(1, len(word)):
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
    for word, count in wordlist.items():
        if count < MIN_WORD_FREQ:
            continue
        for x in range(1, len(word)):
            prefix = word[:x]
            suffix = word[x:]
            if len(suffix) <= MAX_AFFIX_LEN and wordlist[prefix] >= 30 and '-' not in suffix and '\'' not in suffix[1:]:
                if word in wordvectors and prefix in wordvectors:
                    sim = wordvectors.similarity(word, prefix)
                    suffixes[suffix] += sim * (math.log(count) + math.log(wordlist[prefix]))
                else:
                    sim = 0.2
                    suffixes[suffix] += sim * (math.log(count) + math.log(wordlist[prefix]))

            if len(prefix) <= MAX_AFFIX_LEN and wordlist[suffix] >= 30 and '-' not in prefix and '\'' not in prefix[1:]:
                if word in wordvectors and suffix in wordvectors:
                    sim = wordvectors.similarity(word, suffix)
                    prefixes[prefix] += sim * (math.log(count) + math.log(wordlist[suffix]))
                else:
                    sim = 0.2
                    prefixes[prefix] += sim * (math.log(count) + math.log(wordlist[suffix]))

    suff_list = ([s[0] for s in suffixes.most_common(100)])
    pref_list =  ([p[0] for p in prefixes.most_common(100)])
    print suff_list
    print pref_list


def genAffixCorrelation(affixes, wordlist):
    d = {}
    d2 = {}
    for affix in affixes:
        d[affix] = set()
        affixlen = len(affix)
        for word in wordlist:
            if word[-affixlen:] == affix:
                d[affix].add(word[:-affixlen])
    for affix in affixes:
        for affix2 in affixes:
            if affix != affix2:
                d2[(affix, affix2)] = len(d[affix] & d[affix2]) / len(d[affix])








filename = '../data/wordlist-2010.eng.txt'
#filename = '../data/somewords.txt'
#genAffixesList(filename)
wordvectors = fileio.load_wordvectors('../data/vectors_filtered/en/vectors200_filtered.txt')
# wordvectors = fileio.load_wordvectors('data/en-wordvectors200_small.txt')
wordlist = fileio.read_wordcounts(filename)
#genAffixesListOpt(wordlist, wordvectors)
prefix_list = pickle.load(open("../data/prefix_list.p", "rb"))
suffix_list = pickle.load(open("../data/suffix_list.p", "rb"))

