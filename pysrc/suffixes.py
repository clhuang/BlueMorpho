from pysrc import fileio
from collections import Counter, defaultdict
import math
import numpy as np
import numpy.linalg

try:
    import cPickle as pickle
except:
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
    ZERO_VECTOR = lambda: np.zeros(wordvectors.vector_size, dtype='float')
    suffixes = Counter()
    prefixes = Counter()
    suffixvector = defaultdict(ZERO_VECTOR)
    suffixvectorcnt = Counter()
    prefixvector = defaultdict(ZERO_VECTOR)
    prefixvectorcnt = Counter()

    for word, count in wordlist.items():
        if count < MIN_WORD_FREQ:
            continue
        for x in range(1, len(word)):
            prefix = word[:x]
            suffix = word[x:]
            if len(suffix) <= MAX_AFFIX_LEN and wordlist[prefix] >= 30 and '-' not in suffix and '\'' not in suffix[1:]:
                if word in wordvectors and prefix in wordvectors:
                    sim = wordvectors.similarity(word, prefix)
                    wvdiff = wordvectors[word] - wordvectors[prefix]
                    wvdiff /= np.linalg.norm(wvdiff)
                    suffixvector[suffix] += wvdiff
                    suffixvectorcnt[suffix] += 1
                    suffixes[suffix] += sim * (math.log(count) + math.log(wordlist[prefix]))
                else:
                    sim = 0.2
                    suffixes[suffix] += sim * (math.log(count) + math.log(wordlist[prefix]))

            if len(prefix) <= MAX_AFFIX_LEN and wordlist[suffix] >= 30 and '-' not in prefix and '\'' not in prefix[1:]:
                if word in wordvectors and suffix in wordvectors:
                    sim = wordvectors.similarity(word, suffix)
                    wvdiff = wordvectors[word] - wordvectors[suffix]
                    wvdiff /= np.linalg.norm(wvdiff)
                    prefixvector[prefix] += wvdiff
                    prefixvectorcnt[prefix] += 1
                    prefixes[prefix] += sim * (math.log(count) + math.log(wordlist[suffix]))
                else:
                    sim = 0.2
                    prefixes[prefix] += sim * (math.log(count) + math.log(wordlist[suffix]))

    for suffix in suffixvector:
        suffixvector[suffix] /= suffixvectorcnt[suffix]
        suffixes[suffix] *= (1 + 0.3 * np.linalg.norm(suffixvector[suffix]))
    for prefix in prefixvector:
        prefixvector[prefix] /= prefixvectorcnt[prefix]
        prefixes[prefix] *= (1 + 0.3 * np.linalg.norm(prefixvector[prefix]))
    suff_list = [s[0] for s in suffixes.most_common(100)]
    pref_list = [p[0] for p in prefixes.most_common(100)]
    return suff_list, pref_list


def genAffixCorrelation(affixes, wordlist, fname='../data/prefix_corr3.p', suff=True):
    d = {}
    d2 = {}
    for affix in affixes:
        d[affix] = set()
        affixlen = len(affix)
        for word in wordlist:
            word_affix = word[-affixlen:] if suff else word[:affixlen]
            word_root = word[:-affixlen] if suff else word[affixlen:]
            if word_affix == affix:
                d[affix].add(word_root)
    for affix in affixes:
        d2[affix] = []
        for affix2 in affixes:
            if affix != affix2:
                d2[affix].append((affix2, float(len(d[affix] & d[affix2])) / len(d[affix])))
        d2[affix].sort(key = lambda x: x[1])
    with open(fname, 'wb') as f:
        pickle.dump(d2, f)

def genGoldAffix(goldSegsList):
    prefixes = Counter()
    suffixes = Counter()
    for word, goldSegs in goldSegsList.items():
        for q, goldSeg in zip(*goldSegs):
            for thing, seg in zip(q, goldSeg):
                tags = seg.split('_')
                if len(tags) < 2:
                    suffixes[thing] += 1
                elif tags[1] == 's':
                    suffixes[tags[0]] += 1
                elif tags[1] == 'p':
                    prefixes[tags[0]] += 1
    prefixes = [p[0] for p in prefixes.most_common()]
    suffixes = [s[0] for s in suffixes.most_common()]
    return prefixes, suffixes




entr = {'eng': 'en', 'tur': 'tr'}

lang  = 'eng'
size = 'filtered'

filename_w = 'data/wordlist-2010.%s%s.txt' % (lang, '' if size == 'filtered' else size)
filename_v = 'data/%s-wordvectors200_%s.bin' % (entr[lang], size)
wordlist = fileio.read_wordcounts(filename_w)
#wordvectors = fileio.load_wordvectors(filename_v, binary=True)

#suffixes, prefixes = genAffixesListOpt(wordlist, wordvectors)
#with open('data/%s_suffix_list_gold.p' % lang, 'wb') as f:
    #pickle.dump(suffixes, f)
#with open('data/%s_prefix_list_gold.p' % lang, 'wb') as f:
    #pickle.dump(prefixes, f)
#genAffixCorrelation(suffixes, wordlist, fname='data/%s_suffix_corr_gold.p'%lang, suff=True)
#genAffixCorrelation(prefixes, wordlist, fname='data/%s_prefix_corr_gold.p'%lang, suff=False)
#filename = 'data/goldstd_trainset.segmentation.eng.txt'
#corpus = fileio.readCorpus(filename)
#prefixes, suffixes = genGoldAffix(corpus)
#print prefixes
#print suffixes
#with open('data/%s_suffix_list_gold.p' % lang, 'wb') as f:
    #pickle.dump(suffixes, f)
#with open('data/%s_prefix_list_gold.p' % lang, 'wb') as f:
    #pickle.dump(prefixes, f)

#with open('data/%s_suffix_list_gold.p' % lang, 'wb') as f:
    #suffixes = pickle.load(f)
#with open('data/%s_prefix_list_gold.p' % lang, 'wb') as f:
    #prefixes = pickle.load(f)

genAffixCorrelation(suffixes, wordlist, fname='data/%s_suffix_corr_gold.p'%lang, suff=True)
genAffixCorrelation(prefixes, wordlist, fname='data/%s_prefix_corr_gold.p'%lang, suff=False)
