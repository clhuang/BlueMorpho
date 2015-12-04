import fileio
from collections import Counter, defaultdict
import math
import pickle
import numpy as np
import numpy.linalg

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
    print(suff_list)
    print(pref_list)


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
                d2[(affix, affix2)] = float(len(d[affix] & d[affix2])) / len(d[affix])
    pickle.dump(d2, open("prefix_corr.p", "wb"))


filename = '../data/wordlist-2010.eng.txt'

wordlist = fileio.read_wordcounts(filename, True)

#wordvectors = fileio.load_wordvectors('../data/vectors_filtered/en/vectors200_filtered.txt')
# wordvectors = fileio.load_wordvectors('data/en-vectors200_filtered.txt')
# wordvectors = fileio.load_wordvectors('data/en-wordvectors200_small.txt')
wordlist = fileio.read_wordcounts(filename)
#genAffixesListOpt(wordlist, wordvectors)
#prefix_list = pickle.load(open("../data/prefix_list.p", "rb"))
#suffix_list = pickle.load(open("../data/suffix_list.p", "rb"))
suffixes = ['s', "'s", 'ing', 'ed', 'd', 'ly', "'", 'er', 'e', 'es', 'y', 'n', 'ers', 'ness', 'a', 'r', 'i', 'rs', 'o', 't', 'al', 'l', 'man', 'ally', 'ism', 'less', 'able', 'ist', 'en', 'ity', 'on', 'in', 'an', 'h', 'ns', 'ic', 'ment', 'ian', 'ings', 'ion', 'm', 'ie', 'g', 'ists', 'c', 'land', 'men', 'k', 'son', 'is', 'est', 'ful', 'ized', 'ville', 'ship', 'na', 'ting', 'ation', 'ish', 'le', 'ne', 'ies', 'u', 'ry', 'p', 'ia', 'as', 'line', 'ling', 'ments', 'ions', 'ier', 'b', 'like', 'f', 'or', 'ton', 'la', 'hip', 'ping', 'el', 'os', 'side', 'ted', 'us', 'x', 'ize', 'z', 'ter', 'ised', 'izing', 'st', 'ta', 'led', 'house', 'ni', 'ped', 'ee', 'to', 'way']
prefixes = ['un', 're', 's', 'a', 'over', 'de', 'c', 'in', 'non', 'b', 'p', 't', 'dis', 'm', 'd', 'g', 'e', 'k', 'super', 'f', 'h', 'under', 'pre', 'mis', 'inter', 'out', 'i', 'n', 'r', 'w', 'mc', 'sub', 'l', 'o', 'co', 'micro', 'ma', 'la', 'bio', 'multi', 'im', 'be', 'al', 'en', 'v', 'j', 'euro', 'u', 'sa', 'air', 'tele', 'st', 'le', 'anti', 'up', 'sun', 'di', 'ca', 'to', 'mo', 'ba', 'sh', 'back', 'con', 'y', 'mid', 'ka', 'da', 'trans', 'ta', 'sea', 'se', 'bi', 'an', 'z', 'car', 'ro', 'sc', 'ha', 'pro', 'ar', 'na', 'mi', 'home', 'mar', 'fore', 'ra', 'the', 'mega', 'hand', 'pa', 'bar', 'su', 'ad', 'bo', 'mac', 'post', 'mini', 'ch', 'head']
genAffixCorrelation(prefixes, wordlist)
