import fileio
from collections import Counter
import math

MIN_WORD_FREQ = 1
MAX_AFFIX_LEN = 6
def genffixesList(filename):
    suffixes = Counter()
    prefixes = Counter()
    d = fileio.read_wordlist(filename)
    for word, count in d.iteritems():
        if count < MIN_WORD_FREQ:
            continue
        for x in xrange(1, len(word)):
            prefix = word[:x]
            suffix = word[x:]
            if len(suffix) <= MAX_AFFIX_LEN and d.get(prefix, 0) >= MIN_WORD_FREQ:
                suffixes[suffix] += 1
            if len(prefix) <= MAX_AFFIX_LEN and d.get(suffix, 0) >= MIN_WORD_FREQ:
                prefixes[prefix] += 1

    suffixes = [s[0] for s in suffixes.most_common(100)]
    prefixes = [p[0] for p in prefixes.most_common(100)]
    return suffixes, prefixes


def genAffixesListOpt(filename):
    load_wordvectors(filename, fvocab=None, binary=False)
    suffixes = Counter()
    prefixes = Counter()
    d = fileio.read_wordlist(filename)
    for word, count in d.iteritems():
        if count < MIN_WORD_FREQ:
            continue
        for x in xrange(1, len(word)):
            prefix = word[:x]
            suffix = word[x:]
            if len(suffix) <= MAX_AFFIX_LEN and d.get(prefix, 0) >= 30 and word[x] != '-':
                suffixes[suffix] += (math.log(count) + math.log(d[prefix]))
    print [s[0] for s in suffixes.most_common(100)]


#def genAffixesListGold(filename):
    #suffixes = Counter()
    #prefixes = Counter()
    #d = fileio.readCorpus(filename)
    #for seg, tags in d:
        #prefix = True
        #for s in segs:
            #if s == '~':
                #continue
            #if tags ==
                #prefix = False
            #elif prefix:
                #prefixes += 1
            #else:
                #suffixes += 1
    #print [s[0] for s in suffixes.most_common(100)]


filename = '../data/wordlist-2010.eng.txt'
#filename = '../data/somewords.txt'
#genAffixesList(filename)
genAffixesListGold(filename)
