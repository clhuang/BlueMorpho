from fileio import *
from params import *
from morphochain import *
from objective import *
import pprint

en_wordcounts = read_wordcounts('data/wordlist-2010.small.txt')
# en_wordcounts = read_wordcounts('data/wordlist-2010.large.txt')
en_wordvectors = load_wordvectors('data/en-wordvectors200_small.txt')
en_prefixes = pickle.load(open('data/prefix_list.p', 'rb'))
en_suffixes = pickle.load(open('data/suffix_list.p', 'rb'))
en_affixes = (en_prefixes, en_suffixes)

en_morpho = MorphoChain(en_wordvectors, en_wordcounts, en_affixes)
