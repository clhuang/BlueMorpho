from fileio import *
from params import *
from morphochain import *
from objective import *
import pprint

en_wordcounts = read_wordcounts('data/wordlist-2010.large.txt')
en_wordvectors = load_wordvectors('data/en-wordvectors200_small.txt')
en_prefixes = pickle.load(open('data/prefix_list.p', 'rb'))
en_suffixes = pickle.load(open('data/suffix_list.p', 'rb'))
en_prefixes_corr = pickle.load(open('data/prefix_corr2.p', 'rb'))
en_suffixes_corr = pickle.load(open('data/suffix_corr2.p', 'rb'))
en_affixes = (en_prefixes, en_suffixes)
en_affix_corr = (en_prefixes_corr, en_suffixes_corr)

en_morpho = MorphoChain(en_wordvectors, en_wordcounts, en_affixes, en_)
