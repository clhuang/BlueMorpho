from fileio import *
from morphochain import *
from objective import *
import pprint

en_wordcounts = read_wordcounts('data/wordlist-2010.small.txt')
# en_wordcounts = read_wordcounts('data/wordlist-2010.large.txt')
en_wordvectors = load_wordvectors('data/en-vectors200_filtered.txt')
with open('data/prefix_list.p', 'rb') as f:
    en_prefixes = pickle.load(f)
with open('data/suffix_list.p', 'rb') as f:
    en_suffixes = pickle.load(f)
with open('data/prefix_corr2.p', 'rb') as f:
    en_prefixes_corr = pickle.load(f)
with open('data/suffix_corr2.p', 'rb') as f:
    en_suffixes_corr = pickle.load(f)
en_affixes = (en_prefixes, en_suffixes)
en_affix_corr = (en_prefixes_corr, en_suffixes_corr)

en_morpho = MorphoChain(en_wordvectors, en_wordcounts, en_affixes, en_affix_corr)

en_segmentations = readCorpus('data/goldstd_trainset.segmentation.eng.txt')
