from fileio import *
from morphochain import *
from objective import *
import pprint

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

with open('data/corpus.p', 'rb') as f:
    en_segmentations = pickle.load(f)
