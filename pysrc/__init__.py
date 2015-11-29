from fileio import *
from globals import *
from morphochain import *
from objective import *

en_wordcounts = read_wordcounts('data/wordlist.sample')
en_wordvectors = load_wordvectors('data/vectors.sample', fvocab='data/wordlist.sample', binary=False)

example = MorphoChain(en_wordvectors, en_wordcounts)
