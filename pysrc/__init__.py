from fileio import *
from globals import *
from morphochain import *
from objective import *
import pprint

en_wordcounts = read_wordcounts('data/wordlist.sample')
en_wordvectors = load_wordvectors('data/en-wordvectors200_small.txt')
en_affixes = None

example = MorphoChain(en_wordvectors, en_wordcounts, en_affixes)

pprint.pprint(example.genCandidates('squished'))
