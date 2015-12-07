from __init__ import *
import rlcompleter
import readline
import sys

try:
    import cPickle as pickle
except:
    import pickle

readline.parse_and_bind('tab: complete')

"""
args: [optimize | load | run] [now] [wordlist_size] [wordvector_size]

optimize: run the optimizer to find weights
load: load weights to do stuff with
run: like load, but has a simple text interface
now: option for load and run to use dictvectorizer and weights currently being generated
wordlist_size: default is smallw, can also set to medw, largew, or fullw for different sizes of training
wordvector_size: default is smallv, can also set to medv or fullv for different sizes of word2vec
"""

if __name__ == '__main__':
    w_size = 'small'
    file_w = 'data/wordlist-2010.small.txt'
    if 'medw' in sys.argv:
        w_size = 'med'
        file_w = 'data/wordlist-2010.med.txt'
    if 'largew' in sys.argv:
        w_size = 'large'
        file_w = 'data/wordlist-2010.large.txt'
    if 'fullw' in sys.argv:
        w_size = 'full'
        file_w = 'data/wordlist-2010.eng.txt'
    en_wordcounts = read_wordcounts(file_w)

    v_size = 'small'
    file_v = 'data/en-wordvectors200_small.txt'
    if 'medv' in sys.argv:
        v_size = 'med'
        file_v = 'data/en-wordvectors200_med.txt'
    if 'fullv' in sys.argv:
        v_size = 'full'
        file_v = 'data/en-wordvectors200_filtered.txt'
    en_wordvectors = load_wordvectors(file_v)

    en_args = (en_wordvectors, en_wordcounts, en_affixes, en_affix_corr)
    if sys.version_info >= (3, 0):
        raw_input = input  # ghettooooooooo

    if 'optimize' in sys.argv:
        en_morpho = MorphoChain(*en_args)
        print('generating training data')
        train = en_morpho.genTrainingData()
        with open('out_py/dictvectorizer.p', 'wb') as f:
            pickle.dump(en_morpho.dictvectorizer, f)
        with open('out_py/dictvectorizer.%s-%s.p' % (w_size,v_size), 'wb') as f:
            pickle.dump(en_morpho.dictvectorizer, f)
        print('training data saved, optimizing weights')
        weights = optimize_weights(*train)
        en_morpho.setWeightVector(weights)
        with open('out_py/weights.%s-%s.p' % (w_size,v_size), 'wb') as f:
            pickle.dump(weights, f)

    elif 'load' in sys.argv or 'run' in sys.argv:
        def loadweights():
            with open('out_py/weights.p', 'rb') as f:
                weights = pickle.load(f)
                en_morpho.setWeightVector(weights)
        if 'now' in sys.argv:
            with open('out_py/dictvectorizer.p', 'rb') as f:
                dictvectorizer = pickle.load(f)
            en_morpho = MorphoChain(*en_args, dictvectorizer=dictvectorizer)
            loadweights()
        else:
            with open('out_py/dictvectorizer.%s-%s.p' % (w_size,v_size), 'rb') as f:
                dictvectorizer = pickle.load(f)
            with open('out_py/weights.%s-%s.p' % (w_size,v_size), 'rb') as f:
                weights = pickle.load(f)
            en_morpho = MorphoChain(*en_args, dictvectorizer=dictvectorizer, weightvector=weights)

    if 'run' in sys.argv:
        word = raw_input("Enter word: ")
        while True:
            if word == 'RELOAD' and 'now' in sys.argv:
                loadweights()
            elif word == 'EXIT':
                break
            pprint.pprint(en_morpho.predict(word))
            print(en_morpho.genSeg(word))
            word = raw_input("Enter word: ")
