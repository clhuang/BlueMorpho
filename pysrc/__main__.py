from pysrc import *
import rlcompleter
import readline
import sys
import os.path
import argparse

try:
    import cPickle as pickle
except:
    import pickle

readline.parse_and_bind('tab: complete')

"""
args: [optimize | load | run] [--supervised=lamb] [--now]
[--vectors=[small, med, large, full]] [--vocab=[small, med, large, full]]

optimize: run the optimizer to find weights
load: load weights to do stuff with
run: like load, but has a simple text interface
supervised: for the supervised model, if lamb is not 0, by how much to weight the likelihood
now: option for load and run to use dictvectorizer and weights currently being generated
wordlist_size: default is smallw, can also set to medw, largew, or fullw for different sizes of training
wordvector_size: default is smallv, can also set to medv or fullv for different sizes of word2vec
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['optimize', 'load', 'run'])
    parser.add_argument('--now', action='store_true')
    parser.add_argument('--vocab', choices=['small', 'med', 'large', 'full'], default='small')
    parser.add_argument('--vectors', choices=['small', 'med', 'large', 'full'], default='small')
    parser.add_argument('--supervised', type=float, default=0.0)
    args = parser.parse_args()
    file_w = 'data/wordlist-2010.small.txt'
    if args.vocab == 'med':
        file_w = 'data/wordlist-2010.med.txt'
    if args.vocab == 'large':
        file_w = 'data/wordlist-2010.large.txt'
    if args.vocab == 'full':
        file_w = 'data/wordlist-2010.eng.txt'
    en_wordcounts = read_wordcounts(file_w)

    file_v = 'data/en-wordvectors200_small.txt'
    if args.vectors == 'med':
        file_v = 'data/en-wordvectors200_med.txt'
    if args.vectors == 'full':
        file_v = 'data/en-wordvectors200_filtered.txt'

    binfile_v = file_v[:-3] + 'bin'
    if os.path.isfile(binfile_v):
        en_wordvectors = load_wordvectors(binfile_v, binary=True)
    else:
        en_wordvectors = load_wordvectors(file_v)

    if sys.version_info >= (3, 0):
        raw_input = input  # ghettooooooooo

    en_args = (en_wordvectors, en_wordcounts, en_affixes, en_affix_corr)
    en_kwargs = {'segmentations': en_trainsegmentations}

    if args.command == 'optimize':
        en_morpho = MorphoChain(*en_args, **en_kwargs)
        print('generating training data')
        train = en_morpho.genTrainingData()
        with open('out_py/dictvectorizer.p', 'wb') as f:
            pickle.dump(en_morpho.dictvectorizer, f)
        with open('out_py/dictvectorizer.%s-%s.p' % (args.vocab, args.vectors), 'wb') as f:
            pickle.dump(en_morpho.dictvectorizer, f)
        if args.supervised:
            sups = en_morpho.genGoldsegTrainingData()
            print('training data saved, optimizing weights')
            weights = optimize_weights_supervised(*(train + sups), lamb2=args.supervised)
            en_morpho.setWeightVector(weights)
            with open('out_py/weights.supervised%s.%s-%s.p' %
                      (args.supervised, args.vocab, args.vectors), 'wb') as f:
                pickle.dump(weights, f)
        else:
            print('training data saved, optimizing supervised weights')
            weights = optimize_weights(*train)
            en_morpho.setWeightVector(weights)
            with open('out_py/weights.%s-%s.p' % (args.vocab, args.vectors), 'wb') as f:
                pickle.dump(weights, f)

    elif args.command == 'load' or args.command == 'run':
        def loadweights():
            with open('out_py/weights.p', 'rb') as f:
                weights = pickle.load(f)
                en_morpho.setWeightVector(weights)
        if args.now:
            with open('out_py/dictvectorizer.p', 'rb') as f:
                en_kwargs['dictvectorizer'] = pickle.load(f)
            en_morpho = MorphoChain(*en_args, **en_kwargs)
            loadweights()
        else:
            with open('out_py/dictvectorizer.%s-%s.p' % (args.vocab, args.vectors), 'rb') as f:
                en_kwargs['dictvectorizer'] = pickle.load(f)
            with open('out_py/weights%s.%s-%s.p' % ('.supervised%s' % args.supervised
                                                    if args.supervised else '', args.vocab, args.vectors), 'rb') as f:
                en_kwargs['weightvector'] = pickle.load(f)
            en_morpho = MorphoChain(*en_args, **en_kwargs)

    en_morpho.computeAccuracy(en_devsegmentations)

    if args.command == 'run':
        word = raw_input("Enter word: ")
        while True:
            if word == 'ACCURACY_TRAIN':
                print(en_morpho.computeAccuracy())
            elif word =='ACCURACY_DEVEL':
                print(en_morpho.computeAccuracy(en_devsegmentations))
            elif word == 'RELOAD' and args.now:
                loadweights()
            elif word == 'EXIT':
                break
            else:
                pprint.pprint(en_morpho.predict(word))
                print(en_morpho.genSeg(word))
            word = raw_input("Enter word: ")
