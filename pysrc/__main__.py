from pysrc import *
from pysrc.morphochain import MorphoChain
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
    parser.add_argument('--lang', choices=['eng', 'tur'], default='eng')
    parser.add_argument('--supervised', type=float, default=0.0)
    args = parser.parse_args()

    if sys.version_info >= (3, 0):
        raw_input = input  # ghettooooooooo

    trainsegmentations, devsegmentations = get_segmentations(args.lang)
    mc_args = (get_wordvectors(args.vectors, args.lang), get_wordlist(args.vocab, args.lang),
               *get_prefixes_affixes(args.lang))
    mc_kwargs = {'segmentations': trainsegmentations}

    if args.command == 'optimize':
        morpho = MorphoChain(*mc_args, **mc_kwargs)
        print('generating training data')
        train = morpho.genTrainingData()
        with open('out_py/dictvectorizer.p', 'wb') as f:
            pickle.dump(morpho.dictvectorizer, f)
        with open('out_py/dictvectorizer.%s-%s.p' % (args.vocab, args.vectors), 'wb') as f:
            pickle.dump(morpho.dictvectorizer, f)
        if args.supervised:
            sups = morpho.genGoldsegTrainingData()
            print('training data saved, optimizing weights')
            weights = objective.optimize_weights_supervised(*(train + sups), lamb2=args.supervised)
            morpho.setWeightVector(weights)
            with open('out_py/weights.supervised%s.%s-%s.p' %
                      (args.supervised, args.vocab, args.vectors), 'wb') as f:
                pickle.dump(weights, f)
        else:
            print('training data saved, optimizing supervised weights')
            weights = objective.optimize_weights(*train)
            morpho.setWeightVector(weights)
            with open('out_py/weights.%s-%s.p' % (args.vocab, args.vectors), 'wb') as f:
                pickle.dump(weights, f)

    elif args.command == 'load' or args.command == 'run':
        def loadweights():
            with open('out_py/weights.p', 'rb') as f:
                weights = pickle.load(f)
                morpho.setWeightVector(weights)
        if args.now:
            with open('out_py/dictvectorizer.p', 'rb') as f:
                mc_kwargs['dictvectorizer'] = pickle.load(f)
            morpho = MorphoChain(*mc_args, **mc_kwargs)
            loadweights()
        else:
            with open('out_py/dictvectorizer.%s-%s.p' % (args.vocab, args.vectors), 'rb') as f:
                mc_kwargs['dictvectorizer'] = pickle.load(f)
            with open('out_py/weights%s.%s-%s.p' % ('.supervised%s' % args.supervised
                                                    if args.supervised else '', args.vocab, args.vectors), 'rb') as f:
                mc_kwargs['weightvector'] = pickle.load(f)
            morpho = MorphoChain(*mc_args, **mc_kwargs)

    morpho.computeAccuracy(devsegmentations)

    if args.command == 'run':
        word = raw_input("Enter word: ")
        while True:
            if word == 'ACCURACY_TRAIN':
                print(morpho.computeAccuracy())
            elif word =='ACCURACY_DEVEL':
                print(morpho.computeAccuracy(devsegmentations))
            elif word == 'RELOAD' and args.now:
                loadweights()
            elif word == 'EXIT':
                break
            else:
                pprint.pprint(morpho.predict(word))
                print(morpho.genSeg(word))
            word = raw_input("Enter word: ")
