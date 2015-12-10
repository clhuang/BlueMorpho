from pysrc import *
import pysrc.duolingo as duolingo
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
    parser.add_argument('--lang', choices=['eng', 'tur', 'both'], default='eng')
    parser.add_argument('--supervised', type=float, default=0.0)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--recalc', action='store_true')
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--twolang', action='store_true')
    args = parser.parse_args()

    if sys.version_info >= (3, 0):
        raw_input = input  # ghettooooooooo

    trainsegmentations, devsegmentations = get_segmentations(args.lang)
    mc_args = (get_wordvectors(args.vectors, args.lang), get_wordlist(args.vocab, args.lang)) + \
            get_prefixes_affixes(args.lang)
    mc_kwargs = {'segmentations': trainsegmentations}

    MCCls = MorphoChain
    filesuffix = '.%s.%s-%s' % (args.lang, args.vocab, args.vectors)
    ofilesuffix = filesuffix if not args.supervised else \
        '.%s.supervised%s.%s-%s' % (args.lang, args.supervised, args.vocab, args.vectors)

    if args.twolang:
        MCCls = duolingo.TwoLangMorphoChain
        with open('data/eWordToEParents.p', 'rb') as f:
            eWordToEparents = pickle.load(f)
            mc_args = mc_args + (eWordToEparents,)
        filesuffix = '.twolang' + filesuffix
        ofilesuffix = '.twolang' + ofilesuffix

    if args.command == 'optimize':
        morpho = MCCls(*mc_args, **mc_kwargs)
        train_file = 'out_py/%s-train.%s-%s.p' % (args.lang, args.vocab, args.vectors)
        dictvec_file = 'out_py/dictvectorizer%s.p' % filesuffix
        try:
            if args.recalc:
                raise
            with open(dictvec_file, 'rb') as f:
                morpho.dictvectorizer = pickle.load(f)
            with open(train_file, 'rb') as f:
                train = pickle.load(f)
        except:
            print('generating training data')
            train = morpho.genTrainingData()
            with open(train_file, 'wb') as f:
                pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
            with open(dictvec_file, 'wb') as f:
                pickle.dump(morpho.dictvectorizer, f)
        with open('out_py/dictvectorizer.p', 'wb') as f:
            pickle.dump(morpho.dictvectorizer, f)
        if args.supervised:
            sups = morpho.genGoldsegTrainingData()
            print('training data acquired, optimizing weights')
            weights = objective.optimize_weights_supervised(*(train + sups), lamb=args.lamb, lamb2=args.supervised, maxiter=args.maxiter)
            morpho.setWeightVector(weights)
        else:
            print('training data acquired, optimizing supervised weights')
            weights = objective.optimize_weights(*train, lamb=args.lamb, maxiter=args.maxiter)
            morpho.setWeightVector(weights)
        with open('out_py/weights%s.p' % ofilesuffix, 'wb') as f:
            pickle.dump(weights, f)

    elif args.command == 'load' or args.command == 'run':
        def loadweights():
            with open('out_py/weights.p', 'rb') as f:
                weights = pickle.load(f)
                morpho.setWeightVector(weights)
        if args.now:
            with open('out_py/dictvectorizer.p', 'rb') as f:
                mc_kwargs['dictvectorizer'] = pickle.load(f)
            morpho = MCCls(*mc_args, **mc_kwargs)
            loadweights()
        else:
            with open('out_py/dictvectorizer%s.p' % filesuffix, 'rb') as f:
                mc_kwargs['dictvectorizer'] = pickle.load(f)
            with open('out_py/weights%s.p' % ofilesuffix, 'rb') as f:
                mc_kwargs['weightvector'] = pickle.load(f)
            morpho = MCCls(*mc_args, **mc_kwargs)

    if not args.now:
        stdout = sys.stdout
        with open('out_py/results%s.txt' % ofilesuffix, 'w') as sys.stdout:
            print('%s%s %s-%s' % (args.lang, ' supervised(%s)' % args.supervised if args.supervised else '',
                                  args.vocab, args.vectors))
            morpho.computeAccuracy(devsegmentations)
        sys.stdout = stdout
        print('%s%s %s-%s' % (args.lang, ' supervised(%s)' % args.supervised if args.supervised else '',
                              args.vocab, args.vectors))
    morpho.computeAccuracy(devsegmentations)

    if args.command == 'run':
        word = raw_input("Enter word: ")
        while True:
            if word == 'ACCURACY_TRAIN':
                morpho.computeAccuracy(verbose=True)
            elif word =='ACCURACY_DEVEL':
                morpho.computeAccuracy(devsegmentations, True)
            elif word == 'RELOAD' and args.now:
                loadweights()
            elif word == 'EXIT':
                break
            else:
                pprint.pprint(morpho.predict(word))
                print(morpho.genSeg(word))
            word = raw_input("Enter word: ")
