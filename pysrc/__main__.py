from __init__ import *
import rlcompleter
import readline
import sys
import pickle

readline.parse_and_bind('tab: complete')

if __name__ == '__main__':
    # while True:
        # word = input("Input word: ")
        # pprint.pprint(en_morpho.getParentsFeatures(word))
    # pprint.pprint(train)

    if 'optimize' in sys.argv:
        en_morpho = MorphoChain(en_wordvectors, en_wordcounts, en_affixes, en_affix_corr)
        print('generating training data')
        train = en_morpho.genTrainingData()
        with open('out_py/dictvectorizer.p', 'wb') as f:
            pickle.dump(en_morpho.dictvectorizer, f)
        print('training data saved, optimizing weights')
        weights = optimize_weights(*train)
        en_morpho.setWeightVector(weights)

    elif 'load' in sys.argv:
        with open('out_py/dictvectorizer.p', 'rb') as f:
            dictvectorizer = pickle.load(f)
        en_morpho = MorphoChain(en_wordvectors, en_wordcounts, en_affixes, en_affix_corr,
                                dictvectorizer=dictvectorizer)
        def loadweights():
            with open('out_py/weights.p', 'rb') as f:
                weights = pickle.load(f)
                en_morpho.setWeightVector(weights)
        loadweights()
        word = input("Enter word: ")
        while True:
            if word == 'RELOAD':
                loadweights()
            elif word == 'EXIT':
                break
            pprint.pprint(en_morpho.predict(word))
            print(en_morpho.genSeg(word))
            word = input("Enter word: ")
