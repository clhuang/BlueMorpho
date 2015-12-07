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
    train = en_morpho.genTrainingData()
    # pprint.pprint(train)

    if 'optimize' in sys.argv:
        weights = optimize_weights(*train)
        self.setWeightVector(weights)

    elif 'load' in sys.argv:
        with open('out_py/weights.p', 'rb') as f:
            weights = pickle.load(f)
        en_morpho.setWeightVector(weights)
        pprint.pprint(en_morpho.predict('enlistment'))
        print(en_morpho.genSeg('enlistment'))
