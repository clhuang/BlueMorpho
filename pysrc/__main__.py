from __init__ import *
import rlcompleter
import readline

readline.parse_and_bind('tab: complete')

if __name__ == '__main__':
    # while True:
        # word = input("Input word: ")
        # pprint.pprint(en_morpho.getParentsFeatures(word))
    train = en_morpho.genTrainingData()
    pprint.pprint(train)
    # optimize_weights(*train)
