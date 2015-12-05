from __init__ import *
import time

if __name__ == '__main__':
    # while True:
        # word = input("Input word: ")
        # pprint.pprint(en_morpho.getParentsFeatures(word))

    print("start training data gen")
    s = time.time()
    en_morpho.genTrainingData()
    print("time elapsed: " + str(time.time() - s))
