import numpy as np
import pprint


def seg_points(segs):
    return set(np.cumsum([len(i) for i in segs])[:-1])


def score(gold_segs_list, predictions_segs_list, verbose=False):
    correct = 0
    goldTotal = 0
    predTotal = 0
    incorrectSegs = []
    correctSegs = []
    for word, pred_seg in predictions_segs_list:
        pred_segs = pred_seg.split('/')
        if word not in gold_segs_list:
            print('Error: %s' % word)
            continue
        predpoints = len(pred_segs) - 1
        predTotal += predpoints
        bestc = 0
        bestg = 0
        for i, gold_segs in enumerate(gold_segs_list[word][0]):
            gset = seg_points(gold_segs)
            pset = seg_points(pred_segs)
            num_correct_segs = len(gset & pset)
            num_gold_segs = len(gset)
            if (bestc < num_correct_segs) or (bestc == num_correct_segs
                                              and bestg > num_gold_segs):
                bestc = num_correct_segs
                bestg = num_gold_segs
        if bestc == bestg == predpoints:
            correctSegs.append(pred_seg)
        else:
            incorrectSegs.append((pred_seg, "/".join(gold_segs)))

        goldTotal += bestg
        correct += bestc

    if verbose:
        print("Correct: ")
        pprint.pprint(correctSegs)
        print("Incorrect: ")
        pprint.pprint(incorrectSegs)
    precision = float(correct) / predTotal
    recall = float(correct) / goldTotal
    f1 = 2 * precision * recall / (precision + recall)
    print('Precision: %s\nRecall: %s\nF1: %s' % (precision, recall, f1))
    return precision, recall, f1
