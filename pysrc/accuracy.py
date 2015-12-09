import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


def seg_points(segs):
    return set(np.cumsum([len(i) for i in segs])[:-1])


def score(gold_segs_list, predictions_segs_list):
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
        predTotal += len(pred_segs) - 1
        bestc = 0
        bestg = 0
        for gold_segs in gold_segs_list[word][0]:
            gset = seg_points(gold_segs)
            pset = seg_points(pred_segs)
            num_correct_segs = len(gset & pset)
            num_gold_segs = len(gold_segs)
            if (bestc < num_correct_segs) or (bestc == num_correct_segs
                                              and bestg > num_gold_segs):
                bestc = num_correct_segs
                bestg = num_gold_segs
        if bestc == bestg:
            correctSegs.append((word, pred_seg, "-".join(gold_segs)))
        else:
            incorrectSegs.append((word, pred_seg, "-".join(gold_segs)))

        goldTotal += bestg
        correct += bestc
    precision = float(correct) / predTotal
    recall = float(correct) / goldTotal
    f1 = 2 * precision * recall / (precision + recall)
    print('Precision: %s\nRecall: %s\nF1: %s' % (precision, recall, f1))
    return precision, recall, f1
