import pickle


def eval(gold_segs_list, predictions_segs_list):
    correct = 0
    goldTotal = 0
    predTotal = 0
    incorrectSegs = []
    correctSegs = []
    for word, pred_seg in predictions_segs_list:
        pred_segs = pred_seg.split('/')
        if word not in gold_segs_list:
            print "Error: " + word
            continue
        predTotal += len(pred_segs)
        bestc = 0
        bestg = 0
        for gold_segs in gold_segs_list[word][0]:
            gset = set(gold_segs)
            pset = set(pred_segs)
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
    precision = correct / predTotal
    recall = correct / goldTotal
    print "Precision: " + precision + " " + "Recall: " + recall


gold = pickle.load(open('data/corpus.p'))