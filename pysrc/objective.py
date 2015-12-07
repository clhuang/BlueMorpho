import scipy.optimize
import numpy.linalg
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle


def optimize_weights(X, nzs, widsneighbors, lamb=0, output=True):
    '''
    X is a feature matrix, where each row corresponds
    to a (w, z) pair, where the w's appear in order.
    For example the rows might correspond to:

        (fishing, fish)
        (fishing, fishn)
        (fishing, ishing)
        (notanactualword, word)
        (libraries, library)
        (libraries, librari)
        (bananas, ananas)

    nzs is a list corresponding to the number of z's for each word.
    In this case, nzs would contain [3 (for fishing), 1 (for notanactualword), 2, 1].
    and neighbors contains a list containing the indices of the neighbors.

    widsneighbors contains tuples, each tuple contains (the ids of the words, ids of the word's neighbors).
    '''
    idx = 0
    idxs = []  # starting, ending indices for each word/neighbor
    for nz in nzs:
        idxs.append((idx, idx+nz))
        idx += nz

    iteration = [0]

    Gcoo = X.tocoo()
    orow = Gcoo.row
    nrow = np.digitize(orow, np.array(idxs)[:, 1])
    nexamples, nfeatures = X.shape
    nnrow = np.zeros_like(nrow)

    arow = np.digitize(np.arange(nexamples), np.array(idxs)[:, 1])

    fnbridx = np.zeros_like(nzs, dtype=int)
    for i, (widx, nbrs) in enumerate(widsneighbors):
        fnbridx[widx] = len(widsneighbors)
        fnbridx[nbrs] = i
    widx = np.fromiter((i[0] for i in widsneighbors), int)
    idxc = np.zeros_like(nzs, dtype=int)

    import time

    def f(weights):
        stime = time.time()
        Xp = np.exp(X.dot(weights)).flatten()  # e^{\theta*\phi(w, z)}
        F = np.bincount(arow, Xp, minlength=len(idxs))

        fn = np.zeros_like(F)
        fnbrssum = np.bincount(fnbridx, F, minlength=len(widx)+1)

        fv = np.log(F[widx]).sum() - np.log(fnbrssum)[:-1].sum()
        fn = -1 / fnbrssum[fnbridx]
        fn[widx] = 1 / F[widx]

        data = Gcoo.data * Xp[orow] * fn[nrow]
        gv = np.bincount(Gcoo.col, data, minlength=nfeatures)

        fv -= lamb * numpy.linalg.norm(weights)**2
        gv -= 2 * lamb * weights
        iteration[0] += 1
        if output:
            with open('out_py/weights.p', 'wb') as f:
                pickle.dump(weights, f)
            print('Call %s' % iteration[0])
            print('\tWeights range: %s %s' % (weights.min(), weights.max()))
            print('\tWeights norm (d=%s): %s' % (weights.size, np.linalg.norm(weights)))
            print('\tFunction: %s' % fv)
            print('\tGradient range: %s %s' % (gv.min(), gv.max()))
            print('\tTime: %s' % (time.time() - stime))
        # return negative because we want to actually maximize
        return -fv, -gv

    BOUNDS = 50

    return scipy.optimize.fmin_l_bfgs_b(
            f,
            np.zeros_like(X[0].toarray()).T,
            bounds=[(-BOUNDS, BOUNDS)]*X.shape[1],
            approx_grad=False,
            fprime=None
            )[0]
