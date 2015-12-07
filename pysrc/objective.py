import scipy.optimize
import numpy.linalg
import numpy as np
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

    Xcoo = X.tocoo()
    bins = np.array(idxs)[:, 1]

    def f(weights):
        F = np.zeros_like(nzs, dtype='float')  # \sum_z e^{\theta*\phi(w[i], z)}
        Xp = np.exp(X.dot(weights)).flatten()  # e^{\theta*\phi(w, z)}

        for i, (a, b) in enumerate(idxs):
            F[i] = Xp[a:b].sum()

        G = Xcoo.copy()
        G.data *= Xp[G.row]
        G.row = np.digitize(G.row, bins)
        G = G.tocsr()[:len(bins)].toarray()

        fv = 0
        gv = 0
        for widx, nbrs in widsneighbors:
            fv += np.log(F[widx]) - np.log(F[nbrs].sum())
            gv += G[widx] / F[widx] - G[nbrs].sum(0) / F[nbrs].sum()
        fv -= lamb * numpy.linalg.norm(weights)**2
        gv -= 2 * lamb * weights
        iteration[0] += 1
        if output:
            with open('out_py/weights.p', 'wb') as f:
                pickle.dump(weights, f)
            print('Call %s' % iteration[0])
            print('\tWeights range: %s %s' % (weights.min(), weights.max()))
            print('\tWeights norm (d=%s): %s' % (weights.size, np.linalg.norm(weights)))
            print('\tFucntion: %s' % fv)
            print('\tGradient range: %s %s' % (gv.min(), gv.max()))
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
