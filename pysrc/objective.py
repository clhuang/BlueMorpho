import scipy.optimize
import numpy.linalg
import numpy as np

from globals import *


def optimize_weights(X, nzs, neighbors, lamb=0):
    '''
    X is a feature matrix, where each row corresponds
    to a (w, z) pair, where the w's appear in order.
    For example the rows might correspond to:

        (fishing, fish)
        (fishing, fishn)
        (fishing, ishing)
        (libraries, library)
        (libraries, librari)
        (bananas...

    nzs is a list corresponding to the number of z's for each word.
    In this case, nzs would contain [2 (for fishing), 2 (for libraries), ...]
    and neighbors contains a list containing the indices of the neighbors.
    '''
    idx = 0
    idxs = []
    for nz in nzs:
        idxs.append((idx, idx+nz))
        idx += nz

    def f(weights):
        Xp = weights.dot(X.T)
        F = np.zeros_like(nzs)
        G = np.zeros((len(nzs), X.shape[1]))
        for i, (a, b) in enumerate(idxs):
            F[i] = Xp[a:b].sum()
            G[i] = (X[a:b] * Xp[a:b]).sum()
        fv = 0
        gv = 0
        for i, nbrs in enumerate(neighbors):
            fv += np.log(F[i]) - np.log(F[nbrs].sum())
            gv += G[i] / F[i] - G[nbrs].sum(0) / F[nbrs].sum()
        fv -= lamb * numpy.linalg.norm(weights)
        gv -= 2 * lamb * weights
        # return negative because fmin
        return -fv, -gv

    return scipy.optimize.fmin_l_bfgs(f, np.zeros_like(X[0]), approx_grad=False,
                                      fprime=None)[0]
