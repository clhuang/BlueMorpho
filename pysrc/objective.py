import scipy.optimize
import numpy.linalg
import numpy as np


def optimize_weights(X, nzs, widsneighbors, lamb=0):
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

    def f(weights):
        Xp = np.exp(X.dot(weights)).flatten()  # e^{\theta*\phi(w, z)}
        F = np.zeros_like(nzs)  # \sum_z e^{\theta*\phi(w[i], z)}
        G = np.zeros((len(nzs), X.shape[1]))   # \sum_z \phi(w[i], z)*e^{\theta*\phi(w[i], z)}
        for i, (a, b) in enumerate(idxs):
            F[i] = Xp[a:b].sum()
            G[i] = Xp[a:b] * X[a:b]
        fv = 0
        gv = 0
        for widx, nbrs in widsneighbors:
            fv += np.log(F[widx]) - np.log(F[nbrs].sum())
            gv += G[widx] / F[widx] - G[nbrs].sum(0) / F[nbrs].sum()
        fv -= lamb * numpy.linalg.norm(weights)
        gv -= 2 * lamb * weights
        # return negative because we want to actually maximize
        return -fv, -gv

    return scipy.optimize.fmin_l_bfgs_b(f, np.zeros_like(X[0].toarray()).T, approx_grad=False,
                                      fprime=None)[0]
