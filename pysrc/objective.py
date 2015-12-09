import time
import scipy.optimize
import numpy.linalg
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

BOUNDS = 50
MAX_ITERS = 500

def output_decorate(func):
    iteration = [0]

    def newf(weights):
        stime = time.time()
        fv, gv = func(weights)
        iteration[0] += 1
        with open('out_py/weights.p', 'wb') as f:
            pickle.dump(weights, f)
        print('Call %s' % iteration[0])
        print('\tWeights range: %s %s' % (weights.min(), weights.max()))
        print('\tWeights norm (d=%s): %s' % (weights.size, np.linalg.norm(weights)))
        print('\tFunction: %s' % fv)
        print('\tGradient range: %s %s' % (gv.min(), gv.max()))
        print('\tTime: %s' % (time.time() - stime))

        return fv, gv
    return newf


def optimize_weights(X, nzs, widsneighbors, lamb=0, output=True, maxiter=MAX_ITERS):
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
    f = get_optimizer_fn(X, nzs, widsneighbors, lamb)
    if output:
        f = output_decorate(f)
    return scipy.optimize.fmin_l_bfgs_b(
        f,
        np.zeros_like(X[0].toarray()).T,
        bounds=[(-BOUNDS, BOUNDS)]*X.shape[1],
        approx_grad=False,
        fprime=None,
        maxiter=maxiter)[0]


def optimize_weights_supervised(X, nzs, widsneighbors,
                                Xsup, nzs_sup, cxs_sup,
                                lamb=0, lamb2=1, output=True, maxiter=MAX_ITERS):
    fus = get_optimizer_fn(X, nzs, widsneighbors, lamb)
    fsup = get_logprob_fn(Xsup, nzs_sup, cxs_sup)
    def fcomb(weights):
        f1, g1 = fus(weights)
        f2, g2 = fsup(weights)
        return f1 + lamb2 * f2, g1 + lamb2 * g2

    if output:
        fcomb = output_decorate(fcomb)

    return scipy.optimize.fmin_l_bfgs_b(
        fcomb,
        np.zeros_like(X[0].toarray()).T,
        bounds=[(-BOUNDS, BOUNDS)]*X.shape[1],
        approx_grad=False,
        fprime=None,
        maxiter=maxiter)[0]


def get_logprob_fn(X, nzs, cxs):
    wordbounds = np.cumsum(nzs)

    nexamples, nfeatures = X.shape
    bins = np.digitize(np.arange(nexamples), wordbounds)
    Gcoo = X.tocoo()
    wordbins = bins[Gcoo.row]
    ics = np.bincount(cxs, minlength=nexamples)[Gcoo.row]

    def f(weights):
        Xp = X.dot(weights).flatten()
        Xpexp = np.exp(Xp)
        normcs = np.bincount(bins, Xpexp, minlength=len(nzs))
        logprobs = Xp[cxs] - np.log(normcs)

        data = Gcoo.data * (ics - Xpexp[Gcoo.row] / normcs[wordbins])
        grad = np.bincount(Gcoo.col, data, minlength=nfeatures)

        return -logprobs.sum(), -grad  # want to maximize

    return f

def get_optimizer_fn(X, nzs, widsneighbors, lamb=0):
    wordbounds = np.cumsum(nzs)

    Gcoo = X.tocoo()
    orow = Gcoo.row
    nrow = np.digitize(orow, wordbounds)
    nexamples, nfeatures = X.shape

    arow = np.digitize(np.arange(nexamples), wordbounds)

    fnbridx = np.zeros_like(nzs, dtype=int)
    for i, (widx, nbrs) in enumerate(widsneighbors):
        fnbridx[widx] = len(widsneighbors)
        fnbridx[nbrs] = i
    widx = np.fromiter((i[0] for i in widsneighbors), int)

    def f(weights):
        stime = time.time()
        Xp = np.exp(X.dot(weights)).flatten()  # e^{\theta*\phi(w, z)}
        F = np.bincount(arow, Xp, minlength=len(nzs))

        fnbrssum = np.bincount(fnbridx, F, minlength=len(widx)+1)

        fv = np.log(F[widx]).sum() - np.log(fnbrssum)[:-1].sum()
        fn = -1 / fnbrssum[fnbridx]
        fn[widx] = 1 / F[widx]

        data = Gcoo.data * Xp[orow] * fn[nrow]
        gv = np.bincount(Gcoo.col, data, minlength=nfeatures)

        fv -= lamb * weights.dot(weights)
        gv -= 2 * lamb * weights
        # return negative because we want to actually maximize
        return -fv, -gv

    return f
