from scipy.sparse.linalg import eigs, eigsh
import utils.utils as utils
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import mixture
import sys
import numpy as np
import time
import math
import utils.higer_approximation as ha
sys.path.append("..")

from utils.laplacian import calLaplacianMatrix
from sklearn.cluster import KMeans
# import utils.higer_approximation as ha


def run_hSCORE(W, k, c=0.1, beta=0.0026):
    start = time.time()
    W = ha.get_1stplus2nd(W, beta)
    r = k + 1
    Degree = np.sum(W, axis=1)
    delta = c * max(Degree)
    d = 1. / np.sqrt(np.add(delta, Degree))
    sqrtMatrix = np.diag(d)
    L = np.dot(np.dot(sqrtMatrix, W), sqrtMatrix)
    eig_val, eig_vect1 = eigsh(L, r, which='LM')
    eig_vect = np.dot(eig_vect1, np.diag(eig_val))

    F = eig_vect[:, :r]
    for i in range(r):
        F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])

    temp = (eig_val[1] - eig_val[0]) / eig_val[1]
    # print(temp)
    if temp > c:
        F = F[:, 1: r]
        # F = F[:, 1:(r - 1)]

    sp_kmeans = KMeans(n_clusters=k).fit(F)
    # dpgmm = mixture.BayesianGaussianMixture(n_components=k, covariance_type='full').fit(F)
    end = time.time()
    return sp_kmeans.labels_, end - start
    # return dpgmm.predict(F), end - start
