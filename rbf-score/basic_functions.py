import math
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import pandas as pd
from utils.evaluations import *
from pylab import *
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs, eigsh
import scipy.io
from sklearn.metrics.cluster import normalized_mutual_info_score
import networkx as nx
import hSCORE
import utils.higer_approximation as ha
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Times New Roman"
import numpy.matlib


def condi(M):
    return LA.cond(M)


def networkG(W):
    G = nx.from_numpy_matrix(W)

    return G


def get_mat(filename):
    mat = scipy.io.loadmat(filename)
    label = mat['label']
    y = np.asarray(label).reshape(label.shape[0])
    return mat['A'], y


def MQ(data, DM, c):
    MQ = sqrt(DM ** 2 + c ** 2)
    A = MQ * data

    return A


def iMQ(data, DM, c):
    IMQ = 1. / sqrt(DM ** 2 + c ** 2)
    A = IMQ * data

    return A


def gaussianBF(data, DM, c):
    g = np.exp(-DM ** 2 / c ** 2)
    A = g * data

    return A


def DMatrix(data):
    N = data.shape[1]
    x = np.transpose(np.linspace(0, 1, N)).reshape(N, 1)
    a = numpy.matlib.repmat(x, 1, N)
    b = numpy.matlib.repmat(np.transpose(x), N, 1)

    DM = np.sqrt((a - b) ** 2)
    return DM


def get_NMI(y, pre):
    return normalized_mutual_info_score(y, pre)


def get_Q(com, G):
    # print(com)
    comm = [[] for i in range(len(set(com)) + 1)]
    # print(set(com))
    # print(comm)
    if min(com) == 1:
        for m in range(len(com)):
            comm[com[m] - 1].append(m)
    else:
        for m in range(len(com)):
            comm[com[m]].append(m)
    # print(comm)
    # print(nx.algorithms.community.modularity(G, comm))
    return nx.algorithms.community.modularity(G, comm)


def plotNMIQ(condition, NMI, Q, fn):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    # ax1.set_xlabel(r"$\log_2 (condition~number)$")  # Add an x-label to the axes.
    ax1.set_ylabel("NMI")  # Add a y-label to the axes.
    ax1.set_title("Comparisons on condition numbers of " + fn.split('/')[
        1])  # Add a title to the axes.ax1.xlabel(r"$\log_2 (condition~number)$", fontdict={'fontsize': 20})  # Add an x-label to the axes.
    ax2.set_xlabel(r"$\log_2 (condition~number)$")  # Add an x-label to the axes
    ax2.set_ylabel("Modularity")  # Add a y-label to the axes.
    # ax2.set_title("Modularity on condition numbers of an adjacency matrix")  # Add a title to the axes.

    ax1.plot(condition, NMI, '*b')
    ax2.plot(condition, Q, '^m')
    # plt.show()
    fig.savefig(fn + '.pdf')
    print(fn + ' completed...')


def SCOREplus(W, k, c=0.1):
    start = time.time()
    r = k + 1
    Degree = np.sum(W, axis=1)
    delta = c * max(Degree)
    d = 1. / np.sqrt(np.add(delta, Degree))
    sqrtMatrix = np.diag(d)
    L = np.dot(np.dot(sqrtMatrix, W), sqrtMatrix)
    eig_val, eig_vect = eigsh(L, r, which='LM')
    F = eig_vect[:, :r - 1]
    for i in range(r - 1):
        F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])
    temp = (eig_val[0] - eig_val[1]) / eig_val[1]
    if temp <= c:
        F = F[:, 1:(r - 1)]
    sp_kmeans = KMeans(n_clusters=k).fit(F)
    end = time.time()
    return sp_kmeans.labels_, end - start


def import_real_data(fn):
    # for lesmis
    fnn = "../data/lesmis/"

    if fn.strip() == "lesmis":
        data = pd.read_table(fnn + 'lesmis.txt', header=None, sep='\t').to_numpy()
        label = pd.read_table(fnn + 'lesmis_stan.txt', header=None, sep='\t')
        l = label.shape[0]
        y = [label[0][i] for i in range(l)]
        k = len(set(y))

    # for other data
    else:
        folder = '../data/datasets/'
        W, y = get_mat(folder + fn + '.mat')
        data = W
        k = max(y) - min(y) + 1
    return data, y, k


def import_LFR_data(nn, p1, p2, mu):
    folder = '../data/LFR/' + str(nn)

    fn = str(mu) + '_' + str(p1) + '_' + str(p2)
    filename = folder + '/LFR__' + fn
    # G = nx.read_adjlist(filename+'.adjlist', nodetype=int)
    label = pd.read_table(filename + '.labels', header=None, sep='\t')
    W = pd.read_table(filename + '.adj', header=None, sep=' ').to_numpy()
    l = label.shape[0]
    y = [label[0][i] for i in range(l)]
    # W = nx.to_pandas_adjacency(G)
    # G = nx.from_numpy_matrix(W)
    data = W
    # print(len(W))
    k = max(y) - min(y) + 1
    return data, y, k


def run_networks(data, y, k, RBF='MQ'):
    # for lesmis
    # fnn = "../data/lesmis/"
    #
    # if fn.strip() == "lesmis":
    #     data = pd.read_table(fnn + 'lesmis.txt', header=None, sep='\t').to_numpy()
    #     label = pd.read_table(fnn + 'lesmis_stan.txt', header=None, sep='\t')
    #     l = label.shape[0]
    #     y = [label[0][i] for i in range(l)]
    #     k = len(set(y))
    #
    # # for other data
    # else:
    #     folder = '../data/datasets/'
    #     W, y = get_mat(folder + fn + '.mat')
    #     data = W
    #     k = max(y) - min(y) + 1

    G = networkG(data)
    n = 100
    if RBF == 'MQ':
        c = np.linspace(0, 5, n)
    elif RBF == 'iMQ':
        c = np.linspace(0.01, 0.1, n)
    elif RBF == 'gaussian':
        c = linspace(0.01, 1, n)

    condition = [0 for i in range(n)]
    NMI = [0 for i in range(n)]
    Q = [0 for i in range(n)]

    DM = DMatrix(data)
    for i in range(n):
        if RBF == 'MQ':
            A = MQ(data, DM, c[i])
        elif RBF == 'iMQ':
            A = iMQ(data, DM, c[i])
        elif RBF == 'gaussian':
            A = gaussianBF(data, DM, c[i])

        w = np.absolute(np.linalg.eigvals(A))
        W = np.dot(A, data)
        # Algorithms
        splus, t4 = SCOREplus(W, k)

        condition[i] = np.log2(max(w) / min(w))
        NMI4 = round(get_NMI(y, splus), 3)
        Q4 = round(modularity(splus, G), 3)
        NMI[i] = NMI4
        Q[i] = Q4
    # print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"+RBF+"*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    # print("c\tcondition\tNMI\tQ")
    # print("c, condition number for optimal NMI ")
    q = NMI.index(max(NMI))
    # print(round(c[q], 4), round(2 ** condition[q], 4), NMI[q], Q[q])
    #
    # print("\nc, condition number for optimal Modularity - ")
    # q = Q.index(max(Q))
    print(round(c[q], 4), round(2 ** condition[q], 4), NMI[q], Q[q])

    # plotNMIQ(condition, NMI, Q, fn.split('.')[0])


if __name__ == "__main__":
    '''
    Run real-world data set
    fn: data name
    RBF: RBFs, {'MQ', 'iMQ', 'gaussian'}
    '''
    # data, y, k = import_real_data(fn='karate')
    # run_networks(data, y, k, RBF='MQ')
    
    '''
    Run synthetic data set for a specific # Node, p1, p2 on all mu
    '''
    # 
    # nn, p1, p2 = 80, 5, 5
    # nn, p1, p2 = 100, 10, 5
    # nn, p1, p2 = 120, 10, 5
    # nn, p1, p2 = 150, 10, 10
    # nn, p1, p2 = 300, 15, 10
    # nn, p1, p2 = 500, 15, 15
    nn, p1, p2 = 800, 20, 15
    for mu in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        data, y, k = import_LFR_data(nn, p1, p2, mu)
        run_networks(data, y, k, RBF='MQ')
        # run_networks(data, y, k, RBF='gaussian')
    # run_networks(data, y, k, RBF='iMQ')
    # run_networks(data, y, k, RBF='gaussian')

