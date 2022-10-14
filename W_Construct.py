from sklearn.metrics.pairwise import euclidean_distances as EuDist2
import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power
def KNN(X,knn):
    eps = 2.2204e-16
    n, dim = X.shape
    D = EuDist2(X, X, squared=True)
    NN_full = np.argsort(D, axis=1)
    W = np.zeros((n,n))
    for i in range(n):
        id = NN_full[i,1:(knn+2)]
        di = D[i,id]
        W[i,id] = (di[-1]-di)/(knn*di[-1]-sum(di[:-1])+eps)
    A = (W+W.T)/2
    return A

def norm_W(A):
    d = np.sum(A, 1)
    d[d == 0] = 1e-6
    d_inv = 1 / np.sqrt(d)
    tmp = A * np.outer(d_inv, d_inv)
    A2 = np.maximum(tmp, tmp.T)
    return A2
def ancher_W(X,knn,s):
    eps = 2.2204e-16
    N,_ = X.shape
    kmeans = KMeans(n_clusters=knn)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    # index = random.sample(range(N),knn)
    # centers = X[index]
    D = EuDist2(X,centers,squared=True)
    NN_full = np.argsort(D, axis=1)
    Z = np.zeros((N, knn))
    for i in range(N):
        id = NN_full[i, :(s + 1)]
        di = D[i, id]
        Z[i, id] = (di[-1] - di) / (s * di[-1] - sum(di[:-1]) + eps)
    sum_zz = np.sum(Z,axis=0)
    lambda_z = np.diag(sum_zz)
    lambda_z_1 = fractional_matrix_power(lambda_z,-1)
    W = Z@lambda_z_1@Z.T
    return W






