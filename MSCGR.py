import numpy as np
from data_loader import load_mat
import math
from sklearn import metrics
import warnings
import argparse
from W_Construct import norm_W,KNN

warnings.filterwarnings("ignore",category=DeprecationWarning)
parser = argparse.ArgumentParser(description='MSCGR')
parser.add_argument('--epochs', '-te', type=int, default=50, help='number of train_epochs')
parser.add_argument('--dataset', type=str, default='WikipediaArticles', help='choose a dataset')
args = parser.parse_args()
def NMD(X,n_cluster,n_iter,l1,l2,idx,k):
    n_view = X.size
    N, _ = X[0].shape
    alpha = np.ones(n_view) / n_view
    val, vec = np.linalg.eigh(norm_W(KNN(X[idx],k)))
    F = vec[:, -n_cluster:]
    G,obj,Z,R= opt(X,F,alpha,n_iter,n_cluster,l1,l2)
    y_pred = np.argmax(G, axis=1) + 1
    return y_pred,obj,Z,R

def opt(X,F, alpha, NITER,n_cluster,l1,l2):

    n_view = alpha.size
    N, _ = X[0].shape
    X_n = np.zeros((N,N))
    OBJ = []
    G = F
    for iter in range(NITER):
        #update S
        for i in range(n_view):
            X_n +=alpha[i]*X[i]@X[i].T
        S = np.linalg.inv(X_n+l1*np.eye(N))@(X_n.T+l1*F@G.T)
        row, col = np.diag_indices_from(S)
        S[row,col] = 0

        #update F
        U_f, lambda_f, Vf = np.linalg.svd(l1*S.T@G+l2*G)
        A = np.concatenate((np.identity(n_cluster), np.zeros((N - n_cluster, n_cluster))), axis=0)
        F = U_f @ A @ Vf
        # update G
        G = (l1 * S.T @ F + l2 * F) / (l1 + l2)
        G[G < 0] = 0
        #update alpha
        for i in range(n_view):
            alpha[i] = np.power(np.square(np.linalg.norm(X[i].T-X[i].T@S)),-0.5)
        alpha = alpha/np.sum(alpha)
        loss = 0
        l_aplha = 0
        for i in range(n_view):
            loss += alpha[i]*np.square(np.linalg.norm(X[i].T-X[i].T@S))
            l_aplha += np.power(alpha[i],-1)
        obj =loss+l1*np.square(np.linalg.norm(S - F @ G.T))+ l2*np.square(np.linalg.norm(F - G))+l_aplha
        OBJ.append(obj)

    return G,OBJ,S,F@G.T

def EProjSimplex_new(v,k=1):
    ft = 1
    n = v.size
    v_0 = v - np.mean(v) + k/n
    v_min = np.min(v_0)
    if v_min<0:
        f = 1
        lambda_m = 0
        while abs(f)> 1e-10:
            v_1 = v_0 - lambda_m
            posidx = v_1>0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v_1[posidx])-k
            lambda_m = lambda_m-f/g
            ft = ft+1
            if ft>100:
                v_1[v_1<0]=0
                x = v_1
                break
            v_1[v_1<0]=0
            x = v_1
    else:
        x = v_0
    return x


def SimplexQP_ALM(A,b,x,mu=5,beta=1.5):
    N_inter = 500
    threshold = 1e-8
    val = 0
    v = np.ones(x.shape[0])
    lambda_n = np.ones(x.shape[0])
    cnt = 0

    for i in range(N_inter):
        x = EProjSimplex_new(v-1/mu*(lambda_n+A@v-b))
        v = x+1/mu*(lambda_n-A.T@x)
        lambda_n = lambda_n + mu*(x-v)
        mu = beta*mu
        val_old = val
        val = x.T@A@x -x.T@b

        if abs(val - val_old).all() < threshold:
            if cnt >=5:
                break
            else:
                cnt +=1
        else:
            cnt = 0

    return x

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size


def NMI(A,B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)   # Find the intersection of two arrays.
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)


if __name__=="__main__":
    X,GT,idx,n_k = load_mat(args)
    N, _ = X[0].shape
    n_cluster =len(np.unique(GT))
    GT = GT.reshape(np.max(GT.shape), )

    L1 = [10,100,1e3,1e4,1e5]
    L2 = [10, 100, 1e3,1e4,1e5]
    for i in range(5):
        for j in range(5):
            y_pred ,obj,Z,R= NMD(X, n_cluster, args.epochs,L1[i],L2[j],idx,n_k)
            ACC = acc(GT, y_pred)
            NMI = metrics.normalized_mutual_info_score(GT, y_pred)
            Purity = purity_score(GT, y_pred)
            ARI = metrics.adjusted_rand_score(GT, y_pred)
            print( 'clustering accuracy: {}, NMI: {}, Purity: {},ARI: {}，l1:{},l2:{}'.format(ACC, NMI, Purity, ARI, L1[i],L2[j]))
