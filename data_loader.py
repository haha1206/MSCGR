import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from W_Construct import norm_W,KNN
from sklearn import manifold
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

def load_mat(args):
    if args.dataset == 'COIL20':
        index = 1
        n_k = 2
    if args.dataset == 'HW':
        index = 3
        n_k = 15
    if args.dataset == 'bbcsport':
        index = 1
        n_k = 5
    if args.dataset == 'WikipediaArticles':
        index = 1
        n_k = 15

    data = sio.loadmat('data/{}.mat'.format(args.dataset))

    X = np.squeeze(data['X'])

    Y = np.squeeze(data['Y'])

    return X,Y,index,n_k

