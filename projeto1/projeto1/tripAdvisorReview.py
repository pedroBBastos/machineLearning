import split_train_test as stt
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA

import dbscan as dbscan

from mpl_toolkits.mplot3d import Axes3D

# https://archive.ics.uci.edu/ml/datasets/Travel+Reviews#

def executeDBSCAN3D(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))

    clustered = dbscan.DBSCAN(data, matrixDeDistancia, 0.13, 11)
    clusterNumbers = np.unique(clustered[:, 4])

    ax = plt.axes(projection='3d')
    for ci in clusterNumbers:
        ci = clustered[clustered[:, 4] == ci]
        ci = ci[:, :3]
        cix = ci[:, 0]
        ciy = ci[:, 1]
        ciz = ci[:, 2]
        ax.scatter(cix, ciy, ciz, color=np.random.random(3))
    plt.show()


def executeDBSCAN(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))

    clustered = dbscan.DBSCAN(data, matrixDeDistancia, 0.03, 11)
    clusterNumbers = np.unique(clustered[:, 3])

    for ci in clusterNumbers:
        ci = clustered[clustered[:, 3] == ci]
        ci = ci[:, :2]
        cix = ci[:, 0]
        ciy = ci[:, 1]
        plt.plot(cix, ciy, color=np.random.random(3), marker='x', linestyle='')
    plt.show()


if __name__ == '__main__':
    try:
        data = np.genfromtxt('tripadvisor_review.csv', delimiter=',', skip_header=1)
        data = data[:, 1:]
        data = np.nan_to_num(data)

        training_set, test_set = stt.split_train_test(data, 0.1)

        pca = PCA(n_components=2)
        # pca = PCA(n_components=3)
        dataDimensaoReduzida = pca.fit_transform(data)
        print(dataDimensaoReduzida)
        print(dataDimensaoReduzida.shape)

        executeDBSCAN(dataDimensaoReduzida)
        # executeDBSCAN3D(dataDimensaoReduzida)
    except BaseException as e:
        print(e)
