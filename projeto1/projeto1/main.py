import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

from sklearn import preprocessing
import sklearn.metrics as sklMetrics

import kMeans as km
import dbscan as dbscan
import split_train_test as stt
import elbow as em
import scipy.spatial


def executeDBSCAN(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))

    # knee gráfico
    thirdMinDists = np.sort(np.partition(matrixDeDistancia, 4)[:, 4])
    em.plot_elbow_graphic(thirdMinDists, range(0, len(thirdMinDists)), 2)


    eps_values = [.058]
    #eps_values = np.linspace(0.0, 0.5)
    elbow_values = []
    for eps in eps_values:
        clustered = dbscan.DBSCAN(data, matrixDeDistancia, eps, 3)
        clusterNumbers = np.unique(clustered[:, 3])

        silhouette_score = sklMetrics.silhouette_score(data, clustered[:, 3])

        for ci in clusterNumbers:
            ci = clustered[clustered[:, 3] == ci]
            ci = ci[:, :2]
            cix = ci[:, 0]
            ciy = ci[:, 1]
            plt.plot(cix, ciy, color=np.random.random(3), marker='x', linestyle='')
        plt.show()


def executeKMeans(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    k_clusters = [1, 2, 3, 4, 5, 6, 7]
    elbow_values_plot = []
    for k in k_clusters:
        clustered, centroides = km.kMeans(data, k)
        value = em.elbow_value(clustered, centroides)
        elbow_values_plot.append(value)

        cluster0 = clustered[clustered[:, 2] == 0]
        cluster0 = cluster0[:, :2]
        cluster0x = cluster0[:, 0]
        cluster0y = cluster0[:, 1]

        cluster1 = clustered[clustered[:, 2] == 1]
        cluster1 = cluster1[:, :2]
        cluster1x = cluster1[:, 0]
        cluster1y = cluster1[:, 1]

        cluster2 = clustered[clustered[:, 2] == 2]
        cluster2 = cluster2[:, :2]
        cluster2x = cluster2[:, 0]
        cluster2y = cluster2[:, 1]

        cluster3 = clustered[clustered[:, 2] == 3]
        cluster3 = cluster3[:, :2]
        cluster3x = cluster3[:, 0]
        cluster3y = cluster3[:, 1]

        cluster4 = clustered[clustered[:, 2] == 4]
        cluster4 = cluster4[:, :2]
        cluster4x = cluster4[:, 0]
        cluster4y = cluster4[:, 1]

        cluster5 = clustered[clustered[:, 2] == 5]
        cluster5 = cluster5[:, :2]
        cluster5x = cluster5[:, 0]
        cluster5y = cluster5[:, 1]

        cluster6 = clustered[clustered[:, 2] == 6]
        cluster6 = cluster6[:, :2]
        cluster6x = cluster6[:, 0]
        cluster6y = cluster6[:, 1]

        plt.plot(cluster0x, cluster0y, 'rx', cluster1x, cluster1y, 'gx', cluster2x, cluster2y, 'bx',
                 cluster3x, cluster3y, 'yx', cluster4x, cluster4y, 'mx', cluster5x, cluster5y, 'rx',
                 cluster6x, cluster6y, 'kx')
        plt.show()

    em.plot_elbow_graphic(elbow_values_plot, k_clusters, 1)


if __name__ == '__main__':

    # sys.setrecursionlimit(5500)

    data = np.genfromtxt('cluster.dat')
    training_set, test_set = stt.split_train_test(data, 0.1)
    # executeKMeans(training_set)
    executeDBSCAN(training_set)
    # executeKMeans(test_set)
    # executeDBSCAN(test_set)
