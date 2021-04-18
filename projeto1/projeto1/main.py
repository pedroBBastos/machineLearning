import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
import kMeans as km
import dbscan as dbscan
import split_train_test as stt
import elbow as em
import scipy.spatial


def executeDBSCAN(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))

    eps_values = [.1,.2,.3,.4,.5]
    elbow_values = []
    for eps in eps_values:
        clustered = dbscan.DBSCAN(data, matrixDeDistancia, eps, 3)
        clusterNumbers = np.unique(clustered[:, 3])

        mediaDistanciasPorCluster = []
        for ci in clusterNumbers:
            indicesPontosCi = np.where(clustered[:, 3] == ci)[0]
            matrixDeDistanciaCi = matrixDeDistancia[indicesPontosCi][:, indicesPontosCi]
            mediaDistanciasPorCluster.append(np.mean(np.sum(matrixDeDistanciaCi, axis=1)))

        mediaDistanciasPorCluster = np.array(mediaDistanciasPorCluster)
        elbow_values.append(sum(mediaDistanciasPorCluster))

        # cluster0 = clustered[clustered[:, 3] == clusterNumbers[0]]
        # cluster0 = cluster0[:, :2]
        # cluster0x = cluster0[:, 0]
        # cluster0y = cluster0[:, 1]
        #
        # cluster1 = clustered[clustered[:, 3] == clusterNumbers[1]]
        # cluster1 = cluster1[:, :2]
        # cluster1x = cluster1[:, 0]
        # cluster1y = cluster1[:, 1]
        #
        # cluster2 = clustered[clustered[:, 3] == clusterNumbers[2]]
        # cluster2 = cluster2[:, :2]
        # cluster2x = cluster2[:, 0]
        # cluster2y = cluster2[:, 1]
        #
        # plt.plot(cluster0x, cluster0y, 'rx', cluster1x, cluster1y, 'gx', cluster2x, cluster2y, 'bx')
        # plt.show()

    em.plot_elbow_graphic(eps_values, elbow_values, 2)



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
    data = np.genfromtxt('cluster.dat')
    training_set, test_set = stt.split_train_test(data, 0.1)
    # executeKMeans(training_set)
    executeDBSCAN(training_set)
    # executeKMeans(test_set)
    # executeDBSCAN(test_set)
