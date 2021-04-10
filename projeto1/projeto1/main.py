import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import kMeans as km


def executeKMeans():
    data = np.genfromtxt('cluster.dat')
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    clustered = km.kMeans(data)
    print(clustered)

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

    plt.plot(cluster0x, cluster0y, 'rx', cluster1x, cluster1y, 'gx', cluster2x, cluster2y, 'bx')
    plt.show()


if __name__ == '__main__':
    executeKMeans()
