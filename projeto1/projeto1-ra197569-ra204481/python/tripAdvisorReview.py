import split_train_test as stt
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA


import kMeans as km
import dbscan as dbscan


# https://archive.ics.uci.edu/ml/datasets/Travel+Reviews#


if __name__ == '__main__':
    data = np.genfromtxt('datasets/tripadvisor_review.csv', delimiter=',', skip_header=1)
    data = data[:, 1:]
    data = np.nan_to_num(data)

    training_set, test_set = stt.split_train_test(data, 0.1)

    # pca = PCA(n_components=2)
    pca = PCA(n_components=3)
    # pca = PCA(n_components=4)

    dataDimensaoReduzidaTreino = pca.fit_transform(training_set)
    dataDimensaoReduzidaTeste = pca.fit_transform(test_set)

    # print("{}D : pca.explained_variance_ratio_ --->>>> {}".format('9', pca.explained_variance_ratio_))

    # km.executeKMeans(training_set, test_set)
    km.executeKMeans(dataDimensaoReduzidaTreino, dataDimensaoReduzidaTeste)
    # dbscan.executeDBSCAN(dataDimensaoReduzidaTreino, dataDimensaoReduzidaTeste, np.arange(0.0, 0.25, 0.01), 11)
