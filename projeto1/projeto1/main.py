import numpy as np

import kMeans as km
import dbscan as dbscan
import split_train_test as stt

if __name__ == '__main__':
    data = np.genfromtxt('datasets/cluster.dat')
    training_set, test_set = stt.split_train_test(data, 0.1)

    # km.executeKMeans(training_set, test_set)
    dbscan.executeDBSCAN(training_set, test_set, np.arange(0.012, 0.5, 0.012), 3)
