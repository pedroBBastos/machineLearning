import numpy as np

import kMeans as km
import dbscan as dbscan
import split_train_test as stt

if __name__ == '__main__':
    data = np.genfromtxt('datasets/cluster.dat')
    training_set, test_set = stt.split_train_test(data, 0.1)

    # km.executeKMeans(training_set, test_set) 0.058
    dbscan.executeDBSCAN(training_set, test_set, np.arange(0.004, 0.1, 0.002), 3)
