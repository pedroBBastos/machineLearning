import numpy as np
import kMeans as km


def showDataInfo():
    return km.kMeans(np.genfromtxt('cluster.dat'))


if __name__ == '__main__':
    print(showDataInfo())
