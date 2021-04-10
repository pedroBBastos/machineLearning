import numpy as np


def showDataInfo():
    return np.genfromtxt('cluster.dat')


if __name__ == '__main__':
    print(showDataInfo())
