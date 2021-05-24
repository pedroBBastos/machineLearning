import numpy as np
import gradientDescent as gd


if __name__ == '__main__':
    data = np.genfromtxt('kick2.dat')
    entries = data[:, :2]
    results = data[:, 2]

    tethas = gd.initializeTethas(2, [(-3.0, 3.0), (0.0, 2.0)])

    print(tethas)
    for i in range(0, 1000):
        tethas = gd.runEpoch(tethas, entries, results, 0.04)
        print(gd.errorFunction(tethas, entries, results, entries.shape[0]))

