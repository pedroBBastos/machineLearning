import numpy as np
import gradientDescent as gd

if __name__ == '__main__':
    data = np.genfromtxt('kick2.dat')
    entries = data[:, :2]
    results = data[:, 2]

    #################################################################################
    # treinando modelo linear
    #################################################################################
    print("--------------------- treinando modelo linear ----------------------")
    tethas = gd.initializeTethas(2, [(-3.0, 3.0), (0.0, 2.0)])
    linearGD = gd.LinearRegression()
    for i in range(0, 1000):
        tethas = gd.runEpoch(linearGD, tethas, entries, results, 0.04)
        print(gd.errorFunction(linearGD, tethas, entries, results, entries.shape[0]))
    print("final tethas -->>> ")
    print(tethas)

    #################################################################################
    # treinando modelo polinomial
    #################################################################################
    print("--------------------- treinando modelo polinomial ----------------------")
    tethas = gd.initializeTethas(2, [(-3.0, 3.0), (0.0, 2.0)])
    polinomialGD = gd.PolinomialRegression()
    for i in range(0, 1000):
        tethas = gd.runEpoch(polinomialGD, tethas, entries, results, 0.04)
        print(gd.errorFunction(polinomialGD, tethas, entries, results, entries.shape[0]))
    print("final tethas -->>> ")
    print(tethas)
