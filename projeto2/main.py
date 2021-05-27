import numpy as np
import gradientDescent as gd
import matplotlib.pyplot as plt


# def plotPolinomial(dataXY, dataZ, thetas):
#     ax = plt.axes(projection='3d')
#     dataX = dataXY[:, 0]
#     dataY = dataXY[:, 1]
#     ax.scatter(dataX, dataY, dataZ, color=np.random.random(3))
#
#     plt.xlim([-3.0, 3.0])
#     plt.ylim([0.0, 2.0])
#
#     x = np.linspace(-3, 3, 100)
#     y = np.linspace(0, 2, 100)
#     finalFunction = thetas[0] + x * thetas[1] + y ** 2 * thetas[2]
#     ax.plot(x, y, finalFunction)
#     ax.set_zlim((0.0, 0.3))
#     plt.show()
#
#
# def runPolinomial(entries, results):
#     print("--------------------- treinando modelo polinomial ----------------------")
#     thetas = gd.initializethetas(2, [(-3.0, 3.0), (0.0, 2.0)])
#     polinomialGD = gd.PolinomialRegression()
#     for i in range(0, 2000):
#         thetas = gd.runEpoch(polinomialGD, thetas, entries, results, 0.2)
#         print(gd.errorFunction(polinomialGD, thetas, entries, results, entries.shape[0]))
#     print("final thetas -->>> ")
#     print(thetas)
#     plotPolinomial(entries, results, thetas)

def plotLinear(coordenatesArray, thetaX, thetaY, thetaZ):
    ax = plt.axes(projection='3d')
    dataX = coordenatesArray[:, 0]
    dataY = coordenatesArray[:, 1]
    dataZ = coordenatesArray[:, 2]
    ax.scatter(dataX, dataY, dataZ, color=np.random.random(3))

    tempo = np.linspace(1/60, 1, 60)
    x = thetaX[0] + tempo * thetaX[1]
    y = thetaY[0] + tempo * thetaY[1]
    z = thetaZ[0] + tempo * thetaZ[1]
    ax.plot(x, y, z)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.0, 0.3)

    plt.show()


def runLinear(coordenatesArray, time):
    print("--------------------- treinando modelo linear ----------------------")
    thetaX = gd.initializeThetas(2, (-3.0, 3.0))
    thetaY = gd.initializeThetas(2, (0.0, 2.0))
    thetaZ = gd.initializeThetas(2, (0.0, 0.6))

    linearGD = gd.LinearRegression()

    dataX = coordenatesArray[:, 0]
    dataY = coordenatesArray[:, 1]
    dataZ = coordenatesArray[:, 2]

    for i in range(0, 2000):
        thetaX = gd.runEpoch(linearGD, thetaX, time, dataX, 0.2)
        # print("Error X -> {}".format(gd.errorFunction(linearGD, thetaX, time, dataX, time.shape[0])))
        thetaY = gd.runEpoch(linearGD, thetaY, time, dataY, 0.2)
        # print("Error Y -> {}".format(gd.errorFunction(linearGD, thetaY, time, dataY, time.shape[0])))
        thetaZ = gd.runEpoch(linearGD, thetaZ, time, dataZ, 0.2)
        print("Error Z -> {}".format(gd.errorFunction(linearGD, thetaZ, time, dataZ, time.shape[0])))
        # print(" ----------------------------------------- ")

    plotLinear(coordenatesArray, thetaX, thetaY, thetaZ)


if __name__ == '__main__':
    data = np.genfromtxt('kick2.dat')
    runLinear(data, np.linspace(1/60, 1/3, 20))
    # runPolinomial(data[:, :2], data[:, 2])
