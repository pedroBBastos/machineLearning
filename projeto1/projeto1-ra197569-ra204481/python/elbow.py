import numpy as np
import matplotlib.pyplot as plt


def elbow_value(clustered_data, centroides, dimensoes):

    soma = 0
    for ponto in clustered_data:
        # obtendo as distancias para os centroides do ponto atual
        norms_centroides = np.linalg.norm(ponto[:dimensoes] - centroides, axis=1)
        soma += np.amin(norms_centroides)

    return soma


def plot_elbow_graphic(values, k, function):
    xpoints = np.array(k)
    ypoints = np.array(values)
    if function == 1: # elbow method
        plt.xlabel('Number of Clusters')
        plt.ylabel('Cost Function J')
    else: # knee method
        plt.xlabel('Index Points')
        plt.ylabel('Eps Values')
    plt.plot(xpoints, ypoints, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=7)
    plt.show()