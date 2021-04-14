import numpy as np

# metodo para separar conjunto de treinamento e conjunto de teste
# passe os dados e a porcentagem do conjunto de dados que sera o conjunto de test


def split_train_test(data, test_ratio):
    np.random.seed(42) # coloquei 42 porque é o numero do curso :)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]
