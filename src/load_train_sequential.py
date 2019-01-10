import numpy as np
import os


def load_train_sequential():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_sequential_matrix = np.loadtxt(ROOT_DIR + '/../data/train_sequential.csv', delimiter=',', dtype=int, skiprows=1)
    train_sequential_id, train_sequential_songs = zip(*(train_sequential_matrix))
    train_sequential_id = list(train_sequential_id)
    train_sequential_songs = list(train_sequential_songs)

    matrix = []
    dictionary = {}
    temp_data = []

    last_index = 7
    for i in range(len(train_sequential_id)):
        id = train_sequential_id.__getitem__(i)

        if last_index == id:
            temp_data.append(train_sequential_songs.__getitem__(i))
        else:
            dictionary['id'] = last_index
            dictionary['songs'] = temp_data
            matrix.append(dictionary.copy())
            last_index = id
            temp_data = []
            temp_data.append(train_sequential_songs.__getitem__(i))

    dictionary['id'] = last_index
    dictionary['songs'] = temp_data
    matrix.append(dictionary.copy())

    return matrix
