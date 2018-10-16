import numpy as np
import time, sys
import scipy.sparse as sps
from src import utils

check_matrix = utils.check_matrix

class Cosine_Similarity(object):

    def __init__(self, ICM, k=100):
        self.topK = 100
        self.ICM = ICM.copy()
        self._S = None

    def compute(self):
        check_matrix(self.ICM, 'csc')
        S = np.dot(self.ICM, self.ICM)
        np.fill_diagonal(S, 0)
        self._S = S
        return S

    def assign_weights(self, w):
        self._weighted_S = np.dot(self._S, w)
        return  self._weighted_S

    def topK(self):
        if (self._weighted_S != None):
            for row in range(self._weighted_S.shape[0]):
                item_data = self.ICM[row, :]
                item_data = item_data.toarray.squeeze()

