import scipy.sparse as sps
import numpy as np
import scipy as sp

from src import data


def build():
    datafile = np.loadtxt('../input/train.csv', delimiter=',', skiprows=1)
    userList, itemList = zip(*datafile)
    ratings = np.ones(1211791)
    URM_all = sps.coo_matrix((ratings, (userList, itemList)))
    dm = data.Data_manager(URM_all)
    return  dm

