import scipy.sparse as sps
import numpy as np
import scipy as sp
import pandas as p

from src import data


def build():
    datafile = np.loadtxt('../data/train.csv', delimiter=',', skiprows=1, dtype=int)
    userList, itemList = zip(*datafile)
    ratings = np.ones(1211791)
    URM_all = sps.coo_matrix((ratings, (userList, itemList)))
    dm = data.Data_manager(URM_all)
    return  dm


def loadTarget():
    t = p.read_csv('../data/target_playlists.csv', index_col=False)
    res = t.values
    return res
