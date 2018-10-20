import scipy.sparse as sps
import numpy as np
import scipy as sp
import pandas as p
from scipy.sparse import hstack

from src import data


def build_urm():
    datafile = np.loadtxt('../data/train.csv', delimiter=',', skiprows=1, dtype=int)
    userList, itemList = zip(*datafile)
    ratings = np.ones(1211791)
    return sps.coo_matrix((ratings, (userList, itemList)))
    #dm = data.Data_manager(URM_all)
    #return  dm

def build_icm():
    MAX_ALBUM = 12744
    MAX_ARTIST = 6668
    MAX_DURATION = 2115

    datafile = np.loadtxt('../data/tracks.csv', delimiter=',', skiprows=1, dtype=int)

    tracks_list, album_list, artist_list, duration_list = zip(*datafile)
    ratings = np.ones(len(album_list), dtype=int)

    ICM_album = sps.csc_matrix((ratings, (tracks_list, album_list)))
    ICM_artist = sps.csc_matrix((ratings, (tracks_list, artist_list)))
    ICM_duration = sps.csc_matrix((ratings, (tracks_list, duration_list)))

    ICM_partial = hstack((ICM_album, ICM_artist))
    ICM = hstack((ICM_partial, ICM_duration))

    #return ICM
    return ICM_partial


def loadTarget():
    t = p.read_csv('../data/target_playlists.csv', index_col=False)
    res = t.values
    return res