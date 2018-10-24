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

    ALBUM_WEIGHT = 1.5
    ARTIST_WEIGHT = 1
    DURATION_WEIGHT = 0.1

    datafile = np.loadtxt('../data/tracks.csv', delimiter=',', skiprows=1, dtype=int)

    tracks_list, album_list, artist_list, duration_list = zip(*datafile)
    ratings = np.ones(len(album_list), dtype=int)

    ICM_album = sps.csc_matrix((ratings*ALBUM_WEIGHT, (tracks_list, album_list)))
    ICM_artist = sps.csc_matrix((ratings*ARTIST_WEIGHT, (tracks_list, artist_list)))


    duration_class_list = []

    for index in range(len(tracks_list)):
        if duration_list[index] < 106:
            duration_class_list.append(0)
        elif duration_list[index] >= 106 and duration_list[index] < 212 :
            duration_class_list.append(1)
        elif (duration_list[index] >= 212 and duration_list[index] < 318):
            duration_class_list.append(2)
        else:
            duration_class_list.append(3)

    ICM_duration = sps.csc_matrix((ratings*DURATION_WEIGHT, (tracks_list, duration_class_list)))

    ICM_partial = hstack((ICM_album, ICM_artist))
    ICM = hstack((ICM_partial, ICM_duration))

    return ICM.tocsr()


def loadTarget():
    t = p.read_csv('../data/target_playlists.csv', index_col=False)
    res = t.values
    return res
