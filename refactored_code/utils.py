import numpy as np
import scipy.sparse as sps
from scipy.sparse import vstack
import pandas as pd
import os


def load_urm():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    URM_text = np.loadtxt(ROOT_DIR + '/../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    return sps.csr_matrix((rating_list, (user_list, item_list)))


def train_test_holdout(urm_all, train_perc=0.8):

    num_interactions = urm_all.nnz

    urm_all = urm_all.tocoo()
    shape = urm_all.shape

    train_mask = np.random.choice([True, False], num_interactions, p=[train_perc, 1-train_perc])

    urm_train = sps.coo_matrix((urm_all.data[train_mask],
                               (urm_all.row[train_mask], urm_all.col[train_mask])), shape=shape)
    urm_train = urm_train.tocsr()

    test_mask = np.logical_not(train_mask)

    urm_test = sps.coo_matrix((urm_all.data[test_mask],
                              (urm_all.row[test_mask], urm_all.col[test_mask])), shape=shape)
    urm_test = urm_test.tocsr()

    return urm_train, urm_test



def precision(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    if(is_relevant.shape[0] == 0 or relevant_items.shape[0] == 0):

        print('ciao')

    map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])

    return map_score


def evaluate_csv(URM_test, path):

    URM_test = sps.csr_matrix(URM_test)
    csv_file = pd.read_csv(path, index_col=False)

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    # row of the recommendation: csv_file.iloc[0,1].split()

    for i in range(0, len(csv_file)):
        user_id = csv_file.iloc[i,0]

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id+1]

        if end_pos-start_pos>0:

            relevant_items = URM_test.indices[start_pos:end_pos]
            recommended_items = list(map(int,csv_file.iloc[i,1].split()))

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)
            num_eval += 1

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict


def load_icm(album_weight=1, artist_weight=1, duration_weight=1):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    ICM_text = np.loadtxt(ROOT_DIR + '/../data/tracks.csv', delimiter=',', skiprows=1, dtype=int)

    tracks_list, album_list, artist_list, duration_list = zip(*ICM_text)

    ratings = np.ones(len(album_list), dtype=int)

    ICM_album = sps.csc_matrix((ratings, (tracks_list, album_list)))
    ICM_artist = sps.csc_matrix((ratings, (tracks_list, artist_list)))

    ICM_album_flat = ICM_album.toarray().ravel()
    ICM_artist_flat = ICM_artist.toarray().ravel()

    ICM_album = np.multiply(album_weight, ICM_album_flat).reshape(ICM_album.shape)
    ICM_artist = np.multiply(artist_weight, ICM_artist_flat).reshape(ICM_artist.shape)

    duration_class_list = []

    for index in range(len(tracks_list)):
        if duration_list[index] < 106:
            duration_class_list.append(0)

        elif duration_list[index] >= 106 and duration_list[index] < 212:
            duration_class_list.append(1)

        elif (duration_list[index] >= 212 and duration_list[index] < 318):
            duration_class_list.append(2)

        else:
            duration_class_list.append(3)

    ICM_duration = sps.csc_matrix((ratings, (tracks_list, duration_class_list)))

    ICM_duration_flat = ICM_duration.toarray().ravel()
    ICM_duration = np.multiply(duration_weight, ICM_duration_flat).reshape(ICM_duration.shape)

    ICM_partial = np.concatenate((ICM_album, ICM_artist), axis=1)

    return sps.csr_matrix(np.concatenate((ICM_partial, ICM_duration), axis=1))


def load_random_urms(min=3, max=9):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(ROOT_DIR)

    data_index = np.random.randint(min, max)
    test = sps.load_npz(ROOT_DIR + "/../data/validation_mat/TEST_{0}.npz".format(data_index))
    train = sps.load_npz(ROOT_DIR + "/../data/validation_mat/TRAIN_{0}.npz".format(data_index))

    return train, test


def df_sequential_playlists(train_percentage=0.8):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sequential_df = pd.read_csv(ROOT_DIR + '/../data/train_sequential.csv')
    sequential_df.set_index('playlist_id', inplace=True)

    playlist_interaction = sequential_df.groupby('playlist_id').count().copy()

    train_user_list = []
    train_list = []
    test_list = []
    test_user_list = []

    for index, row in playlist_interaction.iterrows():
        num_interactions = playlist_interaction.loc[index].values[0]
        num_trains = int(round(num_interactions * train_percentage, 0))
        num_tests = int(round(num_interactions * (1-train_percentage), 0))
        assert num_trains + num_tests == num_interactions

        for item in sequential_df.loc[index][0:num_trains].values:
            train_list.append(item[0])
            train_user_list.append(index)

        for item in sequential_df.loc[index][num_trains:num_interactions].values:
            test_list.append(item[0])
            test_user_list.append(index)

    return_dictionary = {
        "train_user_list": train_user_list,
        "train_list": train_list,
        "test_user_list": test_user_list,
        "test_list": test_list
    }


    return return_dictionary


def write_train_sequential_files():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    dict = df_sequential_playlists()
    train_file = open(ROOT_DIR + '/../data/train_sequential_train.csv', 'w+')
    test_file = open(ROOT_DIR + '/../data/train_sequential_test.csv', 'w+')

    train_file.write('playlist_id,track_id\n')
    test_file.write('playlist_id,track_id\n')

    for i in range(len(dict["train_user_list"])):
        playlist_elem = dict['train_user_list'][i]
        item_elem = dict['train_list'][i]

        train_file.write(str(playlist_elem) + ',' + str(item_elem) + '\n')

    train_file.close()

    for i in range(len(dict["test_user_list"])):
        playlist_elem = dict['test_user_list'][i]
        item_elem = dict['test_list'][i]

        test_file.write(str(playlist_elem) + ',' + str(item_elem) + '\n')

    test_file.close()


def rewrite_train_csv():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    URM_text = np.loadtxt(ROOT_DIR + '/../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    fivek_playlists = np.loadtxt(ROOT_DIR + '/../data/5k_sequential.csv', delimiter=',', dtype=int, skiprows=1)

    user_list, item_list = zip(*URM_text)
    sequential_playlist_text = np.loadtxt(ROOT_DIR + '/../data/train_sequential_train.csv', delimiter=',', dtype=int, skiprows=1)
    sequential_playlist, sequential_item = zip(*sequential_playlist_text)

    new_train_file = open(ROOT_DIR + '/../data/new_train.csv', 'w+')
    new_train_file.write('playlist_id,track_id\n')

    index = 0
    for i in user_list:
        if i not in fivek_playlists:
            new_train_file.write(str(i) + ',' + str(item_list[index]) + '\n')
        index += 1

    new_train_file.close()


def gen_k_folds_matrix(URM, n):
    for i in range(0,n):
        URM_train, URM_test = utils.train_test_holdout(URM, 0.80)
        if URM_train.shape[0] == URM.shape[0] and URM_train.shape[1] == URM.shape[1]\
                and URM_test.shape[0] == URM.shape[0] and URM_test.shape[1] == URM.shape[1]:
            sps.save_npz("../data/validation_mat/TRAIN_{0}".format(i), URM_train)
            sps.save_npz("../data/validation_mat/TEST_{0}".format(i), URM_test)
        else:
            i = i- 1


def new_train_test_holdout(URM_all, train_perc=0.8):

    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()

    train_mask = np.random.choice([True,False], numInteractions, [train_perc, 1-train_perc])

    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    URM_test = URM_test.tocsr()

    while URM_all.shape[0] != URM_train.shape[0] or \
          URM_all.shape[1] != URM_train.shape[1] or \
          URM_all.shape[0] != URM_test.shape[0] or \
          URM_all.shape[1] != URM_test.shape[1]:
        print('Matrix construction failed, trying another one...')
        train_mask = np.random.choice([True, False], numInteractions, [train_perc, 1 - train_perc])

        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)

        URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
        URM_test = URM_test.tocsr()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    sequential_playlist_text_train = np.loadtxt(ROOT_DIR + '/../data/train_sequential_train.csv', delimiter=',', dtype=int,
                                          skiprows=1)

    sequential_playlist_train, sequential_item_train = zip(*sequential_playlist_text_train)

    sequential_playlist_text_test = np.loadtxt(ROOT_DIR + '/../data/train_sequential_test.csv', delimiter=',',
                                                dtype=int,
                                                skiprows=1)

    sequential_playlist_test, sequential_item_test = zip(*sequential_playlist_text_test)



    fivek_playlists = np.loadtxt(ROOT_DIR + '/../data/5k_sequential.csv', delimiter=',', dtype=int, skiprows=1)

    for row in fivek_playlists:
        URM_train.data[URM_train.indptr[row]:URM_train.indptr[row + 1]] = 0
        URM_test.data[URM_test.indptr[row]:URM_test.indptr[row + 1]] = 0

    URM_train = URM_train.tolil()
    URM_test = URM_test.tolil()

    index = 0
    for item in sequential_playlist_train:
        URM_train[item, sequential_item_train [index]] = 1
        index += 1

    index = 0
    for item in sequential_playlist_test:
        URM_test[item, sequential_item_test[index]] = 1
        index += 1

    print('Matrix successfully constructed')
    return URM_train.tocsr().copy(), URM_test.tocsr().copy()

if __name__ == '__main__':
    new_train_test_holdout(load_urm())
