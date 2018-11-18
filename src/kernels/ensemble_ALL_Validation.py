import pandas as pd
import numpy as np
import scipy as sp
import src.kernels.ALSMF as als
import scipy.sparse as sps
from scipy.sparse import hstack
import time, sys
from src.new_utils import utils

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)

class Cosine_Similarity:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize = True,
                 mode = "cosine"):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:
        :param mode:    "cosine"    computes Cosine similarity
                        "adjusted"  computes Adjusted Cosine, removing the average of the users
                        "pearson"   computes Pearson Correlation, removing the average of the items
                        "jaccard"   computes Jaccard similarity for binary interactions using Tanimoto
                        "tanimoto"  computes Tanimoto coefficient for binary interactions

        """

        super(Cosine_Similarity, self).__init__()

        self.TopK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False

        if mode == "adjusted":
            self.adjusted_cosine = True
        elif mode == "pearson":
            self.pearson_correlation = True
        elif mode == "jaccard" or mode == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif mode == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for paramether 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'jaccard', 'tanimoto'."
                             " Passed value was '{}'".format(mode))



        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns, self.n_columns))




    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')


        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]


        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row= 0

        blockSize = 1000


        while end_row < self.n_rows:

            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize




    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')


        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]


        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col= 0

        blockSize = 1000


        while end_col < self.n_columns:

            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize


    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0

        blockSize = 1000


        while end_pos < len(self.dataMatrix.data):

            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos-start_pos)

            start_pos += blockSize




    def compute_similarity(self):

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient:
            self.useOnlyBooleanInteractions()


        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')


        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not self.tanimoto_coefficient:
            sumOfSquared = np.sqrt(sumOfSquared)


        # Compute all similarities for each item using vectorization
        for columnIndex in range(self.n_columns):

            processedItems += 1

            if time.time() - start_time_print_batch >= 30 or processedItems==self.n_columns:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} secs".format(
                    processedItems, processedItems / self.n_columns * 100, columnPerSec, (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()


            # All data points for a given item
            item_data = self.dataMatrix[:, columnIndex]
            item_data = item_data.toarray().squeeze()

            # Compute item similarities
            this_column_weights = self.dataMatrix.T.dot(item_data)
            this_column_weights[columnIndex] = 0.0

            # Apply normalization and shrinkage, ensure denominator != 0
            if self.normalize:
                denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

            # Apply the specific denominator for Tanimoto
            elif self.tanimoto_coefficient:
                denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

            # If no normalization or tanimoto is selected, apply only shrink
            elif self.shrink != 0:
                this_column_weights = this_column_weights/self.shrink


            if self.TopK == 0:
                self.W_dense[:, columnIndex] = this_column_weights

            else:
                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK-1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix
                values.extend(this_column_weights[top_k_idx])
                rows.extend(top_k_idx)
                cols.extend(np.ones(self.TopK) * columnIndex)

        if self.TopK == 0:
            return self.W_dense

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)


            return W_sparse

class User_CFKNNRecSys():

    def __init__(self, URM_train, k=100, shrink=0):
        self._URM_train = URM_train.tocsr()
        self._k = k
        self._shrink = shrink

    def fit(self):
        self._similarity_matrix = Cosine_Similarity(self._URM_train.T, self._k, self._shrink, normalize=False, mode='cosine').compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        scores = self._similarity_matrix[user_id].dot(self._URM_train).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = int(self._URM_train.indptr[user_id])
        end_pos = int(self._URM_train.indptr[user_id + 1])

        user_profile = self._URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommendALL(self, userList, at=10):
        res = np.array([])
        n=0
        for i in userList:
            n+=1
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

class Item_CFKNNRecSys():

    def __init__(self, URM_train, k=100, shrink=0):
        self._URM_train = URM_train.tocsr()
        self._k = k
        self._shrink = shrink

    def fit(self):
        self._similarity_matrix = Cosine_Similarity(self._URM_train.tocsc(), self._k, self._shrink, normalize=False, mode='cosine').compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        scores = user_profile.dot(self._similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = int(self._URM_train.indptr[user_id])
        end_pos = int(self._URM_train.indptr[user_id + 1])

        user_profile = self._URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommendALL(self, userList, at=10):
        res = np.array([])
        n=0
        for i in userList:
            n+=1
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

class general_ensemble_CFKNNRecSys():
    def __init__(self, URM_train, ICM, k=100,
                 alpha=0.8188,
                 beta=0.7662,
                 gamma=0.3325,
                 epsilon = 0.6212,
                 shrink=5):
        self._URM_train = URM_train.tocsr()
        self._ICM = ICM.tocsr()
        self._k = k
        self._shrink = shrink

        self.UUSCORE = alpha

        self.IISCORE = beta

        self. CBFSCORE = gamma

        self.IALSSCORE = epsilon

    def fit(self, alpha):
        self._similarity_matrixUU = Cosine_Similarity(self._URM_train.T,
                                                      self._k,
                                                      self._shrink,
                                                      normalize=True,
                                                      mode='cosine').compute_similarity()

        self._similarity_matrixII = Cosine_Similarity(self._URM_train.tocsc(),
                                                      self._k,
                                                      self._shrink,
                                                      normalize=True,
                                                      mode='cosine').compute_similarity()

        self._similarity_matrixCBF = Cosine_Similarity(self._ICM.T,
                                                       self._k,
                                                       self._shrink,
                                                       normalize=True,
                                                       mode='cosine').compute_similarity()

        self.lfactors = (als.IALS_numpy(reg=alpha)).fit(self._URM_train)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        scores = (self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray() +
                   self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray() +
                    self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray() +
                  np.dot( self.lfactors[0][user_id], self.lfactors[1].T)).ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = int(self._URM_train.indptr[user_id])
        end_pos = int(self._URM_train.indptr[user_id + 1])

        user_profile = self._URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommendALL(self, userList, at=10):
        res = np.array([])
        n=0
        for i in userList:
            n+=1
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res

def xvalidation_par(elements=1500, folds=10):
    maps = []
    alphas = []
    for i in range(0, elements):
        alpha = np.random.uniform(0.01, 0.05)
        beta = np.random.uniform(0.5,1)
        print('\n \n_____________________________________')
        print('starting iteration {0} with a = {1} and b = {2}'.format(i, alpha, beta))
        print('_____________________________________\n \n')
        data = []
        for j in range(0, folds):
            beta = 1 - alpha
            data_index = np.random.randint(1, 9)
            test = sps.load_npz("../../data/validation_mat/TEST_{0}.npz".format(data_index))
            train = sps.load_npz("../../data/validation_mat/TRAIN_{0}.npz".format(data_index))
            res = main(alpha, beta, URM_train=train, URM_test=test)
            map = res["MAP"]
            data.append(map)
        data_array = np.array(data)
        mean = np.average(data_array)
        alphas.append(alpha)
        maps.append(mean)
        print('\n \n_____________________________________')
        print('finished iteration {0} with a = {1} and b = {2}'.format(i, alpha, beta))
        print('_____________________________________\n \n')
        d = {"alpha": alphas, "map": maps}
        df = pd.DataFrame(data=d, index=None)
        df.to_csv("../../results/evaluation/data_ensembleALL.csv", index=None)


def main(alpha, beta, URM_train, URM_test):
    #URM_text = np.loadtxt('../../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    #user_list, item_list = zip(*URM_text)
    #rating_list = np.ones(len(user_list))
    #URM = sps.csr_matrix((rating_list, (user_list, item_list)))

    #URM_train, URM_test = utils.train_test_holdout(URM, 0.95)

    ARTIST_WEIGHT = 0.3
    ALBUM_WEIGHT = 0.6
    DURATION_WEIGHT = 0.1

    ICM_text = np.loadtxt('../../data/tracks.csv', delimiter=',', skiprows=1, dtype=int)

    tracks_list, album_list, artist_list, duration_list = zip(*ICM_text)

    ratings = np.ones(len(album_list), dtype=int)

    ICM_album = sps.csc_matrix((ratings, (tracks_list, album_list)))
    ICM_artist = sps.csc_matrix((ratings, (tracks_list, artist_list)))

    ICM_album_flat = ICM_album.toarray().ravel()
    ICM_artist_flat = ICM_artist.toarray().ravel()

    ICM_album = np.multiply(ALBUM_WEIGHT, ICM_album_flat).reshape(ICM_album.shape)
    ICM_artist = np.multiply(ARTIST_WEIGHT, ICM_artist_flat).reshape(ICM_artist.shape)

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
    ICM_duration = np.multiply(DURATION_WEIGHT, ICM_duration_flat).reshape(ICM_duration.shape)

    ICM_partial = np.concatenate((ICM_album, ICM_artist), axis=1)

    ICM = np.concatenate((ICM_partial, ICM_duration), axis=1)

    cf = general_ensemble_CFKNNRecSys(URM_train, ICM, 50, epsilon=beta)
    cf.fit(alpha)

    target = pd.read_csv('../../data/target_playlists.csv', index_col=False)
    recommended = cf.recommendALL(target.values)

    playlists = recommended[:, 0]
    recommended = np.delete(recommended, 0, 1)
    i = 0
    res_fin = []
    for j in recommended:
        res = ''
        for k in range(0, len(j)):
            res = res + '{0} '.format(j[k])
        res_fin.append(res)
        i = i + 1
    d = {'playlist_id': playlists, 'track_ids': res_fin}
    df = pd.DataFrame(data=d, index=None)
    df.to_csv("../../results/resultsEnsembleAll.csv", index=None)

    return utils.evaluate_csv(URM_test,"../../results/resultsEnsembleAll.csv")

if __name__ == '__main__':
    xvalidation_par(250, 2)
