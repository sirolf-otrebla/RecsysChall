import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import hstack
import time, sys


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


# We pass as paramether the recommender class

def evaluate_algorithm(userList_unique, URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in userList_unique:

        relevant_items = URM_test[user_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

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

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / self.n_columns * 100, columnPerSec, (time.time() - start_time)/ 60))

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

class CFKNNRecSys():

    def __init__(self, URM_train, k=100, shrink=20):
        self._URM_train = URM_train.tocsr()
        self._k = k
        self._shrink = shrink

    def fit(self):
        self._similarity_matrix = Cosine_Similarity(self._URM_train.T.tocsc(), self._k, self._shrink, normalize=True, mode='cosine').compute_similarity()
        self._estimated_ratings = self._similarity_matrix.dot(self._URM_train).tocsr()

    def recommend(self, user_id, at=10):
        user_real = self._URM_train.getrow(user_id).toarray().squeeze()
        user_estimated = self._estimated_ratings.getrow(user_id).toarray().squeeze()

        user_real = np.argwhere(user_real > 0)

        user_estimated_sorted = np.argsort(-user_estimated)
        recommendation = [x for x in user_estimated_sorted if x not in user_real]

        debug = recommendation[0:at]

        return debug

    def recommendALL(self, userList, at=10):
        res = np.array([])
        self._estimated_ratings.tocsr()
        for i in userList:
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


if __name__ == '__main__':
    URM_text = np.loadtxt('../../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))

    train_test_split = 0.8
    numInteractions = URM.nnz
    train_mask = np.random.choice([True, False], numInteractions, p=[train_test_split, 1 - train_test_split])

    userList = np.array(user_list)
    itemList = np.array(item_list)
    ratingList = np.array(rating_list)

    URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
    test_mask = np.logical_not(train_mask)
    URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))

    cf = CFKNNRecSys(URM, 100)
    cf.fit()

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
    df.to_csv("../../results/recommendedCFtestUserNORMALIZED.csv", index=None)
