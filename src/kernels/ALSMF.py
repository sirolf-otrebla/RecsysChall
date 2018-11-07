import pandas as pd
import numpy as np
import scipy.sparse as sps
import cython
from scipy.sparse import hstack
from MF_RMSE import *

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

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

def csr_row_set_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, sps.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value
# We pass as paramether the recommender class

def evaluate_algorithm( URM_test, recommended, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for row in recommended:

        user_id = row[0]
        relevant_items = URM_test[user_id].indices

        if len(relevant_items) > 0:
            recommended_items = row[1:len(row)]
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

class IALS_numpy(object):
    '''
    binary Alternating Least Squares model (or Weighed Regularized Matrix Factorization)
    Reference: Collaborative Filtering for binary Feedback Datasets (Hu et al., 2008)

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    '''

    # TODO: Add support for multiple confidence scaling functions (e.g. linear and log scaling)
    def __init__(self,
                 num_factors=100,
                 reg=0.030,
                 iters=5,
                 scaling='linear',
                 alpha=40,
                 epsilon=1.0,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: scaling factor to compute confidence scores
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param rnd_seed: random seed
        '''

        super(IALS_numpy, self).__init__()
        assert scaling in ['linear', 'log'], 'Unsupported scaling: {}'.format(scaling)

        self.num_factors = num_factors
        self.reg = reg
        self.iters = iters
        self.scaling = scaling
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_mean = init_mean
        self.init_std = init_std
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "WRMF-iALS(num_factors={},  reg={}, iters={}, scaling={}, alpha={}, episilon={}, init_mean={}, " \
               "init_std={}, rnd_seed={})".format(
            self.num_factors, self.reg, self.iters, self.scaling, self.alpha, self.epsilon, self.init_mean,
            self.init_std, self.rnd_seed
        )

    def _linear_scaling(self, R):
        C = R.copy().tocsr()
        C.data *= self.alpha
        C.data += 1.0
        return C

    def _log_scaling(self, R):
        C = R.copy().tocsr()
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)
        return C

    def fit(self, R):
        self.dataset = R
        # compute the confidence matrix
        if self.scaling == 'linear':
            C = self._linear_scaling(R)
        else:
            C = self._log_scaling(R)

        Ct = C.T.tocsr()
        M, N = R.shape

        # set the seed
        np.random.seed(self.rnd_seed)

        # initialize the latent factors
        self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
        self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))


        for it in range(self.iters):
            self.X = self._lsq_solver_fast(C, self.X, self.Y, self.reg)
            self.Y = self._lsq_solver_fast(Ct, self.Y, self.X, self.reg)
            print('Finished iter {}'.format(it + 1))

        return [self.X, self.Y]

    def recommend(self, user, at=None, exclude_seen=True):
        user_id = user
        user_real = self.dataset.getrow(user_id).toarray().squeeze()
        user_estimated = np.dot(self.X[user_id], self.Y.T)
        user_estimated_sorted = user_estimated.argsort()[::-1]
        # rank items
        if exclude_seen:
            user_real = np.argwhere(user_real > 0)
            user_estimated_sorted = np.argsort(-user_estimated)
            recommendation = [x for x in user_estimated_sorted if x not in user_real]
            return recommendation[0:at]

        return user_estimated_sorted[0:at]

    def _lsq_solver(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            # accumulate Yt*Ci*p(i) in b
            b = np.zeros(factors)

            for j, cij in self._nonzeros(C, i):
                vj = Y[j]
                A += (cij - 1.0) * np.outer(vj, vj)
                b += cij * vj

            X[i] = np.linalg.solve(A, b)
        return X

    def _lsq_solver_fast(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            start, end = C.indptr[i], C.indptr[i + 1]
            j = C.indices[start:end]  # indices of the non-zeros in Ci
            ci = C.data[start:end]  # non-zeros in Ci

            Yj = Y[j]  # only the factors with non-zero confidence
            # compute Yt(Ci-I)Y
            aux = np.dot(Yj.T, np.diag(ci - 1.0))
            A += np.dot(aux, Yj)
            # compute YtCi
            b = np.dot(Yj.T, ci)

            X[i] = np.linalg.solve(A, b)
        return X

    def _nonzeros(self, R, row):
        for i in range(R.indptr[row], R.indptr[row + 1]):
            yield (R.indices[i], R.data[i])


    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]


    def recommendALL(self,true_userList,  userList, at=10):
        res = np.array([])
        for i in range(0, len(userList)):
            recList = self.recommend(userList[i], at, True) #exclude seen
            tuple = np.concatenate(([true_userList[i]], recList))
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
    target = pd.read_csv('../../data/target_playlists.csv', index_col=False)

    numInteractions = URM.nnz
    sums = URM.sum(1).squeeze()
    tgt = target.values.squeeze()
    denoise_mask = [i for i in range(0,sums.shape[1]) if sums[0,i] < 10 and i not in tgt ]
    userList = np.array(user_list)
    itemList = np.array(item_list)
    ratingList = np.array(rating_list)
    denoised_URM = delete_from_csr(URM, denoise_mask)
    denoise_mask = np.array(denoise_mask)
    tgt_reduced = tgt.copy()
    for i in range(0, denoise_mask.shape[0]):
        for j in range(0, len(tgt_reduced)):
            value = tgt_reduced[j]
            if tgt_reduced[-1] == 4247:
                print("porcodio")
            if denoise_mask[i] <= value:
                tgt_reduced[j] -= 1

    duration_class_list = []
    R = denoised_URM.tocsr()
    cf = IALS_numpy()
    cf.fit(R)
#    recommended = cf.recommendALL(target.values)
#    playlists = target.take([0], axis=1)
#    evaluate_algorithm(URM.tocsr(), recommended)
    recommended = cf.recommendALL(tgt, tgt_reduced)
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
    df.to_csv("../../results/recommendedIALSNEW.csv", index=None)
