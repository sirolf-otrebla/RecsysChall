import numpy as np


class IALS_numpy:
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
                 reg=0.015,
                 iters=10,
                 scaling='linear',
                 alpha=80,
                 epsilon=1.0,
                 init_mean=0.5,
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

        return self.X, self.Y

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.X[user_id], self.Y.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

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


    def _filter_seen(self, user_id, ranking):
        ranking = ranking.ravel()
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

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
