import scipy
import numpy as np
import src.similarities as sim
import src.build_data as bd
import scipy.sparse as sps

class TopPop(object):

    def fit(self, URM_train):

        itemPopularity = (URM_train>0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self._popularItems = np.argsort(itemPopularity)
        self._popularItems = np.flip(self._popularItems, axis = 0)

    def recommend(self, user_ID, at=10):
        recommended = self._popularItems[0:at]
        return recommended

    def recommendAll(self, userList, at=10):
        res = np.array([])

        for i in userList:
            recList = self.recommend(i, 10)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


class CBF_Item_Naive(object):
    def __init__(self, k):
        self._cosine = sim.Cosine_Similarity(bd.build_icm(), k)
        self.URM = bd.build_urm()
        self._k = k


    def fit(self, URM_train):
        DURATION_W = 0.1

        self._cosine.compute()
        S_knn = self._cosine.topK(self._k)
        DEBUG = S_knn.getrow(13757).toarray()
        print(DEBUG)
        self._S_knn = S_knn
        self.estimated_ratings = self.URM.dot(S_knn.transpose())


    def recommend(self, user_ID, at = 10):
        user_ID = int(user_ID)
        user_estimated = self.estimated_ratings.getrow(user_ID).toarray().squeeze()
        user_real = self.URM.tocsr().getrow(user_ID).toarray().squeeze()

        user_real = np.argwhere(user_real > 0)

        user_estimated_sorted = np.argsort(-user_estimated)
        DEBUGuser_estimated_values = np.sort(-user_estimated)
        DEBUGurm_row13759 = self.URM.getrow(13759).toarray()
        DEBUGsknn_row13759 = self._S_knn.getrow(13759).toarray()
        recommendation = [x for x in user_estimated_sorted if x not in user_real]
        return recommendation[0:at]

    def recommendAll(self, userList, at = 10):
        res = np.array([])
        for i in userList:
            recList = self.recommend(i, 10)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


class CBF_coldstart(object):

    def __init__(self, k,  coldstart=10):

        self._coldstart = coldstart
        self._cbf = CBF_Item_Naive(k)
        self._URM = bd.build_urm().tocsr()
        self._toppop = TopPop()

    def fit(self):

        self._toppop.fit(self._URM)
        self._cbf.fit(self._URM)

    def recommend(self, userID, at=10, limit=5):

        row_sum = self._URM.getrow(userID).sum()
        if row_sum > limit :
            return self._cbf.recommend(userID, at)
        else:
            return self._toppop.recommend(userID, at)

    def recommendAll(self, userList, at=10, limit=10):
        res = np.array([])
        for i in userList:
            recList = self.recommend(i, 10, limit)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res


class ALS_factorization(object):

    def __init__(self, features):

        self._lambda = 0.1
        self._alpha = 40
        self._URM = bd.build_urm().tocsr()
        self._users = self._URM.shape[0]
        self._items = self._URM.shape[1]
        self._X = sps.csr_matrix(np.random.normal(size=(self._users, features)))
        self._Y = sps.csr_matrix(np.random.normal(size=(self._items, features)))

        self._X_I = sps.eye(self._users)
        self._Y_I = sps.eye(self._items)

        self._I = sps.eye(features)
        self._II = self._I*self._lambda

        self._xTx = self._X.T.dot(self._X)

        self._Cui = self._URM*self._alpha

        self._ALS_ITERS = 100

    # indptr return indexes of nonzero elements in sparse matrix
    # see http://mbatchkarov.github.io/2014/10/10/rate_of_vocab_growth/
    def nonzeros(self, m, row):
        for index in (m.indptr[row], m.indptr[row + 1]):
            yield m.indices[index], m.data[index]


    def fit(self):
        Ciu = self._Cui.T.tocsr()
        for iteration in range(0, self._ALS_ITERS):
            self._X = self._least_squares(self._X, self._Y, self._Cui)
            self._Y = self._least_squares(self._Y, self._X, Ciu)



    def _least_squares(self, X, Y, Cui, cgsteps=3):

        yTy = Y.T.dot(Y) + self._II

        for u in range(0, self._users):
            x = X[u]
            r = - yTy.dot(x.T)
            for i, confidence in self.nonzeros(Cui, u):

                #conjugate gradient method
                yx = Y[i].dot(x)
                r += (confidence - (confidence - 1) * yx) * Y[i]

            p = r.copy()
            rsold = r.dot(r)

            for it in range(0, cgsteps):
                Ap = yTy.dot(p)
                for i, confidence in self.nonzeros(Cui, u):
                    Ap += (confidence -1)*Y.getrow(i).dot(p)*Y.getrow(i)

                alpha = rsold / p.dot(Ap)
                x += alpha *p
                r -= alpha *Ap

                rsnew = r.dot(r)
                p = r + (rsnew/rsold)*p
                rsold = rsnew

            X[u] = x

        return X

    def recommend(self, user_ID, at=10):
        # Get all interactions by the user
        user_interactions = self._URM.getrow(user_ID).toarray()
        user_real = user_interactions
        # We don't want to recommend items the user has consumed. So let's
        # set them all to 0 and the unknowns to 1.
        # TODO -------------------------------------------------------------------------------
        # user_interactions = user_interactions.reshape(-1) + 1  # Reshape to turn into 1D array
        # user_interactions[user_interactions > 1] = 0
        # TODO ===============================================================================

        # This is where we calculate the recommendation by taking the
        # dot-product of the user vectors with the item vectors.
        rec_vector = self._X.getrow(user_ID).dot(self._Y.T).toarray()

        # C R A P
        # Let's scale our scores between 0 and 1 to make it all easier to interpret.
        # min_max = MinMaxScaler()
        # rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        # recommend_vector = user_interactions * rec_vector_scaled
        # E N D  C R A P

        user_estimated = rec_vector
        user_real = np.argwhere(user_real > 0)

        user_estimated_sorted = np.argsort(-user_estimated)
        recommendation = [x for x in user_estimated_sorted if x not in user_real]


        # Get all the artist indices in order of recommendations (descending) and
        # select only the top "num_items" items.
        return recommendation[0:at]

    def recommendAll(self, userList, at=10):
        res = np.array([])
        for i in userList:
            recList = self.recommend(i, 10)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res
