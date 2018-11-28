from Base.Cython.cosine_similarity import Cosine_Similarity
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import pyximport; pyximport.install()
from sklearn import preprocessing
import numpy as np


class GeneralEnsemble:

    def __init__(self, URM_train, URM_test, ICM, k=100,
                 alpha= 0.7662,
                 beta= 0.8188,
                 gamma=0.30,
                 epsilon=0.6212,
                 ro=0.80,
                 mu = 0.45,
                 shrink=15,
                 recommendation_mode='linComb'):

        self._URM_train = URM_train.tocsr()
        self._URM_test = URM_test.tocsr()
        self._ICM = ICM.tocsr()
        self._k = k
        self._shrink = shrink
        self._recommendationMode = recommendation_mode

        self.UUSCORE = alpha

        self.IISCORE = beta

        self. CBFSCORE = gamma

        self.IALSSCORE = epsilon

        self.SLIM_BPR = ro
        self.SLIM_BPRUU = mu

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

        # self.MF_BPRW, self.MF_BPRH = MF_BPR_Cython(self._URM_train, True).fit(epochs=50,
        #                                                                      URM_test=self._URM_test,
        #                                                                      validate_every_N_epochs=9,
        #                                                                      start_validation_after_N_epochs=1,
        #                                                                      num_factors=10,
        #                                                                      sgd_mode='adagrad',
        #                                                                      positive_threshold=0)

        self.latent_x, self.latent_y = (IALS_numpy(reg=alpha)).fit(self._URM_train)

        self.bpr_WII = SLIM_BPR_Cython(self._URM_train, positive_threshold=0, symmetric=True, sparse_weights=False).fit(epochs=10,
                                                                                                   topK=500,
                                                                                                   validate_every_N_epochs=11,
                                                                                                   URM_test=self._URM_test,
                                                                                                    batch_size=500,
                                                                                                   sgd_mode='adagrad',
                                                                                                   learning_rate=1e-4)

        self.bpr_WUU = SLIM_BPR_Cython(self._URM_train.T, positive_threshold=0).fit(epochs=10,
                                                                                    validate_every_N_epochs=2000000,
                                                                                    topK=500,
                                                                                    URM_test=self._URM_test.T,
                                                                                    batch_size=500,
                                                                                    sgd_mode='adagrad',
                                                                                    learning_rate=1e-4)

        # self.bpr_WUU = preprocessing.normalize(self.bpr_WUU, norm='max', axis=0)
        # self.bpr_WII = preprocessing.normalize(self.bpr_WII, norm='max', axis=1)
        # self._similarity_matrixCBF = preprocessing.normalize(self._similarity_matrixCBF, norm='max', axis=1)
        # self._similarity_matrixII = preprocessing.normalize(self._similarity_matrixII, norm='max', axis=1)
        # self._similarity_matrixUU = preprocessing.normalize(self._similarity_matrixUU, norm='max', axis=1)

    def _recommendOthers(self, user_id, at=30, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        # normalized_IALS = np.dot(self.latent_x[user_id], self.latent_y.T)
        normalized_IALS = np.dot(self.latent_x[user_id], self.latent_y.T)
        cfii = self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray()
        cfuu = self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray()
        cbf =  self.CBFSCORE*self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray()

        scores = ( cbf + normalized_IALS + cfuu + cfii).ravel()
        # scores = (self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray() +
        #          self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray() +
        #          self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray() +
        #          self.IALSSCORE*normalized_IALS +
        #          self.SLIM_BPR*user_profile.dot(self.bpr_WII.T).toarray()
        #          #self.SLIM_BPR*self.bpr_WUU[user_id].dot(self._URM_train).toarray()
        #          ).ravel()
        scores = preprocessing.normalize(scores, norm='max')
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at].ravel()

    def _recommendBpr(self, user_id, at=30, exclude_seen=True):
        user_profile = self._URM_train[user_id]
        bprii = self.SLIM_BPR*preprocessing.normalize(user_profile.dot(self.bpr_WII.T).toarray(), norm='max').ravel()
        bpruu = self.SLIM_BPRUU*preprocessing.normalize(self.bpr_WUU[user_id].dot(self._URM_train).toarray(), norm='max').ravel()
        ensemble = bprii + bpruu
        if exclude_seen:
            ensemble = self.filter_seen(user_id, ensemble)

        # rank items
        ranking = ensemble.argsort()[::-1]

        return ranking[:at].ravel()

    def recommendProbabilistic(self, user_id, at=10, exclude_seen=True):
        others = self._recommendOthers(user_id, at=30, exclude_seen=exclude_seen)
        bpr = self._recommendBpr(user_id, at=30, exclude_seen=exclude_seen)
        result = []
        i = 0
        while i < at:
            rand = np.random.uniform(0, 1)
            if rand < 0.40:
                if type(others) is np.ndarray:
                    chosen = others[0]
                else:
                    chosen = others
                others = np.delete(others, 0)
            else:
                if type(bpr) is np.ndarray:
                    chosen = bpr[0]
                else:
                    chosen = bpr
                bpr = np.delete(bpr, 0)
            if chosen in result:
                continue
            else:
                result.append(chosen)
                i += 1

        return np.array(result)
    def recommendLinComb(self, user_id, at=10, exclude_seen=True):

        scores = (self._recommendOthers(user_id, at, exclude_seen) + self._recommendBpr(user_id, at, exclude_seen)).ravel()
        scores = preprocessing.normalize(scores, axis=1, norm='max')
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at].ravel()
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
            if self._recommendationMode == 'linComb':
                recList = self.recommendLinComb(i, at)
            elif self._recommendationMode == 'probabilistic':
                recList = self.recommendProbabilistic(i, at)
            else:
                recList = self.recommendLinComb(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        del self.bpr_WII
        return res
