from Base.Cython.cosine_similarity import Cosine_Similarity

from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.IALS_numpy import IALS_numpy
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from sklearn import preprocessing
import numpy as np


class GeneralEnsemble:

    def __init__(self, URM_train, URM_test, ICM, k=100,
                 alpha= 0.7662,
                 beta= 0.6188,
                 gamma=0.3,
                 epsilon= 0.6212, #0.6212,
                 ro=0.7662,
                 mu = 0.6118,
                 chi = 1,
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

        self.IALS_SCALING = 1
        self.SLIM_BPR_SCALING = 1
        self.SLIM_BPR = ro
        # self.SLIM_BPRUU = mu
        self.MF_BPR = chi

    def fit(self, alpha):
        evaluator_MF = SequentialEvaluator(URM_test_list=self._URM_test, cutoff_list=[10])
        #bprmf = MatrixFactorization_Cython(self._URM_train,
        #                                   positive_threshold=0,
        #                                   algorithm="MF_BPR",
        #                                   )
        # self.MF_BPRW, self.MF_BPRH = bprmf.fit(epochs=200,
        #                                       num_factors=5,
        #                                       batch_size=1,
        #                                       sgd_mode='adagrad'
        #                                       )
        #print(evaluator_MF.evaluateRecommender(bprmf))


        self.bpr_WII = SLIM_BPR_Cython(self._URM_train, positive_threshold=0, symmetric=True).fit(epochs=10,
                                                                                                   topK=200,
                                                                                                    batch_size=200,
                                                                                                   sgd_mode='adagrad',
                                                                                                   learning_rate=1e-2)

        # self.bpr_WUU = SLIM_BPR_Cython(self._URM_train.T, positive_threshold=0).fit(epochs=10,
        #                                                                            topK=200,
        #                                                                            batch_size=200,
        #                                                                            sgd_mode='adagrad',
         #                                                                           learning_rate=1e-2)

        print(self.bpr_WII)
        print("\n \n max bprII: {0}".format(self.bpr_WII.max()))
        #print(self.bpr_WII)
        #print("\n \n max bprUU: {0}".format(self.bpr_WUU.max()))
        # self._similarity_matrixUU = Cosine_Similarity(self._URM_train.T,
        #                                              topK=200,
        #                                              shrink=15,
        #                                              normalize=True,
        #                                              mode='cosine').compute_similarity()
        # print("\n \n max uu: {0}".format(self._similarity_matrixUU.max()))

        # self._similarity_matrixII = Cosine_Similarity(self._URM_train.tocsc(),
        #                                              topK=200,
        #                                              shrink=10,
        #                                              normalize=True,
        #                                              mode='cosine').compute_similarity()

        # print("\n \n max II: {0}".format(self._similarity_matrixII.max()))

        self._similarity_matrixCBF = Cosine_Similarity(self._ICM.T,
                                                       topK=10,
                                                       shrink=10,
                                                       normalize=True,
                                                       mode='cosine').compute_similarity()
        # print(self._similarity_matrixII)
        self.latent_x, self.latent_y = (IALS_numpy()).fit(self._URM_train)
        print(self.latent_x.dot(self.latent_y.T))
        print("\n \n max IALS: {0}".format(self.latent_x.dot(self.latent_y.T).max()))

    def _scoreOthers(self, user_id, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        # normalized_IALS = np.dot(self.latent_x[user_id], self.latent_y.T)
        normalized_IALS = self.IALS_SCALING*self.IALSSCORE*np.dot(self.latent_x[user_id], self.latent_y.T)
        # cfii = self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray()
        # cfuu = self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray()
        cbf =  self.CBFSCORE*self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray()

        scores = ( cbf + normalized_IALS).ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        return scores

    def _scoreBPR(self, user_id, exclude_seen=True):

        user_profile = self._URM_train[user_id]
        bprii = self.SLIM_BPR*user_profile.dot(self.bpr_WII.T).toarray().ravel()
        # bpruu = self.SLIM_BPRUU*self.bpr_WUU[user_id].dot(self._URM_train).toarray().ravel()
        # mfbpr = self.MF_BPR*self.MF_BPRW[user_id].dot(self.MF_BPRH.T)
        ensemble = bprii # + bpruu
        if exclude_seen:
            ensemble = self.filter_seen(user_id, bprii)#+bpruu)

        return ensemble

    def _recommendOthers(self, user_id, at=30, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        # normalized_IALS = np.dot(self.latent_x[user_id], self.latent_y.T)
        normalized_IALS = np.dot(self.latent_x[user_id], self.latent_y.T)
        # cfii = self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray()
        # cfuu = self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray()
        cbf =  self.CBFSCORE*self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray()

        scores = ( cbf + normalized_IALS)
        scores = preprocessing.normalize(scores, norm='max').ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at].ravel()

    def _recommendOthersBPRII(self, user_id, at=30, exclude_seen=True):

        scores = self._scoreOthers(user_id, exclude_seen) + self._scoreBPR(user_id, exclude_seen)
        ranking = scores.argsort()[::-1]
        return ranking[:at].ravel()


    def _recommendBPRUU(self, user_id, at=30, exclude_seen=True):
        user_profile = self._URM_train[user_id]
        bpruu = self.SLIM_BPRUU*self.bpr_WUU[user_id].dot(self._URM_train).toarray().ravel()
        ensemble = bpruu
        if exclude_seen:
            ensemble = self.filter_seen(user_id, ensemble)

        # rank items
        ranking = ensemble.argsort()[::-1]

        return ranking[:at].ravel()


    def recommendProbabilistic(self, user_id, at=10, exclude_seen=True):
        others = self._recommendOthers(user_id, at=30, exclude_seen=exclude_seen)
        # bpr = self._recommendBPRUU(user_id, at=30, exclude_seen=exclude_seen)
        result = []
        i = 0
        while i < at:
            rand = np.random.uniform(0, 1)
            if rand < 0.2:
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

        scores = (self._scoreOthers(user_id) + self.SLIM_BPR_SCALING*self._scoreBPR(user_id))

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
