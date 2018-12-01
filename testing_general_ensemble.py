from Base.Evaluation.Evaluator import SequentialEvaluator
from refactored_code.general_ensemble import GeneralEnsemble
from refactored_code.utils import load_random_urms, load_icm, evaluate_csv, load_urm
import pandas as pd
import numpy as np
from scipy import sparse as sps

def xvalidation_par(elements=500, folds=1):
    maps = []
    alphas = []
    betas = []
    gammas = []
    epsilons = []
    rhos = []
    mus = []
    for i in range(0, elements):
        cosine_cf = np.random.uniform(0, 0.7)
        others = 1 - cosine_cf
        cfii = np.random.uniform(0.3,0.7)
        cfuu = 1 - cfii
        ml = np.random.uniform(0.6, 1)
        cbf = 1 - ml
        ials = np.random.uniform(0.2, 0.8)
        bpr = 1 - ials
        bprii = np.random.uniform(0.2,0.8)
        bpruu = 1 - bprii

        alpha = cosine_cf*cfii
        beta = cosine_cf*cfuu
        gamma = others*cbf
        epsilon = others*ml*ials
        mu = others*ml*bpr*bpruu
        rho = others*ml*bpr*bprii + mu
        print('\n \n_____________________________________')
        print('starting iteration {0} with: \n\n '
              '\t alpha = {1} \n '
              '\t beta = {2} \n'
              '\t gamma = {3} \n'
              '\t epsilon = {4} \n'
              '\t rho = {5} \n'
              '\t mu = {6}\n '
              '\t --------------------------- \n '
              '\t \t SUM = {7} \n'.format(i, alpha, beta, gamma, epsilon, rho, mu, (alpha+beta+gamma+epsilon+rho+mu)))
        print('_____________________________________\n \n')
        data = []
        for j in range(0, folds):
            res = main(alpha, beta, gamma, epsilon, rho, mu)
            map = res["MAP"]
            data.append(map)
        data_array = np.array(data)
        mean = np.average(data_array)
        alphas.append(alpha)
        betas.append(beta)
        gammas.append(gamma)
        epsilons.append(epsilon)
        rhos.append(rho)
        mus.append(mu)
        maps.append(mean)
        print('\n \n_____________________________________')
        print('finished iteration {0} with: \n\n '
              '\t alpha = {1} \n '
              '\t beta = {2} \n'
              '\t gamma = {3} \n'
              '\t epsilon = {4} \n'
              '\t rho = {5} \n'
              '\t mu = {6}\n '.format(i, alpha, beta, gamma, epsilon, rho, mu))
        print('_____________________________________\n \n')

        d = {"alpha": alphas,
             "beta " : betas,
             "gamma" : gammas,
             "epsilon" : epsilons,
             "rho" : rhos,
             "mu" : mus,
             "map": maps}
        df = pd.DataFrame(data=d, index=None)
        df.to_csv("./results/evaluation/ENSEMBLE_WITH_BPR.csv", index=None)

def main( write=True):
    URM_text = np.loadtxt('./data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))
    urm_train, urm_test = load_random_urms()
    icm = load_icm()

    general = GeneralEnsemble(URM, urm_test, icm, recommendation_mode='linComb')
                              # alpha=alpha,
                              # beta=beta,
                              # gamma=gamma,
                              # epsilon=epsilon,
    general.fit(0.001)

    target = pd.read_csv('./data/target_playlists.csv', index_col=False)
    #evaluator_MF = SequentialEvaluator(URM_test_list=urm_test, cutoff_list=[10])
    #print(evaluator_MF.evaluateRecommender(general))

    if write is True:
        recommended = general.recommendALL(target.values)

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
        df.to_csv("./results/TESTING_GENERAL_ENSEMBLE17_nocf.csv", index=None)
        del general
        return evaluate_csv(urm_test, "./results/TESTING_GENERAL_ENSEMBLE17_nocf.csv")

if __name__ == '__main__':
    main( write=True)
