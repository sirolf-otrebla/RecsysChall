from refactored_code.utils import *
from refactored_code.general_ensemble import GeneralEnsemble

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
            train, test = load_random_urms()
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
        df.to_csv("./results/evaluation/data_ensembleALL.csv", index=None)


def main(alpha, beta, URM_train, URM_test):

    ICM = load_icm()

    cf = GeneralEnsemble(URM_train, URM_test, ICM, 50, epsilon=beta)
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
    df.to_csv("./results/resultsEnsembleAll.csv", index=None)

    return evaluate_csv(URM_test, "./results/resultsEnsembleAll.csv")


if __name__ == '__main__':
    xvalidation_par(250, 2)
