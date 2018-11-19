from refactored_code.IALS_numpy import IALS_numpy
from refactored_code.utils import *

if __name__ == '__main__':
    recsys = IALS_numpy(num_factors=200, reg=0.03, iters=10, scaling='log')
    train, test = load_random_urms()
    target = pd.read_csv('../../data/target_playlists.csv', index_col=False)

    recsys.fit(train)

    recommended = recsys.recommendALL(target.values)

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
    df.to_csv("./results/results_als_tuning.csv", index=None)

    evaluate_csv(test, "./results/results_als_tuning.csv")

