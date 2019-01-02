from src.ensembles.AVoltaEnsemble import AVoltaEnsemble
from src.combiners.probabilisticCombiner import ProbabilisticCombiner
from refactored_code.utils import load_random_urms, load_icm, evaluate_csv, create_top_pop_list, train_test_holdout
import json
import io
import pandas as pd
import numpy as np
from scipy import sparse as sps #180344
import gc

POPULARITY_SCALING_EXP = 0.190344
MODE = "TEST"
def load_data():
    URM_text = np.loadtxt('../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))
    urm_train, urm_test = train_test_holdout(urm_all=URM)
    topPop = create_top_pop_list()
    urm_train.tocoo()
    j = 0
    for i in topPop:
        factor = (len(topPop)-j)**POPULARITY_SCALING_EXP
        urm_train[:,i].multiply(1/factor)
        j += 1
    urm_train.tocsr()
    icm = load_icm()
    return { "URM_complete" : URM, "URM_test" : urm_test, "URM_train" : urm_train, "ICM" : icm}

def load_parameters(path):
    file = open(path, mode='r')
    json_parameters = json.load(file)
    file.close()
    return  json_parameters

def run_algorithm(data, parameters):
    if MODE == "SUBMIT":
        recommender = AVoltaEnsemble(data["URM_complete"], data["URM_test"], icm=data["ICM"], parameters=parameters)
    else:
        recommender = AVoltaEnsemble(data["URM_train"], data["URM_test"], icm=data["ICM"], parameters=parameters)
    recommender.fit()
    target = pd.read_csv('../data/target_playlists.csv', index_col=False)
    target  = target['playlist_id']
    recommended = recommender.recommend_batch(user_list=target, combiner=ProbabilisticCombiner())
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
    distrib_data = recommender.get_component_data()
    del recommender
    gc.collect()
    return [recommended, df, distrib_data]

def write_results(id, urm_test, dict, data_frame, component_attributes, encoder=None):
    if encoder is None:
        encoder = json.JSONEncoder()
    filename = "../results/AVolta/result_{0}.csv".format(id)
    data_frame.to_csv(filename, index=None)
    eval = evaluate_csv(urm_test, filename)
    eval_json = encoder.encode(eval)
    param_json = encoder.encode(dict)
    param_file = open('../results/AVolta/params_{0}.json'.format(id), mode='w')
    param_file.write(param_json)
    param_file.close()
    eval_file = open('../results/AVolta/eval_{0}.json'.format(id), mode='w')
    eval_file.write(eval_json)
    eval_file.close()
    return  eval

def print_info(dict, eval):
    print("\n \n PARAMETERS : \n " + str(dict) + "\n ############################################################ \n")
    print("\n \n EVALUATION : \n " + str(eval) + "\n ############################################################ \n")

def main():
    data = load_data()
    param_list = load_parameters('./parameters/AVolta.json')
    encoder = json.JSONEncoder()
    i = 0
    for dict in param_list:
        result = run_algorithm(data, dict)
        eval = write_results(i, data['URM_test'], dict, result[1], result[2], encoder)
        i += 1
        print_info(dict, eval)



if __name__ == "__main__":
    MODE = "SUBMIT"
    main()
