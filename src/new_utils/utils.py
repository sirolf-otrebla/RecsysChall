import numpy as np
import scipy.sparse as sps
import pandas as pd

def precision(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score



def recall(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score



def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    if(is_relevant.shape[0] == 0 or relevant_items.shape[0] == 0):

        print('ciao')

    map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])

    return map_score

def evaluate_csv(URM_test, path):

    URM_test = sps.csr_matrix(URM_test)
    csv_file = pd.read_csv(path, index_col=False)

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    #row of the recommendation: csv_file.iloc[0,1].split()

    for i in range(0, len(csv_file)):
        user_id = csv_file.iloc[i,0]

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id+1]

        if end_pos-start_pos>0:

            relevant_items = URM_test.indices[start_pos:end_pos]
            recommended_items = list(map(int,csv_file.iloc[i,1].split()))

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)
            num_eval += 1

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict

def train_test_holdout(URM_all, train_perc = 0.8):


    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()


    train_mask = np.random.choice([True,False], numInteractions, [train_perc, 1-train_perc])


    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    URM_test = URM_test.tocsr()

    return URM_train, URM_test

if __name__ == '__main__':
    URM_text = np.loadtxt('../../data/train.csv', delimiter=',', dtype=int, skiprows=1)
    user_list, item_list = zip(*URM_text)
    rating_list = np.ones(len(user_list))
    URM = sps.csr_matrix((rating_list, (user_list, item_list)))
    evaluate_csv(URM, '../../results/recommendedCFtest.csv')