import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ADP.a_initailization.initialization import initialization
from ADP.b_static_selection.static_selection import descending_order, sliding_window, \
    static_selection, neighbors_labeling
from ADP.c_danamic_selection.dynamic_selection import uncertainty_selection, query_strategy, \
    dynamic_selection


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data+random_matrix
    return data


def experiemnt_adp(data, real_labels, alpha, l, theta, q=1000):
    ARI_record = []
    ARI_record = {"iter": [], "interaction": [], "ari": []}
    ARI = adjusted_rand_score(real_labels, [0] * len(data))
    interaction = 0
    iter = 0

    ARI_record["iter"].append(iter)
    ARI_record["interaction"].append(interaction)
    ARI_record["ari"].append(ARI)

    # ARI_record.append([{"iter": iter, "interaction": interaction, "ari": ARI}])
    dc, density_vec, distances_vec, center_vec, ascription_tree, nearest_neighbors = initialization(
        alpha, data)
    center_vec_dec = descending_order(center_vec)
    P = sliding_window(l, center_vec_dec, theta)
    neighbors, count = static_selection(P, real_labels)
    predict_labels, result_dict = neighbors_labeling(
        ascription_tree, neighbors)
    while (True) or count < q:
        a = uncertainty_selection(
            predict_labels, data, neighbors, result_dict, nearest_neighbors)
        if a == None:
            print("查询完毕")
            break
        else:
            x = int(a)
        candidates = query_strategy(x, neighbors, data)
        neighbors, count = dynamic_selection(
            real_labels, candidates, neighbors, x, count)
        predict_labels, result_dict = neighbors_labeling(
            ascription_tree, neighbors)
        iter = iter+1
        ARI = adjusted_rand_score(real_labels, predict_labels)
        # print(iter)
        ARI_record["iter"].append(iter)
        ARI_record["interaction"].append(count)
        ARI_record["ari"].append(ARI)
        if ARI == 1:
            break
        # ARI_record.append([{"iter": iter, "interaction": count, "ari": ARI}])
    return ARI_record


def result_to_csv(ARI_record, title):
    # 将所有的交互到加入到里面去`
    record = []
    for i in range(len(ARI_record)):
        if i < len(ARI_record)-1:
            repeat = ARI_record[i+1][0]["interaction"] - \
                ARI_record[i][0]["interaction"]
            for j in range(repeat):
                record.append(ARI_record[i][0]["ari"])
        else:
            record.append(ARI_record[i][0]["ari"])
    # 写入文档
    df = pd.DataFrame({
        'ARI': record,
    })
    df.to_csv("../g_output/%s_result.csv" % title)


if __name__ == '__main__':
    alpha = 0.22
    l = 5
    theta = 0.00001

    datasets = ["balance", "banknote", "breast", "dermatology", "diabetes", "ecoli", "glass", "haberman", "ionosphere", "iris", "led", "musk",
                "pima", "seeds", "segment", "soybean", "thyroid", "vehicle",
                "wine", "mfeat_karhunen", "mfeat_zernike"]
    datasets = ["sonar", "fertility", "plrx", "zoo", "tae"]

    for dataset in datasets:
        func_name = "generate_" + dataset + "_data"
        generate_data_func = getattr(sys.modules[__name__], func_name)
        path = "../../../dataset/small/{}/{}.data".format(dataset, dataset)
        data, real_labels = generate_data_func(path)
        ARI_record = experiemnt_adp(data, real_labels, alpha, l, theta)
        result_to_csv(ARI_record, title="{}".format(dataset))
