import os
import math
import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from ADP.a_initailization.initialization import initialization
from ADP.b_static_selection.static_selection import neighbors_labeling, descending_order, sliding_window, static_selection
from ADP.c_danamic_selection.dynamic_selection import query_strategy, dynamic_selection, uncertainty_selection, uncertainty_selection_m
from myutil import DataLoader
from PRSCSWAP import PRS


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data+random_matrix
    return data


def initialization_m(data, theata=1):
    # label = rdata.iloc[:, -1]
    data = (data - data.mean()) / (data.std())
    # 使用sklearn 给label编码

    # print(f"data shape {data.shape} label shape {label.shape}")
    num_thread = math.ceil(math.ceil(len(real_labels) / (theata * 100)))
    # # print(label)
    prs = PRS(data)
    threshold_clusters = K
    prs.get_clusters(num_thread, threshold_clusters)
    ascription_tree, P = prs.get_final_tree_nx()
    nearest_neighbors = {}
    for node in ascription_tree.nodes:
        nearest_neighbors[node] = list(ascription_tree.neighbors(node))

    # P = prs.final_tree['roots']
    # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
    # distance, nearest_neighbors = neighbors.kneighbors(
    #     data, return_distance=True)
    # distance = distance[:, 1]
    # knn = [[np[0], np[1], d] for a, b in nearest_neighbors]

    return ascription_tree, list(P), nearest_neighbors


def experiemnt_adp(data: pd.DataFrame, real_labels, alpha, l, theta):
    ARI_record = []
    ARI = adjusted_rand_score(real_labels, [0] * len(data))
    interaction = 0
    iter = 0
    ARI_record.append([{"iter": iter, "interaction": interaction, "ari": ARI}])

    # ascription_tree, P,  knn = initialization_m(data)
    data = data.values

    dc, density_vec, distances_vec, center_vec, ascription_tree, nearest_neighbors = initialization(
        alpha, data)
    center_vec_dec = descending_order(center_vec)
    P = sliding_window(l, center_vec_dec, theta)

##############################################################################################
##############################################################################################
##############################################################################################

    neighbors, count = static_selection(P, real_labels)

    predict_labels, result_dict = neighbors_labeling(
        ascription_tree, neighbors)
    # data = data.values

    while True:
        a = uncertainty_selection(
            predict_labels, data, neighbors, result_dict, nearest_neighbors)
        # a = uncertainty_selection(
        #     predict_labels, data, neighbors, result_dict, knn)
        if a == None:
            print("查询完毕")
            break

        x = int(a)
        candidates = query_strategy(x, neighbors, data)
        neighbors, count = dynamic_selection(
            real_labels, candidates, neighbors, x, count)

        predict_labels, result_dict = neighbors_labeling(
            ascription_tree, neighbors)
        iter = iter + 1
        ARI = adjusted_rand_score(real_labels, predict_labels)
        print(iter)
        ARI_record.append([{"iter": iter, "interaction": count, "ari": ARI}])
        print(iter, count, ARI)
        if ARI == 1:
            break
        if count > 20000:
            break
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
    df.to_csv("g_output/%s_result.csv" % title)


if __name__ == '__main__':

    alpha = 0.22
    l = 5
    theta = 0.00001
    datasets = ["letter"]
    for file in os.listdir("exp_disturbed"):
        data, real_labels, K = DataLoader.get_data_from_local(
            "exp_disturbed/"+file, doPerturb=True)
        if len(real_labels) > 1e3:
            continue
        print(f'run on {file}')

        ARI_record = experiemnt_adp(
            data, real_labels, alpha, l, theta)
        result_to_csv(ARI_record, title=file.split(".")[0])
        print(f'fff on {file}')
