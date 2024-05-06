import time
import os
import math

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, rand_score
# import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from DSL.DSL import clustering, data_preprocess, iteration_once
from myutil import DataLoader
from PRSCSWAP import PRS
# import standardize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np


def neiborhood_insertion(N, data, label):
    for i, x_k in enumerate(N):
        if label[data] == label[x_k[0]]:
            x_k.append(data)
            return i, True
    N.append([data])
    return len(N)-1, False


def get_label_from_neiborhood(N, len):
    label = [-1]*len
    for i, x_k in enumerate(N):
        for j in x_k:
            label[j] = i
    return label


def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec


def skeleton_process(Graph):
    clusters = []
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels


def DSL(data, real_labels, title, skeleton=None, representatives=None):
    start_time = time.time()
    # data = data_preprocess(data)
    if skeleton is None or representatives is None:
        skeleton, representatives = clustering(data)
    loop = 10
    loop = 10+len(data)*10000
    constraint_graph = nx.Graph()
    interaction = 0
    predict_labels = skeleton_process(skeleton)
    # ARI = adjusted_rand_score(real_labels, predict_labels)
    ARI = rand_score(real_labels, predict_labels)
    df = pd.DataFrame(
        {"iter": [0], "interaction": [interaction], "ari": [ARI], "time": [0]})
    for i in range(loop):
        skeleton, representatives, constraint_graph, count, suspend = iteration_once(skeleton, representatives, data,
                                                                                     real_labels, constraint_graph)
        interaction = interaction + count
        # print(skeleton.nodes(data=True))
        if suspend == True:
            print("The algorithm is down")
            break
        predict_labels = skeleton_process(skeleton)
        ARI = adjusted_rand_score(real_labels, predict_labels)
        # ARI = rand_score(real_labels, predict_labels)
        duration = time.time() - start_time
        record = [{"iter": i + 1, "interaction": interaction,
                   "ari": ARI, "time": duration}]
        df = df._append(pd.DataFrame(record), ignore_index=True)
        df.to_csv("e_output/%s_result.csv" % title)
        if ARI == 1:
            break
    return df


if __name__ == '__main__':
    dataDir = "data"
    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    for file in os.listdir(dataDir):
        rdata, real_labels, K = DataLoader.get_data_from_local(
            f'{dataDir}/{file}', doPerturb=True)
        if len(real_labels) > 1e3:
            continue

        print(f'run on {file}')
        theta = 1
        data = rdata.copy()
        # data = (data - data.mean()) / (data.std())
        # data = MinMaxScaler().fit_transform(data)
        # data = pd.DataFrame(data)
        # 使用sklearn 给label编码
        # le = LabelEncoder()
        # label = le.fit_transform(label)
        # label = label.tolist()
        # print(f"data shape {data.shape} label shape {label.shape}")
        num_thread = math.ceil(math.ceil(len(real_labels) / (theta * 100)))
        # print(label)
        # data.drop_duplicates(inplace=True)

        prs = PRS(data)
        start = time.time()
        threshold_clusters = K
        threshold_clusters = 2
        prs.get_clusters(num_thread, threshold_clusters)
        ET, roots = prs.get_final_tree_nx()
        roots = list(roots)
        c = 100

        neighbors = NearestNeighbors(n_neighbors=2).fit(data)
        distance, nearest_neighbors = neighbors.kneighbors(
            data, return_distance=True)
        distance = distance[:, 1]
        print(f'min of distance: {np.min(distance)}')
        t = 100
        uc = {}
        # for a, b in enumerate(nx.bfs_layers(ET, roots)):
        #     uc.update({n: t-a for n in b})
        # attrs = {i: {"uncertainty": d} for i, d in uc.items()}
        distance = 1/(distance+1e-7)
        # distance = -distance
        data = data.values
        # attrs = {}
        # for n in ET.nodes:
        #     # distance_between_n_and_neighbors
        #     dis = [np.linalg.norm(data[n]-data[nn])
        #            for nn in list(ET.neighbors(n))]
        #     if dis:
        #         attrs[n] = {"uncertainty": max(dis)}
        #     else:
        #         attrs[n] = {"uncertainty": distance[n]}

        attrs = {i: {"uncertainty": d} for i, d in enumerate(distance)}
        # attrs = {i: {"uncertainty": d + np.random.rand()}
        #          for i, d in ET.out_degree()}
        # print(attrs)
        # print(attrs)
        # c -= 1
        nx.set_node_attributes(ET, attrs)
        # attrs = {i: {"uncertainty": d} for i, d in prs.s_score.items()}
        # nx.set_node_attributes(ET, attrs)
        # print(ET.nodes(data=True))
        # for r in roots:
        #     ET.nodes[r]['uncertainty'] = 0
        # ET.remove_edges_from(nx.selfloop_edges(ET))
        # draw networkx graph
        # pos = nx.spring_layout(ET)
        # nx.draw(ET, pos, with_labels=True)
        # plt.show()

        # print(a, b)

        # dict(enumerate(nx.bfs_layers(ET, [roots])))

        # ARI_record = DSL(data, real_labels, title=file.split(".")[
        #     0], skeleton=nx.DiGraph.reverse(ET), representatives=roots)

        ARI_record = DSL(data, real_labels, title=file.split(".")[
            0], skeleton=None, representatives=None)

        tail = ARI_record.tail(1)
        ari21['dataset'].append(file.split(".")[0])
        ari21['ari'].append(tail['ari'].values[0])
        ari21['interaction'].append(tail['interaction'].values[0])
        print(f'run on {file} finished')
        # 取pd datafream 最后一行

        # ARI_record = DSL(data, real_labels, title=file.split(".")[
        #     0], skeleton=None, representatives=None)

    df = pd.DataFrame(ari21)
    df.to_csv("our.csv")
