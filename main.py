import time
import os
import math


import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, rand_score

from DSL.DSL import clustering, data_preprocess, iteration_once
from myutil import DataLoader
from PRSCSWAP import PRS
# import standardize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


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
        print("clustering using original date")
        skeleton, representatives = clustering(data)

        # attrs = {i: {"uncertainty": np.random.randn()}
        #          for i in list(skeleton)}
        # attrs[representatives[0]]['uncertainty'] = 0
        # nx.set_node_attributes(skeleton, attrs)

    loop = 10
    loop = int(1e10)
    constraint_graph = nx.Graph()
    interaction = 0

    predict_labels = skeleton_process(skeleton)
    ARI = adjusted_rand_score(real_labels, predict_labels)
    # ARI = rand_score(real_labels, predict_labels)
    df = {"iter": [0], "interaction": [interaction], "ari": [ARI], "time": [0]}
    for i in range(loop):
        skeleton, representatives, constraint_graph, count, suspend = iteration_once(skeleton, representatives, data,
                                                                                     real_labels, constraint_graph)
        interaction += count
        # print(skeleton.nodes(data=True))
        if suspend == True:
            # assert i > len(real_labels)
            print(f"The algorithm is down at loop{
                  i} len data{len(real_labels)}")
            break

        predict_labels = skeleton_process(skeleton)
        ARI = adjusted_rand_score(real_labels, predict_labels)
        # ARI = rand_score(real_labels, predict_labels)
        duration = time.time() - start_time
        df['iter'].append(i + 1)
        df['interaction'].append(interaction)
        df['ari'].append(ARI)
        df['time'].append(duration)

        if ARI == 1:
            break
    print(f"the representatives of {title} is {representatives} and k is {
          len(representatives) == len(np.unique(real_labels))}")
    return pd.DataFrame(df), representatives


def set_uncertainty(data: np.array, ET: nx.Graph, roots: list):
    attrs: dict[int, float] = {}
    o = {"uncertainty": 0}
    for ei, ej in ET.edges:
        # distance_between_n_and_neighbors
        dis = np.linalg.norm(data[ei]-data[ej])
        attrs[ei] = {"uncertainty": max(dis+ET.in_degree(ei), attrs.get(
            ei, o)["uncertainty"])+ET.in_degree(ei)}
        attrs[ej] = {"uncertainty": max(dis+ET.in_degree(ej), attrs.get(
            ej, o)["uncertainty"])+ET.in_degree(ej)}

    # min_dis = min([d['uncertainty'] for d in attrs.values()])
    # print(f'min_dis: {min_dis}')

    # centrality = nx.degree_centrality(ET)
    # centrality = nx.in_degree_centrality(ET)
    # centrality = nx.out_degree_centrality(ET)
    # centrality = nx.load_centrality(ET)
    # centrality = nx.trophic_levels(ET)

    # centrality = nx.betweenness_centrality(ET)
    # centrality = nx.closeness_centrality(ET.reverse())
    # centrality = nx.katz_centrality(ET)
    # centrality = nx.harmonic_centrality(ET)
    # centrality = nx.laplacian_centrality(ET)

    # centrality = {n: abs(np.random.randn() + 1e-4) for n in ET.nodes}
    minc = 1000

    # for _, v in centrality.items():
    #     if v < minc:
    #         minc = v
    # print(f'min centrality: {minc}')

    # attrs = {i: {"uncertainty": d + abs(np.random.randn())*1e-5}
    #          for i, d in centrality.items()}

    for n in roots:
        attrs[n]['uncertainty'] = 0
    nx.set_node_attributes(ET, attrs)
    return attrs


def get_auic(path: str):
    auic = {'dataset': [], 'auic': []}
    for file in os.listdir(path):
        auic['dataset'].append(file.split("_")[0])
        df = pd.read_csv(f'{path}/{file}')
        auic['auic'].append(df['ari'].sum()/(df.shape[0]*2))
    pd.DataFrame(auic).to_csv("auic.csv", index=False)


def draw_graph(ET: nx.Graph, data: np.array, roots: list, title: str, real_labels=None):
    # pos = nx.spring_layout(ET)
    pos = nx.kamada_kawai_layout(ET)
    # pos = nx.spectral_layout(ET)
    # pos = nx.shell_layout(ET)
    # pos = nx.circular_layout(ET)
    # pos = nx.planar_layout(ET)
    # pos = nx.random_layout(ET)
    # pos = nx.fruchterman_reingold_layout(ET)
    # pos = nx.bipartite_layout(ET, roots)

    nx.draw(ET, pos, with_labels=True)
    realcluster: dict[int, list] = {}
    for i, l in enumerate(real_labels):
        if l not in realcluster:
            realcluster[l] = []
        realcluster[l].append(i)
    # c = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    # get 20 named colors
    c = plt.cm.tab20.colors
    # get named colors
    for k, v in realcluster.items():
        nx.draw_networkx_nodes(
            ET, pos, nodelist=v, node_color=c[k])
    nx.draw_networkx_nodes(ET, pos, nodelist=roots, node_color='r')
    plt.title(title)
    plt.savefig(f'result/p2/graph/{title}.pdf')
    plt.show()


def get_rank(ET: nx.Graph, data: np.array, roots: list):
    # 将ET变为无向图
    R = []
    visted = set()
    # 使用普利姆算法遍历ET，得到最大生成树，时间复杂度尽量小
    # for r in roots:
    #     visted.add(r)
    #     for n in ET.neighbors(r):
    #         R.append((r, n))
    #         visted.add(n)
    # while len(visted) < len(ET.nodes):
    #     min_dis = 1e10
    #     min_edge = None
    #     for e in ET.edges:
    #         if e[0] in visted and e[1] not in visted:
    #             dis = np.linalg.norm(data[e[0]]-data[e[1]])
    #             if dis < min_dis:
    #                 min_dis = dis
    #                 min_edge = e
    #         if e[1] in visted and e[0] not in visted:
    #             dis = np.linalg.norm(data[e[0]]-data[e[1]])
    #             if dis < min_dis:
    #                 min_dis = dis
    #                 min_edge = e
    #     R.append(min_edge)
    #     visted.add(min_edge[0])
    #     visted.add(min_edge[1])

    raise NotImplementedError


def calculate_uncertainty(ET: nx.Graph, data: np.array, roots: list, N):
    raise NotImplementedError


def get_G(ET: nx.Graph, data: np.array, gamma=1.):
    G = ET.to_undirected()
    for e in ET.edges:
        G.add_edge(
            e[0], e[1], weight=gamma*np.exp(-np.linalg.norm(data[e[0]]-data[e[1]])))
    # 得到邻接矩阵
    A = nx.adjacency_matrix(G).toarray()
    # 判断是否是对称矩阵
    assert np.allclose(A, A.T)
    return A


def refine_graph(ET: nx.Graph, data: np.array, roots: list, title: str, real_labels=None):
    # 将ET变为无向图
    T = ET.to_undirected()
    N: dict[int, set] = {}
    maxiter = len(data)*100
    for i in range(maxiter):
        N[i] = set()
        for j in range(len(data)):
            if i != j:
                N[i].add(j)
    return


def label_propagation(data, real_labels, title):
    start = time.time()
    label_prop_model = LabelPropagation()
    label_prop_model.fit(data, real_labels)
    predict_labels = label_prop_model.transduction_
    ARI = adjusted_rand_score(real_labels, predict_labels)
    duration = time.time() - start
    print(f"the ARI of {title} is {ARI} and time is {duration}")
    return ARI


def run(dataDir: str = "data", uncertainty=False) -> None:
    dataDir = "data"
    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }
    for file in os.listdir(dataDir):
        rdata, real_labels, K = DataLoader.get_data_from_local(
            f'{dataDir}/{file}', doPerturb=True)

        if len(real_labels) > 1e3:
            continue
        info['Dataset'].append(file.split(".")[0])
        info['Class'].append(K)
        info['Samples'].append(len(real_labels))
        info['Features'].append(rdata.shape[1])

        print(f'run on {file} of size {len(real_labels)} and k={
              np.unique(real_labels).shape[0]}')
        theta = 1
        data = rdata.copy()
        data = (data - data.mean()) / (data.std())
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
        # print(prs.boundary_nodes)
        ET, roots = prs.get_final_tree_nx()
        print(f'the len of roots is {len(roots)}')
        ET.remove_edges_from(nx.selfloop_edges(ET))

        # if real_labels[roots[0]] == real_labels[roots[1]]:
        ET.add_edge(roots[1], roots[0])
        roots.pop()
        c = 100

        # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
        # distance, nearest_neighbors = neighbors.kneighbors(
        #     data, return_distance=True)
        # distance = distance[:, 1]

        data = data.values

        attrs = set_uncertainty(data, ET, roots)
        # 去除有向图中的环

        # print('type of ', type(nx.degree_centrality(ET)))
        # print(nx.degree_centrality(ET))

        assert len(data) == len(list(ET.nodes)) and len(data) == len(attrs)

        draw_graph(ET, data, roots, title=file.split(
            ".")[0], real_labels=real_labels)
        get_G(ET, data)
        ARI_record, roots = DSL(data, real_labels, title=file.split(".")[
                                0], skeleton=ET, representatives=roots)

        # draw_graph(ET, data, roots, title=file.split(
        #     ".")[0], real_labels=real_labels)

        # ARI_record = DSL(data, real_labels, title=file.split(".")[
        #     0], skeleton=None, representatives=None)

        fdir = 'result/p2'
        os.makedirs(fdir, exist_ok=True)
        ARI_record.to_csv(
            f'{fdir}/{file.split(".")[0]}_result.csv', index=False)

        tail = ARI_record.tail(1)
        ari21['dataset'].append(file.split(".")[0])
        ari21['ari'].append(tail['ari'].values[0])
        ari21['interaction'].append(tail['interaction'].values[0])
        print(f'run on {file} finished')

    df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    df.to_csv("our.csv", index=False)
    pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')


if __name__ == '__main__':
    arg = {'dataDir': 'data', 'resultDir': 'result'}
    os.makedirs('result/p2/graph', exist_ok=True)
    os.makedirs('result/DSL', exist_ok=True)
    run()
