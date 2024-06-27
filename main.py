import time
import os
import math
from random import shuffle, choice
from itertools import combinations
from collections import deque
import queue

from tqdm.rich import tqdm
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from networkx.drawing.nx_pydot import graphviz_layout
from active_semi_clustering import COPKMeans, MPCKMeans, PCKMeans
from icecream import ic

from DSL.DSLm import clustering, iteration_once
from myutil import DataLoader
from myutil.retry import retry
from PRSCSWAP import PRS


class Neiborhood():
    def __init__(self, label, data):
        self.N: list[list] = []
        self.label = label
        self.data = data

    def insert(self, dataid) -> tuple:
        count = 0
        pair = []
        q = -1
        # 判断dataid是否已经在N中
        for N in self.N:
            if dataid in N:
                return pair, count, q
        for i in range(len(self.N)):
            # choose a random point in N
            N = self.N[i]
            n = choice(N)
            count += 1
            if self.label[dataid] == self.label[n]:
                q = i
                pair = (dataid, n)
                N.append(dataid)
                flag = True
                break
        if not flag:
            self.N.append([dataid])
            q = len(self.N)-1
        return pair, count, q

    def get_N(self):
        return self.N


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

    # for i in range(len(tranversal_dict)):
    #     predict_vec.append(tranversal_dict[i])
    predict_vec = [tranversal_dict[i] for i in range(len(tranversal_dict))]
    return predict_vec


def get_predict_labels(Graph: nx.Graph):
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    predict_labels = np.zeros(len(Graph.nodes), dtype=int)

    for i, s in enumerate(S[1:], 1):
        predict_labels[list(s.nodes)] = i
    return predict_labels


def skeleton_process(Graph: nx.Graph):
    clusters = []
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels


def DSL(data, real_labels, title, skeleton=None, representatives=None, K=2, u_rank=None):
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
    loop = int(data.shape[0]*K)
    constraint_graph = nx.Graph()
    interaction = 0

    predict_labels = skeleton_process(skeleton)
    ARI = adjusted_rand_score(real_labels, predict_labels)
    # ARI = rand_score(real_labels, predict_labels)
    df = {"iter": [0], "interaction": [interaction], "ari": [ARI], "time": [0]}
    # u_rank = get_uncertainty_rank(skeleton)
    if u_rank is None:
        u_rank = get_rank_FUS(skeleton, representatives, data)
    # print('u_rank: ', u_rank)
    for i in tqdm(range(loop)):
        skeleton, representatives, constraint_graph, count, suspend = iteration_once(skeleton, representatives, data,
                                                                                     real_labels, constraint_graph, u_rank)
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


def get_rank_FUS(ET: nx.Graph, roots, data: np.array):
    R = [[int(ei), np.linalg.norm(data[ei]-data[ej])] for ei, ej in ET.edges]
    R = [n for n, d in sorted(R, key=lambda x: x[1], reverse=False)]
    return R


def get_rank_by_layer_(ET: nx.Graph, roots, data: np.array):
    # bfs ET
    # ET = ET.to_undirected()
    assert len(ET.nodes) == len(data), 'len of ET nodes and data is not equal'
    # ET = nx.reverse(ET)
    R = []
    bfs_layers = dict(enumerate(nx.bfs_layers(ET, roots)))
    flayer = bfs_layers.pop(0)
    print(f'flayer is {flayer}')
    for layers, nodes in bfs_layers.items():
        # print(f'layer {layers} has {len(nodes)} nodes')
        # R.extend(nodes)
        nn = []
        R.extend(nodes)
        # for n in nodes:
        #     dis = 0
        #     k = 0
        #     edges = list(ET.edges(n))
        #     if edges:
        #         for ei, ej in edges:
        #             dis += np.linalg.norm(data[ei]-data[ej])
        #             k += 1
        #         nn.append([n, dis/k])
        #     else:
        #         nn.append([n, 0])
        #     # nn.append([n, dis/k])
        # R.extend([n for n, d in sorted(nn, key=lambda x: x[1], reverse=True)])
    # R[0], R[-1] = R[-1], R[0]
    assert len(R) == len(ET.nodes) - \
        len(roots), f'{len(R) = } is not equal to {
            len(ET.nodes) - len(roots) =}, {len(roots) = } of ET nodes'
    print(f'{len(R)=}')
    # R.reverse()
    # R.remove(roots[0])
    return R


def set_uncertainty(data: np.array, ET: nx.Graph, roots: list):
    attrs: dict[int, float] = {}
    o = {"uncertainty": 0}
    # for ei, ej in ET.edges:
    #     # distance_between_n_and_neighbors
    #     dis = np.linalg.norm(data[ei]-data[ej])
    #     attrs[ei] = {"uncertainty": max(dis+ET.in_degree(ei), attrs.get(
    #         ei, o)["uncertainty"])+ET.in_degree(ei)}
    #     attrs[ej] = {"uncertainty": max(dis+ET.in_degree(ej), attrs.get(
    #         ej, o)["uncertainty"])+ET.in_degree(ej)}
    for n in ET.nodes:
        dis = 0
        Nnum = 0
        for nn in ET.neighbors(n):
            dis += np.linalg.norm(data[n]-data[nn])
            Nnum += 1
        attrs[n] = {"uncertainty": Nnum + dis/Nnum if Nnum != 0 else 1e-3}

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

    attrs[roots[0]]['uncertainty'] = 0
    nx.set_node_attributes(ET, attrs)
    return attrs


def get_auic(path: str):
    auic = {'dataset': [], 'auic': []}
    for file in os.listdir(path):
        auic['dataset'].append(file.split("_")[0])
        df = pd.read_csv(f'{path}/{file}')
        auic['auic'].append(df['ari'].sum()/(df.shape[0]*2))
    pd.DataFrame(auic).to_csv("auic.csv", index=False)


def draw_graph_w(ET: nx.Graph, data: np.array, roots: list, title: str, real_labels=None):
    # pos = nx.spring_layout(ET)
    pos = nx.nx_agraph.graphviz_layout(ET, prog="dot")
    # pos = graphviz_layout(ET, prog="twopi")
    # pos = nx.kamada_kawai_layout(ET)
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
    # nx.draw_networkx_nodes(ET, pos, nodelist=roots, node_color='r')
    plt.title(title)
    # plt.savefig(f'result/p2/graph/{title}.pdf')
    plt.show()

# plot scatter for data and label


def draw_scatter(data, label, title):
    # 使用pca降维
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    # 不同的label使用不同的颜色
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.title(title)
    plt.show()
    pass


def calculate_uncertainty(ET: nx.Graph, data: np.array, roots: list, N):
    raise NotImplementedError


def get_label_vec_from_G(ET: nx.Graph):
    y = np.zeros(len(ET.nodes), dtype=int)
    label = nx.get_node_attributes(ET, 'label')
    t = -1
    for k, v in sorted(label.items(), key=lambda x: x[0]):
        assert abs(t-k) == 1
        y[k] = v
        t = k
    return y


def label_all_children(ET: nx.Graph, roots: list):
    # 将ET变为无向图
    T = ET
    if not nx.is_directed(ET):
        T = ET.to_undirected()
    # 将根节点的标签传递给所有的子节点
    for r in roots:
        for n in nx.descendants(T, r):
            # set node attribute
            T.nodes[n]['label'] = T.nodes[r]['label']


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


def IC_activate(graph, seed_set, threshold):
    active = seed_set.copy()
    newly_active = seed_set.copy()
    t = 1
    while newly_active:
        next_active = []
        for node in newly_active:
            neighbors = graph.neighbors(node)
            t = t+1
            for neighbor in neighbors:
                ww = graph.edges[node, neighbor]['weight']
                if neighbor not in active and ww < threshold:
                    next_active.append(neighbor)
        newly_active = next_active
        active.update(newly_active)
        t += 1
    return active


def get_metric(y, T: nx.Graph, y_pred=None):
    if y_pred is None:
        _y_pred = get_predict_labels(T)

    ari = adjusted_rand_score(y, _y_pred)
    ri = rand_score(y, _y_pred)
    nmi = normalized_mutual_info_score(y, _y_pred)
    return ari, ri, nmi


def refine_graph(ET: nx.Graph, data: np.array, roots: list, title: str, real_labels=None):
    df = {"iter": [0], "interaction": [0], "ari": [0], "time": [0]}
    N = []
    T = ET.to_undirected()
    for n in T.nodes:
        T.add_node(n, label=0, activated=False)

    # label_all_children(T, roots)
    Y = get_label_vec_from_G(T)
    print(Y)

    ari, ri, nmi = get_metric(real_labels, T)
    print(f"the ARI of {title} is {ari}")
    print(f"the RI of {title} is {ri}")
    print(f"the NMI of {title} is {nmi}")

    return pd.DataFrame(df), roots


def label_propagation(data, real_labels, title):
    start = time.time()
    label_prop_model = LabelPropagation()
    label_prop_model.fit(data, real_labels)
    predict_labels = label_prop_model.transduction_
    ARI = adjusted_rand_score(real_labels, predict_labels)
    duration = time.time() - start
    print(f"the ARI of {title} is {ARI} and time is {duration}")
    return ARI


def label_propagation(data, real_labels, title):
    start = time.time()
    label_prop_model = LabelPropagation()
    label_prop_model.fit(data, real_labels)
    predict_labels = label_prop_model.transduction_
    ARI = adjusted_rand_score(real_labels, predict_labels)
    duration = time.time() - start
    print(f"the ARI of {title} is {ARI} and time is {duration}")
    return ARI


def get_uncertainty_rank(ET: nx.Graph):
    attrs = nx.get_node_attributes(ET, 'uncertainty')
    attrs = sorted(attrs.items(), key=lambda x: x[1])
    return attrs


def draw_graph(ARIpath: str, pathToSavePic: str = 'result/pic', file='', remove=False, matric='ARI'):
    print('save result to pic')
    os.makedirs(pathToSavePic, exist_ok=True)
    # for file in os.listdir(ARIpath):
    # 判断是否是csv文件
    # if not file.endswith('.csv'):
    #     continue
    ARI = pd.read_csv(f'{ARIpath}/{file}')
    fig, ax = plt.subplots()
    ax.plot(ARI['interaction'], ARI['ari'], label='ARI')
    ax.set_xlabel('Quries')
    ax.set_ylabel(matric)
    ax.set_title(f'{file.split("_")[0]}')
    ax.legend()
    path = f'{pathToSavePic}/{file.split("_")[0]}.png'
    # print(f'save  to {path}')
    plt.savefig(path, dpi=600)
    # plt.show()


def draw_graph_e(ARIpath: str, pathToSavePic: str = 'result/pic', file='', remove=False, matric='ARI'):
    print('save result to pic')
    algpath = ['middle', 'ADP', 'ADPE', 'COBRAS']
    # path = [f'{ARIpath}/{i}' for i in algpath]
    os.makedirs(pathToSavePic, exist_ok=True)

    fig, ax = plt.subplots()
    for i in algpath:
        fpath = f'{ARIpath}/{i}/{file}'
        if not os.path.exists(fpath):
            continue
        ARI = pd.read_csv(fpath)
        ax.plot(ARI['interaction'], ARI['ari'], label=i)

    ax.set_xlabel('Quries')
    ax.set_ylabel(matric)
    ax.set_title(f'{file.split(".")[0]}')
    ax.legend()

    path = f'{pathToSavePic}/{file.split(".")[0]}.png'
    # print(f'save  to {path}')
    plt.savefig(path, dpi=600)


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
        # get_G(ET, data)
        ARI_record, roots = DSL(data, real_labels, title=file.split(".")[
            0], skeleton=ET, representatives=roots)

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


def run2(dataDir: str = "data", uncertainty=False, fdir: str = 'result/p2/small') -> None:
    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }
    for file in os.listdir(dataDir):
        rdata, real_labels, K = DataLoader.get_data_from_local(
            f'{dataDir}/{file}', doPerturb=True)

        # if len(real_labels) < 1e3:
        #     continue
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
        # num_thread = 1
        # print(label)
        # data.drop_duplicates(inplace=True)

        prs = PRS(data)
        start = time.time()
        # threshold_clusters = K
        # threshold_clusters >= 2
        threshold_clusters = 2
        prs.get_clusters(num_thread, threshold_clusters)
        # print(prs.boundary_nodes)
        ET, roots = prs.get_final_tree_nx()
        print(f'the len of roots is {len(roots)}')
        for r in roots:
            ET.remove_edge(r, r)
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
        # draw_graph(ET, data, roots, title=file.split(
        #     ".")[0], real_labels=real_labels)
        # get_G(ET, data)
        # ARI_record, roots = refine_graph(ET, data, roots, title=file.split(".")
        #                                  [0], real_labels=real_labels)

        # draw_graph_w(ET, data, roots, title=file.split(
        #     ".")[0], real_labels=real_labels)

        # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
        #     0], skeleton=None, representatives=None)

        ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
            0], skeleton=ET, representatives=[roots[0]], K=K)

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


def runp(dataDir, file, outdir: str = 'result/p2/small') -> None:
    assert os.path.exists(dataDir)

    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }

    rdata, real_labels, K = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True)

    # if len(real_labels) < 1e3:
    #     continue
    # info['Dataset'].append(file.split(".")[0])
    # info['Class'].append(K)
    # info['Samples'].append(len(real_labels))
    # info['Features'].append(rdata.shape[1])

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
    # num_thread = 1
    # print(label)
    # data.drop_duplicates(inplace=True)

    prs = PRS(data)
    start = time.time()
    threshold_clusters = K
    # threshold_clusters >= 2
    threshold_clusters = 2
    prs.get_clusters(num_thread, threshold_clusters)
    # print(prs.boundary_nodes)
    ET, roots = prs.get_final_tree_nx()
    print(f'the len of roots is {len(roots)}')
    for r in roots:
        ET.remove_edge(r, r)
    # if real_labels[roots[0]] == real_labels[roots[1]]:
    ET.add_edge(roots[1], roots[0])
    roots.pop()
    c = 100

    # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
    # distance, nearest_neighbors = neighbors.kneighbors(
    #     data, return_distance=True)
    # distance = distance[:, 1]
    data = data.values
    # attrs = set_uncertainty(data, ET, roots)

    # 去除有向图中的环
    # print('type of ', type(nx.degree_centrality(ET)))
    # print(nx.degree_centrality(ET))
    # assert len(data) == len(list(ET.nodes)) and len(data) == len(attrs)
    assert len(data) == len(list(ET.nodes))
    # draw_graph(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)
    # get_G(ET, data)
    # ARI_record, roots = refine_graph(ET, data, roots, title=file.split(".")
    #                                  [0], real_labels=real_labels)
    # draw_scatter(data, real_labels, title=file.split(".")[0])
    # draw_graph_w(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=None, representatives=None)

    ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
        0], skeleton=ET, representatives=[roots[0]], K=K)

    os.makedirs(outdir, exist_ok=True)
    ARI_record.to_csv(
        f'{outdir}/{file.split(".")[0]}_result.csv', index=False)

    # tail = ARI_record.tail(1)
    # ari21['dataset'].append(file.split(".")[0])
    # ari21['ari'].append(tail['ari'].values[0])
    # ari21['interaction'].append(tail['interaction'].values[0])
    # print(f'run on {file} finished')

    # df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    # df.to_csv("our.csv", index=False)
    # pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')


def runp2(dataDir, file, outdir: str = 'result/p2/small') -> None:
    assert os.path.exists(dataDir)

    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }

    rdata, real_labels, K = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True)

    # if len(real_labels) < 1e3:
    #     continue
    # info['Dataset'].append(file.split(".")[0])
    # info['Class'].append(K)
    # info['Samples'].append(len(real_labels))
    # info['Features'].append(rdata.shape[1])

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
    # num_thread = 1
    # print(label)
    # data.drop_duplicates(inplace=True)

    prs = PRS(data)
    start = time.time()
    threshold_clusters = K
    # threshold_clusters >= 2
    threshold_clusters = 2
    prs.get_clusters(num_thread, threshold_clusters)
    # print(prs.boundary_nodes)
    ET, roots = prs.get_final_tree_nx()
    print(f'the len of roots is {len(roots)}')
    for r in roots:
        ET.remove_edge(r, r)
    # if real_labels[roots[0]] == real_labels[roots[1]]:
    meg: dict[int, list] = {i: [] for i in range(K)}
    for r in roots:
        meg[real_labels[r]].append(r)
    for k, v in meg.items():
        if len(v) > 1:
            for i in range(1, len(v)):
                ET.add_edge(v[i], v[0])
                roots.remove(v[i])

    # ET.add_edge(roots[1], roots[0])
    # roots.pop()
    c = 100

    # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
    # distance, nearest_neighbors = neighbors.kneighbors(
    #     data, return_distance=True)
    # distance = distance[:, 1]
    data = data.values
    # attrs = set_uncertainty(data, ET, roots)

    # 去除有向图中的环
    # print('type of ', type(nx.degree_centrality(ET)))
    # print(nx.degree_centrality(ET))
    # assert len(data) == len(list(ET.nodes)) and len(data) == len(attrs)
    assert len(data) == len(list(ET.nodes))
    # draw_graph(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)
    # get_G(ET, data)
    # ARI_record, roots = refine_graph(ET, data, roots, title=file.split(".")
    #                                  [0], real_labels=real_labels)
    # draw_scatter(data, real_labels, title=file.split(".")[0])
    # draw_graph_w(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=None, representatives=None)

    ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
        0], skeleton=ET, representatives=roots, K=K)

    os.makedirs(outdir, exist_ok=True)
    ARI_record.to_csv(
        f'{outdir}/{file.split(".")[0]}_result.csv', index=False)

    # tail = ARI_record.tail(1)
    # ari21['dataset'].append(file.split(".")[0])
    # ari21['ari'].append(tail['ari'].values[0])
    # ari21['interaction'].append(tail['interaction'].values[0])
    # print(f'run on {file} finished')

    # df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    # df.to_csv("our.csv", index=False)
    # pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')


def get_node_attrs_by_distance_from_parent(G: nx.Graph, data: np.array):
    attrs = np.zeros(len(G.nodes))
    for edge in G.edges():
        i, j = edge
        attrs[i] = np.linalg.norm(data[i]-data[j])
    return attrs


def get_node_attrs_by_distance_from_root(G: nx.Graph, data: np.array, roots: list):
    # attrs: dict[int, dict] = {}
    attrs = np.zeros(len(G.nodes))
    # attrs[roots[0]] = {'distance': 0}
    # edges = set(G.edges(roots[0]))
    successor = []
    # bfs ET
    queue = deque(roots)
    alpha = 1
    max_dis = 0
    while queue:
        node = queue.popleft()
        successor = list(G.neighbors(node))
        queue.extend(successor)
        for n in successor:
            # if n in attrs:
            #     continue
            # attrs[n] = {'distance': np.linalg.norm(
            #     data[n]-data[node]) + attrs[node]['distance']}
            dist = np.linalg.norm(data[n]-data[node])
            attrs[n] = alpha*dist + attrs[node]
    # nx.set_node_attributes(G, attrs)
    # max_dis = np.max(attrs)
    # attrs = attrs/max_dis
    # for n in range(len(attrs)):
    #     attrs[n] += G.out_degree(n)
    return attrs


def get_rank_by_layer(G: nx.Graph, roots: list, bfs_layers: dict = None, data: np.array = None, reverse=False):
    if bfs_layers is None:
        bfs_layers = dict(enumerate(nx.bfs_layers(G, roots)))
        bfs_layers.pop(0)
    flayer = []
    for i in range(1, 4):
        flayer.extend(bfs_layers[i])
    if reverse:
        flayer.reverse()
    return flayer


def get_rank_by_leaves(G: nx.Graph, roots: list, bfs_layers: dict = None, data: np.array = None, reverse=False):
    if bfs_layers is None:
        bfs_layers = dict(enumerate(nx.bfs_layers(G, roots)))
        bfs_layers.pop(0)
    leaves = []
    leaveInfo = np.zeros(len(G.nodes))
    for l, nodes in bfs_layers.items():
        leaveInfo[nodes] = l
        for n in nodes:
            if G.out_degree(n) == 0:
                leaves.append(n)
    if reverse:
        leaves.reverse()
    return leaves, leaveInfo


def get_leves_rank_que_by_distance(G: nx.Graph, roots: list, att, data: np.array = None, reverse=True):
    bfs_layers = dict(enumerate(nx.bfs_layers(G, roots)))
    bfs_layers.pop(0)
    # sort by distance
    for l, nodes in bfs_layers.items():
        _nodes = [[n, att[n]] for n in nodes]
        _nodes = [n for n, d in sorted(
            _nodes, key=lambda x: x[1], reverse=reverse)]
        bfs_layers[l] = _nodes
    return bfs_layers


def refine_by_h_1(ET: nx.Graph, roots: list, data, real_labels):
    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))
    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}
    c = 0
    visited = set()

    windowSize = 4
    T: nx.Graph = nx.reverse(ET)
    R: list = roots.copy()

    print(f'start refine_by_h ')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    att = set_node_attrs_by_distance_from_root(T, data, roots)
    print(f'{len(att)=}, {len(ET.nodes)=}')
    # if not nx.is_directed(ET):
    #     T = ET.to_undirected()
    bfs_layers = get_leves_rank_que_by_distance(T, roots, att)

    # flayer = get_rank_by_layer(T, roots, bfs_layers)
    flayer = []
    for k, v in bfs_layers.items():
        if k <= 3:
            flayer.extend(v)

    # leaves, leaveInfo = get_rank_by_leaves(T, roots, bfs_layers)

    # flayer_r = [[n, att[n]] for n in flayer]

    # # DEBGU
    # # tf = sorted(
    # #     flayer_r, key=lambda x: x[1], reverse=True)
    # leaves_r = [[n, att[n]] for n in leaves]

    # flayer = [n for n, d in sorted(
    #     flayer_r, key=lambda x: x[1], reverse=True)]

    # # DEBGU
    # # sf = sorted(
    # #     leaves_r, key=lambda x: x[1], reverse=True)
    # leaves = [n for n, d in sorted(
    #     leaves_r, key=lambda x: x[1], reverse=True)]

    FUS = [[n, att[n]] for n in range(len(att))]
    FUS = [n for n, d in sorted(FUS, key=lambda x: x[1], reverse=False)]

    def get_genrator(_Iterabal):
        for n in _Iterabal:
            if n in visited:
                continue
            yield n
    FUS_g = get_genrator(FUS)
    flayer_g = get_genrator(flayer)

    ls = 0
    fs = 0
    i = 0
    levelM = 0
    stopMarker = -1

    for _ in tqdm(range(10)):
        f = []
        l = []
        # _T = nx.reverse(ET)

        # att = set_node_attrs_by_distance_from_root(_T, data, roots)
        # bfs_layers = get_leves_rank_que_by_distance(_T, roots, att)

        # while len(f) < windowSize:
        #     if bfs_layers[levelM+1]:
        #         _n = bfs_layers[levelM+1].pop()
        #         if _n not in visited:
        #             f.append(_n)
        #             levelM += 1
        #     else:
        #         levelM += 1
        #     levelM %= len(bfs_layers)-1

        # while fs < len(flayer) and len(f) < windowSize and flayer[fs] not in visited:
        #     f.append(flayer[fs])
        #     fs += 1

        # bfs_layers = dict(enumerate(nx.bfs_layers(T, roots)))
        # bfs_layers.pop(0)
        # leaves, leaveInfo = get_rank_by_leaves(T, roots, bfs_layers)
        # leaves_r = [[n, att[n]+leaveInfo[n]*100] for n in leaves]
        # leaves = [n for n, d in sorted(
        #     leaves_r, key=lambda x: x[1], reverse=True)]

        # while ls < len(leaves) and leaves[ls] not in visited and len(l) < t:
        #     l.append(leaves[ls])
        #     ls += 1
        # f = []
        upb = math.floor(2*windowSize)

        fl = next(flayer_g, stopMarker)
        while fl != stopMarker and len(f) < windowSize:
            f.append(fl)
            fl = next(flayer_g, stopMarker)

        # while ls < len(FUS) and (len(f) + len(l)) < upb and FUS[ls] not in visited:
        #     l.append(FUS[ls])
        #     ls += 1

        fus = next(FUS_g, stopMarker)
        while fus != stopMarker and (len(f) + len(l)) < upb:
            l.append(fus)
            fus = next(FUS_g, stopMarker)

        assert len(f)+len(l) <= upb, f'{len(f) =}, {len(l) = }, {
            len(f)+len(l) =}, {upb = }'
        Rank = f+l

        for n in Rank:
            if n in visited:
                continue
            s = time.perf_counter()
            count = 0

            # from buttom to top
            e = list(ET.edges(n))
            while e:
                a, b = e[0]
                count += 1
                visited.add(a)
                if real_labels[a] != real_labels[b]:
                    count += refine_dislike(ET, data,
                                            real_labels, R, e, count, att)
                if b in visited:
                    e = []
                else:
                    e = list(ET.edges(b))
            # from top to buttom

            # if e:
            #     count += refine_dislike(ET, data,
            #                             real_labels, R, e, count, att)
            t = time.perf_counter()
            ARI = df['ari'][-1]
            if True and i % 3 == 0:
                predict_labels = get_predict_labels(ET)
                assert len(real_labels) == len(
                    predict_labels), f'len of real_labels is not equal to len of predict_labels'
                ARI = adjusted_rand_score(real_labels, predict_labels)

            c = c + count
            i += 1

            df['iter'].append(i)
            df['interaction'].append(c)
            df['ari'].append(ARI)
            df['time'].append(t-s)

            if ARI == 1:
                print(f'find the best result at {i} iteration')
                return pd.DataFrame(df), R

    return pd.DataFrame(df), R


def refine_by_h_2(ET: nx.Graph, roots: list, data, real_labels, subroots: list):
    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))
    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}
    c = 0
    visited = set()
    arr = np.zeros(len(ET.nodes), dtype='int')
    bool_arr = np.array(arr, dtype='bool')

    windowSize = 10
    T: nx.Graph = nx.reverse(ET)
    R: list = roots.copy()

    print(f'start refine_by_h ')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    att = get_node_attrs_by_distance_from_root(ET, data, roots)
    print(f'{len(att)=}, {len(ET.nodes)=}')
    # if not nx.is_directed(ET):
    #     T = ET.to_undirected()
    bfs_layers = get_leves_rank_que_by_distance(T, roots, att, reverse=True)

    # flayer = get_rank_by_layer(T, roots, bfs_layers)
    subroots = [[n, att[n]] for n in subroots]
    subroots = [n for n, _ in sorted(
        subroots, key=lambda x: x[1], reverse=True)]
    flayer = []
    for k, v in bfs_layers.items():
        if k <= 2:
            flayer.extend(v)
    flayer += subroots
    # flayer.extend(subroots)

    # leaves, leaveInfo = get_rank_by_leaves(T, roots, bfs_layers)

    # flayer_r = [[n, att[n]] for n in flayer]

    FUS = [[n, att[n]] for n in range(len(att))]
    FUS = [n for n, _ in sorted(FUS,
                                key=lambda x: x[1], reverse=True)]
    # FUS = subroots.copy()

    def get_genrator(_Iterabal):
        for n in _Iterabal:
            if n in visited:
                continue
            yield n

    FUS_g = get_genrator(FUS)
    flayer_g = get_genrator(flayer)
    subroots_g = get_genrator(subroots)

    ls = 0
    fs = 0
    i = 0
    levelM = 0
    stopMarker = -1

    for _ in tqdm(range(20)):
        Rank = get_candidate_nodes(windowSize, FUS_g, flayer_g, stopMarker)
        # Rank = flayer+FUS
        # Rank = list(ET.nodes)
        if not Rank:
            print(f'all candidate nodes have been visited')
            return pd.DataFrame(df), R
        for n in Rank:
            if n in visited:
                continue
            s = time.perf_counter()
            count = 0

            # from buttom to top refine the tree
            e = list(ET.edges(n))
            while e:
                a, b = e[0]
                count += 1
                visited.add(a)
                if real_labels[a] != real_labels[b]:
                    count += refine_dislike(ET, data,
                                            real_labels, R, e, att)
                # if b in visited:
                #     e = []
                # else:
                #     e = list(ET.edges(b))
                e = []
            # TODO from top to buttom refine the tree

            t = time.perf_counter()
            # ARI = df['ari'][-1]
            # if True or i % 2 == 0:
            predict_labels = get_predict_labels(ET)

            ARI = adjusted_rand_score(real_labels, predict_labels)
            # RI = rand_score(real_labels, predict_labels)

            c += count
            i += 1

            df['iter'].append(i)
            df['interaction'].append(c)
            # df['ari'].append(RI)
            df['ari'].append(ARI)
            df['time'].append(t-s)

            if ARI == 1:
                print(f'find the best result at {i} iteration')
                return pd.DataFrame(df), R

    return pd.DataFrame(df), R


def get_candidate_nodes(windowSize, FUS_g, flayer_g, stopMarker):
    f = []
    l = []
    # _T = nx.reverse(ET)

    # att = set_node_attrs_by_distance_from_root(_T, data, roots)
    # bfs_layers = get_leves_rank_que_by_distance(_T, roots, att)

    # while len(f) < windowSize:
    #     if bfs_layers[levelM+1]:
    #         _n = bfs_layers[levelM+1].pop()
    #         if _n not in visited:
    #             f.append(_n)
    #             levelM += 1
    #     else:
    #         levelM += 1
    #     levelM %= len(bfs_layers)-1

    # while fs < len(flayer) and len(f) < windowSize and flayer[fs] not in visited:
    #     f.append(flayer[fs])
    #     fs += 1

    # bfs_layers = dict(enumerate(nx.bfs_layers(T, roots)))
    # bfs_layers.pop(0)
    # leaves, leaveInfo = get_rank_by_leaves(T, roots, bfs_layers)
    # leaves_r = [[n, att[n]+leaveInfo[n]*100] for n in leaves]
    # leaves = [n for n, d in sorted(
    #     leaves_r, key=lambda x: x[1], reverse=True)]

    # while ls < len(leaves) and leaves[ls] not in visited and len(l) < t:
    #     l.append(leaves[ls])
    #     ls += 1
    # f = []
    upb = math.floor(1*windowSize)

    fl = next(flayer_g, stopMarker)
    while fl != stopMarker and len(f) < windowSize:
        f.append(fl)
        fl = next(flayer_g, stopMarker)

        # while ls < len(FUS) and (len(f) + len(l)) < upb and FUS[ls] not in visited:
        #     l.append(FUS[ls])
        #     ls += 1

    fus = next(FUS_g, stopMarker)
    while fus != stopMarker and (len(f) + len(l)) < upb:
        l.append(fus)
        fus = next(FUS_g, stopMarker)

    assert len(f)+len(l) <= upb, f'{len(f) =}, {len(l) = }, {
        len(f)+len(l) =}, {upb = }'

    Rank = f+l
    return Rank


class Node(object):
    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __lt__(self, other):
        return self.score > other.score


def get_Rank_queue(G: nx.Graph, data, source, N=1):
    # init
    RankQ = queue.PriorityQueue()

    for n in source:
        e = list(G.neighbors(n))
        if e:
            RankQ.put(Node(n, np.linalg.norm(data[n]-data[e[0]])))

    # update
    while not RankQ.empty():
        node = RankQ.get()
        nbors = list(G.neighbors(node))
        for nb in nbors:
            RankQ.put(Node(nb, np.linalg.norm(data[nb]-data[node])))


def get_N_neighbors_distance(G: nx.Graph, data, nodes, N=1):
    Rank = []
    vist = set(nodes)
    for n in nodes:
        dist = 0
        k = 0
        # no = list(nx.bfs_tree(G, source=nodes, depth_limit=N).nodes())
        # bf_layer = dict(enumerate(nx.bfs_layers(G, n, depth_limit=N)))

        for node in nx.bfs_tree(G, source=n, depth_limit=N).nodes():
            if node in vist:
                continue
            k += 1
            dist += np.linalg.norm(data[n]-data[node])
        Rank.append([n, dist/max(k, 1)])
    Rank = [n for n, _ in sorted(Rank, key=lambda x: x[1], reverse=True)]
    return Rank


def get_N_neighbors_distance(G: nx.Graph, data, nodes, N=1):
    Rank = []
    vist = set(nodes)
    for n in nodes:
        dist = 0
        k = 0
        # no = list(nx.bfs_tree(G, source=nodes, depth_limit=N).nodes())
        # bf_layer = dict(enumerate(nx.bfs_layers(G, n, depth_limit=N)))

        for node in nx.bfs_tree(G, source=n, depth_limit=N).nodes():
            if node in vist:
                continue
            k += 1
            dist += np.linalg.norm(data[n]-data[node])
        Rank.append([n, dist/max(k, 1)])
    Rank = [n for n, _ in sorted(Rank, key=lambda x: x[1], reverse=True)]
    return Rank


def refine_by_h_3(ET: nx.Graph, roots: list, data, real_labels, subroots: list):
    '''
    1. 从根节点开始，按照层次遍历的顺序，依次判断每个节点
    '''
    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))
    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}
    c = 0
    visited = set()
    visited.add(roots[0])
    # arr = np.zeros(len(ET.nodes), dtype='int')
    # bool_arr = np.array(arr, dtype='bool')

    windowSize = 10
    T: nx.Graph = nx.reverse(ET)
    R: list = roots.copy()

    print(f'start refine_by_h ')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    att = get_node_attrs_by_distance_from_root(ET, data, roots)
    print(f'{len(att)=}, {len(ET.nodes)=}')
    # if not nx.is_directed(ET):
    #     T = ET.to_undirected()
    bfs_layers = get_leves_rank_que_by_distance(T, roots, att, reverse=True)

    # flayer = get_rank_by_layer(T, roots, bfs_layers)
    subroots = [[n, att[n]] for n in subroots]
    subroots = [n for n, _ in sorted(
        subroots, key=lambda x: x[1], reverse=True)]
    flayer = []
    for k, v in bfs_layers.items():
        if k <= 1:
            flayer.extend(v)
    # flayer = subroots.copy()
    # flayer.extend(subroots)

    # leaves, leaveInfo = get_rank_by_leaves(T, roots, bfs_layers)

    # flayer_r = [[n, att[n]] for n in flayer]

    # FUS = [[n, att[n]] for n in range(len(att))]
    # FUS = [n for n, _ in sorted(FUS,
    #                             key=lambda x: x[1], reverse=True)]

    # FUS = subroots.copy()

    def get_genrator(_Iterabal):
        for n in _Iterabal:
            if n in visited:
                continue
            yield n

    # FUS_g = get_genrator(FUS)
    # flayer_g = get_genrator(flayer)
    # subroots_g = get_genrator(subroots)

    ls = 0
    fs = 0
    i = 0
    levelM = 0
    stopMarker = -1

    # for _ in tqdm(range(20)):
    # Rank = get_candidate_nodes(windowSize, FUS_g, flayer_g, stopMarker)
    Rank = flayer
    Rank += subroots.copy()
    # Rank = get_N_neighbors_distance(T, data, Rank, N=1)
    # Rank = flayer+FUS
    # Rank = list(ET.nodes)

    # init
    RankQ = queue.PriorityQueue()

    for n in Rank:
        e = list(ET.neighbors(n))
        # assert len(e) == 1, f'{n=} has {len(e)} neighbors'
        if e:
            RankQ.put(Node(n, np.linalg.norm(data[n]-data[e[0]])))

    if not Rank:
        print(f'all candidate nodes have been visited')
        return pd.DataFrame(df), R
    q = 1000

    while not RankQ.empty() and c < q:
        nn = RankQ.get(block=False)
        n = nn.id
        if n in visited:
            continue
        # _ET: nx.Graph = nx.reverse(ET)
        nbors = list(T.neighbors(n))
        for nb in nbors:
            RankQ.put(Node(nb, np.linalg.norm(data[nb]-data[n])))

        s = time.perf_counter()
        count = 0

        # from buttom to top refine the tree
        e = list(ET.edges(n))
        while e:
            a, b = e[0]
            visited.add(a)

            count += 1
            if real_labels[a] != real_labels[b]:
                count += refine_dislike(ET, data,
                                        real_labels, R, e, att)
            # if b in visited:
            #     e = []
            # else:
            #     e = list(ET.edges(b))
            e = []
        # TODO from top to buttom refine the tree

        # ARI = df['ari'][-1]
        # if True or i % 2 == 0:
        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
        t = time.perf_counter()
        # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
        # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            return pd.DataFrame(df), R

    return pd.DataFrame(df), R


def refine_dislike(ET: nx.Graph, data, real_labels, R: list, e, att):
    e = e[0]
    ET.remove_edge(*e)
    n, _ = e
    Find = False
    rl = [[r, np.linalg.norm(data[n]-data[r])] for r in R]
    rl = [r for r, _ in sorted(rl, key=lambda x: x[1])]
    count = 0
    for r in rl:
        count += 1
        if real_labels[n] == real_labels[r]:
            Find = True
            ET.add_edge(n, r)
            # if att[n] > att[r]:
            #     ET.add_edge(n, r)
            # else:
            #     ET.add_edge(r, n)
            #     R.remove(r)
            #     R.append(n)
            break
    if not Find:
        R.append(n)
    return count


def refine_by_h(ET: nx.Graph, roots: list, data, real_labels):
    # 将ET变为无向图
    ari = adjusted_rand_score(real_labels, get_predict_labels(ET))
    df = {"iter": [0], "interaction": [0], "ari": [ari], "time": [0]}
    c = 0
    T: nx.Graph = nx.reverse(ET)
    print(f'start refine_by_h ')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    # if not nx.is_directed(ET):
    #     T = ET.to_undirected()
    R: list = roots.copy()

    bfs_layers = dict(enumerate(nx.bfs_layers(T, roots)))
    bfs_layers.pop(0)
    # flayer = bfs_layers[0] + bfs_layers[1]
    flayer = []
    for l, nodes in bfs_layers.items():
        flayer.extend(nodes)
    Rank = flayer
    # leaves = []
    # for l, nodes in bfs_layers.items():
    #     # print(f'layer {l} has {len(nodes)} nodes')
    #     for n in nodes:
    #         if T.out_degree(n) == 0:
    #             leaves.append(n)
    # print(f'leaves is {leaves}')
    # leaves.reverse()
    # Rank = flayer + leaves
    # Rank = get_rank_by_layer(T, roots, data)
    i = 0
    for n in tqdm(Rank):
        s = time.perf_counter()
        e = list(ET.edges(n))
        if not e:
            # raise ValueError(f'{n=} has no edge')
            continue
        e = e[0]
        count = 1
        if real_labels[e[0]] != real_labels[e[1]]:
            ET.remove_edge(*e)
            notFind = True
            rl = [[r, np.linalg.norm(data[n]-data[r])] for r in R]
            rl = [r for r, d in sorted(rl, key=lambda x: x[1])]
            for r in rl:
                count += 1
                if real_labels[n] == real_labels[r]:
                    notFind = False
                    ET.add_edge(n, r)
                    break
            if notFind:
                R.append(n)
        t = time.perf_counter()
        predict_labels = get_predict_labels(ET)

        assert len(real_labels) == len(
            predict_labels), f'len of real_labels is not equal to len of predict_labels'

        # ARI = adjusted_rand_score(real_labels, predict_labels)
        ARI = rand_score(real_labels, predict_labels)
        c = c + count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    return pd.DataFrame(df), R


def merge_roots(ET: nx.Graph, roots: list):
    if len(roots) == 1:
        return roots

    if ET.in_degree(roots[0]) > ET.in_degree(roots[1]):
        ET.add_edge(roots[1], roots[0])
        roots.pop(1)
        # roots = [roots[0]]
    else:
        ET.add_edge(roots[0], roots[1])
        roots.pop(0)
        # roots = [roots[1]]
    return merge_roots(ET, roots)


@retry(retries=6, delay=0)
def run_PRSC(data, real_labels, K, num_thread):
    start = time.time()
    prs = PRS(data)
    threshold_clusters = K
    threshold_clusters = 2
    prs.get_clusters(num_thread, threshold_clusters)
    # print(prs.boundary_nodes)
    ET, roots = prs.get_final_tree_nx()
    subroots = prs.get_subroots()
    print(f'the len of roots is {len(roots)}')
    print(f'{roots=}')
    roots = merge_roots(ET, roots)
    print(f'roots affter merge: {roots=}')
    assert len(roots) == 1, f'{roots=}'
    assert len([ET.subgraph(c) for c in nx.weakly_connected_components(ET)]) == 1, f'{
        len([ET.subgraph(c) for c in nx.weakly_connected_components(ET)]) =}'
    nods = list(ET.nodes)
    assert len(real_labels) == len(nods), f'{len(real_labels)=}, {len(nods)=}'
    # 检查nodes里面的数是否连续
    nl = np.zeros(len(real_labels))
    nl[nods] = 1
    p1 = get_predict_labels(ET)
    p2: np.ndarray = np.zeros(len(real_labels), dtype=int)
    ari = adjusted_rand_score(real_labels, p1)
    ari2 = adjusted_rand_score(real_labels, p2)
    assert ari == ari2, f'{ari = }, {ari2 = }, {len(nods) = }, {len(real_labels) = }, {
        np.all(p1 == p2)}, {len(roots) =}, len connected_components: {len([ET.subgraph(c) for c in nx.weakly_connected_components(ET)])}'
    # 判断两个数组是否相等
    assert np.all(p1 == p2), f'p1 is not equal to p2'
    # 查看ET有几个联通分量
    print(f'{nx.number_connected_components(ET.to_undirected())=}')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    return ET, roots, subroots


def runp_h(dataDir, file: str, outdir: str = 'result/p2/small') -> None:
    assert os.path.exists(dataDir)

    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }

    rdata, real_labels, K = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True)

    # if len(real_labels) < 1e3:
    #     return
    # info['Dataset'].append(file.split(".")[0])
    # info['Class'].append(K)
    # info['Samples'].append(len(real_labels))
    # info['Features'].append(rdata.shape[1])

    print(f'run on {file} of size {len(real_labels)} and k={
        np.unique(real_labels).shape[0]}')
    theta = 1
    data = rdata.copy()
    data = (data - data.mean()) / (data.std())
    # data = MinMaxScaler().fit_transform(data)
    # data = pd.DataFrame(data)

    num_thread = math.ceil(math.ceil(len(real_labels) / (theta * 100)))

    ET, roots, subroots = run_PRSC(data, real_labels, K, num_thread)

    # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
    # distance, nearest_neighbors = neighbors.kneighbors(
    #     data, return_distance=True)
    # distance = distance[:, 1]
    data = data.values
    # attrs = set_uncertainty(data, ET, roots)
    assert len(data) == len(list(ET.nodes))
    # ARI_record, roots = refine_graph(ET, data, roots, title=file.split(".")
    #                                  [0], real_labels=real_labels)
    # draw_scatter(data, real_labels, title=file.split(".")[0])

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=None, representatives=None)

    # draw_graph_w(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, R = refine_by_h(ET, roots, data, real_labels)
    # print(f'{R=}, {len(R) == K}')

    # ARI_record, R = refine_by_h_1(ET, roots, data, real_labels)

    # ARI_record, R = refine_by_h_2(ET, roots, data, real_labels, subroots)
    ARI_record, R = refine_by_h_3(ET, roots, data, real_labels, subroots)
    print(f'{R=}, {len(R) == K}')

    # draw_graph_w(ET, data, R, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=ET, representatives=roots, K=K)

    # os.makedirs(outdir, exist_ok=True)
    ARI_record.to_csv(
        f'{outdir}/{file.split(".")[0]}.csv', index=False)

    # tail = ARI_record.tail(1)
    # ari21['dataset'].append(file.split(".")[0])
    # ari21['ari'].append(tail['ari'].values[0])
    # ari21['interaction'].append(tail['interaction'].values[0])
    # print(f'run on {file} finished')

    # df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    # df.to_csv("our.csv", index=False)
    # pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')


def runPRSC(data, num_thread, threshold_clusters):
    prs = PRS(data)
    prs.get_clusters(num_thread, threshold_clusters)
    ET, roots = prs.get_final_tree_nx()
    return ET, roots


def getJaccardRank(result: list):
    R = []
    for node in result[0][0].nodes:
        node = int(node)
        ne = []
        for r in result:
            t: nx.Graph = r[0].to_undirected()
            ne.append(set(t.neighbors(node)))
        a: set = ne[0]
        for nn in range(1, len(ne)):
            a = a.intersection(ne[nn])
        b = set()
        for n in a:
            b.update(n)
        if len(b) == 0:
            R.append([node, 0])
        else:
            R.append([node, len(a)/len(b)])
    return [r[0] for r in sorted(R, key=lambda x: x[1], reverse=True)]


def runpJaccard(dataDir, file, outdir: str = 'result/p2/small') -> None:
    assert os.path.exists(dataDir)

    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }

    rdata, real_labels, K = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True)

    # if len(real_labels) < 1e3:
    #     continue
    # info['Dataset'].append(file.split(".")[0])
    # info['Class'].append(K)
    # info['Samples'].append(len(real_labels))
    # info['Features'].append(rdata.shape[1])

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
    # num_thread = 1
    # print(label)
    # data.drop_duplicates(inplace=True)

    prs = PRS(data)
    start = time.time()
    # threshold_clusters = K
    # threshold_clusters >= 2
    threshold_clusters = 2
    result = Parallel(n_jobs=8)(delayed(runPRSC)(data, num_thread, threshold_clusters)
                                for _ in range(10))
    # ET, roots = runPRSC(data, num_thread, threshold_clusters)
    # print(f'the result is {result}')
    ET, roots = result[0]
    uranck = getJaccardRank(result)
    uranck.remove(int(roots[0]))
    print(f'the len of roots is {len(roots)}')
    # for r in roots:
    #     ET.remove_edge(r, r)
    # if real_labels[roots[0]] == real_labels[roots[1]]:

    # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
    # distance, nearest_neighbors = neighbors.kneighbors(
    #     data, return_distance=True)
    # distance = distance[:, 1]
    data = data.values
    # attrs = set_uncertainty(data, ET, roots)

    # 去除有向图中的环
    # print('type of ', type(nx.degree_centrality(ET)))
    # print(nx.degree_centrality(ET))
    # assert len(data) == len(list(ET.nodes)) and len(data) == len(attrs)
    assert len(data) == len(list(ET.nodes))
    # draw_graph(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)
    # get_G(ET, data)
    # ARI_record, roots = refine_graph(ET, data, roots, title=file.split(".")
    #                                  [0], real_labels=real_labels)

    # draw_graph_w(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=None, representatives=None)

    ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
        0], skeleton=ET, representatives=[roots[0]], K=K, u_rank=uranck)

    os.makedirs(outdir, exist_ok=True)
    ARI_record.to_csv(
        f'{outdir}/{file.split(".")[0]}_result.csv', index=False)

    # tail = ARI_record.tail(1)
    # ari21['dataset'].append(file.split(".")[0])
    # ari21['ari'].append(tail['ari'].values[0])
    # ari21['interaction'].append(tail['interaction'].values[0])
    # print(f'run on {file} finished')

    # df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    # df.to_csv("our.csv", index=False)
    # pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')


def run_copkmeans(dataDir, file, outdir: str = 'result/p2/small', cnum=50) -> None:
    assert os.path.exists(dataDir)

    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }

    rdata, real_labels, K = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True)

    # if len(real_labels) < 1e3:
    #     continue
    # info['Dataset'].append(file.split(".")[0])
    # info['Class'].append(K)
    # info['Samples'].append(len(real_labels))
    # info['Features'].append(rdata.shape[1])

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
    # num_thread = 1
    # print(label)
    # data.drop_duplicates(inplace=True)

    prs = PRS(data)
    start = time.time()
    threshold_clusters = K
    # threshold_clusters >= 2
    # threshold_clusters = 2
    prs.get_clusters(num_thread, threshold_clusters)
    # print(prs.boundary_nodes)
    ET, roots = prs.get_final_tree_nx()
    print(f'the len of roots is {len(roots)}')
    for r in roots:
        ET.remove_edge(r, r)

    meg: dict[int, list] = {i: [] for i in range(K)}
    for r in roots:
        meg[real_labels[r]].append(r)
    for k, v in meg.items():
        if len(v) > 1:
            for i in range(1, len(v)):
                ET.add_edge(v[i], v[0])
                roots.remove(v[i])
    # if real_labels[roots[0]] == real_labels[roots[1]]:
    # ET.add_edge(roots[1], roots[0])
    # roots.pop()
    c = 100

    # neighbors = NearestNeighbors(n_neighbors=2).fit(data)
    # distance, nearest_neighbors = neighbors.kneighbors(
    #     data, return_distance=True)
    # distance = distance[:, 1]
    data = data.values
    # attrs = set_uncertainty(data, ET, roots)

    # 去除有向图中的环
    # print('type of ', type(nx.degree_centrality(ET)))
    # print(nx.degree_centrality(ET))
    # assert len(data) == len(list(ET.nodes)) and len(data) == len(attrs)
    assert len(data) == len(list(ET.nodes))
    # draw_graph(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)
    # get_G(ET, data)
    # ARI_record, roots = refine_graph(ET, data, roots, title=file.split(".")
    #                                  [0], real_labels=real_labels)
    # draw_scatter(data, real_labels, title=file.split(".")[0])
    # draw_graph_w(ET, data, roots, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=None, representatives=None)
    N = Neiborhood()
    ARI_record = refine_cluster_COPKmeans(ET, data, real_labels, K, cnum)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=ET, representatives=[roots[0]], K=K)

    os.makedirs(outdir, exist_ok=True)
    ARI_record.to_csv(
        f'{outdir}/{file.split(".")[0]}_result.csv', index=False)

    # tail = ARI_record.tail(1)
    # ari21['dataset'].append(file.split(".")[0])
    # ari21['ari'].append(tail['ari'].values[0])
    # ari21['interaction'].append(tail['interaction'].values[0])
    # print(f'run on {file} finished')

    # df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    # df.to_csv("our.csv", index=False)
    # pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')


def random_choose(label):
    # 随机生成点对
    print(f'label shape {len(label)}')
    pairs = list(combinations(range(len(label)), 2))
    shuffle(pairs)
    print(f'pairs shape {len(pairs)}')
    pairs = [[(p[0], p[1]), 0] for p in pairs]
    return pairs


def refine_cluster_COPKmeans(ET, data, lable, k, N, roots, cnum=20):
    # ml = []
    # cl = []
    # ET.to_undirected()
    weage = []
    df = {"iter": [], "interaction": [], "ari": [], "time": []}
    # for e in ET.edges:
    #     dis = np.linalg.norm(data[e[0]]-data[e[1]])
    #     weage.append([e, dis])
    # weage = sorted(weage, key=lambda x: x[1])
    weage = get_rank_by_layer(ET, roots, data)
    weage.reverse()

    # for e, d in weage:
    #     if len(ml) < mllen and lable[e[0]] == lable[e[1]]:
    #         ml.append(e)
    #     elif len(cl) < cllen and lable[e[0]] != lable[e[1]]:
    #         cl.append(e)
    #     if len(ml) == mllen and len(cl) == cllen:
    #         break
    # for e, d in weage[:cnum]:
    #     if lable[e[0]] == lable[e[1]]:
    #         ml.append(e)
    #     else:
    #         cl.append(e)

    # print(f'ml: {ml} cl: {cl}')
    # weage = random_choose(lable)
    for i in range(1, 10):
        ml = []
        cl = []
        for e in weage[:i]:
            if lable[e[0]] == lable[e[1]]:
                ml.append(e)
            else:
                cl.append(e)
        ts = time.perf_counter()

        @retry(retries=10, delay=0)
        def run_copkmeans_(data, lable, k, ml, cl):
            return COPKMeans(n_clusters=k).fit(
                data, lable, ml=ml, cl=cl)

        cluster = run_copkmeans_(data, lable, k, ml=ml, cl=cl)
        # cluster = MPCKMeans(n_clusters=k).fit(data, lable, ml=ml, cl=cl)
        te = time.perf_counter()

        ari = adjusted_rand_score(lable, cluster.labels_)
        df['iter'].append(i)
        df['interaction'].append(math.ceil((len(lable)-1)*(i*.1)))
        df['ari'].append(ari)
        df['time'].append(te-ts)
        print(f'ml+cl={i} ari: {ari}')

    return pd.DataFrame(df)


def set_run_arg(task: str):
    arg: dict = {'dataDir': 'G:/data/datasets/UCI/small/data',
                 'resultDir': 'result/small',
                 'pathToSavePic': 'result/pic/small'}
    if task == 'small':
        return arg
    elif task == 'middle':
        arg['dataDir'] = 'G:/data/datasets/UCI/middle/data'
        arg['resultDir'] = 'result/OUR'
        arg['pathToSavePic'] = 'result/pic/middle'
    elif task == 'large':
        arg['dataDir'] = 'G:/data/datasets/UCI/large/data'
        arg['resultDir'] = 'result/large'
        arg['pathToSavePic'] = 'result/pic/large'
    else:
        raise ValueError('task must be in [small, middle, large]')
    return arg


def get_file_list(dataDir: str):
    return os.listdir(dataDir)


if __name__ == '__main__':
    # G:/data/datasets/UCI/small/data
    # arg = {'dataDir': 'G:/data/datasets/UCI/small/data',
    #        'resultDir': 'result/COPKmeans/small'}
    os.makedirs('result/p2', exist_ok=True)
    os.makedirs('result/DSL', exist_ok=True)
    os.makedirs('result/COPKmeans/small', exist_ok=True)

    # run2(dataDir=arg['dataDir'], fdir=arg['resultDir'])
    # files = ['Balance Scale.csv', 'Banknote Authentication.csv', 'Dermatology.csv', 'Ecoli.csv', "Haberman's Survival.csv",
    #          'Image Segmentation.csv', 'Ionosphere.csv', 'Iris.csv', 'Musk (Version 1).csv', 'Waveform Database Generator (Version 1).csv', 'Wine.csv']
    # files = ['Iris.csv']
    # arg = set_run_arg('middle')
    # arg = set_run_arg('small')
    arg = set_run_arg('middle')
    # print(files)

    jobs: int = 0
    n_jobs: int = 8
    batch_size = max(1, (jobs + n_jobs-1) // n_jobs)

    files: list[str] = os.listdir(arg['dataDir'])
    #  使用进程池
    # Parallel(n_jobs=6)(delayed(runp_h)(arg['dataDir'], file,
    #                                    arg['resultDir']) for file in files
    #                    if file.endswith('.csv'))

    files: list[str] = os.listdir(arg['resultDir'])
    PathToSaveFig = 'result/pic/t'
    Parallel(n_jobs=6)(delayed(draw_graph_e)(ARIpath='result', pathToSavePic=PathToSaveFig,
                                             file=file) for file in files
                       if file.endswith('.csv'))
