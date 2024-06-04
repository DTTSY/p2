import time
import os
import math
from random import shuffle
from itertools import combinations

from tqdm import tqdm
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from DSL.DSLm import clustering, iteration_once
from myutil import DataLoader
from myutil.retry import retry
from PRSCSWAP import PRS

from active_semi_clustering import COPKMeans, MPCKMeans, PCKMeans


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
        u_rank = get_rank(skeleton, representatives, data)
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


def get_rank(ET: nx.Graph, roots, data: np.array):
    # bfs ET
    ET = ET.to_undirected()
    R = []
    for layers, nodes in dict(enumerate(nx.bfs_layers(ET, roots[0]))).items():
        print(f'layer {layers} has {len(nodes)} nodes')
        # R.extend(nodes)
        nn = []
        for n in nodes:
            dis = 0
            k = 0
            for ne in ET.neighbors(n):
                dis += np.linalg.norm(data[n]-data[ne])
                k += 1
            nn.append([n, dis/k])
        R.extend([n for n, d in sorted(nn, key=lambda x: x[1], reverse=True)])
    # R[0], R[-1] = R[-1], R[0]
    print(f'len of R is {len(R)}')
    R.reverse()
    R.remove(roots[0])
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
        _y_pred = get_label_vec_from_G(T)

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


def draw_graph(ARIpath: str, saveto: str = 'result/pic', file='', remove=False):
    print('save result to pic')
    os.makedirs(saveto, exist_ok=True)
    # for file in os.listdir(ARIpath):
    # 判断是否是csv文件
    # if not file.endswith('.csv'):
    #     continue
    ARI = pd.read_csv(f'{ARIpath}/{file}')
    fig, ax = plt.subplots()
    ax.plot(ARI['interaction'], ARI['ari'], label='ARI')
    ax.set_xlabel('interaction')
    ax.set_ylabel('ARI')
    ax.set_title(f'ARI and interaction on {file.split("_")[0]}')
    ax.legend()
    path = f'{saveto}/{file.split("_")[0]}.png'
    # print(f'save to {path}')
    plt.savefig(path, dpi=600)
    # plt.show()


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
    draw_graph_w(ET, data, roots, title=file.split(
        ".")[0], real_labels=real_labels)

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


def runPRSC(data, num_thread, threshold_clusters):
    prs = PRS(data)
    prs.get_clusters(num_thread, threshold_clusters)
    ET, roots = prs.get_final_tree_nx()
    for r in roots:
        ET.remove_edge(r, r)
    ET.add_edge(roots[1], roots[0])
    roots.pop()
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


def refine_cluster_COPKmeans(ET, data, lable, k, cnum=20):
    # ml = []
    # cl = []
    # ET.to_undirected()
    weage = []
    df = {"iter": [], "interaction": [], "ari": [], "time": []}
    for e in ET.edges:
        dis = np.linalg.norm(data[e[0]]-data[e[1]])
        weage.append([e, dis])
    weage = sorted(weage, key=lambda x: x[1])
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
        for e, d in weage[:i]:
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


if __name__ == '__main__':
    # G:/data/datasets/UCI/small/data
    arg = {'dataDir': 'G:/data/datasets/UCI/small/data',
           'resultDir': 'result/COPKmeans/small'}
    os.makedirs('result/p2', exist_ok=True)
    os.makedirs('result/DSL', exist_ok=True)
    os.makedirs('result/COPKmeans/small', exist_ok=True)

    # run2(dataDir=arg['dataDir'], fdir=arg['resultDir'])
    files = os.listdir(arg['dataDir'])
    files = ['Balance Scale.csv', 'Banknote Authentication.csv', 'Dermatology.csv', 'Ecoli.csv', "Haberman's Survival.csv",
             'Image Segmentation.csv', 'Ionosphere.csv', 'Iris.csv', 'Musk (Version 1).csv', 'Waveform Database Generator (Version 1).csv', 'Wine.csv']
    # files = ['Iris.csv']
    # print(files)
    # batch_size =
    Parallel(n_jobs=8)(delayed(run_copkmeans)(arg['dataDir'], file,
                                              arg['resultDir']) for file in files
                       if file.endswith('.csv'))

    Parallel(n_jobs=8)(delayed(draw_graph)(ARIpath=arg['resultDir'], saveto='result/pic/COPKmeans',
                                           file=file) for file in os.listdir(arg['resultDir'])
                       if file.endswith('.csv'))
