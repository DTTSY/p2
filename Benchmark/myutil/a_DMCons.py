# 最近邻构建子树
import copy
import time
import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from networkx.drawing.nx_pydot import graphviz_layout
# random.seed(100)
# np.random.seed(100)


def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, alpha=0.5, node_color="blue",
            with_labels=True, font_size=20, node_size=30)
    plt.axis("equal")
    plt.show()


def nearest_neighbor_cal(feature_space, k):
    k = k if k < feature_space.shape[0] else feature_space.shape[0]
    neighbors = NearestNeighbors(n_neighbors=k).fit(feature_space)
    distances, nearest_neighbors = neighbors.kneighbors(
        feature_space, return_distance=True)

    # print(nearest_neighbors[:5, :])
    # print(distances[:5, :])

    edges = []
    for i in range(nearest_neighbors.shape[0]):
        for j in range(1, nearest_neighbors.shape[1]):
            edges.append(
                [nearest_neighbors[i, 0], nearest_neighbors[i, j], distances[i, j]])
    print(edges[:5])
    return edges


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data+random_matrix
    return data

# 找到一个小簇的代表点


def representative_cal(sub_S: nx.Graph):
    degree_dict = dict(sub_S.degree())
    # 找到最大度
    max_degree = max(degree_dict.values())
    # 找到具有最大度的所有节点
    nodes_with_max_degree = [
        node for node, degree in degree_dict.items() if degree == max_degree]

    # 在nodes_with_max_degree中找权重最小的那个点
    # min_weighted_degree_sum = float('inf')
    # min_weighted_degree_node = None
    # for node in nodes_with_max_degree:
    #     # 计算当前节点的带权重度之和
    #     weighted_degree_sum = sum(
    #         weight for _, _, weight in sub_S.edges(data='weight', nbunch=node))
    #     # 更新最小值
    #     if weighted_degree_sum < min_weighted_degree_sum:
    #         min_weighted_degree_sum = weighted_degree_sum
    #         min_weighted_degree_node = node
    # 从nodes_with_max_degree列表里随机选一个元素
    # representative = min_weighted_degree_node

    representative = random.choice(nodes_with_max_degree)
    return representative


def clustering_loop(feature_space, dict_mapping, skeleton: nx.Graph, k):
    Graph = nx.Graph()
    representatives = []
    # 1.连边过程
    edges = nearest_neighbor_cal(feature_space, k)
    # print(edges[:5, :])
    Graph.add_weighted_edges_from(edges)
    # 2.找representative node
    S = [Graph.subgraph(c).copy() for c in nx.connected_components(Graph)]
    for sub_S in S:
        representative = representative_cal(sub_S)
        representatives.append(representative)
    # 3.将representatives和edges进行映射
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
    for i in range(len(representatives)):
        representatives[i] = dict_mapping[representatives[i]]
    # 4.将edges加入到skeleton
    skeleton.add_weighted_edges_from(edges)
    # 5.计算新的dict_mapping
    dict_mapping = {}
    for i in range(len(representatives)):
        dict_mapping[i] = representatives[i]
    return representatives, skeleton, dict_mapping


def graph_initialization(data, k):
    feature_space = copy.deepcopy(data)
    dict_mapping = {}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.DiGraph()
    while (True):
        representatives, skeleton, dict_mapping = clustering_loop(
            feature_space, dict_mapping, skeleton, k)
        feature_space = data[representatives]
        if len(representatives) == 1:
            break
    representative = representatives[0]
    return skeleton, representative


if __name__ == '__main__':

    data, labels = generate_iris_data(path="../dataset/small/iris/iris.data")
    data = data_preprocess(data)
    skeleton, representative = graph_initialization(data)

    # numbers=[10000,20000,40000,80000,160000]
    # times=[]
    # for number in numbers:
    #     start=time.time()
    #     data, real_labels = make_classification(random_state=0,n_samples=number, n_features=40, n_redundant=1, n_informative=20, n_clusters_per_class=2, n_classes=10)
    #     skeleton,representative=graph_initialization(data)
    #     end=time.time()
    #     duration=end-start
    #     times.append(duration)
    # print(times)
