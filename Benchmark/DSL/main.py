
from scipy.spatial import distance
from networkx import shortest_path
from matplotlib import pyplot as plt
import time

import networkx as nx
import pandas as pd

from sklearn.metrics import adjusted_rand_score

import copy

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
np.random.seed(0)


def draw_contraint_graph(constraint_G):
    elarge = [(u, v) for (u, v, d) in constraint_G.edges(
        data=True) if d["weight"] == 1]
    esmall = [(u, v) for (u, v, d) in constraint_G.edges(
        data=True) if d["weight"] == 0]
    pos = nx.drawing.nx_agraph.graphviz_layout(constraint_G, prog='neato')
    nx.draw_networkx_nodes(constraint_G, pos, node_size=10)
    nx.draw_networkx_edges(
        constraint_G, pos, edgelist=esmall, width=1, edge_color="r", style="dashed")
    nx.draw_networkx_edges(
        constraint_G, pos, edgelist=elarge, width=1, edge_color="b", style="dashed"
    )
    nx.draw_networkx_labels(
        constraint_G, pos, font_size=10, font_family="sans-serif")
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def anomaly_detection(Graph):
    max_value_node = None
    max_value = float('-inf')
    for node, data in Graph.nodes(data=True):
        if 'uncertainty' in data and data['uncertainty'] > max_value:
            max_value = data['uncertainty']
            max_value_node = node
    uncertain_node = max_value_node
    try:
        anomaly = list(Graph.out_edges(uncertain_node))[0]
    except Exception:
        anomaly = None
    suspend = False
    if Graph.nodes[uncertain_node]["uncertainty"] == 0:
        suspend = True
    return anomaly, suspend


def human_judgement(anomaly, real_labels, constraint_graph):
    node1 = int(anomaly[0])
    node2 = int(anomaly[1])
    if real_labels[node1] == real_labels[node2]:
        result = "like"
        weight = 0
    else:
        result = "dislike"
        weight = 1
    constraint_graph.add_edge(anomaly[0], anomaly[1], weight=weight)
    return constraint_graph, result


def distance_cal(path, G):
    sum = 0
    for i in range(len(path)-1):
        sum = sum+G[path[i]][path[i+1]]["weight"]
    return sum


def constraint_judgement(G, pairwise):
    source = int(pairwise[0])
    target = int(pairwise[1])
    if (source not in list(G.nodes)) or (target not in list(G.nodes)) or nx.has_path(G, source, target) == False:
        result = "unknown"
    else:
        path = shortest_path(G, source=source, target=target,
                             weight='weight', method='dijkstra')
        sum = distance_cal(path, G)
        if sum == 0:
            result = "like"
        elif sum == 1:
            result = "dislike"
        else:
            result = "unknown"
    return result


def judgement(anomaly, constraint_graph, real_labels):
    pairwise = [int(anomaly[0]), int(anomaly[1])]
    result = constraint_judgement(constraint_graph, pairwise)
    if result == "unknown":
        constraint_graph, result = human_judgement(
            pairwise, real_labels, constraint_graph)
        judgement_type = "human"
    else:
        judgement_type = "constraint"
    return constraint_graph, result, judgement_type


def skeleton_reconstruction_like(skeleton, anomaly):
    skeleton.nodes[anomaly[0]]["uncertainty"] = 0
    return skeleton


def connections_cal(edge, representatives, data):
    connections = []
    for representative in representatives:
        euc_distance = distance.euclidean(data[edge[0]], data[representative])
        connections.append([edge[0], representative, euc_distance])
    connections = np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections


def skeleton_reconstruction_dislike(skeleton, anomaly, representatives, data, real_labels, constraint_graph, count):
    skeleton.remove_edge(anomaly[0], anomaly[1])
    connections = connections_cal(anomaly, representatives, data)
    find = False
    for connection in connections:
        constraint_graph, result, judgement_type = judgement(
            connection, constraint_graph, real_labels)
        if judgement_type == "human":
            count += 1
        if result == "like":
            find = True
            node1 = int(connection[0])
            node2 = int(connection[1])
            in_degree_node1 = skeleton.in_degree(node1)
            in_degree_node2 = skeleton.in_degree(node2)
            # 修改两个候选点的uncertainty
            skeleton.nodes[node1]["uncertainty"] = 0
            skeleton.nodes[node2]["uncertainty"] = 0
            if in_degree_node1 > in_degree_node2:
                representatives.remove(node2)
                representatives.append(node1)
                skeleton.add_edge(node2, node1)
            else:
                skeleton.add_edge(node1, node2)
            break
    if find == False:
        representatives.append(anomaly[0])
        skeleton.nodes[anomaly[0]]["uncertainty"] = 0
    return skeleton, representatives, constraint_graph, count


def uncertainty_propagation_like(skeleton, anomaly, alpha):
    result = dict(enumerate(nx.bfs_layers(skeleton.reverse(), [anomaly[0]])))
    count = 1
    while (count < len(result)):
        amptitude = 1-alpha**count
        nodes_layer = result[count]
        for node in nodes_layer:
            skeleton.nodes[node]["uncertainty"] = skeleton.nodes[node]["uncertainty"] * amptitude
        count = count+1
    return skeleton


def uncertainty_propagation_dislike(skeleton, anomaly, beta):
    result = dict(enumerate(nx.bfs_layers(skeleton.reverse(), [anomaly[0]])))
    count = 1
    while (count < len(result)):
        amptitude = 1+beta**count
        nodes_layer = result[count]
        for node in nodes_layer:
            skeleton.nodes[node]["uncertainty"] = skeleton.nodes[node]["uncertainty"] * amptitude
        count = count+1
    return skeleton


def skeleton_reconstruction(skeleton, anomaly, representatives, data, real_labels, constraint_graph, count, result):
    if result == "like":
        skeleton = skeleton_reconstruction_like(skeleton, anomaly)
    if result == "dislike":
        skeleton, representatives, constraint_graph, count = skeleton_reconstruction_dislike(
            skeleton, anomaly, representatives, data, real_labels, constraint_graph, count)
    return skeleton, representatives, constraint_graph, count


def iteration_once(skeleton, representatives, data, real_labels, constraint_graph):
    count = 0
    anomaly, suspend = anomaly_detection(skeleton)
    if anomaly != None:
        constraint_graph, result, judgement_type = judgement(
            anomaly, constraint_graph, real_labels)
        if judgement_type == "human":
            count += 1
        skeleton, representatives, constraint_graph, count = skeleton_reconstruction(
            skeleton, anomaly, representatives, data, real_labels, constraint_graph, count, result)
    return skeleton, representatives, constraint_graph, count, suspend


def nearest_neighbor_cal(feature_space):
    neighbors = NearestNeighbors(n_neighbors=2).fit(feature_space)
    distance, nearest_neighbors = neighbors.kneighbors(
        feature_space, return_distance=True)
    distance = distance[:, 1]
    nearest_neighbors = nearest_neighbors.tolist()
    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i].append(distance[i])
    return nearest_neighbors


def sub_nodes_cal(sub_S):

    points = None
    for edge in sub_S.edges:
        if sub_S.has_edge(edge[1], edge[0]):
            point1 = edge[0]
            point2 = edge[1]
            points = [point1, point2]
            break
    return points


def representative_find_sitation_2(points, skeleton):
    sum1 = 0
    in_edges = skeleton.in_edges(points[0])
    in_edges = list(in_edges)
    for i in range(len(in_edges)):
        sum1 = sum1 + skeleton.nodes[in_edges[i][0]]["uncertainty"]
    sum2 = 0
    in_edges = skeleton.in_edges(points[1])
    in_edges = list(in_edges)
    for i in range(len(in_edges)):
        sum2 = sum2 + skeleton.nodes[in_edges[i][0]]["uncertainty"]
    index = np.argmax([sum1, sum2])
    representative = points[index]
    return index, representative


def clustering_loop(feature_space, dict_mapping, skeleton):
    representatives = []
    edges = nearest_neighbor_cal(feature_space)
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
        uncertainty = edges[i][2]
        skeleton.add_edge(edges[i][0], edges[i][1])
        skeleton.nodes[edges[i][0]]['uncertainty'] = uncertainty
    S = [skeleton.subgraph(c).copy()
         for c in nx.weakly_connected_components(skeleton)]
    for sub_S in S:
        points = sub_nodes_cal(sub_S)
        a = skeleton.in_degree(points[0])
        b = skeleton.in_degree(points[1])
        if a != b:
            index = np.argmax([a, b])
            representative = points[index]
        else:
            index, representative = representative_find_sitation_2(
                points, skeleton)
        representatives.append(representative)
        edge_remove = [points[index], points[1-index],
                       skeleton[points[index]][points[1-index]]]
        skeleton.remove_edge(edge_remove[0], edge_remove[1])
    dict_mapping = {}
    for i in range(len(representatives)):
        dict_mapping[i] = representatives[i]
    return representatives, skeleton, dict_mapping


def clustering(data):
    feature_space = copy.deepcopy(data)
    dict_mapping = {}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.DiGraph()
    while (True):
        representatives, skeleton, dict_mapping = clustering_loop(
            feature_space, dict_mapping, skeleton)
        feature_space = data[representatives]
        if len(representatives) == 1:
            break
    skeleton.nodes[representatives[0]]['uncertainty'] = 0
    return skeleton, representatives


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data+random_matrix
    return data


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


def get_predict_labels(Graph: nx.Graph):
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    predict_labels = np.zeros(len(Graph.nodes), dtype=int)

    for i, s in enumerate(S[1:], 1):
        predict_labels[list(s.nodes)] = i
    return predict_labels


def DSL(data, real_labels, title, q=1000):
    columns = ["iter", "interaction", "ari", "time"]
    df = {"iter": [], "interaction": [], "ari": [], "time": []}
    # df = pd.DataFrame(columns=columns)

    start_time = time.time()
    # data = data_preprocess(data)
    # data = (data - data.mean()) / (data.std())
    skeleton, representatives = clustering(data)
    loop = len(data) + 10
    constraint_graph = nx.Graph()
    interaction = 0
    # predict_labels = skeleton_process(skeleton)
    predict_labels = get_predict_labels(skeleton)
    ARI = adjusted_rand_score(real_labels, predict_labels)
    df["iter"].append(0)
    df["interaction"].append(interaction)
    df["ari"].append(ARI)
    df["time"].append(0)
    # record = [{"iter": 0, "interaction": interaction, "ari": ARI, "time": 0}]
    # df = df._append(pd.DataFrame(record), ignore_index=True)
    for i in range(loop):
        skeleton, representatives, constraint_graph, count, suspend = iteration_once(skeleton, representatives, data,
                                                                                     real_labels, constraint_graph)
        interaction = interaction + count
        if suspend == True:
            print("The algorithm is down")
            break
        # predict_labels = skeleton_process(skeleton)
        predict_labels = get_predict_labels(skeleton)
        ARI = adjusted_rand_score(real_labels, predict_labels)
        duration = time.time() - start_time
        # record = [{"iter": i + 1, "interaction": interaction,
        #            "ari": ARI, "time": duration}]
        # df = df._append(pd.DataFrame(record), ignore_index=True)
        df["iter"].append(i + 1)
        df["interaction"].append(interaction)
        df["ari"].append(ARI)
        df["time"].append(duration)
        if interaction > q:
            break
        # df.to_csv("e_output/%s_result.csv" % title)
    return df


if __name__ == '__main__':
    raise Exception("This file is not meant to be executed")

    data, real_labels = generate_wine_data(
        path="../dataset/small/wine/wine.data")
    ARI_record = DSL(data, real_labels, title="wine")
