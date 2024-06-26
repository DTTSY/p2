import math

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial.distance import euclidean


np.random.seed(0)


def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, alpha=0.5, node_color="blue",
            with_labels=True, font_size=10, node_size=30)
    plt.axis("equal")
    plt.show()


def dc_cal(alpha, data):
    n = len(data)
    # 计算dc坐标
    a = math.floor(0.5*n*(n-1)*alpha+0.5)
    # 计算两两的点对关系
    temp = []
    i = 0
    while (i < n):
        j = i+1
        while (j < n):
            distance = euclidean(data[i], data[j])
            temp.append([i, j, distance])
            j = j+1
        i = i+1
    temp = np.array(temp)
    sorted_indices = np.argsort(temp[:, 2])
    index = sorted_indices[a-1]
    dc = temp[index][2]
    return dc


def local_density_cal(nearest_neighbors, data, dc):
    density_vec = []
    for i in range(len(data)):
        sum = 0
        for j in nearest_neighbors[i]:
            distance = euclidean(data[i], data[j])
            sum = sum + math.exp(-(distance / dc) ** 2)
        density_vec.append(sum)
    density_vec = np.array(density_vec)
    return density_vec


def nearest_higher_vec(density_vec, data):
    distances_vec = []
    for i in range(len(data)):
        # 找打密度比它大的点
        indexes = []
        for j in range(len(data)):
            if density_vec[j] > density_vec[i]:
                indexes.append(j)
        # 计算max_d_{ij}
        max_distance = -1
        for m in range(len(data)):
            distance = euclidean(data[i], data[m])
            if distance > max_distance:
                max_distance = distance
        # 找到密度比他大的最近点
        min_distance = [i, -1, max_distance]
        for k in indexes:
            distance = euclidean(data[i], data[k])
            if distance < min_distance[2]:
                min_distance = [i, k, distance]
        distances_vec.append(min_distance)
    return distances_vec


def ascription_tree_construction(distances_vec):
    ascription_tree = nx.DiGraph()
    ascription_tree.add_weighted_edges_from(distances_vec)
    ascription_tree.remove_node(-1)
    return ascription_tree


def center_probability_cal(density_vec, distances_vec):
    a = np.array(density_vec)
    b = np.array(distances_vec)[:, 2]
    center_vec = []
    max_a = np.max(a)
    max_b = np.max(b)
    for i in range(len(a)):
        center_vec.append([i, (a[i]*b[i])/(max_a*max_b)])
    return center_vec


def nearest_neighbors_cal(dc, data):
    nearest_neighbors = []
    for i in range(len(data)):
        temp = []
        for j in range(len(data)):
            distance = euclidean(data[i], data[j])
            if distance < dc:
                temp.append(j)
        nearest_neighbors.append(temp)
    return nearest_neighbors


def initialization(alpha, data):
    data = data_preprocess(data)
    dc = dc_cal(alpha, data=data)
    nearest_neighbors = nearest_neighbors_cal(dc, data)
    density_vec = local_density_cal(nearest_neighbors, data, dc)
    distances_vec = nearest_higher_vec(density_vec, data)
    center_vec = center_probability_cal(density_vec, distances_vec)
    ascription_tree = ascription_tree_construction(distances_vec)
    return dc, density_vec, distances_vec, center_vec, ascription_tree, nearest_neighbors


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.00000223
    data = data+random_matrix
    return data


if __name__ == '__main__':
    data, real_labels = generate_iris_data(
        path="../../f_datasets/small/iris/iris_result.csv")
    alpha = 0.22
    l = 5
    theta = 0.00001
    data = data_preprocess(data)
    dc, density_vec, distances_vec, center_vec, ascription_tree, nearest_neighbors = initialization(
        alpha, data)
    draw_graph(ascription_tree)
