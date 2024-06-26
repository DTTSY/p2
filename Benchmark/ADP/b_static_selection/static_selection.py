import networkx as nx
import numpy as np

from ADP.a_initailization.initialization import initialization


def descending_order(center_vec):
    center_vec = np.array(center_vec)
    sorted_indices = np.argsort(center_vec[:, 1])[::-1]
    center_vec = center_vec[sorted_indices]
    return center_vec


def sliding_window(l, center_vec_dec, theta):
    index = None
    for i in range(len(center_vec_dec)-l+1):
        indexes = []
        for j in range(l):
            indexes.append(i+j)
        window = center_vec_dec[indexes, :]
        variance = np.var(window[:, 1])
        if variance < theta:
            index = i
            break
    if index == None:
        print("没有找到turning point")
    else:
        centers = center_vec_dec[0:index, :][:, 0].astype(int)
    if list(centers) == []:
        centers = [center_vec_dec[0, 0], center_vec_dec[1, 0]]
        centers = np.array(centers).astype(int)
    return centers


def static_selection(P, real_labels):
    neighbors = []
    neighbors.append([P[0]])
    count = 0
    for i in range(len(P)):
        if i != 0:
            find = False
            for j in range(len(neighbors)):
                count = count+1
                if real_labels[neighbors[j][0]] == real_labels[P[i]]:
                    neighbors[j].append(P[i])
                    find = True
                    break
            if find == False:
                neighbors.append([P[i]])
    return neighbors, count


def neighbors_labeling(ascription_tree, neighbors):

    # 删除出度边
    for i in range(len(neighbors)):
        for j in range(len(neighbors[i])):
            point = neighbors[i][j]
            edge_to_move = list(ascription_tree.out_edges(point))
            if edge_to_move != []:
                ascription_tree.remove_edge(
                    edge_to_move[0][0], edge_to_move[0][1])
    result_dict = {}
    for i in range(len(neighbors)):
        result_dict[i] = []
    for i in range(len(neighbors)):
        for j in range(len(neighbors[i])):
            point = neighbors[i][j]
            result = list(nx.bfs_tree(ascription_tree, point, reverse=True))
            result_dict[i] = result_dict[i]+result
    temp = {value: key for key, values in result_dict.items()
            for value in values}
    # 返回预测结果
    predict_labels = []
    for i in range(len(temp)):
        predict_labels.append(temp[i])
    return predict_labels, result_dict


if __name__ == '__main__':
    data, real_labels = generate_iris_data(
        path="../../f_datasets/small/iris/iris.data")
    alpha = 0.1
    l = 3
    theta = 0.004
    dc, density_vec, distances_vec, center_vec, ascription_tree, nearest_neighbors = initialization(
        alpha, data)
    center_vec_dec = descending_order(center_vec)
    P = sliding_window(l, center_vec_dec, theta)
    neighbors, count = static_selection(P, real_labels)
    # 除去每个点的出度边，然后
    predict_labels = neighbors_labeling(ascription_tree, neighbors)
