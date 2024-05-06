import math

import numpy as np
from scipy.spatial.distance import euclidean
import networkx as nx


def uncertainty_selection(predict_labels: list, data, neighbors, result_dict, nearest_neighbors):
    flattened_neighbors = [item for sublist in neighbors for item in sublist]
    uncertainty_vec = []
    for i in range(len(data)):
        if i not in flattened_neighbors:
            # 计算分母:
            denominator = len(nearest_neighbors[i])
            denominator = denominator if denominator != 0 else 1
            # 计算每个分子
            numerators = []
            for k in range(len(result_dict)):
                numerator = 0
                for j in nearest_neighbors[i]:
                    if predict_labels[j] == k:
                        numerator = numerator + 1
                numerators.append(numerator)
            # 计算每proportion
            sum = 0
            for r in range(len(numerators)):
                proportion = numerators[r]/denominator
                if proportion != 0:
                    sum = sum+proportion*math.log2(proportion)
            sum = -sum
            uncertainty_vec.append([i, sum])
    if uncertainty_vec:
        uncertainty_vec = np.array(uncertainty_vec)
        index = np.argmax(uncertainty_vec[:, 1])
        x = uncertainty_vec[index][0]
    else:
        x = None
    return x


def uncertainty_selection_m(predict_labels, data, neighbors, result_dict, ascription_tree: nx.Graph):
    flattened_neighbors = [item for sublist in neighbors for item in sublist]
    uncertainty_vec = []

    for i in range(len(data)):
        if i not in flattened_neighbors:
            # 计算分母:
            nearest_neighbors = list(ascription_tree.neighbors(i))
            denominator = len(nearest_neighbors[i])
            # 计算每个分子
            numerators = []
            for k in range(len(result_dict)):
                numerator = 0
                for j in nearest_neighbors[i]:
                    if predict_labels[j] == k:
                        numerator = numerator + 1
                numerators.append(numerator)
            # 计算每proportion
            sum = 0
            for r in range(len(numerators)):
                proportion = numerators[r]/denominator
                if proportion != 0:
                    sum = sum+proportion*math.log2(proportion)
            sum = -sum
            uncertainty_vec.append([i, sum])
    if not uncertainty_vec:
        x = None
    else:
        uncertainty_vec = np.array(uncertainty_vec)
        index = np.argmax(uncertainty_vec[:, 1])
        x = uncertainty_vec[index][0]
    return x


def query_strategy(x, neighbors, data):
    # 第一步,删除neighbor中的重复元素
    candidates = []
    for i in range(len(neighbors)):
        temp = []
        for j in range(len(neighbors[i])):
            distance = euclidean(data[x], data[neighbors[i][j]])
            temp.append([x, neighbors[i][j], distance])
        temp = np.array(temp)
        index = np.argmin(temp[:, 2])
        element = list(temp[index])
        candidates.append(element)
    # 第二步：重新对候选列表进行排序
    candidates = np.array(candidates)
    sorted_indices = np.argsort(candidates[:, 2])
    candidates = candidates[sorted_indices]
    return candidates


def dynamic_selection(real_labels, candidates, neighbors, x, count):
    find = False
    for i in range(len(candidates)):
        count = count + 1
        point1 = int(candidates[i][0])
        point2 = int(candidates[i][1])
        if real_labels[point1] == real_labels[point2]:
            for j in range(len(neighbors)):
                if point2 in neighbors[j]:
                    neighbors[j].append(point1)
            find = True
            break
    if find == False:
        neighbors.append([x])
    return neighbors, count
