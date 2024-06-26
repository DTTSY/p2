import random
from scipy.spatial.distance import euclidean
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import numpy as np
import math

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score


np.random.seed(100)

random.seed(100)  # 使用0作为随机数种子


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


def center_probability_cal(density_vec, distances_vec):
    a = np.array(density_vec)
    b = np.array(distances_vec)[:, 2]
    center_vec = []
    max_a = np.max(a)
    max_b = np.max(b)
    for i in range(len(a)):
        center_vec.append([i, (a[i]*b[i])/(max_a*max_b)])
    return center_vec

# 子空间的划分


def subspace_generation(data, xi):
    xi_max = xi[0]
    xi_min = xi[1]
    # 在0.6和0.8之间生成随机数
    extraction_ratio = random.uniform(xi_min, xi_max)
    # 得到总的features需要的数量
    feature_number = np.shape(data)[1]
    # 得到需要随机抽取的features的数量
    number = int(feature_number*extraction_ratio)
    # 抽出number个列作为子空间
    selected_columns = np.random.choice(
        data.shape[1], size=number, replace=False)
    sub_space = data[:, selected_columns]
    return sub_space

# 得到一个树


def ascription_tree_construction(distances_vec):
    ascription_tree = nx.DiGraph()
    ascription_tree.add_weighted_edges_from(distances_vec)
    ascription_tree.remove_node(-1)
    return ascription_tree


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


def static_selection(l, theta, real_labels, center_vec):
    # 按照降序进行排列
    center_vec_dec = descending_order(center_vec)
    # 通过滑动窗口机制来找到前几个代表点
    P = sliding_window(l, center_vec_dec, theta)
    # 通过静态选择找到前几个有用的点。
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


# 从ascription_tree中映射出标签
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
    length = len(ascription_tree.nodes)
    predict_labels = []
    for i in range(length):
        if i not in temp.keys():
            predict_labels.append(0)
        else:
            predict_labels.append(temp[i])
    return predict_labels, result_dict


# 预先对数据进行随机扰动
def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.0001
    data = data+random_matrix
    return data

# 计算点的不确定性


def local_uncertainty_cal(predict_labels, data, result_dict, nearest_neighbors):
    uncertainty_vec = []
    for i in range(len(data)):
        # 计算分母:
        denominator = len(nearest_neighbors[i])
        # 计算每个分子
        numerators = []
        for k in range(len(result_dict)):
            numerator = 0
            for j in nearest_neighbors[i]:
                if predict_labels[j] == k:
                    numerator = numerator + 1
            numerators.append(numerator)
        # 计算每proportion]
        sum = 0
        for r in range(len(numerators)):
            proportion = numerators[r]/denominator
            if proportion != 0:
                sum = sum+proportion*math.log2(proportion)
        sum = -sum
        uncertainty_vec.append([i, sum])
    uncertainty_vec = np.array(uncertainty_vec)
    return uncertainty_vec


def global_uncertainty_cal(predicted_labels):
    global_uncertainty = []
    # 计算有多少个子空间
    s = len(predicted_labels)
    for i in range(len(predicted_labels[0])):
        # 计算每种某个点每种类别的数量
        dict = {}
        for j in range(s):
            varphi_x = predicted_labels[j][i]
            if varphi_x not in dict.keys():
                dict[varphi_x] = 1
            else:
                dict[varphi_x] = dict[varphi_x]+1
        sum = 0
        for k in dict.keys():
            q = dict[k]/s
            if q != 0:
                sum = sum + q * math.log2(q)
        sum = -sum
        global_uncertainty.append(sum)
    global_uncertainty = np.array(global_uncertainty)
    return global_uncertainty


def uncertainty_cal(predicted_labels, subspaces, result_dicts, nearest_neighbors, s):
    local_uncertainty = np.column_stack(
        (np.arange(len(predicted_labels[0])), np.zeros(len(predicted_labels[0]))))
    for i in range(s):
        a = local_uncertainty_cal(
            predicted_labels[i], subspaces[i], result_dicts[i], nearest_neighbors[i])
        local_uncertainty = np.column_stack(
            (local_uncertainty[:, 0], local_uncertainty[:, 1] + a[:, 1]))
    local_uncertainty[:, 1] = local_uncertainty[:, 1] / s
    # 计算global uncertainty
    global_uncertainty = global_uncertainty_cal(predicted_labels)
    # 将两种uncertainty相加
    local_uncertainty[:, 1] += global_uncertainty
    overall_uncertainty = local_uncertainty
    return overall_uncertainty


def labels_and_dict_ouput(ascription_trees, neighborhood, s):
    # 标签和字典打印
    predicted_labels = []
    result_dicts = []
    for i in range(s):
        predicted_label, result_dict = neighbors_labeling(
            ascription_trees[i], neighborhood)
        predicted_labels.append(predicted_label)
        result_dicts.append(result_dict)
    return predicted_labels, result_dicts


# 通过投票机制输出结果
def overall_labeling_output(predicted_labels, subspaces, result_dicts, nearest_neighbors):
    s = len(subspaces)
    # 公式的前半段
    uncertainty_first = []
    for j in range(s):
        # 得到每个此子空间的每个点的不确定集合
        a = local_uncertainty_cal(
            predicted_labels[j], subspaces[j], result_dicts[j], nearest_neighbors[j])
        # 用1来减去得到的不确定性集合
        a[:, 1] = 1 - a[:, 1]
        a = list(a[:, 1])
        uncertainty_first.append(a)
    # 公式的后半段
    dicts = []
    for i in range(len(predicted_labels[0])):
        # 计算每种某个点每种类别的数量
        dict = {}
        for j in range(s):
            varphi_x = predicted_labels[j][i]
            if varphi_x not in dict.keys():
                dict[varphi_x] = [j]
            else:
                dict[varphi_x].append(j)
        dicts.append(dict)
    overall_label = []
    # 每个点的每个类的得分
    for i in range(len(predicted_labels[0])):
        score_dict = {}
        for k in dicts[i].keys():
            sum = 0
            for j in dicts[i][k]:
                sum = sum+uncertainty_first[j][i]
            score_dict[k] = sum
        point = max(score_dict, key=lambda k: score_dict[k])
        overall_label.append(point)
    return overall_label

# 找到除了neighborhood之外不确定度最大的点


def max_uncertainity_node_cal(overall_uncertainty, neighborhood):
    flattened_neighbors = [
        item for sublist in neighborhood for item in sublist]
    temp = np.delete(overall_uncertainty, flattened_neighbors, axis=0)
    if len(temp) == 0:
        x = None
    else:
        index = np.argmax(temp[:, 1])
        x = temp[index, 0]
    return x

# 每一轮的查询顺序


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

# 动态选择更新neighborhood


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


def ADPE(data, labels, xi, alpha, s, l, theta, q=1000):
    # 保存结果的变量:
    ARI_record = []
    ARI_record = {"iter": [], "interaction": [], "ari": []}
    ARI = adjusted_rand_score(labels, [0] * len(data))
    interaction = 0
    iter = 0
    ARI_record["iter"].append(iter)
    ARI_record["interaction"].append(interaction)
    ARI_record["ari"].append(ARI)
    # ARI_record.append([{"iter": iter, "interaction": interaction, "ari": ARI}])
    # 得到子空间
    average_center_vec = np.zeros((len(data),))
    ascription_trees = []
    nearest_neighbors = []
    subspaces = []
    for i in range(s):
        # 计算子空间
        subspace = subspace_generation(data, xi)
        subspaces.append(subspace)
        # 计算alpha
        random_alpha = random.uniform(alpha[0], alpha[1])
        # 计算截止距离
        dc = dc_cal(random_alpha, data=subspace)
        # 计算最近邻
        nearest_neighbor = nearest_neighbors_cal(dc, data=subspace)
        # 将最近邻搜集起来
        nearest_neighbors.append(nearest_neighbor)
        # 计算密度
        density_vec = local_density_cal(nearest_neighbor, data=subspace, dc=dc)
        # 计算parent distance
        distances_vec = nearest_higher_vec(density_vec, data=subspace)
        # 计算每个点作为中心点的概率
        center_vec = center_probability_cal(density_vec, distances_vec)
        # 得出一个ascription_tree
        ascription_tree = ascription_tree_construction(distances_vec)
        ascription_trees.append(ascription_tree)
        # 计算总的中心点概率之和
        average_center_vec = average_center_vec + np.array(center_vec)[:, 1]
    # 计算平局的概率之和
    average_center_vec = average_center_vec / s
    average_center_vec = np.column_stack(
        (np.arange(0, len(data)), average_center_vec))
    # step2 1.静态选择阶段
    neighborhood, count = static_selection(
        l, theta, labels, center_vec=average_center_vec)
    # step2 2.标签和字典打印
    predicted_labels, result_dicts = labels_and_dict_ouput(
        ascription_trees, neighborhood, s)
    # step2 3.通过投票机制输出每个instance的标签
    overall_label = overall_labeling_output(
        predicted_labels, subspaces, result_dicts, nearest_neighbors)

    # 记录；
    iter = iter + 1
    ARI = adjusted_rand_score(labels, overall_label)
    ARI_record["iter"].append(iter)
    ARI_record["interaction"].append(count)
    ARI_record["ari"].append(ARI)

    # ARI_record.append([{"iter": iter, "interaction": count, "ari": ARI}])
    while (True) or count < q:
        # step3 1.计算总的uncertainty
        overall_uncertainty = uncertainty_cal(
            predicted_labels, subspaces, result_dicts, nearest_neighbors, s)
        # step3 2.找到除了overall之外的最大点
        x = max_uncertainity_node_cal(overall_uncertainty, neighborhood)
        # step3 3.设置循环的终止条件
        if x == None:
            print("查询完毕")
            break
        else:
            x = int(x)
        candidates = query_strategy(x, neighborhood, data)
        neighborhood, count = dynamic_selection(
            labels, candidates, neighborhood, x, count)
        # 更新结果
        predicted_labels, result_dicts = labels_and_dict_ouput(
            ascription_trees, neighborhood, s)
        overall_label = overall_labeling_output(
            predicted_labels, subspaces, result_dicts, nearest_neighbors)
        iter = iter + 1
        ARI = adjusted_rand_score(labels, overall_label)

        # print("{iter: %s, interaction: %s, ari: %s}" % (iter, count, ARI))
        ARI_record["iter"].append(iter)
        ARI_record["interaction"].append(count)
        ARI_record["ari"].append(ARI)
        # ARI_record.append([{"iter": iter, "interaction": count, "ari": ARI}])
        if ARI == 1:
            break

    return ARI_record


if __name__ == '__main__':
    # 数据集load
    data, labels = generate_sonar_data(
        path="../f_datasets/small/sonar/sonar.data")
    # 子空间划分
    xi = [0.6, 0.8]
    # 计算local density 密度的
    alpha = [0.15, 0.30]
    # 分为多少个子空间
    s = 10
    # 滑动窗口
    l = 5
    # 计算中线点概率的
    theta = 0.00001
    ARI_record = ADPE(data, labels, xi, alpha, s, l, theta)
