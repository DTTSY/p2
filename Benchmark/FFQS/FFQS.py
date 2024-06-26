import sys

import numpy as np
import pandas as pd
from active_semi_clustering import PCKMeans
from sklearn.metrics import adjusted_rand_score
from skquery.oracle import MLCLOracle
from skquery.pairwise import FFQS

np.random.seed(200)


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.0000001
    data = data+random_matrix
    return data


def ffqs(data, labels, queries, title):
    data = data_preprocess(data)
    # ARI_record = []
    ARI_record = {"iter": [], "interaction": [], "ari": []}
    iter = 0
    for i in range(len(queries)):
        k = len(set(labels))
        constraints = {"ml": [], "cl": []}
        nbhds = None
        pdists = None
        algo = PCKMeans(n_clusters=k)
        algo.fit(data, ml=constraints["ml"], cl=constraints["cl"])
        qs = FFQS(neighborhoods=nbhds, distances=pdists)
        constraints = qs.fit(data, MLCLOracle(truth=labels, budget=queries[i]),
                             partition=algo.labels_, )
        try:
            algo.fit(data, ml=constraints["ml"], cl=constraints["cl"])
            ari = adjusted_rand_score(labels, algo.labels_)
        except:
            # print("exception",ARI_record[-1][0]['ari'])
            ari = ARI_record['ari'][-1]

        # 如何得出的准确率为1,那么就不用进行实验了
        ARI_record["iter"].append(iter)
        ARI_record["interaction"].append(queries[i])
        ARI_record["ari"].append(ari)
        # ARI_record.append(
        #     [{"iter": iter, "interaction": queries[i], "ari": ari}])
        # print("iter%s : interaction:%s : query %s:" % (iter, queries[i], ari))
        # print("iter: %s/50" % iter)
        if ari == 1:
            break
    return ARI_record

    # iter = iter+1
    # if i < len(queries) - 1:
    #     repeat = queries[i+1]-queries[i]
    #     for j in range(repeat):
    #         df = pd.DataFrame([ari])
    #         df.to_csv("g_output/%s_result.csv" %
    #                   title, mode='a', header=False, index=False)
    # else:
    #     df = pd.DataFrame([ari])
    #     df.to_csv("g_output/%s_result.csv" %
    #               title, mode='a', header=False, index=False)

# def result_to_csv(ARI_record, title):
#     # 将所有的交互到加入到里面去`
#     record = []
#     for i in range(len(ARI_record)):
#         if i<len(ARI_record)-1:
#             repeat=ARI_record[i+1][0]["interaction"]-ARI_record[i][0]["interaction"]
#             for j in range(repeat):
#                 record.append(ARI_record[i][0]["ari"])
#         else:
#             record.append(ARI_record[i][0]["ari"])
#     # 写入文档
#     df = pd.DataFrame({
#         'ARI': record,
#     })
#     df.to_csv("g_output/%s_result.csv" % title)


def queries_cal(budget):
    start = 0
    gap = budget//10
    queries = [0]
    while (True):
        start = start+gap
        if start > budget:
            break
        queries.append(int(start))
    return queries


if __name__ == '__main__':

    # 这个budgets如何搞呢,数据集大小的百分之1作为一轮记录

    datasets = ["balance", "banknote", "breast", "dermatology", "diabetes", "ecoli", "glass", "haberman", "ionosphere", "iris", "led", "musk",
                "pima", "seeds", "segment", "soybean", "thyroid", "vehicle",
                "wine", "mfeat_karhunen", "mfeat_zernike"]

    budgets = [600, 350, 400, 150, 1000, 150, 300, 400, 350, 100,
               1000, 500, 1000, 200, 2400, 500, 250, 1200, 100, 1200, 2500]

    # budgets = [1000, 150, 300, 400, 350, 100, 1000, 500, 1000, 200, 2400, 500, 250, 1200, 100, 3500, 1200, 2500]
    # # 第三次
    datasets = ["sonar", "fertility", "plrx", "zoo", "tae"]
    budgets = [250, 100, 300, 75, 400]

    for i in range(len(datasets)):
        func_name = "generate_" + datasets[i] + "_data"
        generate_data_func = getattr(sys.modules[__name__], func_name)
        path = "../../dataset/small/{}/{}.data".format(
            datasets[i], datasets[i])
        data, real_labels = generate_data_func(path)
        queries = queries_cal(budgets[i])
        ffqs(data, real_labels, queries, title="{}".format(datasets[i]))
        # result_to_csv(ARI_record, title="{}".format(datasets[i]))
