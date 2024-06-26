import sys

import numpy as np
import pandas as pd
from active_semi_clustering import PCKMeans
from sklearn.metrics import adjusted_rand_score
from skquery.oracle import MLCLOracle
from skquery.pairwise import MinMax

np.random.seed(100)


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.0000001
    data = data + random_matrix
    return data


def minmax(data, labels, queries, title):
    # data = data_preprocess(data)
    ARI_record = {"iter": [], "interaction": [], "ari": []}
    iter = 0
    count = 0

    for query in queries:
        k = len(set(labels))
        constraints = {"ml": [], "cl": []}
        nbhds = None
        pdists = None
        algo = PCKMeans(n_clusters=k)
        algo.fit(data, ml=constraints["ml"], cl=constraints["cl"])
        qs = MinMax(neighborhoods=nbhds, distances=pdists)
        constraints = qs.fit(data, MLCLOracle(truth=labels, budget=query),
                             partition=algo.labels_, )
        try:
            algo.fit(data, ml=constraints["ml"], cl=constraints["cl"])
            ari = adjusted_rand_score(labels, algo.labels_)
        except:
            # print("exception",ARI_record[-1][0]['ari'])
            ari = ARI_record['ari'][-1]

        if ari == 1:
            break
        ARI_record["iter"].append(iter)
        ARI_record["interaction"].append(query)
        ARI_record["ari"].append(ari)
        # ARI_record.append([{"iter": iter, "interaction": query, "ari": ari}])
        # print("iter%s : interaction:%s : query %s:" % (iter, query, ari))
        # print("iter: %s/50" % iter)
        iter = iter+1
    return ARI_record

    # if i < len(queries) - 1:
    #     repeat = queries[i+1]-queries[i]
    #     for j in range(repeat):
    #         df = pd.DataFrame({count: [ari]})
    #         df.to_csv("g_output/%s_result.csv" %
    #                   title, mode='a', header=False, index=False)
    # else:
    #     df = pd.DataFrame({count: [ari]})
    #     df.to_csv(f"g_output/{title}_result.csv",
    #               mode='a', header=False, index=False)
    # count = count+1


def queries_cal(budget):
    start = 0
    gap = budget/10
    queries = [0]
    while (True):
        start = start+gap
        if start > budget:
            break
        queries.append(int(start))
    return queries


if __name__ == '__main__':
    datasets = ["sonar", "fertility", "plrx", "zoo", "tae"]
    budgets = [250, 100, 300, 75, 400]

    datasets = ["tae"]
    budgets = [400]

    for i in range(len(datasets)):
        func_name = "generate_" + datasets[i] + "_data"
        generate_data_func = getattr(sys.modules[__name__], func_name)
        path = "../../dataset/small/{}/{}.data".format(
            datasets[i], datasets[i])
        data, real_labels = generate_data_func(path)

        queries = queries_cal(budgets[i])
        minmax(data, real_labels, queries, title="{}".format(datasets[i]))
