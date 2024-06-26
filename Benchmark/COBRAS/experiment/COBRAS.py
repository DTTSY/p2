import sys

import numpy
import numpy as np
import pandas as pd
from sklearn import metrics

from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.labelquerier import LabelQuerier
from sklearn.metrics import adjusted_rand_score


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data+random_matrix
    return data


def COBRAS(data, real_labels, budget):
    clusterer = COBRAS_kmeans(data, LabelQuerier(real_labels), budget)
    clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
    # ARI_record = []
    ARI_record = {"interaction": [], "ari": []}
    for i in range(len(intermediate_clusterings)):
        # ARI_record["iter"].append(i)
        ari = adjusted_rand_score(
            real_labels, intermediate_clusterings[i])
        ARI_record["interaction"].append(i)
        ARI_record["ari"].append(ari)
        if ari == 1:
            break
        # ARI_record.append(adjusted_rand_score(
        #     real_labels, intermediate_clusterings[i]))
    return ARI_record


def result_to_csv(ARI_record, title):
    df = pd.DataFrame({
        'ARI': ARI_record,
    })
    df.to_csv("../g_output/%s_result.csv" % title)
    print("%s数据集处理完毕" % title)


if __name__ == '__main__':
    datasets = ["balance", "banknote", "breast", "dermatology", "diabetes", "ecoli",
                "glass", "haberman", "ionosphere", "iris", "led", "musk",
                "pima", "seeds", "segment", "soybean", "thyroid", "vehicle",
                "wine", "yeast", "mfeat_karhunen", "mfeat_zernike"]
    budgets = [600, 350, 400, 150, 1000, 150, 300, 400, 350, 100, 1000,
               500, 1000, 200, 2400, 500, 250, 1200, 100, 3500, 1200, 2500]
    datasets = ["sonar", "fertility", "plrx", "zoo", "tae"]
    budgets = [250, 100, 300, 75, 400]

    for i in range(len(datasets)):
        dataset = datasets[i]
        budget = budgets[i]
        func_name = "generate_" + dataset + "_data"
        generate_data_func = getattr(sys.modules[__name__], func_name)
        path = "../../../dataset/small/{}/{}.data".format(dataset, dataset)
        data, real_labels = generate_data_func(path)
        data = data_preprocess(data)
        ARI_record = COBRAS(data, real_labels, budget)
        # result_to_csv(ARI_record, title="{}".format(dataset))
