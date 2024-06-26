import sys

import pandas as pd
from a import ADPE


def result_to_csv(ARI_record, title):
    # 将所有的交互到加入到里面去`
    record = []
    for i in range(len(ARI_record)):
        if i < len(ARI_record)-1:
            repeat = ARI_record[i+1][0]["interaction"] - \
                ARI_record[i][0]["interaction"]
            for j in range(repeat):
                record.append(ARI_record[i][0]["ari"])
        else:
            record.append(ARI_record[i][0]["ari"])
    # 写入文档
    df = pd.DataFrame({
        'ARI': record,
    })
    df.to_csv("g_output/%s_result.csv" % title)


if __name__ == '__main__':

    datasets = ["balance", "banknote", "breast", "dermatology", "diabetes", "ecoli", "glass", "haberman", "ionosphere", "iris", "led", "musk",
                "pima", "seeds", "segment", "soybean", "thyroid", "vehicle",
                "wine", "mfeat_karhunen", "mfeat_zernike"]

    datasets = ["sonar", "fertility", "plrx", "zoo", "tae"]
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
    for dataset in datasets:
        func_name = "generate_" + dataset + "_data"
        generate_data_func = getattr(sys.modules[__name__], func_name)
        path = "../../dataset/small/{}/{}.data".format(dataset, dataset)
        data, labels = generate_data_func(path)
        # 对数据进行随机扰动
        # data = data_preprocess(data)
        ARI_record = ADPE(data, labels, xi, alpha, s, l, theta)
        result_to_csv(ARI_record, title="{}".format(dataset))
