import numpy as np
import itertools
import time
import sys

import pandas as pd
from future.builtins import input
from sklearn import cluster, metrics


class COBRA:

    def __init__(self, n_super_instances):
        self.n_super_instances = n_super_instances

    def cluster(self, data, true_clustering, train_indices):
        if self.n_super_instances > data.shape[0]:
            self.n_super_instances = data.shape[0]

        self.data = data

        start = time.time()

        # first over-cluster the data into super-instances
        km = cluster.KMeans(self.n_super_instances, n_init="auto")
        km.fit(self.data)
        pred = km.labels_.astype(np.int64)
        clustering = self.build_structures_from_clustering(
            pred, train_indices)  # each cluster will contain one super-instance
        # superinstances without training points are merged with the closest superinstance that does contain a training point
        cluster_without_train_pts = []
        for c in clustering.clusters:
            cluster_pts = c.get_all_points()
            found_train = False
            for pt in cluster_pts:
                if pt in train_indices:
                    found_train = True
                    break
            if not found_train:
                # find nearest super-instance that does contain at least one training instance, and merge these two
                cluster_without_train_pts.append(c)

        for c in cluster_without_train_pts:
            closest_cluster = None
            closest_dist = np.inf

            for c2 in clustering.clusters:
                if c2 in cluster_without_train_pts:
                    continue
                cur_dist = c.super_instances[0].distance_to_all_points(
                    c2.super_instances[0])
                if cur_dist < closest_dist:
                    closest_dist = cur_dist
                    closest_cluster = c2

            closest_cluster.super_instances[0].indices.extend(
                c.super_instances[0].indices)

        for c in cluster_without_train_pts:
            clustering.clusters.remove(c)
        ml, cl, clusterings, runtimes, interactions = clustering.merge_containing_clusters(
            true_clustering, start)
        return clusterings, runtimes, ml, cl, interactions

    def build_structures_from_clustering(self, pred, train_indices):
        clustering = []

        for cluster_label in set(pred):
            new_super_instance = SuperInstance(self.data,
                                               [i for i, cur_label in enumerate(pred) if cur_label == cluster_label], train_indices)
            new_cluster = Cluster([new_super_instance], cluster_label)
            clustering.append(new_cluster)

        return Clustering(clustering)


class SuperInstance:

    def __init__(self, data, indices, train_indices):
        self.data = data
        self.indices = indices
        self.train_indices = [x for x in indices if x in train_indices]
        self.centroid = np.mean(data[self.train_indices, :], axis=0)

    def get_medoid(self):
        try:
            return min(self.train_indices, key=lambda x: np.linalg.norm(self.data[x, :] - self.centroid))
        except:
            raise ValueError('Super instances without training instances')

    def distance_to(self, other_cluster):
        return np.linalg.norm(self.centroid - other_cluster.centroid)

    def distance_to_all_points(self, other_si):
        min_dist = np.inf
        for idx1 in self.indices:
            for idx2 in other_si.indices:
                cur_dist = np.linalg.norm(
                    self.data[idx1, :] - self.data[idx2, :])
                if cur_dist < min_dist:
                    min_dist = cur_dist
        return min_dist


class Cluster:

    def __init__(self, super_instances, origin):
        self.super_instances = super_instances
        self.origin = origin

    def distance_to(self, other_cluster):
        super_instance_pairs = itertools.product(
            self.super_instances, other_cluster.super_instances)
        return min([x[0].distance_to(x[1]) for x in super_instance_pairs])

    def get_comparison_points(self, other_cluster):
        super_instance_pairs = itertools.product(
            self.super_instances, other_cluster.super_instances)
        bc1, bc2 = min(super_instance_pairs,
                       key=lambda p: p[0].distance_to(p[1]))
        return bc1.get_medoid(), bc2.get_medoid()

    def get_all_points(self):
        all_pts = []
        for super_instance in self.super_instances:
            all_pts.extend(super_instance.indices)
        return all_pts


def cannot_link_between_clusters(c1, c2, cl):
    c1_pts = c1.get_all_points()
    c2_pts = c2.get_all_points()

    for c in cl:
        if (c[0] in c1_pts and c[1] in c2_pts) or (c[0] in c2_pts and c[1] in c1_pts):
            return True
    return False


class Clustering:

    def __init__(self, clusters):
        self.clusters = clusters

    def merge_containing_clusters(self, true_clust, start):
        ari = metrics.adjusted_rand_score(
            self.construct_cluster_labeling(), true_clust)
        clusterings = [ari]
        runtimes = [time.time() - start]
        ml = []
        cl = []
        q_asked = 0
        merged = True
        iteractions = [0]
        while merged:
            cluster_pairs = itertools.combinations(self.clusters, 2)
            cluster_pairs = [
                x for x in cluster_pairs if not cannot_link_between_clusters(x[0], x[1], cl)]
            cluster_pairs = sorted(
                cluster_pairs, key=lambda x: x[0].distance_to(x[1]))
            merged = False
            for x, y in cluster_pairs:
                pt1_to_ask, pt2_to_ask = x.get_comparison_points(y)
                pt1 = min([pt1_to_ask, pt2_to_ask])
                pt2 = max([pt1_to_ask, pt2_to_ask])
                q_asked += 1
                if true_clust is None:
                    same_cluster = _query_yes_no(
                        "Should the following instances be in the same cluster?  " + str(pt1) + " and " + str(pt2))
                else:
                    same_cluster = true_clust[pt1] == true_clust[pt2]
                if same_cluster:
                    x.super_instances.extend(y.super_instances)
                    self.clusters.remove(y)
                    ml.append((pt1, pt2))
                    merged = True
                    ari = metrics.adjusted_rand_score(
                        self.construct_cluster_labeling(), true_clust)
                    clusterings.append(ari)
                    # clusterings.append(self.construct_cluster_labeling())
                    runtimes.append(time.time() - start)
                    iteractions.append(q_asked)
                    break
                else:
                    cl.append((pt1, pt2))
                    ari = metrics.adjusted_rand_score(
                        self.construct_cluster_labeling(), true_clust)
                    clusterings.append(ari)
                    # clusterings.append(self.construct_cluster_labeling())
                    runtimes.append(time.time() - start)
                    iteractions.append(q_asked)
        return ml, cl, clusterings, runtimes, iteractions

    def construct_cluster_labeling(self):
        pts_per_cluster = [cluster.get_all_points()
                           for cluster in self.clusters]
        pred = [-1] * sum([len(x) for x in pts_per_cluster])
        for i, pts in enumerate(pts_per_cluster):
            for pt in pts:
                pred[pt] = i
        return pred


def _query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    Taken from: http://code.activestate.com/recipes/577058/
    """

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

# 这个函数只能输出一列


def result_to_csv(ARI_record, title):
    df = pd.DataFrame({
        'ARI': ARI_record,
    })
    df.to_csv("../g_output/%s_result.csv" % title)


def experiment(data, real_labels, title):
    # 初始值设为50
    clusterer = COBRA(50)
    clusterings, runtimes, ml, cl, interactions = clusterer.cluster(
        data, real_labels, range(data.shape[0]))
    ARI_record_percent50 = clusterings
    result_to_csv(ARI_record_percent50, title)


if __name__ == "__main__":
    datasets = ["balance", "banknote", "breast", "dermatology", "diabetes", "ecoli",
                "glass", "haberman", "ionosphere", "iris", "led", "musk",
                "pima", "seeds", "segment", "soybean", "thyroid", "vehicle",
                "wine", "yeast", "mfeat_karhunen", "mfeat_zernike"]
    datasets = ["sonar", "fertility", "plrx", "zoo", "tae"]
    for dataset in datasets:
        func_name = "generate_" + dataset + "_data"
        generate_data_func = getattr(sys.modules[__name__], func_name)
        path = "../../../dataset/small/{}/{}.data".format(dataset, dataset)
        data, real_labels = generate_data_func(path)
        experiment(data, real_labels, title="{}".format(dataset))
        print(dataset)
