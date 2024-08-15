from active_semi_clustering import COPKMeans
from sklearn.metrics import adjusted_rand_score


def runCOPKMEANS(data, lable, K):
    df = {'iter': [], 'interaction': [], 'ari': [], 'time': []}

    COPKMeans(n_clusters=K).fit(
        data, lable, ml=ml, cl=cl)
    cluster = COPKMeans(n_clusters=K).fit(
        data, lable, ml=ml, cl=cl)
    # cluster = MPCKMeans(n_clusters=k).fit(data, lable, ml=ml, cl=cl)
    te = time.perf_counter()

    ari = adjusted_rand_score(lable, cluster.labels_)
    df['iter'].append(i)
    df['interaction'].append(math.ceil((len(lable)-1)*(i*.1)))
    df['ari'].append(ari)
    df['time'].append(te-ts)
