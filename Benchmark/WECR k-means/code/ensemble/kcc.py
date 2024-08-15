import scipy
import io
import numpy as np
import exp_datasets
import sklearn.cluster
from sklearn.cluster import KMeans
from Metrics import normalized_max_mutual_info_score

def create_membership_matrix(cluster_run):

    cluster_run = np.asanyarray(cluster_run)

    cluster_run = cluster_run.reshape(cluster_run.size)

    cluster_ids = np.unique(np.compress(np.isfinite(cluster_run), cluster_run))

    indices = np.empty(0, dtype=np.int32)
    indptr = np.zeros(1, dtype=np.int32)

    for elt in cluster_ids:
        indices = np.append(indices, np.where(cluster_run == elt)[0])
        indptr = np.append(indptr, indices.size)

    data = np.ones(indices.size, dtype=int)

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(cluster_ids.size, cluster_run.size))


def build_hypergraph_adjacency(cluster_runs):

    N_runs = cluster_runs.shape[0]

    hypergraph_adjacency = create_membership_matrix(cluster_runs[0])
    for i in range(1, N_runs):
        hypergraph_adjacency = scipy.sparse.vstack([hypergraph_adjacency,
                                                    create_membership_matrix(cluster_runs[i])],
                                                   format='csr')

    return hypergraph_adjacency



lib_list=
['MNIST4000_50-100_0.7_0.9_100_FSRSNC_pure',
'OPTDIGITS_50-100_0.7_0.9_100_FSRSNC_pure',
'Segmentation_35-70_0.5_0.8_100_FSRSNC_pure',
'COIL20_100-200_0.9_0.2_100_FSRSNC_pure',
'ISOLET_130-260_0.6_0.4_100_FSRSNC_pure',
 'SRBCT_20-40_0.6_0.6_100_FSRSNC_pure',
'Prostate_10-20_0.8_0.3_100_FSRSNC_pure',
'LungCancer_20-40_0.9_0.3_100_FSRSNC_pure',
'Leukemia1_15-30_0.7_0.2_100_FSRSNC_pure',
'Leukemia2_15-30_0.6_0.5_100_FSRSNC_pure']

for  lib_name in   lib_list:

    for count in range(1,10):
        name = lib_name.split('_')[0]
        features, labels = exp_datasets.dataset[name]['data']()
        print(name)
        # 根据名字读入，这里读进来的是一个(#members * #instances)的矩阵
        lib = io.read_matrix('./Results/' + name + '/' + lib_name)

        H_matrix = build_hypergraph_adjacency(lib)
  
        Hmatrix =H_matrix .todent()
        sample_num=Hmatrix .shape[0]
        sumlist=[]
        for i in range(Hmatrix.shape[1]):
            Hmatrix[:,i]=Hmatrix[:,i]-np.sum(Hmatrix[:,i])/sample_num

        ensemble_kcc=KMeans(Hmatrix).labels_
        ensemble_kmeans=KMeans(features).labels_

        nmi_kcc=normalized_max_mutual_info_score(labels,ensemble_kcc)
        nmi_kcmeans=normalized_max_mutual_info_score(labels,ensemble_kmeans)

        wb = openpyxl.load_workbook( 'M2/'+'L'+str(count)+'/Incre_con_fix.xlsx')
        sheet = wb.get_sheet_by_name(name)


        sheet['B''+str(158)] = nmi_kcc
        print('L'+str(count))
        sheet['A'+str(158)] = 'KCC'

        sheet['B''+str(159)] = nmi_kcmeans
        print('L'+str(count))
        sheet['A'+str(159)] = 'K-means'

        wb.save('M2/'+'L'+str(count)+'/Incre_con_fix.xlsx')
