import numpy as np
import time
import evaluation
import utils.io_func as io_func
import utils.exp_datasets as exd

from ensemble import Cluster_Ensembles as ce
from ensemble import spectral_ensemble as spec
from ensemble import spectral_ensemble as spec
from sklearn import preprocessing
#from evaluation import Metrics

_ensemble_method = {'CSPA': ce.cluster_ensembles_CSPAONLY,
                    'HGPA': ce.cluster_ensembles_HGPAONLY,
                    'MCLA': ce.cluster_ensembles_MCLAONLY,
                    'Spectral': spec.weighted_spectral_ensemble,
                    'Lwgp': spec.weighted_spectral_ensemble,
                    'Lwea': spec.weighted_spectral_ensemble}
                      
#_default_ensemble_method = ['Lwgp', 'Lwea']
_default_ensemble_method = [
#        'CSPA']
                            'Spectral']


def _activate_func(value):
    """
    Activate Function (Sigmoid)
    (internal use only)
    """
    return 1/(1+np.exp(-(np.tan(value*np.pi-(np.pi/2)))))


def _activate_func_2(value):
    """
    Activate Function (Sigmoid)
    (internal use only)
    """
    _LOWER_BOUND = -10
    _UPPER_BOUND = 10
    adjusted_value = _LOWER_BOUND + (value * (_UPPER_BOUND - _LOWER_BOUND))
    return 1/(1+np.exp(adjusted_value))


def do_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file,  alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))

    nmis = []
    for alpha in alphas:
      
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            logger.debug(method + ' alpha=' + str(alpha) + ', NMI=' + str(ensemble_nmi))
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
       
    return nmis


def do_2nd_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                         constraint_file, alphas, cons_type='both',
                                         ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
 

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    k_values = []
    expected_cons = {}

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
        k_values.append(len(np.unique(label)))
    k_values = np.array(k_values, dtype=int)
    possible_k = np.unique(k_values)
    cons = np.array(con_clustering)
    for k in possible_k:
        mean_value = np.mean(cons[k_values == k])
        expected_cons[k] = mean_value
    for i in range(0, labels.shape[0]):
        con_clustering[i] /= expected_cons[k_values[i]]
    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))

    nmis = []
    for alpha in alphas:
       
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        
    return nmis


def do_3rd_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file,  alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func(con_clustering[i])
    else:
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func(con_clustering[i])

    nmis = []
    for alpha in alphas:
       
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
          
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
      
    return nmis


def do_4th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file,  alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
   

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        d = evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type)
        for k in d:
            d[k] = _activate_func(d[k])
        con_per_cluster.append(d)
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))


    nmis = []
    for alpha in alphas:
       
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
       
    return nmis


def do_5th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file,alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
   

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        con_per_cluster.append(evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func_2(con_clustering[i])
    else:
        for i in range(0, len(con_clustering)):
            con_clustering[i] = _activate_func_2(con_clustering[i])

    nmis = []
    for alpha in alphas:
       
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
       
    return nmis


def do_6th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                     constraint_file,alphas, cons_type='both',
                                     ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    for label in labels:
        d =evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type)
        for k in d:
            d[k] = _activate_func_2(d[k])
        con_per_cluster.append(d)
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))


    nmis = []
    for alpha in alphas:
        
        cur_nmis = []
        for method in ensemble_method:
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
        
    return nmis


def do_7th_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                         constraint_file,  alphas, internals, cons_type='both',
                                        ensemble_method=_default_ensemble_method, scale=False):
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
    
    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)
    k_values = []
    expected_cons = {}

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)

    # get cluster/clustering level weights
    con_per_cluster = []
    con_clustering = []
    

    for label in labels:
        con_per_cluster.append(evaluation.Metrics.consistency_per_cluster(label, mlset, nlset, cons_type=cons_type))
    for label in labels:
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
        k_values.append(len(np.unique(label)))
    k_values = np.array(k_values, dtype=int)
    possible_k = np.unique(k_values)
    cons = np.array(con_clustering)
    for k in possible_k:
        mean_value = np.mean(cons[k_values == k])
        if mean_value == 0:
            mean_value = 1
        expected_cons[k] = mean_value
    for i in range(0, labels.shape[0]):
        con_clustering[i] /= expected_cons[k_values[i]]
        con_clustering[i] *= internals[i]
    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering))

    nmis = []
    for alpha in alphas:
         
        cur_nmis = []
        for method in ensemble_method:
#       
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=alpha)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            
            cur_nmis.append(ensemble_nmi)
        nmis.append(cur_nmis)
       
    return nmis







def _get_g_gamma(cluster_instances, cluster_constraints_i, n_constraints, n_instances, gamma):
    expected_constraints = (float(cluster_instances) / n_instances) * n_constraints
    if cluster_constraints_i <= expected_constraints:
        return cluster_constraints_i / expected_constraints * gamma
    else:
        return gamma + (1 - gamma) * (cluster_constraints_i - expected_constraints) / (n_constraints - expected_constraints)


def get_g_gamma(cluster_constraints, labels, n_constraints, n_instances, gamma):
    g_gammas = []
    for label, cluster_constraints_c in zip(labels, cluster_constraints):
        g_gamma = {}
        clusters = np.unique(label)
        for cluster in clusters:
            g_gamma[cluster] = _get_g_gamma(len(label[label == cluster]),
                                            cluster_constraints_c[cluster],
                                            n_constraints, n_instances, gamma)
        g_gammas.append(g_gamma)
    return g_gammas


def _build_pesudo_internal(labels):
    internal = []
    for label in labels:
        inside = {}
        cluster_ids = np.unique(label)
        for cluster in cluster_ids:
            inside[cluster] = 1
        internal.append(inside)
    return internal


def do_new_weighted_ensemble_for_library(library_folder, library_name, class_num, target,
                                         constraint_file, gammas, internals=None, cons_type='both',
                                         ensemble_method=_default_ensemble_method, scale=False):
                                                
    """

    :param library_folder:
    :param library_name:
    :param class_num:
    :param target:
    :param constraint_file:
    :param logger:
    :param alphas:
    :param cons_type:
    :param ensemble_method
    :return:
    """
#    

    labels = np.loadtxt(library_folder + library_name + '.res', delimiter=',')
    labels = labels.astype(int)

    # if the library is not pure, i.e, ensemble results and targets are also included.
    # then, last 5 rows should be removed (single kmeans, cspa, hgpa, mcla, real labels)
    if 'pure' not in library_name:
        labels = labels[0:-5]
    mlset, nlset = io_func.read_constraints(constraint_file)
    n_instances = labels.shape[1]
    if cons_type == 'both':
        n_constraints = len(mlset) + len(nlset)
    if cons_type == 'cannot':
        n_constraints = len(nlset)
    else:
        n_constraints = len(mlset)
#    if internals is None:
#    t1=time.clock()
    internals = _build_pesudo_internal(labels)
#    t2=time.clock()
#        
#    print("#############################internals##################")
#    print("################internals*100############################")
#    print("#################internals*100#########################")
#    print("##############################internals##############")
#    print(t2-t1)     
    # get cluster/clustering level weights
    # constraints in each cluster of all clusterings are also obtained to get g_gamma
    con_per_cluster = []
    constraints_num = []
    con_clustering = []
#    cluster_time_sum = 0.0
#    clustering_time_sum = 0.0
    
    for label in labels:
#        t1 = time.clock()       
        weight, cluster_cons_num = evaluation.Metrics.consistency_per_cluster_efficient(label, mlset, nlset, cons_type=cons_type)
        con_per_cluster.append(weight)
        constraints_num.append(cluster_cons_num)
#        t2 = time.clock()
#        cluster_time_sum += (t2 - t1)
        
#    print("#############################consistency_per_cluster#################")
#    print("###############consistency_per_cluster#########################")
#    print("################consistency_per_cluster#######################")
#    print("###########################consistency_per_cluster############")
#    print(t2-t1)
    
    for label in labels:
#        t1 = time.clock()
        con_clustering.append(evaluation.Metrics.consistency(label, mlset, nlset, cons_type=cons_type))
#        t2 = time.clock()
#        clustering_time_sum += (t2 - t1)

#    print 'library size='+str(labels.shape[0])
#    print 'cluster avg='+str(cluster_time_sum/labels.shape[0])
#    print 'clustering avg=' + str(clustering_time_sum / labels.shape[0])

    if scale:
        scaler = preprocessing.MinMaxScaler()
        con_clustering = scaler.fit_transform(np.array(con_clustering).reshape(-1, 1))
        
#    if add_consrain:
#        

    nmis = []
    cnmis=[]
    labels_set=[]
    for gamma in gammas:
        
        cur_g_gamma = get_g_gamma(constraints_num, labels, n_constraints, n_instances, gamma)
        cur_nmis = []
        constrain_nmis=[]
        label_set=[]
        for method in ensemble_method:
#            ggama
#            t1=time.clock()
            ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
                                                       weighted=True, clustering_weights=con_clustering,
                                                       cluster_level_weights=con_per_cluster, alpha=cur_g_gamma,
                                                       new_formula=True,internal=internals)
#            t2=time.clock()
#            print(len(np.unique(ensemble_labels)))
#            print(len(np.unique(target)))
            
            # ensemble_labels = _ensemble_method[method](labels, N_clusters_max=class_num,
            #                                            weighted=True, clustering_weights=con_clustering,
            #                                            cluster_level_weights=con_per_cluster, alpha=cur_g_gamma,
            #                                            new_formula=True, internal=internals, ml=mlset, cl=nlset)
#            print("lable")
#            print(ensemble_labels)
            ensemble_nmi = evaluation.Metrics.normalized_max_mutual_info_score(ensemble_labels, target)
            constrain_nmi=evaluation.Metrics.constrain_score(library_name,ensemble_labels,mlset, nlset,cur_g_gamma)
            
            cur_nmis.append(ensemble_nmi)
            constrain_nmis.append(constrain_nmi)
            label_set.append(ensemble_labels)
        nmis.append(cur_nmis)
        cnmis.append(constrain_nmis)
        labels_set.append(label_set)
        
    return nmis,cnmis,labels_set