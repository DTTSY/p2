import numpy as np
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as metrics

def top_5ensemble(t,class_num,nmi,con,ensemble_labels_sets):

    top=5
#    print(len(nmi))
#    print(len(con))
#    print(len(ensemble_labels_sets))
    re=[]
    nmi_max=max(nmi)
    con_nmi=[]
    top_index=[]
    
    con_nmi.append(nmi[con.index(max(con))])
    for i in range(top):       
        top_index.append(con.index(max(con)))
        con[con.index(max(con))]=0
    

#    make lib
    lib=ensemble_labels_sets[top_index[0]]
    for ind in range(1,top):
        lib=np.vstack((lib,ensemble_labels_sets[top_index[ind]]))
        
    print(lib.shape)    
    top_nmi=metrics.normalized_max_mutual_info_score(t, ce.cluster_ensembles_CSPAONLY(lib, N_clusters_max=class_num))    
    
    re.append(nmi_max)
    re.extend(con_nmi)
    re.append(top_nmi)
    return re
    



    