"""
Wrapper function for executing several experiments together.
Author: Zhijie Lin
"""
import numpy as np
import ensemble
#import time
import ensemble.weighted_ensemble as weighted
import ensemble.propagation_ensemble as prop
import utils.exp_datasets as exp_data
import evaluation.internal_metrics as im
import ensemble.top_ensemble as nt


_default_constraints_folder = '../data/Constraints/'
_default_result_folder = '../data/libs/'
_default_constraints_postfix = ['constraints_n']
_default_ensemble_performance_folder = '../data/libs//Exp_Results/nmi/'
_default_ensemble_performance_folder_new = '../data/libs/Exp_Results/constrain/'
_default_ensemble_performance_folder_spec = '../data/libs/Exp_Results/top/spec/'
_default_ensemble_performance_folder_cspa = '../data/libs/Exp_Results/top/cspa/'
#changed
_default_cons_types = ['both']
_default_alpha_range = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
_default_prop_alphas_range = [0.1]


def do_ensemble_different_constraints(library_name, scale, cons_types=None, constraints_files_postfix=None,
                                      precomputed_internals=None, additional_postfix=''):
    """
    do weighted ensemble for a library using different constraints

    :param library_name:
    :param scale:
    :param cons_types:
    :param constraints_files_postfix:
    :param precomputed_internals:
    :param additional_postfix:
    :return:
    """
#    dataset_name = library_name.split('_')[0]
    dataset_name = "fasion_mnist"
    library_folder = _default_result_folder + dataset_name + '/'
    class_num = exp_data.dataset[dataset_name]['k']
    d, t = exp_data.dataset[dataset_name]['data']()
    scale_tag = '_scaled' if scale else ''
    weighted_cons_types = _default_cons_types if cons_types is None else cons_types
    files_postfix = _default_constraints_postfix if constraints_files_postfix is None else constraints_files_postfix
#    SI
    if precomputed_internals is not None:
        internals = precomputed_internals
    else:
        internals = im.cal_internal_weights_for_library_as_array(d, library_name)
        
#       ['constraints_quarter_n', 'constraints_half_n', 'constraints_n', 'constraints_onehalf_n',
#                                'constraints_2n']
    for postfix in files_postfix:
        performances = []
#  weighted_cons_types=[both must]
        for cons_type in weighted_cons_types:

            constraints_file_name = _default_constraints_folder + dataset_name + '_' + postfix + additional_postfix + '.csv'
            # it should be changed to a generalized version later
#   ll_perf =[both(cspa,spe),must(cspa,spe)]
            performance = weighted.do_7th_weighted_ensemble_for_library(library_folder,
                                                                        library_name, class_num, t,
                                                                        constraints_file_name,
                                                                        
                                                                        _default_alpha_range,
                                                                        internals,
                                                                        cons_type=cons_type, scale=scale)
            performances.append(np.array(performance))
        all_perf = np.hstack(performances)
        np.savetxt(_default_ensemble_performance_folder + dataset_name + '_' + postfix + scale_tag + additional_postfix + '.csv', all_perf,
                   fmt='%.6f', delimiter=',')
    return


def do_ensemble_different_constraints_new_exp(library_name, scale, cons_types=None, constraints_files_postfix=None,
                                              internal=True,
                                              precomputed_internals=None, additional_postfix=''):
    """
    do weighted ensemble for a library using different constraints

    :param library_name:
    :param scale:
    :param cons_types:
    :param constraints_files_postfix:
    :param precomputed_internals:
    :param additional_postfix:
    :return:
    """
    dataset_name = library_name.split('_')[0]
    library_folder = _default_result_folder 
    class_num = exp_data.dataset[dataset_name]['k']
    d, t = exp_data.dataset[dataset_name]['data']()
    scale_tag = '_scaled' if scale else ''
    weighted_cons_types = _default_cons_types if cons_types is None else cons_types
    files_postfix = _default_constraints_postfix if constraints_files_postfix is None else constraints_files_postfix
    if precomputed_internals is not None:
        internals = precomputed_internals
    elif internal:
        internals = im.cal_internal_weights_for_library_cluster_as_array(d, library_name)
    else:
        internals = None
    for postfix in files_postfix:
        real_performances = []
        constrain_performances=[]
        ensemble_labels_sets=[]
        
        for cons_type in weighted_cons_types:
            constraints_file_name = _default_constraints_folder + dataset_name + '_' + postfix + additional_postfix + '.txt'

            real_performance,constrain_performance,ensemble_labels_set = weighted.do_new_weighted_ensemble_for_library(library_folder,
                                              library_name, class_num, t,
                                                                        constraints_file_name,
                                                                        
                                                                        _default_alpha_range,
                                                                        internals=internals
                                                                        )
            
                                                                                                                                  
                                                   
#            
            real_performances.extend(np.array(real_performance))
            constrain_performances.extend(np.array(constrain_performance))
            ensemble_labels_sets.extend(ensemble_labels_set)
        
#        np.savetxt(_default_ensemble_performance_folder_new + dataset_name + '_' + postfix + scale_tag + #additional_postfix + '.csv', constrain_performances,
 #                  fmt='%.6f', delimiter=',')    
        
  #      np.savetxt(_default_ensemble_performance_folder + dataset_name + '_' + postfix + scale_tag + #additional_postfix + '.csv', real_performances,
 #                  fmt='%.6f', delimiter=',')  
    return real_performances,constrain_performances


def do_prop_ensemble_different_constraints(library_name, constraints_files_postfix=None
                                           , additional_postfix=''):
    """
    do weighted ensemble for a library using different constraints

    :param library_name:
    :param scale:
    :param cons_types:
    :param constraints_files_postfix:
    :param precomputed_internals:
    :param additional_postfix:
    :return:
    """
    dataset_name = library_name.split('_')[0]
    library_folder = _default_result_folder + dataset_name + '/'
    class_num = exp_data.dataset[dataset_name]['k']
    d, t = exp_data.dataset[dataset_name]['data']()

    files_postfix = _default_constraints_postfix if constraints_files_postfix is None else constraints_files_postfix

    for postfix in files_postfix:
        performances = []

        constraints_file_name = _default_constraints_folder + dataset_name + '_' + postfix + additional_postfix + '.txt'
        performance = prop.do_propagation_ensemble(library_folder,
                                                   library_name, class_num, t,
                                                   constraints_file_name,
                                                   
                                                   _default_prop_alphas_range)
        performances.append(np.array(performance))
        all_perf = np.hstack(performances)
        np.savetxt(_default_ensemble_performance_folder + dataset_name + '_' + postfix + '_' + additional_postfix +  '_prop.csv', all_perf,
                   fmt='%.6f', delimiter=',')
    return
