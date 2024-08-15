import member_generation.library_generation as lg
import utils.io_func as io_func
import utils.exp_datasets as exd
import evaluation.Metrics as Metrics
import constrained_methods.generate_constraints_link as gcl
import ensemble.ensemble_wrapper as ew
import evaluation.comparision_methods as ed

import numpy as np
from  utils.deal_constrains import deal_constrains
_noise_postfix = ['noise_n_1']


"""
This example set constrain_num=n  dtaset=COIL20
"""
"""
========================================================================================================================
Generate Basic Library
========================================================================================================================
"""
"""
generate basic library
(double-random kmeans)

"""

baseclustering_NUM=100
lg.generate_libs_by_sampling_rate('COIL20', baseclustering_NUM)
"""
========================================================================================================================
Generate Constraints
========================================================================================================================
"""
"""
generate constraints [part1: different amount of constraints]
"""

gcl.generate_diff_amount_constraints_wrapper('COIL20')
deal_constrains(['COIL20'])

"""
generate constraints [part2: constraints with different level of noise constraints]
"""
gcl.generate_noise_constraints_wrapper('COIL20')

"""
===========================================================================================================
new experiments, 12th Oct, 2019.
updates: new formula for calculating weights (g_gamma introduced)
12.27 modified: internals included
===========================================================================================================
"""
"""
experiment
[part1: different amount of constraints]
"""
real_performances,constrain_performances=ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.7_0.7_100_FSRSNC_pure',True)
index=constrain_performances.index(np.max(constrain_performances))

print("============Result============")
print("the best gama=",index*0.1+0.1)
print("the NMI score=",real_performances[index])

"""
experiment
[part2: constraints with different level of noise constraints]
"""

real_performances,constrain_performances=ew.do_ensemble_different_constraints_new_exp('COIL20_100-200_0.7_0.7_100_FSRSNC_pure', True, constraints_files_postfix=_noise_postfix)
index=constrain_performances.index(np.max(constrain_performances))
print("============Result============")
print("the best gama=",index*0.1+0.1)
print("the NMI score=",real_performances[index])
