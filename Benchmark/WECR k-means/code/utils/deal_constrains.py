import numpy as np
import utils.io_func as io_func

_default_dataset = ['semeion']

def deal_constrains(dataset_list):

    for dataset_name in dataset_list:
#        mlset, nlset = io_func.read_constraints('../data/Constraints/'+dataset_name+'_constraints_2n.txt')
    
        
        mlset, nlset = io_func.read_constraints('../data/Constraints/'+ dataset_name+'_constraints_2n.txt')
        io_func.store_constraints('../data/Constraints//' + dataset_name + '_constraints_n.txt',mlset[0:int(0.5*mlset.shape[0]),:],nlset[0:int(0.5*mlset.shape[0]),:])
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_constraints_onehalf_n.txt',mlset[0:int(0.75*mlset.shape[0]),:],nlset[0:int(0.75*mlset.shape[0]),:])
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_constraints_half_n.txt',mlset[0:int(0.25*mlset.shape[0]),:],nlset[0:int(0.25*mlset.shape[0]),:])
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_constraints_quarter_n.txt',mlset[0:int(0.125*mlset.shape[0]),:],nlset[0:int(0.125*mlset.shape[0]),:])
    
 
    
    for dataset_name in dataset_list:
        mlset, nlset = io_func.read_constraints('../data/Constraints/'+dataset_name+'_constraints_n.txt')
    
    #    0%
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_noise_n_1.txt',mlset,nlset)
    #5%    
        num=int(0.05*mlset.shape[0])
        
        must=np.vstack((mlset[0:int(0.95*mlset.shape[0]),:],nlset[0:int(0.05*nlset.shape[0]),:]))
        cannot=np.vstack((nlset[0:int(0.95*nlset.shape[0]),:],mlset[0:int(0.05*mlset.shape[0]),:]))
#        print(must)
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_noise_n_2.txt',must,cannot)
        
    #10%    
        num=int(0.1*mlset.shape[0])
        
        must=np.vstack((mlset[0:int(0.9*mlset.shape[0]),:],nlset[0:int(0.1*nlset.shape[0]),:]))
        cannot=np.vstack((nlset[0:int(0.9*nlset.shape[0]),:],mlset[0:int(0.1*mlset.shape[0]),:]))
        
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_noise_n_3.txt',must,cannot)
        
    #15%    
        num=int(0.15*mlset.shape[0])
        
        must=np.vstack((mlset[0:int(0.85*mlset.shape[0]),:],nlset[0:int(0.15*nlset.shape[0]),:]))
        cannot=np.vstack((nlset[0:int(0.85*nlset.shape[0]),:],mlset[0:int(0.15*mlset.shape[0]),:]))
        
        
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_noise_n_4.txt',must,cannot)
        
    #20%    
        num=int(0.2*mlset.shape[0])
        
        must=np.vstack((mlset[0:int(0.8*mlset.shape[0]),:],nlset[0:int(0.2*nlset.shape[0]),:]))
        cannot=np.vstack((nlset[0:int(0.8*nlset.shape[0]),:],mlset[0:int(0.2*mlset.shape[0]),:]))
        
        io_func.store_constraints('../data/Constraints/' + dataset_name + '_noise_n_5.txt',must,cannot)
        
     