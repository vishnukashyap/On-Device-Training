'''
	This file contains the constants to be used during the process of training
'''

# Parameters for Error Map Pruning

error_map_pruning_ratio = 0.9 # Precent of channels to be retained after pruning
error_map_pruning_gamma_1 = 1 # Scaling factor for weights during importance score calculation
error_map_pruning_gamma_2 = 2 # Scaling factor for output gradient during importance score calculation

# Parameters for Early Instance Filtering

R_set = 0.10 # Precent of instances to be selected for backpropagation
adaptive_threshold = 0.2 # The adaptive threshold value for the loss
eif_threshold = 0.3 # Threshold for the confidence of eif prediction of the class of the loss
entropy_threshold = 0.5 # Threshold value for uncertainity sampling
loss_threshold_aplha_1 = 1.005 # The loss_threshold step for adaptive threshold for the case when R_th < R_set
loss_threshold_aplha_2 = 0.995 # The loss_threshold step for adaptive threshold for the case when R_th > R_set