import torch
from tqdm import tqdm

import utils

def validate(dataloader,main_network,main_network_loss_criterion,device,num_classes):
	'''
		This function will validate the accuracy and performance and calculate the loss of the network on the validation set
	'''
	val_pbar = tqdm(enumerate(dataloader))
	val_pbar.set_description("Validation:")

	correct = 0
	total = 0

	main_network.eval()

	with torch.no_grad():
		for batch_idx,(inputs,targets) in val_pbar:
			index_targets = targets

			output = main_network(inputs.to(device))

			targets = utils.convert_index_to_target(targets,num_classes)

			loss = main_network_loss_criterion(output,targets.to(device))

			batch_correct,batch_total = utils.get_classification_accuracy(output,index_targets.to(device))

			correct += batch_correct
			total += batch_total

			val_pbar.set_postfix_str("Main Network Loss: "+str(loss.item()))

		accuracy = (correct/total)*100

		print("\nAccuracy of the model as predicted on the validation dataset = " + str(accuracy) + "\n")

def validate_with_eif_and_emp(dataloader,eif_model,main_network,eif_loss_criterion,main_network_loss_criterion,device,num_classes):
	'''
		This function will validate the accuracy and performance and calculate the loss of the network on the validation set
	'''
	R_th = 0
	TH_count = 0
	correct = 0
	total = 0
	val_pbar = tqdm(enumerate(dataloader))
	val_pbar.set_description("Validation:")

	eif_model.eval()
	main_network.eval()

	with torch.no_grad():
		for batch_idx,(inputs,targets) in val_pbar:

			inputs = inputs.to(device)
			targets = utils.convert_index_to_target(targets,num_classes).to(device)
			eif_targets = torch.zeros(batch_size,2,dtype=torch.float32).to(device)

			# EIF Training
			loss_class_output = EIF(inputs)
			HL_selected_samples = (loss_class_output[:,0] > constants.eif_threshold) # These samples are the high loss samples
			selected_samples = inputs[HL_selected_samples,:,:,:]

			# Uncertainty Sampling
			LL_selected_samples = utils.entropy_calculation(loss_class_output[:,1]) > constants.entropy_threshold # These are the samples which are selected as part of uncertainity sampling
			uncertain_samples = inputs[LL_selected_samples,:,:,:]

			# Main Network Training
			output = main_network(selected_samples)

			loss = main_network_loss_criterion(output,targets[HL_selected_samples,:,:,:])

			eif_target_indices = (loss > constants.adaptive_threshold)

			TH_count += torch.sum(eif_target_indices)

			R_th = (TH_count/((batch_idx+1)*batch_size))

			# Uncertainty samples
			uncertain_output = main_network(uncertain_samples)
			uncertainty_loss = main_network_loss_criterion(uncertain_output,targets[LL_selected_samples,:,:,:])
			uncertain_sample_indices = (uncertainty_loss > constants.adaptive_threshold)

			# EIF Tragets for training

			# Among the samples selected as high loss class, some are true high classes and others are false high, the same needs to be learnt by the network
			# Hence the targets are being made to help the network learn the same, and those which aren't classified as high are automatically classified as low
			eif_targets[HL_selected_samples,0][eif_target_indices] = 1
			eif_targets[HL_selected_samples,1][~eif_target_indices] = 1
			eif_targets[~HL_selected_samples,1] = 1

			# Among the samples classified as low, it is possible that the network miss classified some instances as low loss class without much confidence
			# These samples after thier loss is evalueted by the main network will be either given as high or low as per the adaptive threshold so that the network
			# may learn and predict the correct class for such samples in the future
			eif_targets[LL_selected_samples,0][uncertain_sample_indices] = 1
			eif_targets[LL_selected_samples,1][uncertain_sample_indices] = 0

			# EIF Loss Calculation:
			eif_loss = torch.sum(wH*eif_loss_criterion(eif_targets[HL_selected_samples,:][eif_target_indices],loss_class_output[HL_selected_samples,:][eif_target_indices]))
			eif_loss += torch.sum(wL*eif_loss_criterion(eif_targets[HL_selected_samples,:][~eif_target_indices],loss_class_output[HL_selected_samples,:][~eif_target_indices]))
			eif_loss += torch.sum(wH*eif_loss_criterion(eif_targets[LL_selected_samples,:][uncertain_sample_indices],loss_class_output[LL_selected_samples,:][uncertain_sample_indices]))
			eif_loss += torch.sum(wL*(torch.sum(eif_loss_criterion(eif_targets[~HL_selected_samples,:],loss_class_output[~HL_selected_samples,:])) - 
			torch.sum(eif_loss_criterion(eif_targets[LL_selected_samples,:][uncertain_sample_indices],loss_class_output[LL_selected_samples,:][uncertain_sample_indices]))))

			val_pbar.set_postfix_str("Main Network Loss: "+str(loss.item())+" EIF Loss: "+str(eif_loss.item())+" R_th: "+str(R_th)+ " Adaptive Threshold: "+str(constants.adaptive_threshold))

			batch_correct,batch_total = utils.get_classification_accuracy(output,target.to(device))

			correct += batch_correct
			total += batch_total

		accuracy = (correct/total)*100

		print("\nAccuracy of the model as predicted on the validation dataset = " + str(accuracy) + "\n")
