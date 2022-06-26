import torch
import emp_conv_module

def entropy_calculation(uncertainty_sample_selection_probabilities):

	entropy = -1*torch.sum(uncertainty_sample_selection_probabilities*torch.log(uncertainty_sample_selection_probabilities),dim=1)
	return entropy

def get_classification_accuracy(output,target):
	'''
		This function returns the number of correct predictions and the total number of predictions
	'''
	confidence_threshold = 0.7

	_ , output_idx = torch.topk((output>confidence_threshold).type(torch.int),k=1,dim=1)
	output_idx = output_idx.squeeze(1)
	
	correct = (output_idx==target).sum().item()

	total = int(target.shape[0])

	return int(correct),total

def compute_model_accuracy(validation_dataloader,model,model_type,device):
	'''
		This function calculates the validation accuracy and loss of the provided model on the given dataset
	'''
	model.eval()

	loss_criterion = torch.nn.BCEWithLogitsLoss()

	with torch.no_grad():

		correct = 0.
		total = 0.
		total_loss = 0.
		count = 0

		for batch_idx,(data,target) in enumerate(validation_dataloader):
			data = data.to(device)
			target = target.to(device)

			if model_type ==  "Linear" or model_type == "qnn_Linear":
				data = data.reshape(data.shape[0],-1)

			output,_ = model(data)

			loss = loss_criterion(output,target)
			batch_correct,batch_total = get_classification_accuracy(output,target)

			correct += batch_correct
			total += batch_total
			
			total_loss = (count*total_loss + loss.item())/(count+1)
			count += 1

		accuracy = (correct/total)*100

		return accuracy,total_loss

def convert_index_to_target(target_indices,num_class):

	targets = torch.zeros(target_indices.shape[0],num_class)

	for i in range(target_indices.shape[0]):
		targets[i,target_indices[i]] = 1

	return targets