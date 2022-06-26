import torch
import torchvision
import yaml
import datetime
import os
from tqdm import tqdm
import argparse

import models
import constants
import utils
import validation

parser = argparse.ArgumentParser()
parser.add_argument("--training_type",type=str,help="Type of training to be done")

def train_standard_method(dataset_name,dataset_dir,batch_size,epochs,lr,momentum,optimizer_type,lr_scheduler,train_val_split,num_workers,weight_decay,device,train_tensorboard,val_tensorboard,load_checkpoints=None,checkpoint_path=None,checkpoint_save_interval=1):
	'''
		This function implements the standard methodology used during training
	'''

	# Dataset Loader
	if dataset_name == 'CIFAR10':
		normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
		train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
		dataset = torchvision.datasets.CIFAR10(dataset_dir,train=True,download=True,transform=train_transform)
		indices = len(dataset)
		num_classes = 10

	elif dataset_name == 'CIFAR100':
		normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],std=[0.2009, 0.1984, 0.2023])
		train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
		dataset = torchvision.datasets.CIFAR100(dataset_dir,train=True,download=True,transform=train_transform)
		indices = len(dataset)
		num_classes = 100

	train_indices = range(int(indices*train_val_split))
	train_data_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
	train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=train_data_sampler)

	val_indices = range(int(indices*train_val_split),indices)
	val_data_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
	val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=val_data_sampler)

	# Model Initialization

	# Main Network

	main_network = models.ResNet34(num_classes).to(device)
	if load_checkpoints['main_network_checkpoint_path'] is not None:
		main_network_checkpoint = torch.load(load_checkpoints['main_network_checkpoint_path'])
		if not main_network.FullyConnected.weight.shape == main_network_checkpoint['main_network']['FullyConnected.weight'].shape:
			main_network_checkpoint['main_network']['FullyConnected.weight'] = main_network.FullyConnected.weight
		main_network.load_state_dict(main_network_checkpoint['main_network'])
	main_network.train()

	# Loss Criterion Initialization
	main_network_loss_criterion = torch.nn.BCEWithLogitsLoss()

	# Optimizer Initialization
	if optimizer_type == "SGD":
		main_network_optimizer = torch.optim.SGD(main_network.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
	elif optimizer_type == "Adam":
		main_network_optimizer = torch.optim.Adam(main_network.parameters(),lr=lr,weight_decay=weight_decay)
	elif optimizer_type == "NAdam":
		main_network_optimizer = torch.optim.NAdam(main_network.parameters(),lr=lr,weight_decay=weight_decay)

	# Starting Model Training

	for epoch in range(epochs):
		pbar = tqdm(enumerate(train_dataloader))
		pbar.set_description("Training:")
		for batch_idx,(inputs,targets) in pbar:

			targets = utils.convert_index_to_target(targets,num_classes)

			output = main_network(inputs.to(device))

			loss = main_network_loss_criterion(output,targets.to(device))

			main_network_optimizer.zero_grad()
			loss.backward()
			main_network_optimizer.step()

			pbar_string = "Training:: Epoch:"+str(epoch)+"  Main Network Loss: "+str(loss.item())
			pbar.set_description(pbar_string)

		validation.validate(val_dataloader,main_network,main_network_loss_criterion,device,num_classes)

		if epoch%checkpoint_save_interval==0:

			current_timestamp = str(datetime.datetime.now()).split(" ")
			checkpoint_filename = "Standard_Epoch_"+str(epoch)+"_"+current_timestamp[1][:5].split(".")[0].replace(":","_")+"_"+current_timestamp[0].replace("-","_")+".pth"
			checkpoint_filepath = os.path.join(checkpoint_path,checkpoint_filename)

			checkpoint = {}
			checkpoint["main_network"] = main_network.state_dict()
			checkpoint["main_network_optimizer"] = main_network_optimizer.state_dict()
			checkpoint["epoch"] = epoch

			torch.save(checkpoint,checkpoint_filepath)
			print("---------------------Checkpoint Saved---------------------")

def train_using_eif_and_emp(dataset_name,dataset_dir,batch_size,epochs,lr,momentum,optimizer_type,lr_scheduler,train_val_split,num_workers,weight_decay,device,train_tensorboard,val_tensorboard,load_checkpoints=None,checkpoint_path=None,checkpoint_save_interval=1):
	'''
		This function implements:
			- The self supervised training code for early instance filtering 
			- The training code for the main network which has error map pruning modules
	'''

	# Dataset Loader
	if dataset_name == 'CIFAR10':
		normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
		train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
		dataset = torchvision.datasets.CIFAR10(dataset_dir,train=True,download=True,transform=train_transform)
		indices = len(dataset)
		num_classes = 10

	elif dataset_name == 'CIFAR100':
		normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],std=[0.2009, 0.1984, 0.2023])
		train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
		dataset = torchvision.datasets.CIFAR100(dataset_dir,train=True,download=True,transform=train_transform)
		indices = len(dataset)
		num_classes = 100

	train_indices = range(int(indices*train_val_split))
	train_data_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
	train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=train_data_sampler)

	val_indices = range(int(indices*train_val_split),indices)
	val_data_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
	val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=val_data_sampler)

	# Model Initialization

	# Early Filter Initialization
	EIF = models.ResNet8().to(device)
	if load_checkpoints['eif_checkpoint_path'] is not None:
		eif_checkpoint = torch.load(load_checkpoints['eif_checkpoint_path'])
		EIF.load_state_dict(eif_checkpoint['eif_network'])

	# Main Network
	main_network = models.ResNet34_with_EMP(num_classes).to(device)

	if load_checkpoints['main_network_checkpoint_path'] is not None:		
		main_network_checkpoint = torch.load(load_checkpoints['main_network_checkpoint_path'])
		if not main_network.FullyConnected.weight.shape == main_network_checkpoint['main_network']['FullyConnected.weight'].shape:
			main_network_checkpoint['main_network']['FullyConnected.weight'] = main_network.FullyConnected.weight
			main_network_checkpoint['main_network']['FullyConnected.bias'] = main_network.FullyConnected.bias
		main_network.load_state_dict(main_network_checkpoint['main_network'])
	main_network.train()
	EIF.train()

	# Loss Criterion Initialization
	eif_loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
	main_network_loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

	# Optimizer Initialization
	if optimizer_type == "SGD":
		eif_optimizer = torch.optim.SGD(EIF.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
		main_network_optimizer = torch.optim.SGD(main_network.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
	elif optimizer_type == "Adam":
		eif_optimizer = torch.optim.Adam(EIF.parameters(),lr=lr,weight_decay=weight_decay)
		main_network_optimizer = torch.optim.Adam(main_network.parameters(),lr=lr,weight_decay=weight_decay)
	elif optimizer_type == "NAdam":
		eif_optimizer = torch.optim.NAdam(EIF.parameters(),lr=lr,weight_decay=weight_decay)
		main_network_optimizer = torch.optim.NAdam(main_network.parameters(),lr=lr,weight_decay=weight_decay)

	# Parameters
	wH = 1/constants.R_set
	wL = 1/(1 - constants.R_set)

	# Starting Model Training

	for epoch in range(epochs):

		R_th = 0
		TH_count = 0

		pbar = tqdm(enumerate(train_dataloader))
		pbar.set_description("Training:")

		for batch_idx,(inputs,targets) in pbar:

			inputs = inputs.to(device)
			targets = utils.convert_index_to_target(targets,num_classes).to(device)
			eif_targets = torch.zeros(batch_size,2,dtype=torch.float32).to(device)

			# EIF Training
			loss_class_output = EIF(inputs)
			HL_selected_samples = (loss_class_output[:,0].sigmoid() > constants.eif_threshold) # These samples are the high loss samples
			selected_samples = inputs[HL_selected_samples,:,:,:]
			num_selected_samples = selected_samples.shape[0]

			# Uncertainty Sampling
			LL_selected_samples = utils.entropy_calculation(loss_class_output[:,1].unsqueeze(-1).sigmoid()) > constants.entropy_threshold # These are the samples which are selected as part of uncertainity sampling
			uncertain_samples = inputs[LL_selected_samples,:,:,:]

			if num_selected_samples > 0:
				# Main Network Training
				output = main_network(selected_samples)

				loss = main_network_loss_criterion(output,targets[HL_selected_samples,:])

				eif_target_indices = torch.sum(loss,dim=1)>constants.adaptive_threshold

				TH_count += torch.sum(eif_target_indices)

				R_th = (TH_count/((batch_idx+1)*batch_size))

				if (R_th >= constants.R_set):
					constants.adaptive_threshold = constants.loss_threshold_aplha_1*constants.adaptive_threshold
				else:
					constants.adaptive_threshold = constants.loss_threshold_aplha_2*constants.adaptive_threshold

			# Uncertainty samples
			uncertain_output = main_network(uncertain_samples)
			uncertainty_loss = main_network_loss_criterion(uncertain_output,targets[LL_selected_samples,:])
			uncertain_sample_indices = torch.sum((uncertainty_loss > constants.adaptive_threshold),dim=1)

			# EIF Tragets for training

			if num_selected_samples > 0:
				# Among the samples selected as high loss class, some are true high classes and others are false high, the same needs to be learnt by the network
				# Hence the targets are being made to help the network learn the same, and those which aren't classified as high are automatically classified as low
				eif_targets[HL_selected_samples,0][eif_target_indices] = 1
				eif_targets[HL_selected_samples,1] = 1
				eif_targets[HL_selected_samples,1][eif_target_indices] = 0
				eif_targets[~HL_selected_samples,1] = 1
			elif num_selected_samples==0:
				eif_targets[:,1] = 1

			# Among the samples classified as low, it is possible that the network miss classified some instances as low loss class without much confidence
			# These samples after thier loss is evalueted by the main network will be either given as high or low as per the adaptive threshold so that the network
			# may learn and predict the correct class for such samples in the future
			eif_targets[LL_selected_samples,0][uncertain_sample_indices] = 1
			eif_targets[LL_selected_samples,1][uncertain_sample_indices] = 0

			# EIF Loss Calculation:
			if num_selected_samples > 0:
				eif_loss = torch.sum(wH*eif_loss_criterion(eif_targets[HL_selected_samples,:][eif_target_indices],loss_class_output[HL_selected_samples,:][eif_target_indices]))
				eif_loss += torch.sum(wL*eif_loss_criterion(eif_targets[HL_selected_samples,:][~eif_target_indices],loss_class_output[HL_selected_samples,:][~eif_target_indices]))
				eif_loss += torch.sum(wH*eif_loss_criterion(eif_targets[LL_selected_samples,:][uncertain_sample_indices],loss_class_output[LL_selected_samples,:][uncertain_sample_indices]))
				eif_loss += torch.sum(wL*(torch.sum(eif_loss_criterion(eif_targets,loss_class_output)) - 
					torch.sum(eif_loss_criterion(eif_targets[LL_selected_samples,:][uncertain_sample_indices],loss_class_output[LL_selected_samples,:][uncertain_sample_indices]))))
			elif num_selected_samples == 0:
				eif_loss = 0
				eif_loss += torch.sum(wH*eif_loss_criterion(eif_targets[LL_selected_samples,:][uncertain_sample_indices],loss_class_output[LL_selected_samples,:][uncertain_sample_indices]))
				eif_loss += torch.sum(wL*(torch.sum(eif_loss_criterion(eif_targets,loss_class_output)) - 
					torch.sum(eif_loss_criterion(eif_targets[LL_selected_samples,:][uncertain_sample_indices],loss_class_output[LL_selected_samples,:][uncertain_sample_indices]))))

			# Back Propagation
			eif_optimizer.zero_grad()
			main_network_optimizer.zero_grad()

			torch.mean(eif_loss).backward()
			torch.mean(loss).backward()

			eif_optimizer.step()
			main_network_optimizer.step()

			pbar_string = "Training:: Epoch: "+str(epoch)+" Main Network Loss: "+str(torch.mean(loss).item())+" EIF Loss: "+str(torch.mean(eif_loss).item())+" R_th: "+str(R_th)+ " Adaptive Threshold: "+str(constants.adaptive_threshold)
			pbar.set_description(pbar_string)

		validation.validate_with_eif_and_emp(val_dataloader,EIF,main_network,eif_loss_criterion,main_network_loss_criterion,device,num_classes)

		# Save Checkpoint
		if epoch%checkpoint_save_interval==0:

			current_timestamp = str(datetime.datetime.now()).split(" ")
			checkpoint_filename = "EIF_EMP_Epoch_"+str(epoch)+"_"+current_timestamp[1][:5].split(".")[0].replace(":","_")+"_"+current_timestamp[0].replace("-","_")+".pth"
			checkpoint_filepath = os.path.join(checkpoint_path,checkpoint_filename)

			checkpoint = {}
			checkpoint["main_network"] = main_network.state_dict()
			checkpoint["eif_network"] = EIF.state_dict()
			checkpoint["eif_optimizer"] = eif_optimizer.state_dict()
			checkpoint["main_network_optimizer"] = main_network_optimizer.state_dict()
			checkpoint["epoch"] = epoch

			torch.save(checkpoint,checkpoint_filepath)
			print("---------------------Checkpoint Saved---------------------")

def main():

	args = parser.parse_args()
	training_type = args.training_type

	config_file_path = "config.yaml"
	config_file = open(config_file_path,'r')
	config_dict = yaml.load(config_file,Loader=yaml.FullLoader)

	dataset_name = config_dict["dataset_name"]
	dataset_dir = config_dict["dataset_dir"]
	batch_size = config_dict["batch_size"]
	device = config_dict["device"]
	epochs = config_dict["epochs"]
	main_network_checkpoint_path = config_dict["main_network_checkpoint_path"]
	eif_checkpoint_path = config_dict["eif_checkpoint_path"]
	lr = config_dict["lr"]
	momentum = config_dict["momentum"]
	optimizer_type = config_dict["optimizer_type"]
	lr_scheduler = config_dict["lr_scheduler"]
	train_val_split = config_dict["train_val_split"]
	num_workers = config_dict["num_workers"]
	weight_decay = config_dict["weight_decay"]
	checkpoint_path = config_dict["checkpoint_path"]
	checkpoint_save_interval = config_dict["checkpoint_save_interval"]

	load_checkpoints = {}
	load_checkpoints["main_network_checkpoint_path"] = main_network_checkpoint_path
	load_checkpoints["eif_checkpoint_path"] = eif_checkpoint_path

	if training_type == "standard":
		train_standard_method(dataset_name,dataset_dir,batch_size,epochs,lr,momentum,optimizer_type,lr_scheduler,train_val_split,num_workers,weight_decay,device,None,None,load_checkpoints,checkpoint_path,checkpoint_save_interval)
	elif training_type == "eif_emp":
		train_using_eif_and_emp(dataset_name,dataset_dir,batch_size,epochs,lr,momentum,optimizer_type,lr_scheduler,train_val_split,num_workers,weight_decay,device,None,None,load_checkpoints,checkpoint_path,checkpoint_save_interval)
	else:
		print(" Please select valid training type.\n Valid training types are \n\t1) standard \n\t2) eif_emp")


if __name__ == '__main__':
	main()