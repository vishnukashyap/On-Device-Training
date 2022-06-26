import torch
import torch.nn.functional as F
import numpy as np

import constants

def prune_channels(output_gradient,weights):
	'''
		The procedure for error map pruning is implemented as defined as in the paper:
		Enabling On-Device CNN Training by Self-Supervised Instance Filtering and Error Map Pruning
		https://arxiv.org/pdf/2007.03213.pdf
	'''

	output_gradient_importance_score = output_gradient.norm(p=1,dim=(2,3))
	weights_importance_score = weights.norm(p=1,dim=(1,2,3))

	importance_score = constants.error_map_pruning_gamma_2*output_gradient_importance_score + constants.error_map_pruning_gamma_1*weights_importance_score

	mini_bacth_importance_score = torch.sum(importance_score,dim=0)

	# Since the number of channels are C_out*(1-pruning_ratio), the top C_out*pruning_ratio channels are selected and passed for gradient calculation
	number_of_channels_to_be_retained = int((constants.error_map_pruning_ratio)*mini_bacth_importance_score.shape[0])

	_,indices_of_channels_to_be_retained = torch.topk(mini_bacth_importance_score,k=number_of_channels_to_be_retained)

	pruned_output_gradients = output_gradient[:,indices_of_channels_to_be_retained.type(torch.long),:,:]
	pruned_weights = weights[indices_of_channels_to_be_retained.type(torch.long),:,:,:]

	return pruned_output_gradients,pruned_weights,indices_of_channels_to_be_retained

'''
	The implementation of the below function of 2d convolution is mentioned in:
	https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/21
'''
class conv2d_with_error_map_pruning(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, use_emp=False):
		# Save arguments to context to use on backward
		# WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
		if padding == 'same':
			confs = torch.from_numpy(np.array([stride, 1, dilation, groups, use_emp]))
		elif padding == 'valid':
			confs = torch.from_numpy(np.array([stride, 0, dilation, groups, use_emp]))
		else:
			confs = torch.from_numpy(np.array([stride, padding, dilation, groups, use_emp]))
		ctx.save_for_backward(input, weight, bias, confs)

		# Compute Convolution
		return F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
	
	@staticmethod
	def backward(ctx, grad_output):
		# Load saved tensors
		input, weight, bias, confs = ctx.saved_variables
		confs = confs.numpy()
		stride, padding, dilation, groups, use_emp= confs[0], confs[1], confs[2], confs[3], confs[4]

		if use_emp:
			
			pruned_grad_output,pruned_weights,indices_of_channels_to_be_retained = prune_channels(grad_output,weight)

			# Calculate Gradient
			grad_input = grad_weight = grad_bias = None
			grad_weight = torch.zeros(weight.shape).to(weight.device.type)
			grad_bias = torch.zeros(bias.shape).to(bias.device.type)

			pruned_grad_weight = pruned_grad_bias = None

			if ctx.needs_input_grad[0]:
				grad_input = torch.nn.grad.conv2d_input(input.shape, pruned_weights, pruned_grad_output, stride, padding, dilation, groups)
				
			if ctx.needs_input_grad[1]:
				pruned_grad_weight = torch.nn.grad.conv2d_weight(input, pruned_weights.shape, pruned_grad_output, stride, padding, dilation, groups)
				grad_weight[indices_of_channels_to_be_retained.type(torch.long),:,:,:] = pruned_grad_weight
					
			if bias is not None and ctx.needs_input_grad[2]:
				pruned_grad_bias = pruned_grad_output.sum(dim=(0,2,3))
				grad_bias[indices_of_channels_to_be_retained] = pruned_grad_bias

			if bias is not None:
				return grad_input, grad_weight, grad_bias, None, None, None, None, None
			else:
				return grad_input, grad_weight, None, None, None, None, None, None

		else:

			# Calculate Gradient
			grad_input = grad_weight = grad_bias = None
			grad_weight = torch.zeros(weight.shape).to(weight.device.type)
			grad_bias = torch.zeros(bias.shape).to(bias.device.type)

			if ctx.needs_input_grad[0]:
				grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)

			if ctx.needs_input_grad[1]:
				grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)

			if bias is not None and ctx.needs_input_grad[2]:
				grad_bias = grad_output.sum(dim=(0,2,3))

			if bias is not None:
				return grad_input, grad_weight, grad_bias, None, None, None, None, None
			else:
				return grad_input, grad_weight, None, None, None, None, None, None


class Conv2d_with_EMP(torch.nn.Module):

	def __init__(self,in_channels,out_channels,kernel_size,bias=True,stride=1, padding=0, dilation=1, groups=1, use_emp=False):
		super(Conv2d_with_EMP,self).__init__()

		if type(kernel_size) == tuple:
			self.weight = torch.nn.Parameter(torch.empty(out_channels,in_channels//groups,kernel_size[0],kernel_size[1]))
		elif type(kernel_size) == int:
			self.weight = torch.nn.Parameter(torch.empty(out_channels,in_channels//groups,kernel_size,kernel_size))

		if bias == True:
			self.bias = torch.nn.Parameter(torch.empty(out_channels))
		else:
			self.bias = None

		torch.nn.init.xavier_normal_(self.weight)

		if self.bias is not None:
			self.bias.data.fill_(0.01)

		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.use_emp = use_emp

		self.Conv2d_Function = conv2d_with_error_map_pruning.apply

	def forward(self,input_tensor):
		return self.Conv2d_Function(input_tensor,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups,self.use_emp)