import torch
import torch.nn.functional as F
import numpy as np

import constants

class Conv2dFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
		# Save arguments to context to use on backward
		# WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
		confs = torch.from_numpy(np.array([stride, padding, dilation, groups]))
		ctx.save_for_backward(input, weight, bias, confs)

		# Compute Convolution
		return F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
	
	@staticmethod
	def backward(ctx, grad_output):
		# Load saved tensors
		input, weight, bias, confs = ctx.saved_variables
		confs = confs.numpy()
		stride, padding, dilation, groups= confs[0], confs[1], confs[2], confs[3]

		# Calculate Gradient
		grad_input = grad_weight = grad_bias = None
		if ctx.needs_input_grad[0]:
			grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
			
		if ctx.needs_input_grad[1]:
			grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
				
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(dim=(0,2,3))

		if bias is not None:
			return grad_input, grad_weight, grad_bias, None, None, None, None
		else:
			return grad_input, grad_weight, None, None, None, None, None