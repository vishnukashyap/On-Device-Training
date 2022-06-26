import torch
import emp_conv_module

'''
	Implementation of Residual Block with Error Map Pruning
'''
class Residual_Block_with_EMP(torch.nn.Module):
	def __init__(self,in_channels,out_channels,input_stride=1,padding='same'):
		super(Residual_Block_with_EMP,self).__init__()

		self.Conv1 = emp_conv_module.Conv2d_with_EMP(in_channels,out_channels,kernel_size=(3,3),stride=input_stride,padding=padding,use_emp=True)
		self.BatchNorm1 = torch.nn.BatchNorm2d(out_channels)
		self.Conv2 = emp_conv_module.Conv2d_with_EMP(out_channels,out_channels,kernel_size=(3,3),stride=1,padding='same',use_emp=True)
		self.BatchNorm2 = torch.nn.BatchNorm2d(out_channels)

		self.Indentity_Conv = emp_conv_module.Conv2d_with_EMP(in_channels,out_channels,kernel_size=(3,3),stride=input_stride,padding=padding,use_emp=True)
		self.Indentity_BatchNorm = torch.nn.BatchNorm2d(out_channels)

		self.ReLU = torch.nn.LeakyReLU()

	def forward(self,input_tensor,batch_size):

		if batch_size == 1:
			output = self.ReLU(self.Conv1(input_tensor))
			output = self.ReLU(self.Conv2(output))

			residual_output = self.ReLU(self.Indentity_Conv(input_tensor))
		else:	
			output = self.ReLU(self.BatchNorm1(self.Conv1(input_tensor)))
			output = self.ReLU(self.BatchNorm2(self.Conv2(output)))

			residual_output = self.ReLU(self.Indentity_BatchNorm(self.Indentity_Conv(input_tensor)))

		return residual_output + output

'''
	Implementation of Residual Block
'''
class Residual_Block(torch.nn.Module):
	def __init__(self,in_channels,out_channels,input_stride=1,padding='same'):
		super(Residual_Block,self).__init__()

		self.Conv1 = emp_conv_module.Conv2d_with_EMP(in_channels,out_channels,kernel_size=(3,3),stride=input_stride,padding=padding,use_emp=False)
		self.BatchNorm1 = torch.nn.BatchNorm2d(out_channels)
		self.Conv2 = emp_conv_module.Conv2d_with_EMP(out_channels,out_channels,kernel_size=(3,3),stride=1,padding='same',use_emp=False)
		self.BatchNorm2 = torch.nn.BatchNorm2d(out_channels)

		self.Indentity_Conv = emp_conv_module.Conv2d_with_EMP(in_channels,out_channels,kernel_size=(3,3),stride=input_stride,padding=padding,use_emp=False)
		self.Indentity_BatchNorm = torch.nn.BatchNorm2d(out_channels)

		self.ReLU = torch.nn.LeakyReLU()

	def forward(self,input_tensor,batch_size):

		if batch_size == 1:
			output = self.ReLU(self.Conv1(input_tensor))
			output = self.ReLU(self.Conv2(output))

			residual_output = self.ReLU(self.Indentity_Conv(input_tensor))
		else:
			output = self.ReLU(self.BatchNorm1(self.Conv1(input_tensor)))
			output = self.ReLU(self.BatchNorm2(self.Conv2(output)))

			residual_output = self.ReLU(self.Indentity_BatchNorm(self.Indentity_Conv(input_tensor)))

		return residual_output + output

'''
	Implementation of ResNet-34 with Error Map Pruning Convolution Modules
'''
class ResNet34_with_EMP(torch.nn.Module):
	def __init__(self,num_classes):
		super(ResNet34_with_EMP,self).__init__()

		self.Conv1 = emp_conv_module.Conv2d_with_EMP(3,64,kernel_size=(7,7),stride=2,padding=3,use_emp=True)
		self.BatchNorm = torch.nn.BatchNorm2d(64)
		self.ReLU = torch.nn.ReLU()
		self.Pool = torch.nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)

		self.ResBlock1_1 = Residual_Block_with_EMP(64,64)
		self.ResBlock1_2 = Residual_Block_with_EMP(64,64)
		self.ResBlock1_3 = Residual_Block_with_EMP(64,64)

		self.ResBlock2_1 = Residual_Block_with_EMP(64,128,input_stride=2,padding=1)
		self.ResBlock2_2 = Residual_Block_with_EMP(128,128)
		self.ResBlock2_3 = Residual_Block_with_EMP(128,128)
		self.ResBlock2_4 = Residual_Block_with_EMP(128,128)

		self.ResBlock3_1 = Residual_Block_with_EMP(128,256,input_stride=2,padding=1)
		self.ResBlock3_2 = Residual_Block_with_EMP(256,256)
		self.ResBlock3_3 = Residual_Block_with_EMP(256,256)
		self.ResBlock3_4 = Residual_Block_with_EMP(256,256)
		self.ResBlock3_5 = Residual_Block_with_EMP(256,256)
		self.ResBlock3_6 = Residual_Block_with_EMP(256,256)

		self.ResBlock4_1 = Residual_Block_with_EMP(256,512,input_stride=2,padding=1)
		self.ResBlock4_2 = Residual_Block_with_EMP(512,512)
		self.ResBlock4_3 = Residual_Block_with_EMP(512,512)
		
		self.FullyConnected = torch.nn.Linear(512,num_classes)

	def forward(self,input_tensor):
		batch_size = input_tensor.shape[0]

		if batch_size == 1:
			output = self.Pool(self.ReLU(self.Conv1(input_tensor)))
		else:
			output = self.Pool(self.ReLU(self.BatchNorm(self.Conv1(input_tensor))))

		output = self.ResBlock1_1(output,batch_size)
		output = self.ResBlock1_2(output,batch_size)
		output = self.ResBlock1_3(output,batch_size)

		output = self.ResBlock2_1(output,batch_size)
		output = self.ResBlock2_2(output,batch_size)
		output = self.ResBlock2_3(output,batch_size)
		output = self.ResBlock2_4(output,batch_size)

		output = self.ResBlock3_1(output,batch_size)
		output = self.ResBlock3_2(output,batch_size)
		output = self.ResBlock3_3(output,batch_size)
		output = self.ResBlock3_4(output,batch_size)
		output = self.ResBlock3_5(output,batch_size)
		output = self.ResBlock3_6(output,batch_size)

		output = self.ResBlock4_1(output,batch_size)
		output = self.ResBlock4_2(output,batch_size)
		output = self.ResBlock4_3(output,batch_size)

		output = torch.flatten(output,start_dim=1)
		output = self.FullyConnected(output)

		return output

'''
	Implementation of ResNet-34 with Error Map Pruning Convolution Modules
'''
class ResNet34(torch.nn.Module):
	def __init__(self,num_classes):
		super(ResNet34,self).__init__()

		self.Conv1 = emp_conv_module.Conv2d_with_EMP(3,64,kernel_size=(7,7),stride=2,padding=3,use_emp=False)
		self.BatchNorm = torch.nn.BatchNorm2d(64)
		self.ReLU = torch.nn.ReLU()
		self.Pool = torch.nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)

		self.ResBlock1_1 = Residual_Block(64,64)
		self.ResBlock1_2 = Residual_Block(64,64)
		self.ResBlock1_3 = Residual_Block(64,64)

		self.ResBlock2_1 = Residual_Block(64,128,input_stride=2,padding=1)
		self.ResBlock2_2 = Residual_Block(128,128)
		self.ResBlock2_3 = Residual_Block(128,128)
		self.ResBlock2_4 = Residual_Block(128,128)

		self.ResBlock3_1 = Residual_Block(128,256,input_stride=2,padding=1)
		self.ResBlock3_2 = Residual_Block(256,256)
		self.ResBlock3_3 = Residual_Block(256,256)
		self.ResBlock3_4 = Residual_Block(256,256)
		self.ResBlock3_5 = Residual_Block(256,256)
		self.ResBlock3_6 = Residual_Block(256,256)

		self.ResBlock4_1 = Residual_Block(256,512,input_stride=2,padding=1)
		self.ResBlock4_2 = Residual_Block(512,512)
		self.ResBlock4_3 = Residual_Block(512,512)
		
		self.FullyConnected = torch.nn.Linear(512,num_classes)

	def forward(self,input_tensor):
		batch_size = input_tensor.shape[0]

		if batch_size == 1:
			output = self.Pool(self.ReLU(self.Conv1(input_tensor)))
		else:
			output = self.Pool(self.ReLU(self.BatchNorm(self.Conv1(input_tensor))))

		output = self.ResBlock1_1(output,batch_size)
		output = self.ResBlock1_2(output,batch_size)
		output = self.ResBlock1_3(output,batch_size)

		output = self.ResBlock2_1(output,batch_size)
		output = self.ResBlock2_2(output,batch_size)
		output = self.ResBlock2_3(output,batch_size)
		output = self.ResBlock2_4(output,batch_size)

		output = self.ResBlock3_1(output,batch_size)
		output = self.ResBlock3_2(output,batch_size)
		output = self.ResBlock3_3(output,batch_size)
		output = self.ResBlock3_4(output,batch_size)
		output = self.ResBlock3_5(output,batch_size)
		output = self.ResBlock3_6(output,batch_size)

		output = self.ResBlock4_1(output,batch_size)
		output = self.ResBlock4_2(output,batch_size)
		output = self.ResBlock4_3(output,batch_size)

		output = torch.flatten(output,start_dim=1)
		output = self.FullyConnected(output)

		return output

'''
	Implementation of ResNet8
	Early Instance Filtering(EIF) Network
'''
class ResNet8(torch.nn.Module):
	def __init__(self):
		super(ResNet8,self).__init__()

		self.Conv1 = emp_conv_module.Conv2d_with_EMP(3,16,kernel_size=(7,7),stride=2,padding=1,use_emp=False)
		self.Pool = torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)

		self.ResBlock1 = Residual_Block(16,32,padding=0)
		self.ResBlock2 = Residual_Block(32,64,padding=0)
		self.ResBlock3 = Residual_Block(64,64,padding=0)

		self.Avg_Pool = torch.nn.AvgPool2d(1)
		self.FullyConnected = torch.nn.Linear(64,2)

	def forward(self,input_tensor):

		batch_size = input_tensor.shape[0]

		output = self.Pool(self.Conv1(input_tensor))

		output = self.ResBlock1(output,batch_size)
		output = self.ResBlock2(output,batch_size)
		output = self.ResBlock3(output,batch_size)

		output = self.Avg_Pool(output)
		output = torch.flatten(output,start_dim=1)
		output = self.FullyConnected(output)

		return output