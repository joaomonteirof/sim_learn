import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torch.nn.init as init
import torch
from collections import OrderedDict


class AlexNet(nn.Module):
	def __init__(self, n_classes = 7, n_domains = 3, pretrained_path = './alexnet_caffe.pth.tar'):
		super(AlexNet, self).__init__()
		
		self.n_classes = n_classes
		self.n_domains = n_domains

		self.features = nn.Sequential(OrderedDict([
			("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
			("relu1", nn.ReLU(inplace=True)),
			("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
			("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
			("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
			("relu2", nn.ReLU(inplace=True)),
			("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
			("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
			("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
			("relu3", nn.ReLU(inplace=True)),
			("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
			("relu4", nn.ReLU(inplace=True)),
			("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
			("relu5", nn.ReLU(inplace=True)),
			("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
		]))
		
		self.classifier = nn.Sequential(OrderedDict([
			("fc6", nn.Linear(256 * 6 * 6, 4096)),
			("relu6", nn.ReLU(inplace=True)),
			("drop6", nn.Dropout()),
			("fc7", nn.Linear(4096, 4096)),
			("relu7", nn.ReLU(inplace=True)),
			("drop7", nn.Dropout()),
			("fc8", nn.Linear(4096, self.n_classes*self.n_domains))]))
				
		self.initialize_params()

		print('\nLoading pretrained encoder from: {}\n'.format(pretrained_path))
		state_dict = torch.load(pretrained_path)
		del state_dict["classifier.fc8.weight"]
		del state_dict["classifier.fc8.bias"]
		not_loaded = self.load_state_dict(state_dict, strict = False)
		print(not_loaded, '\n')
		
	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Linear):
				init.xavier_uniform_(layer.weight, 0.1)
				layer.bias.data.zero_()	

	def forward(self, x):
		x = self.features(x*57.6)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x
		
if __name__=='__main__':

	state_dict = torch.load(pretrained_path)
	feature_extractor = AlexNet(baseline=True)

	not_loaded = feature_extractor.load_state_dict(state_dict, strict = False)

	print(not_loaded)