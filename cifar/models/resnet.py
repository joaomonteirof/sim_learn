'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_centroids
from models.losses import AMSoftmax, Softmax


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes) )

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes))

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class PreActBlock(nn.Module):
	'''Pre-activation version of the BasicBlock.'''
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out += shortcut
		return out


class PreActBottleneck(nn.Module):
	'''Pre-activation version of the original Bottleneck module.'''
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out += shortcut
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, nh, n_h, sm_type, num_classes=10, dropout_prob=0.25, centroids_lambda=0.9):
		super(ResNet, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.centroids_lambda = centroids_lambda

		self.centroids = torch.rand(num_classes, 512*block.expansion)
		self.centroids.requires_grad = False

		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=512*block.expansion, output_features=num_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=512*block.expansion, output_features=num_classes)
		else:
			raise NotImplementedError

		self.similarity = self.make_bin_layers(n_in=2*512*block.expansion, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)

		return out

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))
		classifier.append(nn.Sigmoid())

		return classifier

	def forward_bin(self, centroids, embeddings):

		z = torch.cat((centroids, embeddings), -1)

		for l in self.similarity:
			z = l(z)
		
		return z

	def update_centroids(self, embeddings, targets):

		self.centroids =  self.centroids.to(embeddings.device)

		new_centroids, mask = get_centroids(embeddings, targets, 10)

		with torch.no_grad():
			mask *= 1.-self.centroids_lambda
			self.centroids = (1.-mask)*self.centroids + mask*new_centroids

		self.centroids.requires_grad = False

	def compute_logits(self, embeddings, ablation=False):

		centroids = self.centroids.unsqueeze(0)
		emb = embeddings.unsqueeze(1)

		centroids = centroids.repeat(embeddings.size(0), 1, 1)
		emb = emb.repeat(1, self.centroids.size(0), 1)

		if ablation:
			return -((centroids-emb).pow(2).sum(-1).sqrt()).transpose(1,-1)
		else:
			return self.forward_bin(centroids, emb).squeeze(-1).transpose(1,-1)

def ResNet18(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
	return ResNet(PreActBlock, [2,2,2,2], nh=nh, n_h=n_h, sm_type=sm_type, dropout_prob=dropout_prob, centroids_lambda=centroids_lambda)

def ResNet34(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
	return ResNet(PreActBlock, [3,4,6,3], nh=nh, n_h=n_h, sm_type=sm_type, dropout_prob=dropout_prob, centroids_lambda=centroids_lambda)

def ResNet50(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
	return ResNet(PreActBottleneck, [3,4,6,3], nh=nh, n_h=n_h, sm_type=sm_type, dropout_prob=dropout_prob, centroids_lambda=centroids_lambda)

def ResNet101(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
	return ResNet(PreActBottleneck, [3,4,23,3], nh=nh, n_h=n_h, sm_type=sm_type, dropout_prob=dropout_prob, centroids_lambda=centroids_lambda)

def ResNet152(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
	return ResNet(PreActBottleneck, [3,8,36,3], nh=nh, n_h=n_h, sm_type=sm_type, dropout_prob=dropout_prob, centroids_lambda=centroids_lambda)
