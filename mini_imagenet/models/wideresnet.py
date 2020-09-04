## Adapted from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import get_centroids
from models.losses import AMSoftmax, Softmax


_bn_momentum = 0.1


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_uniform_(m.weight, gain=np.sqrt(2))
		init.constant_(m.bias, 0)
	elif classname.find('BatchNorm') != -1:
		init.constant_(m.weight, 1)
		init.constant_(m.bias, 0)


class WideBasic(nn.Module):
	def __init__(self, in_planes, planes, dropout_rate, stride=1):
		super(WideBasic, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
		self.dropout = nn.Dropout(p=dropout_rate)
		self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
			)

	def forward(self, x):
		out = self.dropout(self.conv1(F.relu(self.bn1(x))))
		out = self.conv2(F.relu(self.bn2(out)))
		out += self.shortcut(x)

		return out


class WideResNet(nn.Module):
	def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, n_classes=10, nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
		super(WideResNet, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.n_classes = n_classes
		self.centroids_lambda = centroids_lambda

		self.in_planes = 16

		assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
		n = int((depth - 4) / 6)
		k = widen_factor

		nStages = [16, 16*k, 32*k, 64*k]

		self.centroids = torch.rand(n_classes, nStages[3])
		self.centroids.requires_grad = False

		self.conv1 = conv3x3(3, nStages[0])
		self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
		self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
		self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
		self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=nStages[3], output_features=n_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=nStages[3], output_features=n_classes)
		else:
			raise NotImplementedError

		self.similarity = self.make_bin_layers(n_in=2*nStages[3], n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

	def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []

		for stride in strides:
			layers.append(block(self.in_planes, planes, dropout_rate, stride))
			self.in_planes = planes

		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = F.relu(self.bn1(out))
		# out = F.avg_pool2d(out, 8)
		out = F.adaptive_avg_pool2d(out, (1, 1))
		out = out.view(out.size(0), -1)

		return out

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))

		return classifier

	def forward_bin(self, centroids, embeddings):

		z = torch.cat((centroids, embeddings), -1)

		for l in self.similarity:
			z = l(z)
		
		return z

	def update_centroids(self, embeddings, targets):

		self.centroids =  self.centroids.to(embeddings.device)

		new_centroids, mask = get_centroids(embeddings, targets, self.n_classes)

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
