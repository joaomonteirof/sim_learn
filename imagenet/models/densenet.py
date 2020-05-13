'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_centroids
from models.losses import AMSoftmax, Softmax


class Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(4*growth_rate)
		self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat([out,x], 1)
		return out


class Transition(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Transition, self).__init__()
		self.bn = nn.BatchNorm2d(in_planes)
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

	def forward(self, x):
		out = self.conv(F.relu(self.bn(x)))
		out = F.avg_pool2d(out, 2)
		return out


class DenseNet(nn.Module):
	def __init__(self, block, nblocks, nh, n_h, sm_type, growth_rate=12, reduction=0.5, num_classes=1000, dropout_prob=0.25, centroids_lambda=0.9):
		super(DenseNet, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.n_classes = num_classes
		self.centroids_lambda = centroids_lambda
		self.growth_rate = growth_rate

		self.centroids = torch.rand(10, 1024*6*6)
		self.centroids.requires_grad = False

		num_planes = 2*growth_rate
		self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, padding=1, bias=False)

		self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
		num_planes += nblocks[0]*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans1 = Transition(num_planes, out_planes)
		num_planes = out_planes

		self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
		num_planes += nblocks[1]*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans2 = Transition(num_planes, out_planes)
		num_planes = out_planes

		self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
		num_planes += nblocks[2]*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans3 = Transition(num_planes, out_planes)
		num_planes = out_planes

		self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
		num_planes += nblocks[3]*growth_rate

		self.bn = nn.BatchNorm2d(num_planes)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=1024*6*6, output_features=num_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=1024*6*6, output_features=num_classes)
		else:
			raise NotImplementedError

		self.similarity = self.make_bin_layers(n_in=2*1024*6*6, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

	def _make_dense_layers(self, block, in_planes, nblock):
		layers = []
		for i in range(nblock):
			layers.append(block(in_planes, self.growth_rate))
			in_planes += self.growth_rate
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.trans1(self.dense1(out))
		out = self.trans2(self.dense2(out))
		out = self.trans3(self.dense3(out))
		out = self.dense4(out)
		out = F.avg_pool2d(F.relu(self.bn(out)), 4)
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

def DenseNet121(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=1000, centroids_lambda=0.9):
	return DenseNet(Bottleneck, [6,12,24,16], nh, n_h, sm_type, dropout_prob=dropout_prob, growth_rate=32, num_classes=n_classes, centroids_lambda=centroids_lambda)

def DenseNet169(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=1000, centroids_lambda=0.9):
	return DenseNet(Bottleneck, [6,12,32,32], nh, n_h, sm_type, dropout_prob=dropout_prob, growth_rate=32, num_classes=n_classes, centroids_lambda=centroids_lambda)

def DenseNet201(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=1000, centroids_lambda=0.9):
	return DenseNet(Bottleneck, [6,12,48,32], nh, n_h, sm_type, dropout_prob=dropout_prob, growth_rate=32, num_classes=n_classes, centroids_lambda=centroids_lambda)

def DenseNet161(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=1000, centroids_lambda=0.9):
	return DenseNet(Bottleneck, [6,12,36,24], nh, n_h, sm_type, dropout_prob=dropout_prob, growth_rate=48, num_classes=n_classes, centroids_lambda=centroids_lambda)

def densenet_cifar(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=1000, centroids_lambda=0.9):
	return DenseNet(Bottleneck, [6,12,24,16], nh, n_h, sm_type, dropout_prob=dropout_prob, growth_rate=12, num_classes=n_classes, centroids_lambda=centroids_lambda)
