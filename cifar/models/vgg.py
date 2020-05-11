import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.losses import AMSoftmax, Softmax

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', centroids_lambda=0.9):
		super(VGG, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type

		self.centroids_lambda = centroids_lambda

		self.centroids = torch.rand(10, 512)
		self.centroids.requires_grad = False

		self.features = self._make_layers(cfg[vgg_name])

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=512, output_features=10)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=512, output_features=10)
		else:
			raise NotImplementedError

		self.similarity = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

	def forward(self, x):
		features = self.features(x)
		features = features.view(features.size(0), -1)
		return features

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)

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

	def update_centroids(self, centroids, targets):

		self.centroids.to(centroids.device)

		with torch.no_grad():
			new_centroids = torch.zeros_like(self.centroids).scatter_(0, targets.unsqueeze(-1).expand_as(centroids), centroids)

			mask = torch.ones_like(self.centroids).scatter_(0, targets.unsqueeze(-1).expand_as(centroids), self.centroids_lambda*torch.ones_like(self.centroids))

			self.centroids = mask*self.centroids + (1.-mask)*new_centroids

		self.centroids.requires_grad = False

	def compute_logits(self, embeddings):

		centroids = self.centroids.unsqueeze(0)
		emb = embeddings.unsqueeze(1)

		centroids = centroids.repeat(embeddings.size(0), 1, 1)
		emb = emb.repeat(1, self.centroids.size(0), 1)

		return self.forward_bin(centroids, emb).squeeze(-1).transpose(1,-1)