import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import get_centroids
from models.losses import AMSoftmax, Softmax


class cnn(nn.Module):
	def __init__(self, nh=1, n_h=64, dropout_prob=0.25, sm_type='softmax', n_classes=10, centroids_lambda=0.9):
		super(cnn, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.n_classes = n_classes
		self.centroids_lambda = centroids_lambda

		self.centroids = torch.rand(self.n_classes, 50)
		self.centroids.requires_grad = False

		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=50, output_features=n_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=50, output_features=n_classes)
		else:
			raise NotImplementedError

		self.similarity = self.make_bin_layers(n_in=2*50, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

	def forward(self, x, pre_softmax = False):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))

		return x

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
			return F.cosine_similarity(centroids, emb, dim=-1).squeeze(-1).transpose(1,-1)
		else:
			return self.forward_bin(centroids, emb).squeeze(-1).transpose(1,-1)
