import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.losses import AMSoftmax, Softmax
from utils.utils import get_centroids

class StatisticalPooling(nn.Module):

	def forward(self, x):
		# x is 3-D with axis [B, feats, T]
		noise = torch.rand(x.size()).to(x.device)*1e-6
		x = x + noise 
		mu = x.mean(dim=2, keepdim=False)
		std = x.std(dim=2, keepdim=False)
		return torch.cat((mu, std), dim=1)

class TDNN(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, ncoef=23, sm_type='softmax', n_speakers=5994, dropout_prob=0.25, centroids_lambda=0.9):
		super(TDNN, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.emb_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef
		self.nspeakers = n_speakers

		self.centroids_lambda = centroids_lambda

		self.centroids = torch.rand(self.nspeakers, self.emb_size)
		self.centroids.requires_grad = False

		self.model = nn.Sequential( nn.Conv1d(ncoef, 512, 5, padding=2, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Conv1d(512, 512, 5, padding=2, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Conv1d(512, 512, 5, padding=3, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Conv1d(512, 512, 7, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Conv1d(512, 1500, 1, bias=False),
			nn.ReLU(inplace=True), 
			nn.BatchNorm1d(1500) )

		self.pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Linear(3000, 512, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )

		self.post_pooling_2 = nn.Sequential(nn.Linear(512, 512, bias=False),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Linear(512, self.emb_size) )

		self.similarity = self.make_bin_layers(n_in=2*self.emb_size, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=self.emb_size, output_features=self.nspeakers)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=self.emb_size, output_features=self.nspeakers)

	def forward(self, x):
		x = self.model(x.squeeze(1))
		x = self.pooling(x)
		x = self.post_pooling_1(x)
		x = self.post_pooling_2(x)

		return x.squeeze(-1)

	def forward_bin(self, centroids, embeddings):

		z = torch.cat((centroids, embeddings), -1)

		for l in self.similarity:
			z = l(z)
		
		return z

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))

		return classifier

	def update_centroids(self, embeddings, targets):

		self.centroids =  self.centroids.to(embeddings.device)

		new_centroids, mask = get_centroids(embeddings, targets, self.nspeakers)

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

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Conv1d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

class TDNN_multipool(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, ncoef=23, sm_type='softmax', n_speakers=5994, dropout_prob=0.25, centroids_lambda=0.9):
		super(TDNN_multipool, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.emb_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef
		self.nspeakers = n_speakers

		self.centroids_lambda = centroids_lambda

		self.centroids = torch.rand(self.nspeakers, self.emb_size)
		self.centroids.requires_grad = False

		self.model_1 = nn.Sequential( nn.Conv1d(ncoef, 512, 5, padding=2, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_2 = nn.Sequential( nn.Conv1d(512, 512, 5, padding=2, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_3 = nn.Sequential( nn.Conv1d(512, 512, 5, padding=3, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_4 = nn.Sequential( nn.Conv1d(512, 512, 7, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )
		self.model_5 = nn.Sequential( nn.Conv1d(512, 512, 1, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )

		self.stats_pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Linear(2048, 512, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512) )

		self.post_pooling_2 = nn.Sequential(nn.Linear(512, 512, bias=False),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Linear(512, self.emb_size) )

		self.similarity = self.make_bin_layers(n_in=2*self.emb_size, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=self.emb_size, output_features=self.nspeakers)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=self.emb_size, output_features=self.nspeakers)

	def forward(self, x):

		x_pool = []

		x = x.squeeze(1)

		x_1 = self.model_1(x)
		x_pool.append(self.stats_pooling(x_1).unsqueeze(2))

		x_2 = self.model_2(x_1)
		x_pool.append(self.stats_pooling(x_2).unsqueeze(2))

		x_3 = self.model_3(x_2)
		x_pool.append(self.stats_pooling(x_3).unsqueeze(2))

		x_4 = self.model_4(x_3)
		x_pool.append(self.stats_pooling(x_4).unsqueeze(2))

		x_5 = self.model_5(x_4)
		x_pool.append(self.stats_pooling(x_5).unsqueeze(2))

		x_pool = torch.cat(x_pool, -1)

		x = self.stats_pooling(x_pool)

		x = self.post_pooling_1(x)
		x = self.post_pooling_2(x)

		return x

	def forward_bin(self, centroids, embeddings):

		z = torch.cat((centroids, embeddings), -1)

		for l in self.similarity:
			z = l(z)
		
		return z

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))
		classifier.append(nn.Sigmoid())

		return classifier

	def update_centroids(self, embeddings, targets):

		self.centroids =  self.centroids.to(embeddings.device)

		new_centroids, mask = get_centroids(embeddings, targets, self.nspeakers)

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

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Conv1d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()