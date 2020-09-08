import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from models.losses import AMSoftmax, Softmax
from utils import get_centroids

## adapted from https://github.com/kjunelee/MetaOptNet

class DropBlock(nn.Module):
	def __init__(self, block_size):
		super(DropBlock, self).__init__()

		self.block_size = block_size
		#self.gamma = gamma
		#self.bernouli = Bernoulli(gamma)

	def forward(self, x, gamma):
		# shape: (bsize, channels, height, width)

		if self.training:
			batch_size, channels, height, width = x.shape
			
			bernoulli = Bernoulli(gamma)
			mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
			block_mask = self._compute_block_mask(mask)
			countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
			count_ones = block_mask.sum()

			return block_mask * x * (countM / count_ones)
		else:
			return x

	def _compute_block_mask(self, mask):
		left_padding = int((self.block_size-1) / 2)
		right_padding = int(self.block_size / 2)
		
		batch_size, channels, height, width = mask.shape
		non_zero_idxs = mask.nonzero()
		nr_blocks = non_zero_idxs.shape[0]

		offsets = torch.stack(
			[
				torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
				torch.arange(self.block_size).repeat(self.block_size), #- left_padding
			]
		).t().cuda()
		offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
		
		if nr_blocks > 0:
			non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
			offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
			offsets = offsets.long()

			block_idxs = non_zero_idxs + offsets
			#block_idxs += left_padding
			padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
			padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
		else:
			padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
			
		block_mask = 1 - padded_mask#[:height, :width]
		return block_mask

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.LeakyReLU(0.1)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv3x3(planes, planes)
		self.bn3 = nn.BatchNorm2d(planes)
		self.maxpool = nn.MaxPool2d(stride)
		self.downsample = downsample
		self.stride = stride
		self.drop_rate = drop_rate
		self.num_batches_tracked = 0
		self.drop_block = drop_block
		self.block_size = block_size
		self.DropBlock = DropBlock(block_size=self.block_size)

	def forward(self, x):
		self.num_batches_tracked += 1

		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		out = self.maxpool(out)
		
		if self.drop_rate > 0:
			if self.drop_block == True:
				feat_size = out.size()[2]
				keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
				gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
				out = self.DropBlock(out, gamma=gamma)
			else:
				out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

		return out


class ResNet(nn.Module):
	def __init__(self, nh, n_h, sm_type, block, dropout_prob=0.25, centroids_lambda=0.9, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, num_classes=1000):
		self.inplanes = 3
		super(ResNet, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.n_classes = num_classes

		self.centroids_lambda = centroids_lambda

		self.centroids = torch.rand(self.n_classes, 640)
		self.centroids.requires_grad = False

		self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
		self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
		self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
		self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
		if avg_pool:
			self.avgpool = nn.AvgPool2d(5, stride=1)
		self.keep_prob = keep_prob
		self.keep_avg_pool = avg_pool
		self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
		self.drop_rate = drop_rate

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=640, output_features=num_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=640, output_features=num_classes)
		else:
			raise NotImplementedError

		self.similarity = self.make_bin_layers(n_in=2*640, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
		self.inplanes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		if self.keep_avg_pool:
			x = self.avgpool(x)
		x = x.view(x.size(0), -1)

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

	def update_centroids_eval(self, centroids, embeddings, targets, update_lambda=None):

			centroids =  centroids.to(embeddings.device)
			if not update_lambda: update_lambda = self.centroids_lambda

			new_centroids, mask = get_centroids(embeddings, targets, self.n_classes)

			with torch.no_grad():
				mask *= 1.-update_lambda
				centroids = (1.-mask)*centroids + mask*new_centroids

			return centroids

	def compute_logits_eval(self, centroids, embeddings, ablation=False):

			centroids = centroids.unsqueeze(0)
			emb = embeddings.unsqueeze(1)

			centroids = centroids.repeat(embeddings.size(0), 1, 1)
			emb = emb.repeat(1, centroids.size(1), 1)

			if ablation:
				return -((centroids-emb).pow(2).sum(-1).sqrt()).transpose(1,-1)
			else:
				return self.forward_bin(centroids, emb).squeeze(-1).transpose(1,-1)

def ResNet12(nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=100, centroids_lambda=0.9, keep_prob=1.0, avg_pool=True, **kwargs):
	"""Constructs a ResNet-12 model.
	"""
	model = ResNet(block=BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, num_classes=n_classes, nh=nh, n_h=n_h, sm_type=sm_type, dropout_prob=dropout_prob, centroids_lambda=centroids_lambda, **kwargs)
	return model