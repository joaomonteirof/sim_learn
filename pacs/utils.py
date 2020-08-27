import torch
import math
from torch import nn
from scipy.special import binom
import torch.nn.functional as F

def parse_args_for_log(args):
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		if args_dict[arg_key] is None:
			args_dict[arg_key] = 'None'

	return args_dict

def compare_parameters(model1, model2):

	for p1, p2 in zip(model1.parameters(), model2.parameters()):
		try: 
			if p1.allclose(p2): 
				print('equal')
			else:
				print('diff')	
			
		except:
	 		print('diff layer')

def get_centroids(embeddings, targets, num_classes):

	all_ones, counts = torch.ones(embeddings.size(0)).to(embeddings.device), torch.zeros(num_classes).to(embeddings.device)
	centroids = torch.zeros(num_classes, embeddings.size(-1)).to(embeddings.device)

	with torch.no_grad():

		counts.scatter_add_(dim=0, index=targets, src=all_ones)
		counts_corrected = torch.max(counts, torch.ones_like(counts))
		mask = 1.-torch.abs(counts_corrected-counts).unsqueeze(-1).expand_as(centroids)
		centroids.scatter_add_(dim=0, index=targets.unsqueeze(-1).expand_as(embeddings), src=embeddings).div_(counts_corrected.unsqueeze_(-1))

	return centroids, mask


class LabelSmoothingWithLogitsLoss(nn.Module):
	def __init__(self, label_smoothing, lbl_set_size, dim=1):
		super(LabelSmoothingWithLogitsLoss, self).__init__()
		self.confidence = 1.0 - label_smoothing
		self.smoothing = label_smoothing
		self.cls = lbl_set_size
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class LabelSmoothingNLLLoss(nn.Module):
	def __init__(self, label_smoothing, lbl_set_size, dim=1):
		super(LabelSmoothingNLLLoss, self).__init__()
		self.confidence = 1.0 - label_smoothing
		self.smoothing = label_smoothing
		self.cls = lbl_set_size
		self.dim = dim

	def forward(self, pred, target):
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
		

## Copied from https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py		
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):

	""" Gradually warm-up(increasing) learning rate in optimizer.
	Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

	Args:
		optimizer (Optimizer): Wrapped optimizer.
		multiplier: target learning rate = base lr * multiplier
		total_epoch: target learning rate is reached at total_epoch, gradually
		after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
	"""

	def __init__(self, optimizer, total_epoch, init_lr=1e-7, after_scheduler=None):
		self.init_lr = init_lr
		assert init_lr>0, 'Initial LR should be greater than 0.'
		self.total_epoch = total_epoch
		self.after_scheduler = after_scheduler
		self.finished = False
		super().__init__(optimizer)

	def get_lr(self):
		if self.last_epoch > self.total_epoch:
			if self.after_scheduler:
				if not self.finished:
					self.finished = True
				return self.after_scheduler.get_lr()
			return self.base_lrs

		return [(((base_lr - self.init_lr)/self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in self.base_lrs]

	def step_ReduceLROnPlateau(self, metrics, epoch=None):
		if epoch is None:
			epoch = self.last_epoch + 1
		self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
		if self.last_epoch <= self.total_epoch:
			warmup_lr = [(((base_lr - self.init_lr)/self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in self.base_lrs]
			for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
				param_group['lr'] = lr
		else:
			if epoch is None:
				self.after_scheduler.step(metrics, None)
			else:
				self.after_scheduler.step(metrics, epoch - self.total_epoch)

	def step(self, epoch=None, metrics=None):
		if type(self.after_scheduler) != ReduceLROnPlateau:
			if (self.finished and self.after_scheduler) or self.total_epoch==0:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.total_epoch)
			else:
				return super(GradualWarmupScheduler, self).step(epoch)
		else:
			self.step_ReduceLROnPlateau(metrics, epoch)		
