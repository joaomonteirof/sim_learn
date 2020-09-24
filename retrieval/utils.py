import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn import metrics

import torch
import itertools
import os
import sys
import pickle
from time import sleep
import random

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class add_noise(object):
	"""add noise
	"""

	def __call__(self, pic):
		"""
		Args:
			pic (Torch tensor of arbitrary shape): Image to be distorted.
		Returns:
			Tensor: Distorted image.
		"""
		if random.random()>0.5:
			pic += torch.rand_like(pic)*random.choice([0.1, 0.2, 0.3])

		pic = torch.clamp(pic, min=0.0, max=1.0)

		return pic

	def __repr__(self):
		return self.__class__.__name__ + '()'

def get_centroids(embeddings, targets, num_classes):

	all_ones, counts = torch.ones(embeddings.size(0)).to(embeddings.device), torch.zeros(num_classes).to(embeddings.device)
	centroids = torch.zeros(num_classes, embeddings.size(-1)).to(embeddings.device)

	with torch.no_grad():

		counts.scatter_add_(dim=0, index=targets, src=all_ones)
		counts_corrected = torch.max(counts, torch.ones_like(counts))
		mask = 1.-torch.abs(counts_corrected-counts).unsqueeze(-1).expand_as(centroids)
		centroids.scatter_add_(dim=0, index=targets.unsqueeze(-1).expand_as(embeddings), src=embeddings).div_(counts_corrected.unsqueeze_(-1))

	return centroids, mask

def adjust_learning_rate(optimizer, epoch, base_lr, n_epochs=30, lr_factor=0.1, min_lr=1e-8):
	"""Sets the learning rate to the initial LR decayed by 10 every n_epochs epochs"""
	lr = max( base_lr * (lr_factor ** (epoch // n_epochs)), min_lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def correct_topk(output, target, topk=(1,)):
	"""Computes the number of correct predicitions over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k)
		return res

def parse_args_for_log(args):
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		if args_dict[arg_key] is None:
			args_dict[arg_key] = 'None'
		elif isinstance(args_dict[arg_key], type([])):
			args_dict[arg_key] = str(args_dict[arg_key])

	return args_dict

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(2)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def strided_app(a, L, S):
	nrows = ( (len(a)-L) // S ) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

def get_classifier_config_from_cp(ckpt):
	keys=ckpt['model_state'].keys()
	classifier_params=[]
	out_proj_params=[]
	for x in keys:
		if 'classifier' in x:
			classifier_params.append(x)
		elif 'out_proj' in x:
			out_proj_params.append(x)
	
	n_hidden, hidden_size, softmax = max(len(classifier_params)//2 - 1, 1), ckpt['model_state']['classifier.0.weight'].size(0), 'am_softmax' if len(out_proj_params)==1 else 'softmax'

	if softmax == 'am_softmax':
		n_classes = ckpt['model_state']['out_proj.w'].size(1)
	elif softmax == 'softmax':
		n_classes = ckpt['model_state']['out_proj.w.weight'].size(0)

	return n_hidden, hidden_size, softmax, n_classes

def create_trials_labels(labels_list, max_n_trials=1e8):

	enroll_ex, test_ex, labels = [], [], []

	for i, prod_exs in enumerate(itertools.combinations(list(range(len(labels_list))), 2)):

		enroll_ex.append(prod_exs[0])
		test_ex.append(prod_exs[1])

		if labels_list[prod_exs[0]]==labels_list[prod_exs[1]]:
			labels.append(1)
		else:
			labels.append(0)

		if i>=max_n_trials: break

	return enroll_ex, test_ex, labels

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(5)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def compute_eer(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr

	t = np.nanargmin(np.abs(fnr-fpr))
	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	return eer

def compute_metrics(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	t = np.nanargmin(np.abs(fnr-fpr))

	eer_threshold = thresholds[t]

	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	auc = metrics.auc(fpr, tpr)

	avg_precision = metrics.average_precision_score(y, y_score)

	pred = np.asarray([1 if score > eer_threshold else 0 for score in y_score])
	acc = metrics.accuracy_score(y ,pred)

	return eer, auc, avg_precision, acc, eer_threshold

def read_trials(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	enroll_utt_list, test_utt_list, labels_list = [], [], []

	for line in utt_labels:
		enroll_utt, test_utt, label = line.split(' ')
		enroll_utt_list.append(enroll_utt)
		test_utt_list.append(test_utt)
		labels_list.append(1 if label=='target\n' else 0)

	return enroll_utt_list, test_utt_list, labels_list

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict
