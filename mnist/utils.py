import numpy as np
from sklearn import metrics

import torch
import itertools
import os
import sys
import pickle
from time import sleep
import random

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
			pic += torch.randn_like(pic)*random.choice([1e-1])

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

def get_classifier_config_from_model(model):
	x=0
	for layer in model.classifier:
		if isinstance(layer, torch.nn.Linear):
			x+=1
	return max(x-1, 1), model.classifier[0].weight.size(0)

def get_classifier_config_from_cp(ckpt):
	keys=ckpt['model_state'].keys()
	classifier_params=[]
	out_proj_params=[]
	for x in keys:
		if 'classifier' in x:
			classifier_params.append(x)
		elif 'out_proj' in x:
			out_proj_params.append(x)
	return max(len(classifier_params)//2 - 1, 1), ckpt['model_state']['classifier.0.weight'].size(0), 'am_softmax' if len(out_proj_params)==1 else 'softmax'

def create_trials_labels(labels_list):

	enroll_ex, test_ex, labels = [], [], []

	for prod_exs in itertools.combinations(list(range(len(labels_list))), 2):

		enroll_ex.append(prod_exs[0])
		test_ex.append(prod_exs[1])

		if labels_list[prod_exs[0]]==labels_list[prod_exs[1]]:
			labels.append(1)
		else:
			labels.append(0)

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

if __name__=='__main__':


	emb = torch.rand(8,5)
	nclasses = 10
	idx = torch.Tensor([1, 2, 4, 7, 4, 1, 1, 5]).long()
	prot = get_centroids(emb, idx, nclasses)

	print(prot)