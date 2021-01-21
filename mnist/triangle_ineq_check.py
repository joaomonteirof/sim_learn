from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import cnn
import numpy as np
import os
import sys
import itertools
from tqdm import tqdm
from utils import *
import random

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Triangle Inequality check')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['cnn'], default='cnn')
	parser.add_argument('--sample-size', type=int, default=1000, metavar='N', help='Sample size')
	parser.add_argument('--out-path', type=str, default='', metavar='Path', help='Path for saving outputs')
	parser.add_argument('--out-prefix', type=str, default='', metavar='Path', help='Prefix to be added to output file name')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-histogram', action='store_true', default=False, help='Disables histogram plot')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	validset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
	idx_list = random.sample(range(len(validset)), min(len(validset), args.sample_size))

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
	except KeyError as err:
		print("Key Error: {0}".format(err))

	if args.model == 'cnn':
		model = cnn.cnn(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	
	try:
		print(model.load_state_dict(ckpt['model_state'], strict=True), '\n')
		model.centroids = ckpt['centroids']
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = torch.device('cpu')

	model = model.to(device)
	model.centroids = model.centroids.to(device)

	scores_dif = []

	mem_embeddings = {}
	mem_dists = {}

	model.eval()

	with torch.no_grad():

		print('\nPreparing distance dictionary.')

		pairs = itertools.combinations(range(len(idx_list)), 2)
		iterator = tqdm(pairs, total=len(idx_list)*(len(idx_list)-1)/2)

		for i, j in iterator:

			anchor_ex = str(i)

			try:
				emb_anchor = mem_embeddings[anchor_ex]
			except KeyError:

				anchor_ex_data = validset[idx_list[i]][0].unsqueeze(0)

				if args.cuda:
					anchor_ex_data = anchor_ex_data.cuda(device)

				emb_anchor = model.forward(anchor_ex_data).detach()
				mem_embeddings[anchor_ex] = emb_anchor

			a_ex = str(j)

			try:
				emb_a = mem_embeddings[a_ex]
			except KeyError:

				a_ex_data = validset[idx_list[j]][0].unsqueeze(0)

				if args.cuda:
					a_ex_data = a_ex_data.cuda(device)

				emb_a = model.forward(a_ex_data).detach()
				mem_embeddings[a_ex] = emb_a

			mem_dists[anchor_ex+'_'+a_ex] = 1.0-torch.sigmoid(model.forward_bin(emb_anchor, emb_a)).squeeze().item()
			mem_dists[a_ex+'_'+anchor_ex] = 1.0-torch.sigmoid(model.forward_bin(emb_a, emb_anchor)).squeeze().item()


		print('\nComputing scores differences.')

		triplets = itertools.combinations(range(len(idx_list)), 3)
		iterator = tqdm(triplets, total=len(idx_list)*(len(idx_list)-1)*(len(idx_list)-2)/6)

		for i, j, k in iterator:

			total_dist = mem_dists[str(i)+'_'+str(j)] + mem_dists[str(i)+'_'+str(k)]
			local_dist = mem_dists[str(j)+'_'+str(k)]

			scores_dif.append( max(local_dist-total_dist, 0.0) )

	print('\nScoring done.')

	print('Avg: {}'.format(np.mean(scores_dif)))
	print('Std: {}'.format(np.std(scores_dif)))
	print('Median: {}'.format(np.median(scores_dif)))
	print('Max: {}'.format(np.max(scores_dif)))
	print('Min: {}'.format(np.min(scores_dif)))

	if not args.no_histogram:
		import matplotlib
		matplotlib.rcParams['pdf.fonttype'] = 42
		matplotlib.rcParams['ps.fonttype'] = 42
		matplotlib.use('agg')
		import matplotlib.pyplot as plt
		plt.hist(scores_dif, density=True, bins=30)
		plt.savefig(os.path.join(args.out_path, args.out_prefix+'triang_hist_mnist.pdf'), bbox_inches='tight')
