from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import resnet, wideresnet
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Symmetry check')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['resnet', 'wideresnet'], default='resnet')
	parser.add_argument('--sample-size', type=int, default=5000, metavar='N', help='Sample size')
	parser.add_argument('--max-nontarget', type=int, default=50000, metavar='N', help='Maximum number of non target trials to be considered')
	parser.add_argument('--out-path', type=str, default='', metavar='Path', help='Path for saving outputs')
	parser.add_argument('--out-prefix', type=str, default='', metavar='Path', help='Prefix to be added to output file name')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-histogram', action='store_true', default=False, help='Disables histogram plot')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	validset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	labels_list = [x[1] for x in validset]

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
	except KeyError as err:
		print("Key Error: {0}".format(err))

	if args.model == 'resnet':
		model = resnet.ResNet18(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'wideresnet':
		model = wideresnet.WideResNet(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	
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

	idxs_enroll, idxs_test, labels = create_trials_labels(labels_list, max_non_target_trials=args.max_nontarget)
	print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

	target_scores_e2e = []
	non_target_scores_e2e = []
	target_scores_cos = []
	non_target_scores_cos = []

	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		iterator = tqdm(range(len(labels)), total=len(labels))
		for i in iterator:

			enroll_ex = str(idxs_enroll[i])

			try:
				emb_enroll = mem_embeddings[enroll_ex]
			except KeyError:

				enroll_ex_data = validset[idxs_enroll[i]][0].unsqueeze(0)

				if args.cuda:
					enroll_ex_data = enroll_ex_data.cuda(device)

				emb_enroll = model.forward(enroll_ex_data).detach()
				mem_embeddings[str(idxs_enroll[i])] = emb_enroll

			test_ex = str(idxs_test[i])

			try:
				emb_test = mem_embeddings[test_ex]
			except KeyError:

				test_ex_data = validset[idxs_test[i]][0].unsqueeze(0)

				if args.cuda:
					test_ex_data = test_ex_data.cuda(device)

				emb_test = model.forward(test_ex_data).detach()
				mem_embeddings[str(idxs_test[i])] = emb_test

			e2e_score = torch.sigmoid(model.forward_bin(emb_enroll, emb_test)).squeeze().item()
			cos_score = torch.nn.functional.cosine_similarity(emb_enroll, emb_test).squeeze().item()
	
			if labels[i] == 1:
				target_scores_e2e.append(e2e_score)
				target_scores_cos.append(cos_score)
			elif labels[i] == 0:
				non_target_scores_e2e.append(e2e_score)
				non_target_scores_cos.append(cos_score)
			else:
				print(f'Unknown label: {labels[i]}')

	print('\nScoring done')

	if not args.no_histogram:
		import matplotlib
		matplotlib.rcParams['pdf.fonttype'] = 42
		matplotlib.rcParams['ps.fonttype'] = 42
		matplotlib.use('agg')
		import matplotlib.pyplot as plt
		plt.hist((target_scores_e2e, non_target_scores_e2e), density=True, bins=30, color=('blue', 'red'))
		plt.savefig(os.path.join(args.out_path, args.out_prefix+'e2e_scores_hist_cifar.pdf'), bbox_inches='tight')
		plt.hist((target_scores_cos, non_target_scores_cos), density=True, bins=30, color=('blue', 'red'))
		plt.savefig(os.path.join(args.out_path, args.out_prefix+'cos_scores_hist_cifar.pdf'), bbox_inches='tight')
