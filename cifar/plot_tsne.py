from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import resnet, wideresnet, wrapper_racc
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
	parser.add_argument('--inf-mode', choices=['sim', 'ce', 'fus'], default='sim', help='Inference mode')
	parser.add_argument('--out-path', type=str, default='', metavar='Path', help='Path for saving outputs')
	parser.add_argument('--out-prefix', type=str, default='', metavar='Path', help='Prefix to be added to output file name')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-histogram', action='store_true', default=False, help='Disables histogram plot')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	validset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	preprocessing = dict(mean=[x / 255 for x in [125.3, 123.0, 113.9]], std=[x / 255 for x in [63.0, 62.1, 66.7]], axis=-3)

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

	model = wrapper_racc.wrapper(base_model=model, inf_mode=args.inf_mode)

	model.eval()

	fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=device)

	clean_embeddings, attack_embeddings = [], []

	model.eval()

	with torch.no_grad():

		iterator = tqdm(range(len(validset)), total=len(validset))
		for i in iterator:

			enroll_ex_data = validset[i][0].unsqueeze(0)

			if args.cuda:
				enroll_ex_data = enroll_ex_data.cuda(device)

			emb_enroll = model.forward(enroll_ex_data).detach()

			scores_dif.append( 1.-torch.sigmoid(model.forward_bin(emb_enroll, emb_enroll)).squeeze().item() )

	print('\nScoring done')

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
		plt.savefig(os.path.join(args.out_path, args.out_prefix, 'met_hist_cifar.pdf'), bbox_inches='tight')
