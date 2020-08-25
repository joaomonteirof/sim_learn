from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import resnet, wideresnet
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Cifar10 ACC Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['resnet', 'wideresnet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--ablation-sim', action='store_true', default=False, help='Computes similarities as negative Euclidean distances')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
	except KeyError as err:
		print("Key Error: {0}".format(err))
		print('\nProbably old cp has no info regarding classifiers arch!\n')
		n_hidden, hidden_size, softmax = get_classifier_config_from_cp(ckpt)
		dropout_prob = args.dropout_prob

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

	model.eval()

	correct_ce, correct_sim, correct_mix = 0, 0, 0

	with torch.no_grad():

		iterator = tqdm(test_loader, total=len(test_loader))
		for batch in iterator:

			x, y = batch

			x = x.to(device)
			y = y.to(device).squeeze()

			embeddings = model.forward(x)

			out_ce = F.softmax(model.out_proj(embeddings), dim=1)
			pred_ce = out_ce.max(1)[1].long()
			correct_ce += pred_ce.squeeze().eq(y).sum().item()

			out_sim = F.softmax(model.compute_logits(embeddings, ablation=args.ablation_sim), dim=1)
			pred_sim = out_sim.max(1)[1].long()
			correct_sim += pred_sim.squeeze().eq(y).sum().item()

			out_mix = 0.5*out_ce+0.5*out_sim
			pred_mix = out_mix.max(1)[1].long()
			correct_mix += pred_mix.squeeze().eq(y).sum().item()

	print('\nCE Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_ce/len(testset)))
	print('\nSIM Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_sim/len(testset)))
	print('\nMIX Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_mix/len(testset)))

