from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
from data_load import Loader, collater
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='ImageNet Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--hdf-path', type=str, default=None, metavar='Path', help='Path to data stored in hdf. Has priority over data path if set')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--ablation-sim', action='store_true', default=False, help='Computes similarities as negative Euclidean distances')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.hdf_path:
		transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
		testset = Loader(args.hdf_path, transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True, collate_fn=collater)
	else:
		transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
		testset = datasets.ImageFolder(args.data_path, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['centroids'].size(0)
	except KeyError as err:
		print("Key Error: {0}".format(err))

	if args.model == 'vgg':
		model = vgg.VGG('VGG16', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'resnet':
		model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'densenet':
		model = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	
	print(model.load_state_dict(ckpt['model_state'], strict=False), '\n')
	model.centroids = ckpt['centroids']

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = torch.device('cpu')

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