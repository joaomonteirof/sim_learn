from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='ImageNet Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--ablation-sim', action='store_true', default=False, help='Computes similarities as negative Euclidean distances')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	testset = datasets.ImageFolder(args.data_path, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

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

	model = model.to(device)
	model.centroids = model.centroids.to(device)

	model.eval()

	correct_ce_1, correct_ce_5, correct_sim_1, correct_sim_5, correct_mix_1, correct_mix_5 = 0, 0, 0, 0, 0, 0
	total = 0

	with torch.no_grad():

		iterator = tqdm(test_loader, total=len(test_loader))
		for batch in iterator:

			x, y = batch

			x = x.to(device)
			y = y.to(device)

			embeddings = model.forward(x)

			pred_ce = F.softmax(model.out_proj(embeddings), dim=1)
			(correct_ce_1_, correct_ce_5_) = correct_topk(pred_ce, y, (1,5))

			pred_sim = F.softmax(model.compute_logits(embeddings, ablation=args.ablation_sim), dim=1)
			(correct_sim_1_, correct_sim_5_) = correct_topk(pred_sim, y, (1,5))

			pred_mix = 0.5*pred_ce+0.5*pred_sim
			(correct_mix_1_, correct_mix_5_) = correct_topk(pred_mix, y, (1,5))

			correct_ce_1 += correct_ce_1_
			correct_ce_5 += correct_ce_5_
			correct_sim_1 += correct_sim_1_
			correct_sim_5 += correct_sim_5_
			correct_mix_1 += correct_mix_1_
			correct_mix_5 += correct_mix_5_
			total += x.size(0)

	print('\nCE Top 1 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*float(correct_ce_1)/total))
	print('\nCE Top 5 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*float(correct_ce_5)/total))
	print('\nSIM Top 1 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*float(correct_sim_1)/total))
	print('\nSIM Top 5 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*float(correct_sim_5)/total))
	print('\nMIX Top 1 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*float(correct_mix_1)/total))
	print('\nMIX Top 5 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*float(correct_mix_5)/total))

