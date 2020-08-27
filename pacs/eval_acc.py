from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from data_load import Loader_validation
from models import resnet
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
	parser.add_argument('--model', choices=['resnet'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--ablation-sim', action='store_true', default=False, help='Computes similarities as negative Euclidean distances')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.Resize(size=222), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	testset = Loader_validation(hdf_path=args.data_path, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['centroids'].size(0)
	except KeyError as err:
		print("Key Error: {0}".format(err))

	if args.model == 'resnet':
		model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	
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

	correct_ce_1, correct_ce_3, correct_sim_1, correct_sim_3, correct_mix_1, correct_mix_3 = 0, 0, 0, 0, 0, 0

	with torch.no_grad():

		iterator = tqdm(test_loader, total=len(test_loader))
		for batch in iterator:

			x, y = batch

			x = x.to(device)
			y = y.to(device).squeeze()

			embeddings = model.forward(x)

			pred_ce = F.softmax(model.out_proj(embeddings), dim=1)
			(correct_ce_1_, correct_ce_3_) = correct_topk(pred_ce, y, (1,3))

			pred_sim = F.softmax(model.compute_logits(embeddings, ablation=args.ablation_sim), dim=1)
			(correct_sim_1_, correct_cesim_3_) = correct_topk(pred_sim, y, (1,3))

			pred_mix = 0.5*pred_ce+0.5*pred_sim
			(correct_mix_1_, correct_mix_3_) = correct_topk(pred_mix, y, (1,3))

			correct_ce_1 += correct_ce_1_
			correct_ce_3 += correct_ce_3_
			correct_sim_1 += correct_sim_1_
			correct_sim_3 += correct_sim_3_
			correct_mix_1 += correct_mix_1_
			correct_mix_3 += correct_mix_3_

	print('\nCE Top 1 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_ce_1/len(testset)))
	print('\nCE Top 3 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_ce_3/len(testset)))
	print('\nSIM Top 1 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_sim_1/len(testset)))
	print('\nSIM Top 3 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_sim_3/len(testset)))
	print('\nMIX Top 1 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_mix_1/len(testset)))
	print('\nMIX Top 3 Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct_mix_3/len(testset)))

