from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from data_load import fewshot_eval_builder
from torchvision import transforms
from torch.utils.data import DataLoader
from models import resnet, resnet12, wideresnet
import os
import sys
from tqdm import tqdm
from utils import *
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Mini-Imagenet few shot classification evaluation with k-nn classifiers')
	parser.add_argument('--model', choices=['resnet', 'resnet_12', 'wideresnet'], default='resnet')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (default: 5)')
	parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (default: 5)')
	parser.add_argument('--num-queries', type=int, default=15, help='Number of data points per class on test partition (default: 15)')
	parser.add_argument('--num-runs', type=int, default=600, help='Number of evaluation runs (default: 600)')
	parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='batch size(default: 24)')
	parser.add_argument('--report-every', type=int, default=50, metavar='N', help='Number of runs to wait before reporting current results (default: 50)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False


	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	task_builder = fewshot_eval_builder(hdf5_name=args.data_path, train_transformation=transform_test, test_transformation=transform_test, k_shot=args.num_shots, n_way=args.num_ways, n_queries=args.num_queries)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['centroids'].size(0)
	emb_size = ckpt['centroids'].size(1)

	print('\n')
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		print('{}: {}'.format(arg_key, args_dict[arg_key]))
	print('\n')

	if args.model == 'resnet':
		model = resnet.ResNet18(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'resnet_12':
		model = resnet12.ResNet12(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'wideresnet':
		model = wideresnet.WideResNet(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	
	print(model.load_state_dict(ckpt['model_state'], strict=False))
	model.n_classes = args.num_ways

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = torch.device('cpu')

	model = model.to(device).eval()

	results = {'acc_list_sim':[], 'acc_list_cos':[], 'acc_list_fus':[]}

	for i in range(args.num_runs):
		
		train_dataset, test_dataset = task_builder.get_task_loaders()

		### Use the train split to compute the centroids

		dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

		embeddings_train = []
		labels_train = []

		with torch.no_grad():

			for batch in dataloader_train:

				x, y = batch

				if args.cuda:
					x = x.to(device)

				emb = model.forward(x).detach().cpu()

				embeddings_train.append(emb)
				labels_train.append(y)

		embeddings_train = torch.cat(embeddings_train, 0).cpu().numpy()
		labels_train = torch.cat(labels_train, 0).squeeze(-1).cpu().numpy()

		### Eval on test split

		correct_sim, correct_cos = 0, 0

		dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

		embeddings_test = []
		labels_test = []

		with torch.no_grad():

			for batch in dataloader_test:

				x, y = batch

				if args.cuda:
					x = x.to(device)

				emb = model.forward(x).detach().cpu()

				embeddings_test.append(emb)
				labels_test.append(y)

		embeddings_test = torch.cat(embeddings_test, 0).cpu().numpy()
		labels_test = torch.cat(labels_test, 0).to(device).squeeze(-1).cpu()

		def dist_metric_sim(a,b):
			a, b = torch.Tensor(a).float().to(device).unsqueeze(0), torch.Tensor(b).float().to(device).unsqueeze(0)
			return (1.-torch.sigmoid(model.forward_bin(a,b))).squeeze().cpu()

		def dist_metric_cos(a,b):
			a, b = torch.Tensor(a).float(), torch.Tensor(b).float()
			return -F.cosine_similarity(a,b,dim=0).squeeze()

		def dist_metric_fus(a,b):
			a, b = torch.Tensor(a).float(), torch.Tensor(b).float()
			sim_cos = ((1.-torch.sigmoid(model.forward_bin(a,b))).squeeze().cpu() + -0.5*(1+F.cosine_similarity(a,b,dim=0).squeeze().cpu()))*0.5
			return sim_cos

		neigh_sim = KNeighborsClassifier(n_neighbors=args.num_shots//2+1, metric=dist_metric_sim, algorithm='brute')
		neigh_sim.fit(embeddings_train, labels_train)
		pred_sim = torch.Tensor(neigh_sim.predict(embeddings_test)).long()

		neigh_cos = KNeighborsClassifier(n_neighbors=args.num_shots//2+1, metric=dist_metric_cos, algorithm='brute')
		neigh_cos.fit(embeddings_train, labels_train)
		pred_cos = torch.Tensor(neigh_cos.predict(embeddings_test)).long()

		neigh_fus = KNeighborsClassifier(n_neighbors=args.num_shots//2+1, metric=dist_metric_fus, algorithm='brute')
		neigh_fus.fit(embeddings_train, labels_train)
		pred_fus = torch.Tensor(neigh_fus.predict(embeddings_test)).long()

		results['acc_list_sim'].append(100.*pred_sim.eq(labels_test).sum().item()/labels_test.size(0))
		results['acc_list_cos'].append(100.*pred_cos.eq(labels_test).sum().item()/labels_test.size(0))
		results['acc_list_fus'].append(100.*pred_fus.eq(labels_test).sum().item()/labels_test.size(0))

		if i % args.report_every == 0:
			print('\nAccuracy at round {}/{}:\n'.format(i+1,args.num_runs))
			for el in results:
				mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(i + 1)
				print('{} --- Current ACC: {:.2f} \t Accumulated: {:.2f} +- {:.2f}'.format(el, results[el][-1], mean, ci95))

	print('\n\nFinal accuracy:\n')
	for el in results:
		mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(len(results[el]))
		print('{} --- Current ACC: {:.2f} \t Accumulated: {:.2f} +- {:.2f}'.format(el, results[el][-1], mean, ci95))

