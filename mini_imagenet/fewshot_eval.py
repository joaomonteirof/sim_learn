from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from data_load import fewshot_eval_builder
from torchvision import transforms
from RandAugment import RandAugment
from torch.utils.data import DataLoader
from models import resnet, resnet12, wideresnet
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Mini-Imagenet few shot classification evaluation')
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
	### fine tuning config
	parser.add_argument('--sgd-epochs', type=int, default=0, metavar='N', help='number of epochs to adapt centroids with SGD (default: 0)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum', help='Momentum (default: 0.9)')
	parser.add_argument('--aug-M', type=int, default=15, metavar='AUGM', help='Augmentation hp. Default is 15')
	parser.add_argument('--aug-N', type=int, default=1, metavar='AUGN', help='Augmentation hp. Default is 1')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False


	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	transform_train.transforms.insert(1, RandAugment(args.aug_N, args.aug_M))
	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

	## transfor is set to trasnform_test here for both case and the train one is updated only when fine tuning is performed
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
	if args.sgd_epochs>0:
		results.update({'acc_list_sim_sgdsim':[], 'acc_list_cos_sgdsim':[], 'acc_list_fus_sgdsim':[]})
		results.update({'acc_list_sim_sgdcos':[], 'acc_list_cos_sgdcos':[], 'acc_list_fus_sgdcos':[]})

	for i in range(args.num_runs):
		
		train_dataset, test_dataset = task_builder.get_task_loaders()

		### Use the train split to compute the centroids

		dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

		embeddings = []
		labels = []

		with torch.no_grad():

			for batch in dataloader_train:

				x, y = batch

				if args.cuda:
					x = x.to(device)

				emb = model.forward(x).detach()

				embeddings.append(emb)
				labels.append(y)

		embeddings = torch.cat(embeddings, 0).to(device)
		labels = torch.cat(labels, 0).to(device).squeeze(-1)

		centroids, _ = get_centroids(embeddings, labels, args.num_ways)

		if args.sgd_epochs>0:

			task_builder.train_transformation = transform_train

			centroids_sgd_sim, centroids_sgd_cos = centroids.clone(), centroids.clone()
			centroids_sgd_sim.requires_grad, centroids_sgd_cos.requires_grad = True, True
			optimizer_sim = optim.SGD([centroids_sgd_sim], lr=args.lr, momentum=args.momentum, weight_decay=0.0)
			optimizer_cos = optim.SGD([centroids_sgd_cos], lr=args.lr, momentum=args.momentum, weight_decay=0.0)

			for epoch in range(args.sgd_epochs):
				for batch in dataloader_train:

					optimizer_sim.zero_grad()
					optimizer_cos.zero_grad()

					x, y = batch
					x = x.to(device)
					y = y.to(device).squeeze()

					embeddings = model.forward(x).detach()

					out_sim = torch.sigmoid(model.compute_logits_eval(centroids_sgd_sim, embeddings, ablation=False))
					out_cos = model.compute_logits_eval(centroids_sgd_cos, embeddings, ablation=True)

					loss_sim = torch.nn.CrossEntropyLoss()(out_sim, y)
					loss_cos = torch.nn.CrossEntropyLoss()(out_cos, y)

					loss_sim.backward()
					optimizer_sim.step()

					loss_cos.backward()
					optimizer_cos.step()

			task_builder.train_transformation = transform_test

		else:
			centroids_sgd_sim, centroids_sgd_cos = None, None

		### Eval on test split

		correct_sim, correct_cos, correct_fus = 0, 0, 0
		correct_sim_sgdsim, correct_cos_sgdsim, correct_fus_sgdsim = 0, 0, 0
		correct_sim_sgdcos, correct_cos_sgdcos, correct_fus_sgdcos = 0, 0, 0

		dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

		with torch.no_grad():

			for batch in dataloader_test:

				x, y = batch

				x = x.to(device)
				y = y.to(device).squeeze()

				embeddings = model.forward(x)

				out_sim = torch.sigmoid(model.compute_logits_eval(centroids, embeddings))
				pred_sim = out_sim.max(1)[1].long()
				correct_sim += pred_sim.squeeze().eq(y).sum().item()
				out_cos = model.compute_logits_eval(centroids, embeddings, ablation=True)
				pred_cos = out_cos.max(1)[1].long()
				correct_cos += pred_cos.squeeze().eq(y).sum().item()
				out_fus = (F.softmax(out_sim, dim=1)+F.softmax(out_cos, dim=1))*0.5
				pred_fus = out_fus.max(1)[1].long()
				correct_fus += pred_fus.squeeze().eq(y).sum().item()

				if centroids_sgd_sim is not None:
					out_sim_sgd = model.compute_logits_eval(centroids_sgd_sim, embeddings)
					pred_sim_sgd = out_sim_sgd.max(1)[1].long()
					correct_sim_sgdsim += pred_sim_sgd.squeeze().eq(y).sum().item()
					out_cos_sgd = model.compute_logits_eval(centroids_sgd_sim, embeddings, ablation=True)
					pred_cos_sgd = out_cos_sgd.max(1)[1].long()
					correct_cos_sgdsim += pred_cos_sgd.squeeze().eq(y).sum().item()
					out_fus_sgd = (F.softmax(out_sim_sgd, dim=1)+F.softmax(out_cos_sgd, dim=1))*0.5
					pred_fus_sgd = out_fus_sgd.max(1)[1].long()
					correct_fus_sgdsim += pred_fus_sgd.squeeze().eq(y).sum().item()

				if centroids_sgd_cos is not None:
					out_sim_sgd = model.compute_logits_eval(centroids_sgd_cos, embeddings)
					pred_sim_sgd = out_sim_sgd.max(1)[1].long()
					correct_sim_sgdcos += pred_sim_sgd.squeeze().eq(y).sum().item()
					out_cos_sgd = model.compute_logits_eval(centroids_sgd_cos, embeddings, ablation=True)
					pred_cos_sgd = out_cos_sgd.max(1)[1].long()
					correct_cos_sgdcos += pred_cos_sgd.squeeze().eq(y).sum().item()
					out_fus_sgd = (F.softmax(out_sim_sgd, dim=1)+F.softmax(out_cos_sgd, dim=1))*0.5
					pred_fus_sgd = out_fus_sgd.max(1)[1].long()
					correct_fus_sgdcos += pred_fus_sgd.squeeze().eq(y).sum().item()

		results['acc_list_sim'].append(100.*correct_sim/len(test_dataset))
		results['acc_list_cos'].append(100.*correct_cos/len(test_dataset))
		results['acc_list_fus'].append(100.*correct_fus/len(test_dataset))

		if centroids_sgd_sim is not None:
			results['acc_list_sim_sgdsim'].append(100.*correct_sim_sgdsim/len(test_dataset))
			results['acc_list_cos_sgdsim'].append(100.*correct_cos_sgdsim/len(test_dataset))
			results['acc_list_fus_sgdsim'].append(100.*correct_fus_sgdsim/len(test_dataset))

		if centroids_sgd_cos is not None:
			results['acc_list_sim_sgdcos'].append(100.*correct_sim_sgdcos/len(test_dataset))
			results['acc_list_cos_sgdcos'].append(100.*correct_cos_sgdcos/len(test_dataset))
			results['acc_list_fus_sgdcos'].append(100.*correct_fus_sgdcos/len(test_dataset))

		if i % args.report_every == 0:
			print('\nAccuracy at round {}/{}:\n'.format(i+1,args.num_runs))
			for el in results:
				mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(i + 1)
				print('{} --- Current ACC: {:.2f} \t Accumulated: {:.2f} +- {:.2f}'.format(el, results[el][-1], mean, ci95))

	print('\n\nFinal accuracy:\n')
	for el in results:
		mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(len(results[el]))
		print('{} --- Current ACC: {:.2f} \t Accumulated: {:.2f} +- {:.2f}'.format(el, results[el][-1], mean, ci95))

