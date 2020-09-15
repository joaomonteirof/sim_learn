from __future__ import print_function
import argparse
import torch
import torch.optim as optim
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

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Mini-Imagenet few shot classification evaluation')
	parser.add_argument('--model', choices=['resnet', 'resnet_12', 'wideresnet'], default='resnet')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (default: 5)')
	parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (default: 5)')
	parser.add_argument('--num-queries', type=int, default=15, help='Number of data points per class on test partition (default: 15)')
	parser.add_argument('--num-runs', type=int, default=600, help='Number of evaluation runs (default: 600)')
	parser.add_argument('--sgd-epochs', type=int, default=0, metavar='N', help='number of epochs to adapt centroids with SGD (default: 0)')
	parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='batch size(default: 24)')
	parser.add_argument('--report-every', type=int, default=50, metavar='N', help='Number of runs to wait before reporting current results (default: 50)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	task_builder = fewshot_eval_builder(hdf5_name=args.data_path, train_transformation=transform, test_transformation=transform, k_shot=args.num_shots, n_way=args.num_ways, n_queries=args.num_queries)


	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['centroids'].size(0)
	emb_size = ckpt['centroids'].size(1)

	print('\n', args, '\n')

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
		results.update({'acc_list_sim_sgd':[], 'acc_list_cos_sgd':[], 'acc_list_fus_sgd':[]})

	for i in range(args.num_runs):

		centroids = torch.rand(args.num_ways, emb_size).to(device)
		
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

				embeddings.append(emb.detach().cpu())
				labels.append(y)

		embeddings = torch.cat(embeddings, 0)
		labels = torch.cat(labels, 0).to(device)

		centroids = get_centroids(embeddings, labels, args.num_ways)

		if args.sgd_epochs>0:
			centroids_sgd = centroids.clone()
			optimizer = optim.SGD([centroids_sgd], lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=True)
			for epoch in range(args.sgd_epochs):
				for batch in dataloader_train:

					optimizer.zero_grad()

					x, y = batch
					x = x.to(device)
					y = y.to(device).squeeze()

					embeddings = model.forward(x).detach()
					out = model.compute_logits_eval(centroids, embeddings)
					loss = torch.nn.CrossEntropyLoss()(out, y)
					loss.backward()
					optimizer.step()
		else:
			centroids_sgd = None

		### Eval on test split

		correct_sim, correct_cos, correct_fus = 0, 0, 0
		correct_sim_sgd, correct_cos_sgd, correct_fus_sgd = 0, 0, 0

		dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

		with torch.no_grad():

			for batch in dataloader_test:

				x, y = batch

				x = x.to(device)
				y = y.to(device).squeeze()

				embeddings = model.forward(x)

				out_sim = model.compute_logits_eval(centroids, embeddings)
				pred_sim = out_sim.max(1)[1].long()
				correct_sim += pred_sim.squeeze().eq(y).sum().item()
				out_cos = model.compute_logits_eval(centroids, embeddings, ablation=True)
				pred_cos = out_cos.max(1)[1].long()
				correct_cos += pred_cos.squeeze().eq(y).sum().item()
				out_fus = (F.softmax(out_sim, dim=1)+F.softmax(out_cos, dim=1))*0.5

				if centroids_sgd is not None:
					out_sim_sgd = model.compute_logits_eval(centroids_sgd, embeddings)
					pred_sim_sgd = out_sim_sgd.max(1)[1].long()
					correct_sim_sgd += pred_sim_sgd.squeeze().eq(y).sum().item()
					out_cos_sgd = model.compute_logits_eval(centroids_sgd, embeddings, ablation=True)
					pred_cos_sgd = out_cos_sgd.max(1)[1].long()
					correct_cos_sgd += pred_cos_sgd.squeeze().eq(y).sum().item()
					out_fus_sgd = (F.softmax(out_sim_sgd, dim=1)+F.softmax(out_cos_sgd, dim=1))*0.5

		results['acc_list_sim'].append(100.*correct_sim/len(test_dataset))
		results['acc_list_cos'].append(100.*correct_cos/len(test_dataset))
		results['acc_list_fus'].append(100.*correct_fus/len(test_dataset))

		if centroids_sgd is not None:
			results['acc_list_sim_sgd'].append(100.*correct_sim_sgd/len(test_dataset))
			results['acc_list_cos_sgd'].append(100.*correct_cos_sgd/len(test_dataset))
			results['acc_list_fus_sgd'].append(100.*correct_fus_sgd/len(test_dataset))

		if i % args.report_every == 0:
			print('Accuracy at round {}/{}:\n'.format(i+1,args.num_runs))
			for el in results:
				mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(i + 1)
				print('el: {:.2f} +- {:.2f}'.format(mean, ci95))

