from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from data_load import fewshot_eval_builder
from torchvision import transforms
from torch.utils.data import DataLoader
from RandAugment import RandAugment
from models import resnet, resnet12, wideresnet
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Mini-Imagenet few shot classification evaluation')
	parser.add_argument('--model', choices=['resnet', 'resnet_12', 'wideresnet'], default='resnet')
	parser.add_argument('--centroid-smoothing', type=float, default=0.9, metavar='Lamb', help='Moving average parameter for centroids')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (default: 5)')
	parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (default: 5)')
	parser.add_argument('--num-queries', type=int, default=15, help='Number of data points per class on test partition (default: 15)')
	parser.add_argument('--num-runs', type=int, default=600, help='Number of evaluation runs (default: 600)')
	parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to adapt centroids (default: 500)')
	parser.add_argument('--sgd-epochs', type=int, default=0, metavar='N', help='number of epochs to adapt centroids with SGD (default: 0)')
	parser.add_argument('--aug-M', type=int, default=15, metavar='AUGM', help='Augmentation hp. Default is 15')
	parser.add_argument('--aug-N', type=int, default=1, metavar='AUGN', help='Augmentation hp. Default is 1')
	parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='batch size(default: 24)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_train_eval = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(84, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(), add_noise(), transforms.Normalize(mean=mean, std=std)])
	transform_train_eval.transforms.insert(1, RandAugment(args.aug_N, args.aug_M))
	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	task_builder = fewshot_eval_builder(hdf5_name=args.data_path, train_transformation=transform_train_eval, test_transformation=transform_test, k_shot=args.num_shots, n_way=args.num_ways, n_queries=args.num_queries)


	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['centroids'].size(0)
	emb_size = ckpt['centroids'].size(1)

	print('\n', args, '\n')

	if args.model == 'resnet':
		model = resnet.ResNet18(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, centroids_lambda=args.centroid_smoothing, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'resnet_12':
		model = resnet12.ResNet12(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, centroids_lambda=args.centroid_smoothing, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'wideresnet':
		model = wideresnet.WideResNet(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, centroids_lambda=args.centroid_smoothing, sm_type=softmax, n_classes=n_classes)
	
	print(model.load_state_dict(ckpt['model_state'], strict=False))
	model.n_classes = args.num_ways

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = torch.device('cpu')

	model = model.to(device).eval()

	acc_list = []

	with torch.no_grad():

		for i in range(args.num_runs):

			centroids = torch.rand(args.num_ways, emb_size).to(device)
			
			train_dataset, test_dataset = task_builder.get_task_loaders()

			### Use the train split to compute the centroids

			dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

			for epoch in range(args.epochs):
				for batch in dataloader_train:

					x, y = batch
					x = x.to(device)
					y = y.to(device).squeeze()

					embeddings = model.forward(x)
					centroids = model.update_centroids_eval(centroids, embeddings, y, update_lambda=args.centroid_smoothing)

			if args.sgd_epochs>0:
				optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=True)
				for epoch in range(args.sgd_epochs):
					for batch in dataloader_train:

						optimizer.zero_grad()

						x, y = batch
						x = x.to(device)
						y = y.to(device).squeeze()

						embeddings = model.forward(x)
						out = model.compute_logits_eval(centroids, embeddings)
						loss = torch.nn.CrossEntropyLoss()(out, y)
						loss.backward()
						optimizer.step()

			### Eval on test split

			correct = 0

			dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

			for batch in dataloader_test:

				x, y = batch

				x = x.to(device)
				y = y.to(device).squeeze()

				embeddings = model.forward(x)
				out = model.compute_logits_eval(centroids, embeddings)
				pred = out.max(1)[1].long()
				correct += pred.squeeze().eq(y).sum().item()

			acc_list.append(100.*correct/len(test_dataset))
			print('Accuracy at round {}: {}'.format(i, acc_list[-1]))

	print('Accuracy: {} +- {}'.format(np.mean(acc_list), np.std(acc_list)))