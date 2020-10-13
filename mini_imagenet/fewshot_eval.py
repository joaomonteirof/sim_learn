from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
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
import copy

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
	parser.add_argument('--transductive', action='store_true', default=False, help='Enables use query set for centering')
	### fine tuning config
	parser.add_argument('--finetune-epochs', type=int, default=0, metavar='N', help='number of epochs to adapt centroids (default: 0)')
	parser.add_argument('--centroid-smoothing', type=float, default=0.9, metavar='Lamb', help='Moving average parameter for centroids')
	parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate (default: 0.00001)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='momentum', help='momentum (default: 0.5)')
	parser.add_argument('--l2', type=float, default=1e-3, metavar='lambda', help='L2 wheight decay coefficient (default: 0.001)')
	parser.add_argument('--aug-M', type=int, default=15, metavar='AUGM', help='Augmentation hp. Default is 15')
	parser.add_argument('--aug-N', type=int, default=2, metavar='AUGN', help='Augmentation hp. Default is 2')
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
	if args.finetune_epochs>0:
		results.update({'acc_list_sim_finetune':[], 'acc_list_cos_finetune':[], 'acc_list_fus_finetune':[]})

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

		if args.finetune_epochs>0:

			dataloader_train.dataset.transformation = transform_train

			model_finetune = copy.deepcopy(model).to(device).train()
			model_finetune.load_state_dict(model.state_dict())
			model_finetune.centroids = centroids.clone()
			model_finetune.n_classes = args.num_ways
			optimizer = optim.SGD(model_finetune.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

			for epoch in range(args.finetune_epochs):
				for batch in dataloader_train:

					x, y = batch
					x = x.to(device)
					y = y.to(device).squeeze()

					embeddings = model.forward(x).detach()

					model_finetune.update_centroids(embeddings, y)

					sim_loss = torch.nn.CrossEntropyLoss()(model_finetune.compute_logits(embeddings), y)

					optimizer.zero_grad()
					sim_loss.backward()
					optimizer.step()

			dataloader_train.dataset.transformation = transform_test
			centroids_finetune = model_finetune.centroids

		else:
			centroids_finetune = None

		### Eval on test split

		correct_sim, correct_cos, correct_fus = 0, 0, 0
		correct_sim_finetune, correct_cos_finetune, correct_fus_finetune = 0, 0, 0

		dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

		with torch.no_grad():

			if args.transductive:
				centroids_copy = centroids.clone()
				if centroids_finetune:
					centroids_finetune_copy = centroids_finetune.clone()

			for batch in dataloader_test:

				x, y = batch

				x = x.to(device)
				y = y.to(device).squeeze()

				embeddings = model.forward(x)

				if args.transductive:
					centroids = centroids_copy - embeddings.mean(0, keepdim=True).repeat(centroids_copy.size(0),1)
					if centroids_finetune:
						centroids_finetune = centroids_finetune_copy - embeddings.mean(0, keepdim=True).repeat(centroids_finetune_copy.size(0),1)
					embeddings -= embeddings.mean(0, keepdim=True).repeat(embeddings.size(0),1)

				out_sim = model.compute_logits_eval(centroids, embeddings)
				pred_sim = out_sim.max(1)[1].long()
				correct_sim += pred_sim.squeeze().eq(y).sum().item()
				out_cos = model.compute_logits_eval(centroids, embeddings, ablation=True)
				pred_cos = out_cos.max(1)[1].long()
				correct_cos += pred_cos.squeeze().eq(y).sum().item()
				out_fus = (F.softmax(out_sim, dim=1)+F.softmax(out_cos, dim=1))*0.5
				pred_fus = out_fus.max(1)[1].long()
				correct_fus += pred_fus.squeeze().eq(y).sum().item()

				if centroids_finetune is not None:
					model_finetune.eval()
					out_sim_finetune = model_finetune.compute_logits_eval(centroids_finetune, embeddings)
					pred_sim_finetune = out_sim_finetune.max(1)[1].long()
					correct_sim_finetune += pred_sim_finetune.squeeze().eq(y).sum().item()
					out_cos_finetune = model_finetune.compute_logits_eval(centroids_finetune, embeddings, ablation=True)
					pred_cos_finetune = out_cos_finetune.max(1)[1].long()
					correct_cos_finetune += pred_cos_finetune.squeeze().eq(y).sum().item()
					out_fus_finetune = (F.softmax(out_sim_finetune, dim=1)+F.softmax(out_cos_finetune, dim=1))*0.5
					pred_fus_finetune = out_fus_finetune.max(1)[1].long()
					correct_fus_finetune += pred_fus_finetune.squeeze().eq(y).sum().item()

		results['acc_list_sim'].append(100.*correct_sim/len(test_dataset))
		results['acc_list_cos'].append(100.*correct_cos/len(test_dataset))
		results['acc_list_fus'].append(100.*correct_fus/len(test_dataset))

		if centroids_finetune is not None:
			results['acc_list_sim_finetune'].append(100.*correct_sim_finetune/len(test_dataset))
			results['acc_list_cos_finetune'].append(100.*correct_cos_finetune/len(test_dataset))
			results['acc_list_fus_finetune'].append(100.*correct_fus_finetune/len(test_dataset))

		if i % args.report_every == 0:
			print('\nAccuracy at round {}/{}:\n'.format(i+1,args.num_runs))
			for el in results:
				mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(i + 1)
				print('{} --- Current ACC: {:.2f} \t Accumulated: {:.2f} +- {:.2f}'.format(el, results[el][-1], mean, ci95))

	print('\n\nFinal accuracy:\n')
	for el in results:
		mean, ci95 = np.mean(results[el]), 1.96 * np.std(results[el]) / np.sqrt(len(results[el]))
		print('{} --- Current ACC: {:.2f} \t Accumulated: {:.2f} +- {:.2f}'.format(el, results[el][-1], mean, ci95))

