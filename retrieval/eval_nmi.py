from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *
from sklearn.metrics.cluster import normalized_mutual_info_score


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Clustering Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-path', type=str, default=None, metavar='Path', help='Path to output embeddings.')
	parser.add_argument('--emb-path', type=str, default=None, metavar='Path', help='Path to precomputed embedding.')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	print(args)

	transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

	labels_list = [x[1] for x in validset]
	pred_list_e2e, pred_list_cos = [], []

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['n_classes']
		else:
			rproj_size = -1
	except KeyError as err:
		print("Key Error: {0}".format(err))
		print('\nProbably old cp has no info regarding classifiers arch!\n')
		n_hidden, hidden_size, softmax, n_classes = get_classifier_config_from_cp(ckpt)
		dropout_prob = args.dropout_prob
		emb_size = 350
		rproj_size = -1

	if args.model == 'vgg':
		model = vgg.VGG('VGG19', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'resnet':
		model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'densenet':
		model = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)

	if ckpt['sm_type'] == 'am_softmax':
		del(ckpt['model_state']['out_proj.w'])
	elif ckpt['sm_type'] == 'softmax':
		del(ckpt['model_state']['out_proj.w.weight'])
		del(ckpt['model_state']['out_proj.w.bias'])

	print(model.load_state_dict(ckpt['model_state'], strict=False))

	if args.cuda:
		device = get_freer_gpu()
		model = model.cuda(device)

	class_center = {}
	class_count = {}

	model.classifier = model.classifier[:-1]
	model.eval()

	if args.emb_path:

		emb_labels = torch.load(args.emb_path)
		embeddings, labels = emb_labels['embeddings'], emb_labels['labels']
		del emb_labels
		emb_labels = None

		print('\nEmbeddings loaded')

	else:

		embeddings = []
		labels = []

		iterator = tqdm(valid_loader, total=len(valid_loader))

		with torch.no_grad():

			for batch in iterator:

				x, y = batch

				if args.cuda:
					x = x.to(device)

				emb = model.forward(x).detach()

				embeddings.append(emb.detach().cpu())
				labels.append(y)

		embeddings = torch.cat(embeddings, 0)
		labels = list(torch.cat(labels, 0).squeeze().numpy())

		if args.out_path:
			if os.path.isfile(args.out_path):
				os.remove(args.out_path)
				print(args.out_path+' Removed')
			torch.save({'embeddings':embeddings, 'labels':labels}, args.out_path)

		print('\nEmbedding done')



	with torch.no_grad():

		iterator = tqdm(enumerate(labels), total=len(labels))
		for i, label in iterator:

			example = str(i)

			emb = embeddings[i].unsqueeze(0).to(device)

			try:
				class_center[label] += emb
				class_count[label] += 1
			except KeyError:
				class_center[label] = emb
				class_count[label] = 1

		for k in class_center:
			class_center[k] /= class_count[k]

		for i, label in enumerate(labels_list):
			class_scores_e2e = []
			class_scores_cos = []
			emb = embeddings[i].unsqueeze(0).to(device)
			for k in class_center:
				class_scores_e2e.append( [model.forward_bin(class_center[k], emb).squeeze().item(), k] )
				class_scores_cos.append( [torch.nn.functional.cosine_similarity(class_center[k], emb).squeeze().item(), k] )

			pred_list_e2e.append(max(class_scores_e2e)[1])
			pred_list_cos.append(max(class_scores_cos)[1])

	print('\nScoring done')

	print('\n NMI - E2E: {}'.format(normalized_mutual_info_score(labels_list, pred_list_e2e)))
	print('\n NMI - COS: {}'.format(normalized_mutual_info_score(labels_list, pred_list_cos)))
