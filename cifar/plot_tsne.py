from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import resnet, wideresnet, wrapper_racc
import numpy as np
import os
import sys
from tqdm import tqdm
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
from utils import *
from sklearn.manifold import TSNE

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Plot embeddings of clean and attack samples')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['resnet', 'wideresnet'], default='resnet')
	parser.add_argument('--inf-mode', choices=['sim', 'ce', 'fus'], default='sim', help='Inference mode')
	parser.add_argument('--sample-size', type=int, default=200, metavar='N', help='Number of images to plot')
	parser.add_argument('--out-path', type=str, default='', metavar='Path', help='Path for saving outputs')
	parser.add_argument('--out-prefix', type=str, default='', metavar='Path', help='Prefix to be added to output file name')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-histogram', action='store_true', default=False, help='Disables histogram plot')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	validset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, num_workers=args.workers)
	preprocessing = dict(mean=[x / 255 for x in [125.3, 123.0, 113.9]], std=[x / 255 for x in [63.0, 62.1, 66.7]], axis=-3)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
	except KeyError as err:
		print("Key Error: {0}".format(err))

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

	attack = fa.LinfPGD(steps=7)
	epsilons = [8.0/255.0]

	model.eval()

	fmodel = PyTorchModel(wrapper_racc.wrapper(base_model=model, inf_mode=args.inf_mode).eval(),
		bounds=(0, 1),
		preprocessing=preprocessing,
		device=device)

	clean_embeddings, attack_embeddings, label_list = [], [], []

	model.eval()

	success_counter = 0
	iterator = tqdm(test_loader)
	for input_image, labels in iterator:

		input_image = input_image.to(device)
		labels = labels.to(device)

		_, attack_input, success = attack(fmodel, input_image, labels, epsilons=epsilons)
		success = success.squeeze().item()

		if success:
			success_counter += 1
			with torch.no_grad():
				clean_embeddings.append( model(input_image).detach().cpu().numpy() )
				attack_embeddings.append( model(attack_input[0]).detach().cpu().numpy() )
				label_list.append( labels.squeeze().item() )

		if success_counter == args.sample_size:
			break

	clean_embeddings, attack_embeddings = np.concatenate(clean_embeddings, axis=0,), np.concatenate(attack_embeddings, axis=0)
	centroids = model.centroids.detach().clone().cpu().numpy()

	print('\n\nDone creating attacks.\n')

	if success_counter != args.sample_size:
		print(f'\nDesired number of attackers not achieved. Computed {success_counter} attacks.\n')


	embeddings = np.concatenate((centroids, clean_embeddings, attack_embeddings), axis=0)
	tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)

	plt.scatter(tsne_embeddings[:centroids.shape[0], 0],
		tsne_embeddings[:centroids.shape[0], 1],
		marker='X',
		color='black',
		s=60.0,
		label='Centroids'
		)
	plt.scatter(tsne_embeddings[centroids.shape[0]:(centroids.shape[0]+success_counter), 0],
		tsne_embeddings[centroids.shape[0]:(centroids.shape[0]+success_counter), 1],
		c=label_list,
		label='Test instances'
		)
	plt.scatter(tsne_embeddings[(centroids.shape[0]+success_counter):, 0],
		tsne_embeddings[(centroids.shape[0]+success_counter):, 1],
		marker='*',
		color='red',
		label='attacks'
		)
	plt.legend()
	plt.savefig(os.path.join(args.out_path, args.out_prefix+'tsne_adv_cifar.pdf'), bbox_inches='tight')
