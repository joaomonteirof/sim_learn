from __future__ import print_function
import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet, wrapper_racc
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
from utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='ImageNet Robust ACC Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--ce-layer', action='store_true', default=False, help='Uses standard output layer to compute and evaluate attacks')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	print('\n', args, '\n')

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

	model = wrapper_racc.wrapper(base_model=model, ce_layer=args.ce_layer)

	model.eval()

	fmodel = PyTorchModel(model, bounds=(0, 1), device=device)

	attacks = [
		fa.FGSM(),
		fa.LinfBasicIterativeAttack(steps=10),
		fa.LinfPGD(steps=10),
		fa.LinfPGD(steps=20),
		fa.L2CarliniWagnerAttack(binary_search_steps=10),
	]

	epsilons = [
		0.0,
		8.0/255.0,
		16.0/255.0,
		32.0/255.0,
		48.0/255.0,
		80.0/255.0,
		128.0/255.0,
	]

	print("\nepsilons\n")
	print(epsilons, '\n')

	## Computing clean accuracy

	correct = 0

	with torch.no_grad():

		for batch in test_loader:

			x, y = batch

			x = x.to(device)
			y = y.to(device).squeeze()

			out = model.forward(x)
			pred = out.max(1)[1].long()
			correct += pred.squeeze().eq(y).sum().item()

	print('\nClean Accuracy of model {}: {}\n'.format(args.cp_path.split('/')[-1], 100.*correct/len(testset)))


	# Computing robust accuracy for each attack

	for attack in attacks:

		incorrect = torch.zeros(len(epsilons))

		for batch in test_loader:

			x, y = batch

			x = x.to(device)
			y = y.to(device).squeeze()

			_, _, success = attack(fmodel, x, y, epsilons=epsilons)

			incorrect += success.float().cpu().sum(-1)

		print('\nRobust Accuracy of attack: {}\n'.format(attack))
		for i, eps in enumerate(epsilons):
			print('Eps: {}, ACC: {}'.format(eps, 1.0 - incorrect[i].item()/len(testset)))