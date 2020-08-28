from __future__ import print_function
import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.utils.data
from models import resnet, wideresnet, wrapper_racc
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
from utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Cifar10 Robust ACC Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['resnet', 'wideresnet'], default='resnet')
	parser.add_argument('--inf-mode', choices=['sim', 'ce', 'fus'], default='sim', help='Inference mode')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--full-eval', action='store_true', default=False, help='Enables use of large list of epsilons per attack')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	print('\n', args, '\n')

	transform_test = transforms.Compose([transforms.ToTensor()])
	testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
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

	model = wrapper_racc.wrapper(base_model=model, inf_mode=args.inf_mode)

	model.eval()

	fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=device)

	attacks = [
		fa.FGSM(),
		fa.LinfBasicIterativeAttack(steps=7),
		fa.LinfPGD(steps=7),
		fa.LinfPGD(steps=20),
		fa.L2CarliniWagnerAttack(binary_search_steps=30),
	]

	if args.full_eval:

		epsilons = [
			0.0,
			8.0/255.0,
			16.0/255.0,
			32.0/255.0,
			48.0/255.0,
			80.0/255.0,
			128.0/255.0,
		]

	else:

		epsilons = [
			0.0,
			8.0/255.0,
			16.0/255.0,
		]

	print("\nepsilons\n")
	print(epsilons, '\n')

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