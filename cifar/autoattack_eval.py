from __future__ import print_function
import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from autoattack import AutoAttack
from models import resnet, wideresnet, wrapper_racc

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Cifar10 Robust ACC Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['resnet', 'wideresnet'], default='resnet')
	parser.add_argument('--inf-mode', choices=['sim', 'ce', 'fus'], default='sim', help='Inference mode')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()

	print('\n', args, '\n')

	transform_test = transforms.Compose([transforms.ToTensor()])
	testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.workers)

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

	model.centroids = model.centroids.cuda()

	model = model.eval().cuda()

	model = wrapper_racc.wrapper(base_model=model, inf_mode=args.inf_mode, normalize=True, use_softmax=False).eval().cuda()

	adversary = AutoAttack(model, norm='Linf', eps=8./255., version='standard')

	l = [x for (x, y) in test_loader]
	x_test = torch.cat(l, 0)
	l = [y for (x, y) in test_loader]
	y_test = torch.cat(l, 0)

	adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
	torch.save({'adv_complete': adv_complete}, '{}_{}_{}.pt'.format(
		args.model, 'aa_std_linf_8-255', args.inf_mode))
