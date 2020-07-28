from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from data_load import Loader, collater
from torchvision import datasets, transforms
from RandAugment import RandAugment
from models import resnet, resnet12
import numpy as np
from time import sleep
import os
import sys
from utils import mean, std, set_np_randomseed, get_freer_gpu, parse_args_for_log

# Training settings
parser = argparse.ArgumentParser(description='Mini Imagenet')
parser.add_argument('--model', choices=['resnet', 'resnet_12'], default='resnet')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.9, metavar='beta1', help='Beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='beta2', help='Beta2 (default: 0.9)')
parser.add_argument('--l2', type=float, default=5e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--centroid-smoothing', type=float, default=0.9, metavar='Lamb', help='Moving average parameter for centroids')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--aug-M', type=int, default=15, metavar='AUGM', help='Augmentation hp. Default is 15')
parser.add_argument('--aug-N', type=int, default=1, metavar='AUGN', help='Augmentation hp. Default is 1')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data_train', metavar='Path', help='Path to data')
parser.add_argument('--hdf-path', type=str, default=None, metavar='Path', help='Path to data stored in hdf. Has priority over data path if set')
parser.add_argument('--valid-data-path', type=str, default='./data_val', metavar='Path', help='Path to data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to valid data stored in hdf. Has priority over valid data path if set')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--ablation-sim', action='store_true', default=False, help='Disables similarity learning')
parser.add_argument('--ablation-ce', action='store_true', default=False, help='Disables auxiliary classification loss')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	torch.backends.cudnn.benchmark=True

if args.hdf_path:
	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	transform_train.transforms.insert(1, RandAugment(args.aug_N, args.aug_M))
	trainset = Loader(args.hdf_path, transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True, collate_fn=collater)
else:
	transform_train = transforms.Compose([transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	transform_train.transforms.insert(0, RandAugment(args.aug_N, args.aug_M))
	trainset = datasets.ImageFolder(args.data_path, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)


if args.valid_hdf_path:
	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = Loader(args.valid_hdf_path, transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True, collate_fn=collater)
else:
	transform_test = transforms.Compose([transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = datasets.ImageFolder(args.valid_data_path, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

args.nclasses = trainset.n_classes if isinstance(trainset, Loader) else len(trainset.classes)

if args.model == 'resnet':
	model = resnet.ResNet18(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing, n_classes=args.nclasses)
elif args.model == 'resnet_12':
	model = resnet12.ResNet12(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing, n_classes=args.nclasses)

if args.verbose >0:
	print('\n', model, '\n')

if args.cuda:
	device = get_freer_gpu()
	model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm, label_smoothing=args.smoothing,
			verbose=args.verbose, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path,
			checkpoint_epoch=args.checkpoint_epoch, ablation_sim=args.ablation_sim, ablation_ce=args.ablation_ce, cuda=args.cuda)

if args.verbose >0:
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		print('{}: {}'.format(arg_key, args_dict[arg_key]))
	print('\n')

trainer.train(n_epochs=args.epochs, save_every=args.save_every)