from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from data_load import Loader, collater, fewshot_eval_builder
from torchvision import datasets, transforms
from RandAugment import RandAugment
from models import resnet, resnet12, wideresnet
import numpy as np
from time import sleep
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from utils import mean, std, set_np_randomseed, get_freer_gpu, parse_args_for_log, add_noise

# Training settings
parser = argparse.ArgumentParser(description='Mini Imagenet')
parser.add_argument('--model', choices=['resnet', 'resnet_12', 'wideresnet'], default='resnet')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--lr-steps', nargs='+', type=int, default=[100, 180], help='List of epochs to reduce lr by lr-factor')
parser.add_argument('--lr-factor', type=float, default=0.1, metavar='LR', help='Factor to reduce learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='beta1', help='momentum (default: 0.9)')
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
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
###Validation config
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to valid data stored in hdf. Has priority over valid data path if set')
parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (default: 5)')
parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (default: 5)')
parser.add_argument('--num-queries', type=int, default=15, help='Number of data points per class on test partition (default: 15)')
parser.add_argument('--num-runs', type=int, default=10, help='Number of evaluation runs (default: 10)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--eval-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

eval_config={'num_shots':args.num_shots, 
'num_ways':args.num_ways, 
'num_runs':args.num_runs, 
'batch_size':args.valid_batch_size, 
'workers':args.eval_workers}

if args.cuda:
	torch.backends.cudnn.benchmark=True

if args.hdf_path:
	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), add_noise(), transforms.Normalize(mean=mean, std=std)])
	transform_train.transforms.insert(1, RandAugment(args.aug_N, args.aug_M))
	trainset = Loader(args.hdf_path, transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True, collate_fn=collater)
else:
	transform_train = transforms.Compose([transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), add_noise(), transforms.Normalize(mean=mean, std=std)])
	transform_train.transforms.insert(0, RandAugment(args.aug_N, args.aug_M))
	trainset = datasets.ImageFolder(args.data_path, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)

transform_test = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(84), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
valid_loader = fewshot_eval_builder(hdf5_name=args.valid_hdf_path, train_transformation=transform_test, test_transformation=transform_test, k_shot=args.num_shots, n_way=args.num_ways, n_queries=args.num_queries)

args.nclasses = trainset.n_classes if isinstance(trainset, Loader) else len(trainset.classes)

if args.model == 'resnet':
	model = resnet.ResNet18(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing, n_classes=args.nclasses)
elif args.model == 'resnet_12':
	model = resnet12.ResNet12(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing, n_classes=args.nclasses)
elif args.model == 'wideresnet':
	model = wideresnet.WideResNet(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing, n_classes=args.nclasses)

if args.verbose >0:
	print('\n', model, '\n')

if args.cuda:
	device = get_freer_gpu()
	model = model.to(device)

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model, purge_step=0 if args.checkpoint_epoch is None else int(args.checkpoint_epoch*len(train_loader)))
	args_dict = parse_args_for_log(args)
	writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_acc':0.0})
else:
	writer = None

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, eval_config=eval_config, max_gnorm=args.max_gnorm, label_smoothing=args.smoothing,
			lr_steps=args.lr_steps, lr_factor=args.lr_factor, verbose=args.verbose, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path, 
			checkpoint_epoch=args.checkpoint_epoch, ablation_sim=args.ablation_sim, ablation_ce=args.ablation_ce, cuda=args.cuda, logger=writer)

if args.verbose >0:
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		print('{}: {}'.format(arg_key, args_dict[arg_key]))
	print('\n')

trainer.train(n_epochs=args.epochs, save_every=args.save_every)