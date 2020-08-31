from __future__ import print_function
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import transforms
from RandAugment import RandAugment
from models import resnet
from data_load import Loader_training, Loader_validation, collater
import numpy as np
from time import sleep
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from utils import mean, std, set_np_randomseed, get_freer_gpu, parse_args_for_log, add_noise

# Training settings
parser = argparse.ArgumentParser(description='PACS out of domain classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.9, metavar='beta1', help='Beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='beta2', help='Beta2 (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--centroid-smoothing', type=float, default=0.9, metavar='Lamb', help='Moving average parameter for centroids')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data_train', metavar='Path', help='Path to data')
parser.add_argument('--target', choices=['photo', 'cartoon', 'sketch', 'artpainting'], default='artpainting', help='Choice of left out (target) domain')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--model', choices=['resnet'], default='resnet')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--aug-M', type=int, default=15, metavar='AUGM', help='Augmentation hp. Default is 15')
parser.add_argument('--aug-N', type=int, default=1, metavar='AUGN', help='Augmentation hp. Default is 1')
parser.add_argument('--pretrained', action='store_true', default=False, help='Get pretrained weights on imagenet. Encoder only')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path to trained model. Discards output layer')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before saving checkpoints. Default is 1')
parser.add_argument('--eval-every', type=int, default=1000, metavar='N', help='how many iterations to wait before evaluatiing models. Default is 1000')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
parser.add_argument('--ablation-sim', action='store_true', default=False, help='Disables similarity learning')
parser.add_argument('--ablation-ce', action='store_true', default=False, help='Disables auxiliary classification loss')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	torch.backends.cudnn.benchmark=True

source_domains = [x for x in ['photo', 'cartoon', 'sketch', 'art_painting'] if x != args.target]

print('\nSource domains: {}\n'.format(source_domains))

train_source_1 = args.data_path + 'train_' + source_domains[0] + '.hdf'
train_source_2 = args.data_path + 'train_' + source_domains[1] + '.hdf'
train_source_3 = args.data_path + 'train_' + source_domains[2] + '.hdf'
target_path = args.data_path + 'test_' + args.target + '.hdf'

transform_train = transforms.Compose([transforms.RandomGrayscale(p=0.10), transforms.RandomResizedCrop(222, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), add_noise(), transforms.Normalize(mean=mean, std=std)])	
transform_train.transforms.insert(0, RandAugment(args.aug_N, args.aug_M))
trainset = Loader_training(hdf_path1=train_source_1, hdf_path2=train_source_2, hdf_path3=train_source_3, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True, collate_fn=collater)

transform_test = transforms.Compose([transforms.Resize(size=222), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
validset = Loader_validation(hdf_path=target_path, transform=transform_test)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

print(args, '\n')

if args.pretrained_path:
	print('\nLoading pretrained model from: {}\n'.format(args.pretrained_path))
	ckpt=torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)
	args.dropout_prob, args.n_hidden, args.hidden_size = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size']
	print('\nUsing pretrained config for discriminator. Ignoring args.')

if args.model == 'resnet':
	model = resnet.ResNet50(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing)

if args.pretrained_path:
	print(model.load_state_dict(ckpt['model_state'], strict=False))
	model.centroids = ckpt['centroids']
	print('\n')
elif args.pretrained:
	print('\nLoading pretrained encoder from torchvision\n')
	if args.model == 'resnet':
		model_pretrained = torchvision.models.resnet50(pretrained=True)
	print(model.load_state_dict(model_pretrained.state_dict(), strict=False))
	print('\n')

if args.verbose >0:
	print(model)

if args.cuda:
	device = get_freer_gpu()
	model = model.to(device)

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model, purge_step=0 if args.checkpoint_epoch is None else int(args.checkpoint_epoch*len(train_loader)))
	args_dict = parse_args_for_log(args)
	writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_eer':0.0})
else:
	writer = None

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm,
		label_smoothing=args.smoothing, verbose=args.verbose, save_cp=(not args.no_cp), 
		checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, 
		ablation_sim=args.ablation_sim, ablation_ce=args.ablation_ce, cuda=args.cuda, logger=writer)

if args.verbose >0:
	print('\n')
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		print('{}: {}'.format(arg_key, args_dict[arg_key]))
	print('\n')

trainer.train(n_epochs=args.epochs, save_every=args.save_every, eval_every=args.eval_every)