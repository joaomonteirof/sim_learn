from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_valid, collater
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from utils.utils import set_np_randomseed, get_freer_gpu, parse_args_for_log
from utils.optimizer import TransformerOptimizer

# Training settings
parser = argparse.ArgumentParser(description='Speaker embbedings with combined loss')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--model', choices=['TDNN', 'TDNN_multipool'], default='TDNN', help='Model')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--emb-size', type=int, default=256, metavar='S', help='embbeding dimension (default: 256)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='hidden layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--n-frames', type=int, default=1000, metavar='N', help='maximum number of frames per utterance (default: 1000)')
parser.add_argument('--warmup', type=int, default=4000, metavar='N', help='Iterations until reach lr (default: 4000)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--centroid-smoothing', type=float, default=0.9, metavar='Lamb', help='Moving average parameter for centroids')
parser.add_argument('--ablation-sim', action='store_true', default=False, help='Disables similarity learning')
parser.add_argument('--ablation-ce', action='store_true', default=False, help='Disables auxiliary classification loss')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--verbose', type=int, default=2, metavar='N', help='Verbose is activated if > 0')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

if args.cuda:
	device = get_freer_gpu()
else:
	device = torch.device('cpu')

train_dataset = Loader(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed, collate_fn=collater)

if args.valid_hdf_file is not None:
	valid_dataset = Loader_valid(hdf5_name = args.valid_hdf_file, max_nb_frames = args.n_frames)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed, collate_fn=collater)
else:
	valid_loader=None

if args.pretrained_path is not None:
	print('\nLoading pretrained model from: {}\n'.format(args.pretrained_path))
	ckpt=torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)
	args.dropout_prob, args.n_hidden, args.hidden_size, args.emb_size = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['emb_size']
	print('\nUsing pretrained config for discriminator. Ignoring args.')

args_dict = parse_args_for_log(args)

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model, purge_step=True if args.checkpoint_epoch is None else False)
	writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_eer':0.0})
else:
	writer = None

if args.model == 'TDNN':
	model = model_.TDNN(n_z=args.emb_size, nh=args.n_hidden, n_h=args.hidden_size, n_speakers=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing)
elif args.model == 'TDNN_multipool':
	model = model_.TDNN_multipool(n_z=args.emb_size, nh=args.n_hidden, n_h=args.hidden_size, n_speakers=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax, centroids_lambda=args.centroid_smoothing)

if args.verbose > 0:
	print(model)

if args.pretrained_path is not None:

	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

model = model.to(device)

optimizer = TransformerOptimizer(optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True), 
			lr=args.lr, 
			warmup_steps=args.warmup)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm, 
			label_smoothing=args.smoothing, verbose=args.verbose, device=device, save_cp=(not args.no_cp), 
			checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, 
			ablation_sim=args.ablation_sim, ablation_ce=args.ablation_ce, cuda=args.cuda, logger=writer)

if args.verbose > 0:
	print('\n')
	for arg_key in args_dict:
		print('{}: {}'.format(arg_key, args_dict[arg_key]))
	print('\n')

best_eer = trainer.train(n_epochs=args.epochs, save_every=args.save_every)

if args.logdir:
	writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_eer':best_eer[0]})
