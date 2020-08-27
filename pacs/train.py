import argparse
import os
import sys
import random
from tqdm import tqdm

import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import PIL
import pandas

import models as models
from train_loop import TrainLoop
from data_loader import Loader_validation, Loader_unif_sampling
from torch.utils.tensorboard import SummaryWriter
import utils

parser = argparse.ArgumentParser(description='Domain conditional models for domain generalization')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='momentum (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001')
parser.add_argument('--lr-factor', type=float, default=0.1, metavar='f', help='LR decrease factor (default: 0.1')
parser.add_argument('--lr-threshold', type=float, default=0.001, metavar='f', help='LR threshold (default: 0.001')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default='./', metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default='./alexnet_caffe.pth.tar', metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./prepared_data/', metavar='Path', help='Data path')
parser.add_argument('--source1', type=str, default='photo', metavar='Path', help='Path to source1 file')
parser.add_argument('--source2', type=str, default='cartoon', metavar='Path', help='Path to source2 file')
parser.add_argument('--source3', type=str, default='sketch', metavar='Path', help='Path to source3 file')
parser.add_argument('--target', type=str, default='artpainting', metavar='Path', help='Path to target data')
parser.add_argument('--n-classes', type=int, default=7, metavar='N', help='number of classes (default: 7)')
parser.add_argument('--n-domains', type=int, default=3, metavar='N', help='number of available training domains (default: 3)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--patience', type=int, default=20, metavar='N', help='number of epochs to wait before reducing lr (default: 20)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--n-runs', type=int, default=1, metavar='n', help='Number of repetitions (default: 3)')
parser.add_argument('--out-file-name', type=str, default='out', metavar='out', help='Output file with all results')
parser.add_argument('--no-combined-loss', action='store_true', default=False, help='Disables the loss over the combined simplex')
parser.add_argument('--class-loss', action='store_true', default=False, help='Enables loss over class simplex')
parser.add_argument('--domain-loss', action='store_true', default=False, help='Enables loss over domain simplex')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
args.combined_loss = True if not args.no_combined_loss else False

assert args.combined_loss or args.class_loss or args.domain_loss, 'One of the losses has to be enabled!!!'

print(args, '\n')

print('Source domains: {}, {}, {}'.format(args.source1, args.source2, args.source3))
print('Target domain:', args.target)
print('Number of classes and domains: {}, {}'.format(args.n_classes, args.n_domains))
print('Cuda Mode: {}'.format(args.cuda))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('L2: {}'.format(args.l2))
print('Momentum: {}'.format(args.momentum))
print('Patience: {}'.format(args.patience))
print('Smoothing: {}'.format(args.smoothing))
print('LR reduction factor: {}'.format(args.lr_factor))
print('LR threshold: {}'.format(args.lr_threshold))
print('Train mode (combined, class, and domains losses): {}, {}, {}'.format(args.combined_loss, args.class_loss, args.domain_loss))

acc_runs = []
acc_blind_runs = []
seeds = [1, 10, 100]

assert args.n_runs<=len(seeds), "n-runs can be at most {}.".format(len(seeds))

out_file_name = os.path.join(args.checkpoint_path, 'out_' + args.target + '.txt')

for run in range(args.n_runs):
	print('Run {}'.format(run))

	# Setting seed
	random.seed(seeds[run])
	torch.manual_seed(seeds[run])
	checkpoint_path = os.path.join(args.checkpoint_path, args.target+'_seed'+str(seeds[run]))
	
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	img_transform_train = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	img_transform_test = transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	train_source_1 = args.data_path + 'train_' + args.source1 + '.hdf'
	train_source_2 = args.data_path + 'train_' + args.source2 + '.hdf'
	train_source_3 = args.data_path + 'train_' + args.source3 + '.hdf'
	test_source_1 = args.data_path + 'val_' + args.source1 + '.hdf'
	test_source_2 = args.data_path + 'val_' + args.source2 + '.hdf'
	test_source_3 = args.data_path + 'val_' + args.source3 + '.hdf'
	target_path = args.data_path + 'test_' + args.target + '.hdf'

	source_dataset = Loader_unif_sampling(hdf_path1=train_source_1, hdf_path2=train_source_2, hdf_path3=train_source_3, transform=img_transform_train)
	source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	test_source_dataset = Loader_unif_sampling(hdf_path1=test_source_1, hdf_path2=test_source_2, hdf_path3=test_source_3, transform=img_transform_test)
	test_source_loader = torch.utils.data.DataLoader(dataset=test_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	target_dataset = Loader_validation(hdf_path=target_path, transform=img_transform_test)
	target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
		
	model = models.AlexNet(n_classes=args.n_classes, n_domains=args.n_domains, pretrained_path=args.pretrained_path)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.l2)

	if args.cuda:
		model = model.cuda()
		torch.backends.cudnn.benchmark = True

	if args.logdir:
		writer = SummaryWriter(log_dir=os.path.join(args.logdir,'run_{}'.format(run)), purge_step=True if args.checkpoint_epoch is None else False)
		args_dict = utils.parse_args_for_log(args)
		writer.add_hparams(hparam_dict=args_dict, metric_dict={'src_acc':0.0, 'tgt_acc':0.0, 'tgt_acc_1':0.0, 'tgt_acc_2':0.0})
	else:
		writer = None

	trainer = TrainLoop(model=model,
						optimizer=optimizer,
						source_loader=source_loader,
						test_source_loader=test_source_loader,
						target_loader=target_loader,
						patience=args.patience,
						factor=args.lr_factor,
						label_smoothing=args.smoothing,
						lr_threshold=args.lr_threshold,
						combined_loss=args.combined_loss,
						class_loss=args.class_loss,
						domain_loss=args.domain_loss,
						checkpoint_path=checkpoint_path,
						checkpoint_epoch=args.checkpoint_epoch,
						cuda=args.cuda,
						logger=writer )

	_, results_acc, results_epoch = trainer.train(n_epochs=args.epochs, save_every=args.save_every)

	if args.logdir:
		writer.add_hparams(hparam_dict=args_dict, metric_dict={'src_acc':results_acc[-2],
																'tgt_acc':results_acc[-1],
																'tgt_acc_1':results_acc[0],
																'tgt_acc_2':results_acc[1]})

	acc_runs.append(results_acc[-1])
	acc_blind_runs.append(results_acc[-3])

	# Logging results on text file
	with open(out_file_name, 'w') as out_file:
		out_file.write('Run {}\n'.format(run))
		out_file.write('Source domains: {}, {}, {}'.format(args.source1, args.source2, args.source3))
		out_file.write('Target domain {}:'.format(args.target))
		out_file.write('Cuda Mode: {}'.format(args.cuda))
		out_file.write('Batch size: {}'.format(args.batch_size))
		out_file.write('LR: {}'.format(args.lr))
		out_file.write('L2: {}'.format(args.l2))
		out_file.write('Momentum: {}'.format(args.momentum))
		out_file.write('Patience: {}'.format(args.patience))
		out_file.write('Smoothing: {}'.format(args.smoothing))
		out_file.write('LR factor: {}'.format(args.lr_factor))
		out_file.write('LR threshold: {}'.format(args.lr_threshold))
		out_file.write('Train mode (combined, class, and domains losses): {}, {}, {}'.format(args.combined_loss, args.class_loss, args.domain_loss))

		out_file.write('\nSources: {}, {}, {}'.format(args.source1, args.source2, args.source3))
		out_file.write('\nTarget: {}'.format(args.target))
		out_file.write('\nBest source acc: {:0.4f}, epoch: {}'.format(results_acc[-2], results_epoch[-2]))
		out_file.write('\nBest target acc:{:0.4f}, epoch: {}'.format(results_acc[-1], results_epoch[-1]))
		out_file.write('\nTarget acc when best source acc: {:0.4f}, epoch: {}'.format(results_acc[0], results_epoch[-2]))
		out_file.write('\nTarget acc when best total loss: {:0.4f}, epoch: {}'.format(results_acc[1], results_epoch[0]))


df = pandas.DataFrame(data={'Acc-{}'.format(args.target): acc_runs, 'Seed': seeds[:args.n_runs]})
df.to_csv('./accuracy_runs_'+args.target+'.csv', sep=',', index = False)

df = pandas.DataFrame(data={'Acc-{}'.format(args.target): acc_blind_runs, 'Seed': seeds[:args.n_runs]})
df.to_csv('./accuracy_runs_'+args.target+'_blind.csv', sep=',', index = False)