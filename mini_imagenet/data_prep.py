import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to root folder with sub-folders containing files')
	parser.add_argument('--path-to-more-data', type=str, default=None, metavar='Path', help='Path to root folder with sub-folders containing files')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	args = parser.parse_args()

	if os.path.isfile(args.out_path+args.out_name):
		os.remove(args.out_path+args.out_name)
		print(args.out_path+args.out_name+' Removed')

	transform = transforms.Compose([transforms.ToTensor()])
	dataset = datasets.ImageFolder(args.path_to_data, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)

	if args.path_to_more_data is not None:
		extra_dataset = datasets.ImageFolder(args.path_to_more_data, transform=transform)
	else:
		extra_dataset = None

	print('Start of data preparation')

	hdf = h5py.File(args.out_path+args.out_name, 'a')

	for i, data_label in enumerate(dataloader):

		x, y = data_label
		y = str(y.squeeze().item())

		if not y in hdf:
			hdf.create_group(y)

		hdf[y].create_dataset(y+'_'+str(i), data=x.squeeze(0))

	if extra_dataset is not None:

		extra_dataloader = torch.utils.data.DataLoader(extra_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)

		for j, data_label in enumerate(extra_dataloader):

			x, y = data_label
			y = str(y.squeeze().item())

			if not y in hdf:
				hdf.create_group(y)

			hdf[y].create_dataset(y+'_'+str(i+j+1), data=x.squeeze(0))


	hdf.close()