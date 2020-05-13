import argparse
import numpy as np
import glob
import torch
import os
import sys
import pathlib
from kaldi_io import read_mat_scp, open_or_fd, write_vec_flt
import model as model_
import scipy.io as sio

from utils.utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute embeddings')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to input data')
	parser.add_argument('--path-to-more-data', type=str, default=None, metavar='Path', help='Path to input data')
	parser.add_argument('--utt2spk', type=str, default=None, metavar='Path', help='Optional path for utt2spk')
	parser.add_argument('--more-utt2spk', type=str, default=None, metavar='Path', help='Optional path for utt2spk')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model', choices=['TDNN', 'TDNN_multipool'], default='TDNN', help='Model')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--eps', type=float, default=0.0, metavar='eps', help='Add noise to embeddings')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	pathlib.Path(args.out_path).mkdir(parents=True, exist_ok=True)

	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cuda:
		device = get_freer_gpu()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	if args.model == 'TDNN':
		model = model_.TDNN(n_z=ckpt['emb_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], ncoef=ckpt['ncoef'])
	if args.model == 'TDNN_multipool':
		model = model_.TDNN_multipool(n_z=ckpt['emb_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], ncoef=ckpt['ncoef'])

	print(model.load_state_dict(ckpt['model_state'], strict=False))

	model.eval()
	if args.cuda:
		model = model.to(device)

	scp_list = glob.glob(args.path_to_data + '*.scp')

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	if args.path_to_more_data:
		more_scp_list = glob.glob(args.path_to_more_data + '*.scp')

		if len(more_scp_list)<1:
			print('Nothing found at {}.'.format(args.path_to_more_data))
			exit(1)
		else:
			scp_list = scp_list + more_scp_list

	if args.utt2spk:
		utt2spk = read_utt2spk(args.utt2spk)
		if args.more_utt2spk:
			utt2spk.update(read_utt2spk(args.more_utt2spk))

	scp_list = glob.glob(args.path_to_data + '*.scp')

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	print('Start of data embeddings computation')

	embeddings = {}

	with torch.no_grad():

		for file_ in scp_list:

			data = { k:m for k,m in read_mat_scp(file_) }

			for i, utt in enumerate(data):

				if args.utt2spk:
					if not utt in utt2spk:
						print('Skipping utterance '+ utt)
						continue

				feats = prep_feats(data[utt])

				try:
					if args.cuda:
						feats = feats.to(device)
						model = model.to(device)

					emb = model.forward(feats)

				except:
					feats = feats.cpu()
					model = model.cpu()

					emb = model.forward(feats)

				embeddings[utt] = emb.detach().cpu().numpy().squeeze()

				if args.eps>0.0:
					embeddings[utt] += args.eps*np.random.randn(embeddings[utt].shape[0])

	print('Storing embeddings in output file')

	out_name = args.path_to_data.split('/')[-2] if not args.utt2spk else args.utt2spk.split('/')[-2]
	file_name = args.out_path+out_name+'.ark'

	if os.path.isfile(file_name):
		os.remove(file_name)
		print(file_name + ' Removed')

	with open_or_fd(file_name,'wb') as f:
		for k,v in embeddings.items(): write_vec_flt(f, v, k)

	print('End of embeddings computation.')