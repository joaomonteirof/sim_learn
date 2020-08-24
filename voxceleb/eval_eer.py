import argparse
import numpy as np
import torch
from kaldi_io import read_mat_scp
from sklearn import metrics
import scipy.io as sio
import model as model_
import glob
import pickle
import os
import sys
import pathlib

from utils.utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--test-data', type=str, default='./data/test/', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default=None, help='Path to trials file. If None, will be created from spk2utt')
	parser.add_argument('--spk2utt', type=str, default=None, metavar='Path', help='Path to spk2utt file. Will be used in case no trials file is provided')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--model', choices=['TDNN', 'TDNN_multipool'], default='TDNN', help='Model')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--out-prefix', type=str, default=None, metavar='Path', help='Prefix to be added to score files')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	pathlib.Path(args.out_path).mkdir(parents=True, exist_ok=True)

	print('Cuda Mode is: {}\n'.format(args.cuda))

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

	test_data = None

	files_list = glob.glob(args.test_data+'*.scp')

	for file_ in files_list:
		if test_data is None:
			test_data = { k:v for k,v in read_mat_scp(file_) }
		else:
			for k,v in read_mat_scp(file_):
				test_data[k] = v

	if args.trials_path:
		utterances_enroll, utterances_test, labels = read_trials(args.trials_path)
	else:
		spk2utt = read_spk2utt(args.spk2utt)
		utterances_enroll, utterances_test, labels = create_trials(spk2utt)

	print('\nAll data ready. Start of scoring')

	cos_scores = []
	e2e_scores = []
	fus_scores = []
	out_e2e = []
	out_cos = []
	out_fus = []
	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		for i in range(len(labels)):

			enroll_utt = utterances_enroll[i]

			try:
				emb_enroll = mem_embeddings[enroll_utt]
			except KeyError:

				enroll_utt_data = prep_feats(test_data[enroll_utt])

				if args.cuda:
					enroll_utt_data = enroll_utt_data.to(device)

				emb_enroll = model.forward(enroll_utt_data).detach()
				mem_embeddings[enroll_utt] = emb_enroll

			test_utt = utterances_test[i]

			try:
				emb_test = mem_embeddings[test_utt]
			except KeyError:

				test_utt_data = prep_feats(test_data[test_utt])

				if args.cuda:
					enroll_utt_data = enroll_utt_data.to(device)
					test_utt_data = test_utt_data.to(device)

				emb_test = model.forward(test_utt_data).detach()
				mem_embeddings[test_utt] = emb_test

			pred = model.forward_bin(emb_enroll, emb_test)

			e2e_scores.append( pred.squeeze().item() )
			cos_scores.append( 0.5*(torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item()+1.) )
			fus_scores.append( (torch.sigmoid(torch.Tensor([e2e_scores[-1]])).item()+cos_scores[-1])*0.5 )

			out_e2e.append([enroll_utt, test_utt, e2e_scores[-1]])
			out_cos.append([enroll_utt, test_utt, cos_scores[-1]])
			out_fus.append([enroll_utt, test_utt, cos_scores[-1]])

	print('\nScoring done')

	with open(args.out_path+args.out_prefix+'e2e_scores.out' if args.out_prefix is not None else args.out_path+'e2e_scores.out', 'w') as f:
		for el in out_e2e:
			item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
			f.write("%s" % item)

	with open(args.out_path+args.out_prefix+'cos_scores.out' if args.out_prefix is not None else args.out_path+'cos_scores.out', 'w') as f:
		for el in out_cos:
			item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
			f.write("%s" % item)

	with open(args.out_path+args.out_prefix+'fus_scores.out' if args.out_prefix is not None else args.out_path+'fus_scores.out', 'w') as f:
		for el in out_fus:
			item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
			f.write("%s" % item)

	e2e_scores = np.asarray(e2e_scores)
	cos_scores = np.asarray(cos_scores)
	fus_scores = np.asarray(fus_scores)
	labels = np.asarray(labels)

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, e2e_scores)
	print('\nE2E eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, cos_scores)
	print('\nCOS eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, fus_scores)
	print('\nFUS eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))