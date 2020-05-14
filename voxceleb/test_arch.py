from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['TDNN', 'TDNN_multipool'], default='TDNN', help='Model')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--batch-size', type=int, default=5, metavar='N', help='batch size')
parser.add_argument('--ablation-sim', action='store_true', default=False, help='Disables similarity learning')
parser.add_argument('--emb-size', type=int, default=256, metavar='S', help='latent layer dimension (default: 256)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--n-speakers', type=int, default=100, metavar='N', help='Number of speakers')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
args = parser.parse_args()

if args.model == 'TDNN':
	model = model_.TDNN(n_z=args.emb_size, nh=args.n_hidden, n_h=args.hidden_size, n_speakers=args.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'TDNN_multipool':
	model = model_.TDNN_multipool(n_z=args.emb_size, nh=args.n_hidden, n_h=args.hidden_size, n_speakers=args.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)

x, y = torch.rand(args.batch_size, 1, args.ncoef, 200), torch.randint(args.n_speakers, (args.batch_size,)).long()

emb = model.forward(x)

print('\nEmbeddings: ', emb.size())

out = model.out_proj(emb)

print('Auxiliary outputs: ', out.size())

print('Centroids prior to update: ', model.centroids.size())

model.update_centroids(emb, y)

print('Centroids post update: ', model.centroids.size())

logits = model.compute_logits(emb, ablation=args.ablation_sim)

print('Logits: ', logits.size(), '\n')

loss_ce = torch.nn.functional.cross_entropy(out, y)
loss_sim = torch.nn.functional.cross_entropy(logits, y)
loss = loss_ce+loss_sim
loss.backward()

print('Losses (ce, sim, all): {}, {}, {} \n'.format(loss_ce, loss_sim, loss))
