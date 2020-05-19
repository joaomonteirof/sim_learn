from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import resnet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--batch-size', type=int, default=5, metavar='N', help='batch size')
parser.add_argument('--ablation-sim', action='store_true', default=False, help='Disables similarity learning')
args = parser.parse_args()

model = resnet.ResNet12(nh=args.n_hidden, n_h=args.hidden_size)

x, y = torch.rand(args.batch_size, 3, 84, 84), torch.randint(10, (args.batch_size,)).long()

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