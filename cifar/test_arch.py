from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG16', nh=args.n_hidden, n_h=args.hidden_size)
elif args.model == 'resnet':
	model = resnet.ResNet18(nh=args.n_hidden, n_h=args.hidden_size)
elif args.model == 'densenet':
	model = densenet.densenet_cifar(nh=args.n_hidden, n_h=args.hidden_size)

batch = torch.rand(3, 3, 32, 32)

emb = model.forward(batch)

print('\nEmbeddings: ', emb.size())

out = model.out_proj(emb)

print('Auxiliary outputs: ', out.size())

print('Centroids prior to update: ', model.centroids.size())

model.update_centroids(emb, torch.zeros(emb.size(0)).long())

print('Centroids post update: ', model.centroids.size())

logits = model.compute_logits(emb)

print('Logits: ', logits.size(), '\n')

loss_ce = torch.nn.functional.cross_entropy(out, torch.ones(out.size(0)).long())
loss_sim = torch.nn.functional.cross_entropy(logits, torch.ones(out.size(0)).long())
loss = loss_ce+loss_sim
loss.backward()

print('Losses (ce, sim, all): {}, {}, {} \n'.format(loss_ce, loss_sim, loss))