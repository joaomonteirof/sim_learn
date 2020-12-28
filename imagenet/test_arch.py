from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--batch-size', type=int, default=5, metavar='N', help='batch size')
parser.add_argument('--ablation-sim', action='store_true', default=False, help='Disables similarity learning')
parser.add_argument('--pretrained', action='store_true', default=False, help='Get pretrained weights on imagenet. Encoder only')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG19', nh=args.n_hidden, n_h=args.hidden_size)
elif args.model == 'resnet':
	model = resnet.ResNet50(nh=args.n_hidden, n_h=args.hidden_size)
elif args.model == 'densenet':
	model = densenet.DenseNet121(nh=args.n_hidden, n_h=args.hidden_size)

if args.pretrained:
	print('\nLoading pretrained encoder from torchvision\n')
	if args.model == 'vgg':
		model_pretrained = torchvision.models.VGG('VGG19', pretrained=True)
		pretrained_state_dict = model_pretrained.state_dict()
	elif args.model == 'resnet':
		model_pretrained = torchvision.models.resnet50(pretrained=True)
		pretrained_state_dict = model_pretrained.state_dict()
	elif args.model == 'densenet':
		model_pretrained = torchvision.models.densenet121(pretrained=True)
		pretrained_state_dict = model_pretrained.state_dict()

	print(model.load_state_dict(pretrained_state_dict, strict=False))

x, y = torch.rand(args.batch_size, 3, 224, 224), torch.randint(1000, (args.batch_size,)).long()

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