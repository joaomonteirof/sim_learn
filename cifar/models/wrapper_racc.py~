import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class wrapper(nn.Module):
	def __init__(self, base_model, inf_mode='sim', normalize=False):
		super(wrapper, self).__init__()

		if normalize:
			self.normalization = transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
		else:
			self.normalization = None

		self.base_model = base_model
		self.inf_mode = inf_mode

	def forward(self, x):

		if self.normalization is not None:
			x = self.normalization(x)

		embeddings = self.base_model.forward(x)

		if self.inf_mode=='sim':
			logits = self.base_model.compute_logits(embeddings)
		elif self.inf_mode=='ce':
			logits = self.base_model.out_proj(embeddings)
		elif self.inf_mode=='fus':
			logits_sim = self.base_model.compute_logits(embeddings)
			logits_ce = self.base_model.out_proj(embeddings)
			logits = (logits_sim+logits_ce)*0.5

		return F.softmax(logits, dim=1)