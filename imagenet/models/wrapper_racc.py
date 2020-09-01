import torch
import torch.nn as nn
import torch.nn.functional as F

class wrapper(nn.Module):
	def __init__(self, base_model, inf_mode='sim'):
		super(wrapper, self).__init__()

		self.base_model = base_model
		self.inf_mode = inf_mode

	def forward(self, x):

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