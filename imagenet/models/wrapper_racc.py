import torch
import torch.nn as nn
import torch.nn.functional as F

class wrapper(nn.Module):
	def __init__(self, base_model, ce_layer=False):
		super(wrapper, self).__init__()

		self.base_model = base_model
		self.ce_layer = ce_layer

	def forward(self, x):

		embeddings = self.base_model.forward(x)

		if self.ce_layer:
			logits = self.base_model.out_proj(embeddings)
		else:
			logits = self.base_model.compute_logits(embeddings)

		return F.softmax(logits, dim=1)