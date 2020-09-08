import torch
import torch.nn.functional as F

import numpy as np
import random

import os
from tqdm import tqdm

from harvester import AllTripletSelector
from models.losses import LabelSmoothingLoss
from utils import correct_topk
from data_load import Loader

class TrainLoop(object):
	def __init__(self, model, optimizer, train_loader, valid_loader, max_gnorm, label_smoothing, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, ablation_sim=False, ablation_ce=False, cuda=True, logger=None):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.ablation_sim = ablation_sim
		self.ablation_ce = ablation_ce
		self.model = model
		self.optimizer = optimizer
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 400], gamma=0.1)
		self.max_gnorm = max_gnorm
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.harvester = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = next(self.model.parameters()).device
		self.logger = logger
		self.history = {'train_loss': [], 'train_loss_batch': [], 'ce_loss': [], 'ce_loss_batch': [], 'sim_loss': [], 'sim_loss_batch': [], 'bin_loss': [], 'bin_loss_batch': []}
		self.best_r_at_3 = -np.inf

		if label_smoothing>0.0:
			self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=self.model.n_classes)
			self.disc_label_smoothing = label_smoothing
		else:
			self.ce_criterion = torch.nn.CrossEntropyLoss()
			self.disc_label_smoothing = 0.0

		if self.valid_loader is not None:
			self.history['r@3'] = []

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1, eval_every=1000):

		while (self.cur_epoch < n_epochs):

			self.cur_epoch += 1
			np.random.seed()
			if isinstance(self.train_loader.dataset, Loader):
				self.train_loader.dataset.update_lists()

			if self.logger:
				self.logger.add_scalar('Info/Epoch', self.cur_epoch, self.total_iters)

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			self.save_epoch_cp = False
			train_loss_epoch=0.0
			ce_loss_epoch=0.0
			sim_loss_epoch=0.0
			bin_loss_epoch=0.0
			for t, batch in train_iter:
				train_loss, ce_loss, sim_loss, bin_loss = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				self.history['ce_loss_batch'].append(ce_loss)
				self.history['sim_loss_batch'].append(sim_loss)
				self.history['bin_loss_batch'].append(bin_loss)
				train_loss_epoch+=train_loss
				ce_loss_epoch+=ce_loss
				sim_loss_epoch+=sim_loss
				bin_loss_epoch+=bin_loss

				self.total_iters += 1

				if self.logger:
					self.logger.add_scalar('Train/Total train Loss', train_loss, self.total_iters)
					self.logger.add_scalar('Train/Similarity class. Loss', sim_loss, self.total_iters)
					self.logger.add_scalar('Train/Cross enropy', ce_loss, self.total_iters)
					self.logger.add_scalar('Train/Bin. loss', bin_loss, self.total_iters)
					self.logger.add_scalar('Info/LR', self.optimizer.param_groups[0]['lr'], self.total_iters)

				if self.total_iters % eval_every == 0:
					self.evaluate()
					if self.save_cp and self.history['r@3'][-1] > np.max([-np.inf]+self.history['r@3'][:-1]):
							self.checkpointing()
							self.save_epoch_cp = True

			self.history['train_loss'].append(train_loss_epoch/(t+1))
			self.history['ce_loss'].append(ce_loss_epoch/(t+1))
			self.history['sim_loss'].append(sim_loss_epoch/(t+1))
			self.history['bin_loss'].append(bin_loss_epoch/(t+1))

			if self.verbose>0:
				print('\nTotal train loss: {:0.4f}'.format(self.history['train_loss'][-1]))
				print('CE loss: {:0.4f}'.format(self.history['ce_loss'][-1]))
				print('Sim loss: {:0.4f}'.format(self.history['sim_loss'][-1]))
				print('Bin loss: {:0.4f}'.format(self.history['bin_loss'][-1]))
				print('Current LR: {}\n'.format(self.optimizer.param_groups[0]['lr']))

			if self.save_cp and self.cur_epoch % save_every == 0 and not self.save_epoch_cp:
					self.checkpointing()

			self.scheduler.step()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			return [self.history['r@3']]
		else:
			return [np.min(self.history['train_loss'])]

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		x = x.to(self.device)
		y = y.to(self.device)

		embeddings = self.model.forward(x)

		self.model.update_centroids(embeddings, y)

		if not self.ablation_ce:
			ce_loss = self.ce_criterion(self.model.out_proj(embeddings, y), y)
		else:
			ce_loss = 0.0

		sim_loss = self.ce_criterion(self.model.compute_logits(embeddings, ablation=self.ablation_sim), y)

		# Get all triplets now for bin classifier
		triplets_idx = self.harvester.get_triplets(embeddings.detach(), y)
		triplets_idx = triplets_idx.to(self.device, non_blocking=True)

		emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

		pred_bin_p, pred_bin_n = self.model.forward_bin(emb_a, emb_p).squeeze(), self.model.forward_bin(emb_a, emb_n).squeeze()

		if self.ablation_sim:
			loss_bin = (torch.nn.functional.cosine_similarity(emb_a, emb_n) - torch.nn.functional.cosine_similarity(emb_a, emb_p) ).mean()
		else:
			loss_bin = torch.nn.BCEWithLogitsLoss()(pred_bin_p, torch.rand_like(pred_bin_p)*self.disc_label_smoothing+(1.0-self.disc_label_smoothing)) + torch.nn.BCEWithLogitsLoss()(pred_bin_n, torch.rand_like(pred_bin_n)*self.disc_label_smoothing)

		loss = ce_loss + sim_loss + loss_bin
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

		return loss.item(), 0.0 if self.ablation_ce else ce_loss.item(), sim_loss.item(), loss_bin.item()

	def valid(self):

		self.model.eval()
		r_at_3_e2e = 0
		embeddings = []
		labels = []

		with torch.no_grad():

			for batch in self.valid_loader:

				x, y = batch

				if args.cuda:
					x = x.to(self.device)

				emb = self.model.forward(x).detach().cpu()

				embeddings.append(emb)
				labels.append(y)

		embeddings = torch.cat(embeddings, 0)
		labels = torch.cat(labels, 0)

		for i, label in labels:

			enroll_emb = embeddings[i].unsqueeze(0).to(self.device)

			e2e_scores = torch.zeros(len(labels))

			for j in range(0, len(labels), self.valid_loader.batch_size):

				test_emb = embeddings[j:(min(j+self.valid_loader.batch_size, len(embeddings))),:].to(self.device)
				enroll_emb_repeated = enroll_emb.repeat(test_emb.size(0), 1)

				dist_e2e = self.model.forward_bin(enroll_emb_repeated, test_emb).squeeze(-1)
				
				for l in range(dist_e2e.size(0)):

					if i==(j+l): continue ## skip same example

					e2e_scores[j+l] = dist_e2e[l].item()

			_, topk_e2e_idx = torch.topk(torch.Tensor(e2e_scores), max(args.k_list)+1)

			sorted_e2e_classes = labels[topk_e2e_idx]

			if label in sorted_e2e_classes[:3]:
				r_at_3_e2e+=1

			r_at_3_e2e/=len(labels)

			return r_at_3_e2e



	def evaluate(self):

		if self.verbose>0:
			print('\nIteration {} - Epoch {}'.format(self.total_iters, self.cur_epoch))

		r_at_3 = self.valid()

		self.history['r@3'].append(r_at_3)

		if self.history['r@3'][-1]>self.best_r_at_3:
			self.best_r_at_3 = self.history['r@3'][-1]
			self.best_r_at_3_epoch = self.cur_epoch
			self.best_r_at_3_iteration = self.total_iters

		if self.logger:
			self.logger.add_scalar('Valid/r@3', self.history['r@3'][-1], self.total_iters)
			self.logger.add_scalar('Valid/Best r@3', np.max(self.history['r@3']), self.total_iters)
			self.logger.add_embedding(mat=self.model.centroids.detach().cpu().numpy(), metadata=np.arange(self.model.centroids.size(0)), global_step=self.total_iters)

		if self.verbose>0:
			print('\nCurrent r@3, best r@3, and epoch - iteration: {:0.4f}, {:0.4f}, {}, {}'.format(self.history['r@3'][-1], np.max(self.history['r@3']), self.best_r_at_3_epoch, self.best_r_at_3_iteration))

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'dropout_prob': self.model.dropout_prob,
		'n_hidden': self.model.n_hidden,
		'hidden_size': self.model.hidden_size,
		'sm_type': self.model.sm_type,
		'optimizer_state': self.optimizer.state_dict(),
		'scheduler_state': self.scheduler.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'centroids': self.model.centroids}
		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			self.model.centroids = ckpt['centroids']
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))
