import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm
from utils.losses import LabelSmoothingLoss
from utils.harvester import AllTripletSelector
from utils.utils import compute_eer

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, max_gnorm=10.0, label_smoothing=0.0, verbose=-1, device=0, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, ablation_sim=False, ablation_ce=False, cuda=True, logger=None):
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
		self.max_gnorm = max_gnorm
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.harvester = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = device
		self.logger = logger
		self.history = {'train_loss': [], 'train_loss_batch': [], 'ce_loss': [], 'ce_loss_batch': [], 'sim_loss': [], 'sim_loss_batch': [], 'bin_loss': [], 'bin_loss_batch': []}

		if label_smoothing>0.0:
			self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=train_loader.dataset.n_speakers)
			self.disc_label_smoothing = label_smoothing*0.5
		else:
			self.ce_criterion = torch.nn.CrossEntropyLoss()
			self.disc_label_smoothing = 0.0

		if self.valid_loader is not None:
			self.history['e2e_eer'] = []
			self.history['cos_eer'] = []
			self.history['fus_eer'] = []

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):

			np.random.seed()
			self.train_loader.dataset.update_lists()

			if self.logger:
				self.logger.add_scalar('Info/Epoch', self.cur_epoch+1, self.total_iters)

			if self.verbose>1:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				print('Number of training examples given new list: {}'.format(len(self.train_loader.dataset)))
				train_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

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
				if self.logger:
					self.logger.add_scalar('Train/Total train Loss', train_loss, self.total_iters)
					self.logger.add_scalar('Train/Similarity class. Loss', sim_loss, self.total_iters)
					self.logger.add_scalar('Train/Cross enropy', ce_loss, self.total_iters)
					self.logger.add_scalar('Train/Bin. Loss', bin_loss, self.total_iters)
					self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)

				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))
			self.history['ce_loss'].append(ce_loss_epoch/(t+1))
			self.history['sim_loss'].append(sim_loss_epoch/(t+1))
			self.history['bin_loss'].append(bin_loss_epoch/(t+1))

			if self.verbose>1:
				print(' ')
				print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))
				print('CE loss: {:0.4f}'.format(self.history['ce_loss'][-1]))
				print('Similarity classification loss: {:0.4f}'.format(self.history['sim_loss'][-1]))
				print('Bin. loss: {:0.4f}'.format(self.history['bin_loss'][-1]))
				print(' ')

			if self.valid_loader is not None:

				e2e_scores, cos_scores, labels = None, None, None

				for t, batch in enumerate(self.valid_loader):
					e2e_scores_batch, cos_scores_batch, labels_batch = self.valid(batch)

					try:
						e2e_scores = np.concatenate([e2e_scores, e2e_scores_batch], 0)
						cos_scores = np.concatenate([cos_scores, cos_scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						e2e_scores, cos_scores, labels = e2e_scores_batch, cos_scores_batch, labels_batch

				fus_scores = (e2e_scores + 0.5*(cos_scores+1.))*0.5

				self.history['e2e_eer'].append(compute_eer(labels, e2e_scores))
				self.history['cos_eer'].append(compute_eer(labels, cos_scores))
				self.history['fus_eer'].append(compute_eer(labels, fus_scores))

				if self.logger:
					self.logger.add_scalar('Valid/E2E EER', self.history['e2e_eer'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best E2E EER', np.min(self.history['e2e_eer']), self.total_iters-1)
					self.logger.add_scalar('Valid/Cosine EER', self.history['cos_eer'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best Cosine EER', np.min(self.history['cos_eer']), self.total_iters-1)
					self.logger.add_scalar('Valid/Fus EER', self.history['fus_eer'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best Fus EER', np.min(self.history['fus_eer']), self.total_iters-1)
					self.logger.add_pr_curve('E2E ROC', labels=labels, predictions=e2e_scores, global_step=self.total_iters-1)
					self.logger.add_pr_curve('Cosine ROC', labels=labels, predictions=cos_scores, global_step=self.total_iters-1)
					self.logger.add_pr_curve('Fus ROC', labels=labels, predictions=fus_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/COS_Scores', values=cos_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/E2E_Scores', values=e2e_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/FUS_Scores', values=fus_scores, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/Labels', values=labels, global_step=self.total_iters-1)
					self.logger.add_embedding(mat=self.model.centroids.detach().cpu().numpy(), metadata=np.arange(self.model.centroids.size(0)), global_step=self.total_iters-1)

				if self.verbose>1:
					print(' ')
					print('Current e2e EER, best e2e EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['e2e_eer'][-1], np.min(self.history['e2e_eer']), 1+np.argmin(self.history['e2e_eer'])))
					print('Current cos EER, best cos EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['cos_eer'][-1], np.min(self.history['cos_eer']), 1+np.argmin(self.history['cos_eer'])))
					print('Current fus EER, best fus EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['fus_eer'][-1], np.min(self.history['fus_eer']), 1+np.argmin(self.history['fus_eer'])))

			if self.verbose>1:
				print('Current LR: {}'.format(self.optimizer.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.valid_loader is not None and self.save_cp and (self.cur_epoch % save_every == 0 or self.history['e2e_eer'][-1] < np.min([np.inf]+self.history['e2e_eer'][:-1]) or self.history['cos_eer'][-1] < np.min([np.inf]+self.history['cos_eer'][:-1])):
					self.checkpointing()
			elif self.save_cp and self.cur_epoch % save_every == 0:
					self.checkpointing()

		if self.verbose>1:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>1:
				print('Best e2e eer and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['e2e_eer']), 1+np.argmin(self.history['e2e_eer'])))
				print('Best cos eer and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['cos_eer']), 1+np.argmin(self.history['cos_eer'])))
				print('Best fus eer and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['fus_eer']), 1+np.argmin(self.history['fus_eer'])))

			return [np.min(self.history['e2e_eer']), np.min(self.history['cos_eer'])]
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

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			x, y = batch

			if self.cuda_mode:
				x = x.to(self.device)
				y = y.to(self.device)

			embeddings = self.model.forward(x)

			# Get all triplets now for bin classifier
			triplets_idx = self.harvester.get_triplets(embeddings.detach(), y)
			triplets_idx = triplets_idx.to(self.device)

			emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

			e2e_scores_p = self.model.forward_bin(emb_a, emb_p).squeeze()
			e2e_scores_n = self.model.forward_bin(emb_a, emb_n).squeeze()

			cos_scores_p = torch.nn.functional.cosine_similarity(emb_a, emb_p)
			cos_scores_n = torch.nn.functional.cosine_similarity(emb_a, emb_n)

		return np.concatenate([e2e_scores_p.detach().cpu().numpy(), e2e_scores_n.detach().cpu().numpy()], 0), np.concatenate([cos_scores_p.detach().cpu().numpy(), cos_scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(e2e_scores_p.size(0)), np.zeros(e2e_scores_n.size(0))], 0)

	def checkpointing(self):

		# Checkpointing
		if self.verbose>1:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'dropout_prob': self.model.dropout_prob,
		'n_hidden': self.model.n_hidden,
		'hidden_size': self.model.hidden_size,
		'emb_size': self.model.emb_size,
		'sm_type': self.model.sm_type,
		'ncoef': self.model.ncoef,
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
			self.optimizer.step_num = ckpt['total_iters']
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.to(self.device)
		else:
			print('No checkpoint found at: {}'.format(ckpt))
