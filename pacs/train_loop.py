import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from test import test
from utils import LabelSmoothingWithLogitsLoss, LabelSmoothingNLLLoss, GradualWarmupScheduler


class TrainLoop(object):

	def __init__(self, model, optimizer, source_loader, test_source_loader,
				target_loader, patience, factor, label_smoothing,
				lr_threshold, combined_loss=True, class_loss=False, domain_loss=False,
				verbose=-1, cp_name=None, save_cp=True, checkpoint_path=None, 
				checkpoint_epoch=None, cuda=True, logger=None):

		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				try:
					os.mkdir(self.checkpoint_path)
				except OSError:
					pass	

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.source_loader = source_loader
		self.test_source_loader = test_source_loader
		self.target_loader = target_loader
		self.history = {'train_loss':[0.0], 'accuracy_source':[], 'accuracy_target':[], 'loss_task_val_source':[]}
		self.cur_epoch = 0
		self.total_iter = 0
		self.device = next(self.model.parameters()).device
		self.writer = logger

		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=patience, gamma=factor)

		self.verbose = verbose
		self.save_cp = save_cp
		
		if self.writer:
			self.logging = True
		else:
			self.logging = False

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)

		if combined_loss:
			if label_smoothing > 0.0:
				self.comb_criterion = LabelSmoothingWithLogitsLoss(label_smoothing, lbl_set_size = self.model.n_classes*self.model.n_domains)
			else:
				self.comb_criterion = torch.nn.CrossEntropyLoss()
			if not 'comb_loss' in self.history:
				self.history['comb_loss'] = [0.0]
		else:
			self.comb_criterion = None

		if class_loss and self.model.n_domains>1:
			if label_smoothing > 0.0:
				self.class_criterion = LabelSmoothingNLLLoss(label_smoothing, lbl_set_size = self.model.n_classes)
			else:
				self.class_criterion = torch.nn.NLLLoss()

			if not 'class_loss' in self.history:
				self.history['class_loss'] = [0.0]
		else:
			self.class_criterion = None

		if domain_loss and self.model.n_domains>1:
			if label_smoothing > 0.0:
				self.domain_criterion = LabelSmoothingNLLLoss(label_smoothing, lbl_set_size = self.model.n_domains)
			else:
				self.domain_criterion = torch.nn.NLLLoss()

			if not 'domain_loss' in self.history:
				self.history['domain_loss'] = [0.0]
		else:
			self.domain_criterion = None

	def train(self, n_epochs=1, save_every=1):

		while self.cur_epoch < n_epochs:

			print('Epoch {}/{} \n'.format(self.cur_epoch + 1, n_epochs))
			self.source_loader.dataset.update_lists()
			self.test_source_loader.dataset.update_lists()
			
			source_iter = tqdm(enumerate(self.source_loader), total=len(self.source_loader), disable=False)

			for t, batch in source_iter:

				loss_dict = self.train_step(batch)

				self.total_iter += 1
				
				if self.logging:
					for key in loss_dict:
						self.writer.add_scalar('Iteration/'+key, loss_dict[key], self.total_iter)
					self.writer.add_scalar('misc/LR', self.optimizer.param_groups[0]['lr'], self.total_iter)
					self.writer.add_scalar('misc/Epoch', self.cur_epoch + 1, self.total_iter)

			self.scheduler.step()

			for key in loss_dict:
				self.history[key][-1] /= t+1
				if self.logging:
					self.writer.add_scalar('Epoch/'+key, self.history[key][-1], self.cur_epoch)
				self.history[key].append(0.0)
			
			acc_source, loss_task_val_source = test(self.test_source_loader, self.model, self.device, source_target = 'source',
												epoch = self.cur_epoch, tb_writer = self.writer if self.logging else None)
			acc_target, loss_task_target = test(self.target_loader, self.model, self.device, source_target = 'target',
											epoch = self.cur_epoch, tb_writer = self.writer if self.logging else None)
			
			self.history['accuracy_source'].append(acc_source)
			self.history['accuracy_target'].append(acc_target)
			self.history['loss_task_val_source'].append(loss_task_val_source)
			
			self.source_epoch_best_loss_task = np.argmin(self.history['train_loss'][:-1])
			self.source_epoch_best_loss_task_val = np.argmin(self.history['loss_task_val_source'])
			self.source_epoch_best = np.argmax(self.history['accuracy_source'])
			self.target_epoch_best = np.argmax(self.history['accuracy_target'])

			self.source_best_acc = np.max(self.history['accuracy_source'])
			self.target_best_loss_task = self.history['accuracy_target'][self.source_epoch_best_loss_task]
			self.target_best_source_acc = self.history['accuracy_target'][self.source_epoch_best]
			self.target_best_acc = np.max(self.history['accuracy_target'])
			self.target_best_acc_loss_task_val = self.history['accuracy_target'][self.source_epoch_best_loss_task_val]

			self.print_results()

			if self.logging:
				self.writer.add_scalar('Epoch/Loss-task-val', self.history['loss_task_val_source'][-1], self.cur_epoch)
				self.writer.add_scalar('Epoch/Acc-Source', self.history['accuracy_source'][-1], self.cur_epoch)
				self.writer.add_scalar('Epoch/Acc-target', self.history['accuracy_target'][-1], self.cur_epoch)				
																
			self.cur_epoch += 1

			if self.save_cp and (self.cur_epoch % save_every == 0 or self.history['accuracy_target'][-1] > np.max([-np.inf]+self.history['accuracy_target'][:-1])):
				self.checkpointing()
		
		if self.logging:
			self.writer.close()

		results_acc = [self.target_best_loss_task,
						self.target_best_source_acc,
						self.source_best_acc,
						self.target_best_acc]

		results_epochs = [self.source_epoch_best_loss_task,
		self.source_epoch_best_loss_task_val,
		self.source_epoch_best,
		self.target_epoch_best]

		return np.min(self.history['loss_task_val_source']), results_acc, results_epochs
		
	def train_step(self, batch):
		self.model.train()

		x_1, x_2, x_3, y_task_1, y_task_2, y_task_3, y_domain_1, y_domain_2, y_domain_3 = batch

		x = torch.cat((x_1, x_2, x_3), dim=0)
		y_task = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
		y_domain = torch.cat((y_domain_1, y_domain_2, y_domain_3), dim=0)
		
		if self.cuda_mode:
			x = x.to(self.device)
			y_task = y_task.to(self.device)
			y_domain = y_domain.to(self.device)

		y_comb = y_task+(self.model.n_domains-1)*y_domain

		out = self.model.forward(x)

		loss = 0.0
		loss_dict = {}

		if self.comb_criterion is not None:
			comb_loss = self.comb_criterion(out, y_comb)
			loss += comb_loss
			self.history['comb_loss'][-1] += comb_loss.item()
			loss_dict['comb_loss'] = comb_loss.item()

		if self.class_criterion is not None:
			out_sm = F.softmax(out, dim=1)
			out_comb = out_sm.reshape([out_sm.size(0), self.model.n_domains, self.model.n_classes])
			log_prob = torch.log(out_comb.sum(1) + 1e-7)
			class_loss = self.class_criterion(log_prob, y_task)
			loss += class_loss
			self.history['class_loss'][-1] += class_loss.item()
			loss_dict['class_loss'] = class_loss.item()

		if self.domain_criterion is not None:
			try:
				log_prob = torch.log(out_comb.sum(1) + 1e-7)
				domain_loss = self.domain_criterion(log_prob, y_domain)
			except NameError:
				out_sm = F.softmax(out, dim=1)
				out_comb = out_sm.reshape([out_sm.size(0), self.model.n_domains, self.model.n_classes])
				log_prob = torch.log(out_comb.sum(1) + 1e-7)
				domain_loss = self.domain_criterion(log_prob, y_domain)

			loss += domain_loss
			self.history['domain_loss'][-1] += domain_loss.item()
			loss_dict['domain_loss'] = domain_loss.item()

		self.history['train_loss'][-1] += loss.item()
		loss_dict['train_loss'] = loss.item()
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss_dict

	def checkpointing(self):
		if self.verbose>0:
			print(' ')	
			print('Checkpointing...')
			
		ckpt = {'model_state': self.model.state_dict(),
				'n_classes': self.model.n_classes,
				'n_domains': self.model.n_domains,
				'optimizer_state': self.optimizer.state_dict(),
				'scheduler_state': self.scheduler.state_dict(),
				'history': self.history,
				'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, epoch):
		ckpt = self.save_epoch_fmt.format(epoch)

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_results(self):
		print('\nCurrent loss: {}.'.format(self.history['train_loss'][-2]))

		print('\nVALIDATION ON SOURCE DOMAINS')
		print('Current, best, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['accuracy_source'][-1], self.source_best_acc, self.source_epoch_best+1))
		
		print('\nVALIDATION ON TARGET DOMAIN')
		print('Current, best, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['accuracy_target'][-1], self.target_best_acc, self.target_epoch_best+1))
		
		print('\nVALIDATION ON TARGET DOMAIN - BEST SOURCE VAL ACC')
		print('Best and epoch: {:0.4f}, {}'.format(self.target_best_source_acc, self.source_epoch_best+1))
		
		print('\nVALIDATION ON TARGET DOMAIN - BEST TASK LOSS')		
		print('Best and epoch: {:0.4f}, {}'.format(self.target_best_loss_task, self.source_epoch_best_loss_task+1))
		
		print('\nVALIDATION ON TARGET DOMAIN - BEST VAL TASK LOSS')
		print('Best and epoch: {:0.4f}, {}'.format(self.target_best_acc_loss_task_val, self.source_epoch_best_loss_task_val+1))						
		
