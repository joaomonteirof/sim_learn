import os
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm


def test(dataloader, model, device, source_target, epoch = 0, tb_writer=None):

	model = model.eval()
	
	with torch.no_grad():

		model = model.to(device)

		target_iter = tqdm(enumerate(dataloader), total=len(dataloader))

		n_total = 0
		n_correct = 0
		predictions_domain = []
		labels_domain = []
		cross_entropy = torch.nn.CrossEntropyLoss()

		for t, batch in target_iter:

			if source_target == 'source':
				x_1, x_2, x_3, y_task_1, y_task_2, y_task_3, y_domain_1, y_domain_2, y_domain_3 = batch
				x = torch.cat((x_1, x_2, x_3), dim=0)
				y = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
				y_domain = torch.cat((y_domain_1, y_domain_2, y_domain_3), dim=0)
				y_domain.to(device)
			else:
				x, y, _ = batch

			x = x.to(device)
			y = y.to(device)
			
			# Task 
			out = F.softmax(model.forward(x), dim=1)
			class_output = out.reshape([out.size(0), model.n_domains, model.n_classes]).sum(1)
			log_prob = torch.log(class_output + 1e-7)
			loss = cross_entropy(log_prob, y)
			pred_task = class_output.data.max(1, keepdim=True)[1]
			n_correct += pred_task.eq(y.data.view_as(pred_task)).cpu().sum()
			n_total += x.size(0)

			try:
				predictions_task = torch.cat((predictions_task, pred_task), 0)
			except:
				predictions_task = pred_task

		acc = n_correct.item() * 1.0 / n_total			
		
		if tb_writer is not None:
			predictions_task_numpy = predictions_task.cpu().numpy()
			tb_writer.add_histogram('Test/'+source_target, predictions_task_numpy, epoch)
			tb_writer.add_scalar('Test/'+source_target+'_accuracy', acc, epoch)
			tb_writer.add_scalar('Test/'+source_target+'_val_loss', loss, epoch)

		return acc, loss


