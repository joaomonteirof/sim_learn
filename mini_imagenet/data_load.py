import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os
import subprocess
import shlex
from utils import strided_app

def collater(batch):

	examples, labels = [], []

	for el in batch:
		examples_sample, y = el[:-1], el[-1]

		examples.append( torch.cat([ex.unsqueeze(0) for ex in examples_sample], dim=0) )
		labels.append( torch.cat(len(examples_sample)*[y], dim=0).squeeze().contiguous() )

	examples, labels = torch.cat(examples, dim=0), torch.cat(labels, dim=0)

	return examples, labels

class Loader(Dataset):

	def __init__(self, hdf5_name, transformation):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.transformation = transformation

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		example_1, example_2, example_3, example_4, example_5, clss, y = self.example_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		example_1_data = self.transformation( torch.from_numpy(self.open_file[clss][example_1][:,...]) )
		example_2_data = self.transformation( torch.from_numpy(self.open_file[clss][example_2][:,...]) )
		example_3_data = self.transformation( torch.from_numpy(self.open_file[clss][example_3][:,...]) )
		example_4_data = self.transformation( torch.from_numpy(self.open_file[clss][example_4][:,...]) )
		example_5_data = self.transformation( torch.from_numpy(self.open_file[clss][example_5][:,...]) )

		return example_1_data.contiguous(), example_2_data.contiguous(), example_3_data.contiguous(), example_4_data.contiguous(), example_5_data.contiguous(), y

	def __len__(self):
		return len(self.example_list)

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.class2file = {}
		self.clss2label = {}

		for i, clss in enumerate(open_file):
			clss_example_list = list(open_file[clss])
			self.class2file[clss] = clss_example_list
			self.clss2label[clss] = torch.LongTensor([i])

		open_file.close()

		self.n_classes = len(self.class2file)

	def update_lists(self):

		self.example_list = []

		for i, clss in enumerate(self.class2file):
			clss_file_list = np.random.permutation(self.class2file[clss])

			idxs = strided_app(np.arange(len(clss_file_list)), 5, 5)

			for idxs_list in idxs:
				if len(idxs_list)==5:
					self.example_list.append([clss_file_list[file_idx] for file_idx in idxs_list])
					self.example_list[-1].append(clss)
					self.example_list[-1].append(self.clss2label[clss])

class Loader_list(Dataset):

	def __init__(self, hdf5_name, file_list, transformation):
		super(Loader_list, self).__init__()
		self.hdf5_name = hdf5_name
		self.transformation = transformation
		self.example_list = file_list
		self.open_file = None

	def __getitem__(self, index):

		example, clss, y = self.example_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		example_data = self.transformation( torch.from_numpy(self.open_file[clss][example][:,...]) )

		return example_data.contiguous(), y

	def __len__(self):
		return len(self.example_list)

class fewshot_eval_builder(Object):

	def __init__(self, hdf5_name, train_transformation, test_transformation, k_shot=5, n_way=5, n_queries=15):
		super(fewshot_eval_builder, self).__init__()
		self.hdf5_name = hdf5_name
		self.train_transformation = train_transformation
		self.test_transformation = test_transformation
		self.k_shot = k_shot
		self.n_way = n_way
		self.n_queries = n_queries

		self.create_lists()

		def get_task_loaders():

			task_classes = random.sample(self.class_list, self.n_way)

			train_list, test_list = [], []

			for i, clss in enumerate(task_classes):
				ex_list = random.sample(self.class2file, self.k_shot+self.n_queries)

				sub_train_list = ex_list[:self.k_shot]
				sub_test_list = ex_list[self.k_shot:]

				for train_ex in sub_train_list:
					train_list.append([train_ex, clss, torch.LongTensor([i])])

				for test_ex in sub_test_list:
					test_list.append([test_ex, clss, torch.LongTensor([i])])

			train_loader = Loader_list(hdf5_name=self.hdf5_name, file_list=train_list, transformation=self.train_transformation)
			test_loader = Loader_list(hdf5_name=self.hdf5_name, file_list=test_list, transformation=self.test_transformation)

			return train_loader, test_loader

		def create_lists(self):

			open_file = h5py.File(self.hdf5_name, 'r')

			self.class2file = {}
			self.class_list = []

			for i, clss in enumerate(open_file):
				self.class_list.append(clss)
				clss_example_list = list(open_file[clss])
				self.class2file[clss] = clss_example_list

			open_file.close()

			self.n_classes = len(self.class2file)

if __name__=='__main__':

	import torch.utils.data
	from torchvision import transforms
	import argparse

	parser = argparse.ArgumentParser(description='Test data loader')
	parser.add_argument('--hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
	args = parser.parse_args()

	transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])	
	dataset = Loader(args.hdf_file, transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

	loader.dataset.update_lists()

	print('Dataset length: {}, {}'.format(len(loader.dataset), len(loader.dataset.example_list)))

	for batch in loader:
		utt_1, utt_2, utt_3, utt_4, utt_5, y = batch

	print(utt_1.size(), utt_2.size(), utt_3.size(), utt_4.size(), utt_5.size(), y.size())

	print(y)