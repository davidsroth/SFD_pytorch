from __future__ import print_function

import torch
import torch.optim as optim

from torch.utils.data.dataset import Dataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import torchvision
import torchvision.transforms as transforms

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

from net_s3fd import *
# from s3fd import *
from bbox import *
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

class Joint_Loss(torch.nn.Module):
	def __init__(self, w1=1, w2=2,w3=3):
		super(Joint_Loss, self).__init__()
		
		self.w1 = w1
		self.w2 = w2
		self.w3 = w3

	def forward(self,input,target):
		mse_loss = torch.nn.MSELoss()
		cross_entropy_loss = torch.nn.BCELoss()

		# calculate cross-entropy loss on face and gender classificaition and MSE on regression
		cls_loss_face = [cross_entropy_loss(a,b) for a,b in zip(input[0], target[0])]
		reg_loss 	  = [mse_loss(a,b) for a,b in zip(input[1], target[1])]
		cls_loss_gen  = [cross_entropy_loss(a,b) for a,b in zip(input[2], target[2])]

		total = w1*sum(cls_loss_face) + w2*sum(reg_loss) + w3*sum(cls_loss_face)

		return total
		

class AFLW_Dataset(Dataset):
	"""Dataset wrapping images and target labels
	Arguments:
		path to ground truth labels
		path to images
		image extension
		PIL transforms
	"""

	def __init__(self, annot_csv, transform=None):

		tmp_df = pd.read_csv(annot_csv)

		self.mlb = MultiLabelBinarizer()
		self.img_path = img_path
		self.img_ext = img_ext
		self.transform = transform

		self.x_train = tmp_df["filepath"]
		self.y_trainRect = tmp_df[["x", "y", "w", "w"]]
		
		self.y_trainGen = self.mlb.fit_transform(tmp_df["Gender"].str.split()).astype(np.float32)


	def __len__(self):
		return len(self.x_train.index)

	def __getitem__(self, index):
		img = cv2.imread(self.x_train[index])

		rect = self.y_trainRect.iloc[index]
		gen  = self.y_trainGen.iloc[index]

		labels = [rect, gen]

		sample = {'image ': image, 'labels': labels}

		if self.transform:
			sample = self.transform(sample)

		return sample

class Rescale(object):
	"""Rescale an image to a given size.

	args:
		output_size
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, labels = sample['image'], sample['labels']

		h,w = image.shape[:2]

		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = int(new_h), int(new_w)

		img = cv2.resize(img, (new_h, new_w))

		# relocated bounding box
		labels[0][0] = labels[0][0] * (new_w/w)
		labels[0][1] = labels[0][1] * (new_h/h)
		labels[0][2] = labels[0][2] * (new_w/w)
		labels[0][3] = labels[0][3] * (new_h/h)

		return {'image': img, 'labels': labels}

class ToTensor(object):
	"""converts ndarrays to Tensors"""

	def __call__(self, sample):

		image, labels = sample['image'], sample['labels']

		image = image.transpose((2,0,1))

		return {"image": torch.from_numpy(image), 'labels': torch.from_numpy(labels)}




def save(model, optimizer, loss, filename):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.data[0]
        }
    torch.save(save_dict, filename)


train_data = "index.csv"
img_path = "/Volumes/Seagate Expansion Drive/AFLW_Images/aflw/data/flickr/0"
img_ext = ".jpg"




def train_model(model, criterion, optimizer, num_classes, num_epochs = 100):
	for epoch in range(num_epochs):

		model.train()
		running_loss = 0.0



def load_base_layers(model, source):

	pretrained_weights = torch.load(source)
	full_parameters = model.state_dict()

	base_parameters = {k:v for k,v in pretrained_weights.items() if k in full_parameters}
	full_parameters.update(base_parameters)
	model.load_state_dict(full_parameters)

def freeze_layers(model):
	for param in model.parameters():
		param.requires_grad = False


def main():
	transformations = transforms.Compose(
		[
			Rescale(256),
			ToTensor()
		])

	dataset = AFLW_Dataset('/Volumes/Seagate Expansion Drive/AFLW_Images/aflw/data/AFLW_rect.csv', transform=transformations)

	for i in range(10):
		sample = dataset[i]

	# dataloader = DataLoader()


	# num_classes = 2
	# model = s3fd(num_classes)

	# load_base_layers(model, 's3fd_convert.pth')

	# freeze_layers(model)

	# model.conv3_3_norm_mbox_gen  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
	# model.conv4_3_norm_mbox_gen  = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
	# mode.conv5_3_norm_mbox_gen   = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
	# model.fc7_mbox_gen      	 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
	# model.conv6_2_mbox_gen 	 	 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
	# model.conv7_2_mbox_gen  	 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)

	# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

if __name__ == "__main__":
	main()












