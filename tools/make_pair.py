from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math

from Preprocess_data import is_image_file, load_img, save_img, get_test_set, get_training_set

# Testing settings
parser = argparse.ArgumentParser(description='RestoNet-PyTorch-implementation')
parser.add_argument('--dataset', default='facades_subdir', help='facades')
parser.add_argument('--model', type=str, default='3000', help='model file to use')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--datasetPath', default='../dataset/Facade', help='facades')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.dataset):
    os.mkdir(opt.dataset)
if not os.path.exists(os.path.join("{}/train".format(opt.dataset))):
    os.mkdir(os.path.join("{}/train".format(opt.dataset)))
    os.mkdir(os.path.join("{}/train/A".format(opt.dataset)))
    os.mkdir(os.path.join("{}/train/B".format(opt.dataset)))
if not os.path.exists(os.path.join("{}/test".format(opt.dataset))):
    os.mkdir(os.path.join("{}/test".format(opt.dataset)))
    os.mkdir(os.path.join("{}/test/A".format(opt.dataset)))
    os.mkdir(os.path.join("{}/test/B".format(opt.dataset)))

train_set = get_training_set(opt.datasetPath)
train_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=1, shuffle=True)
test_set = get_test_set(opt.datasetPath)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)


criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

i=0
'''
for x in range(10):
    for batch in train_data_loader:
        input, target, input_masked = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(
            batch[2], volatile=True)



        input = input.cpu()
        target = target.cpu()
        in_img = input.data[0]
        target = target.data[0]

        merged_result = torch.cat((in_img,target), 2)

        #save_img(merged_result,"{}/train/{}_{}.jpg".format(opt.dataset, opt.dataset, i))
        save_img(in_img, "{}/train/A/{}_{}.jpg".format(opt.dataset, opt.dataset, i))
        save_img(target, "{}/train/B/{}_{}.jpg".format(opt.dataset, opt.dataset, i))
        i=i+1
'''
i=0
for batch in testing_data_loader:
    input, target, input_masked = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(
        batch[2], volatile=True)



    input = input.cpu()
    target = target.cpu()
    in_img = input.data[0]
    target = target.data[0]

    merged_result = torch.cat((in_img,target), 2)

    #save_img(merged_result,"{}/test/{}_{}.jpg".format(opt.dataset, opt.dataset, i))
    save_img(in_img, "{}/test/A/{}_{}.jpg".format(opt.dataset, opt.dataset, i))
    save_img(target, "{}/test/B/{}_{}.jpg".format(opt.dataset, opt.dataset, i))
    i=i+1

