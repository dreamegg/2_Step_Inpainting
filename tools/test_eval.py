from __future__ import print_function
import argparse
import os

import torch
from ssim_torch import ssim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math

from Preprocess_data import is_image_file, load_img, save_img, get_test_set

# Testing settings
parser = argparse.ArgumentParser(description='RestoNet-PyTorch-implementation')
parser.add_argument('--dataset', default='facades_l1', help='facades')
parser.add_argument('--model', type=str, default='3000', help='model file to use')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--datasetPath', default='../dataset/Facade', help='facades')
opt = parser.parse_args()
print(opt)

if not os.path.exists("result"):
    os.mkdir("result")
if not os.path.exists(os.path.join("result/{}".format(opt.dataset, opt.model))):
    os.mkdir(os.path.join("result/{}".format(opt.dataset, opt.model)))

f= open("result/{}_{}.txt".format(opt.dataset, opt.model),'w')
avg_psnr = 0
sum_psnr = 0
avg_ssim = 0
sum_ssim = 0
prediction = 0
count = 0

test_set = get_test_set(opt.datasetPath)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

torch.cuda.set_device(1)
netG = torch.load("checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset ,opt.model))
i =0

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

for batch in testing_data_loader:
    input, target, input_masked = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(
        batch[2], volatile=True)

    if opt.cuda:
        netG = netG.cuda()
        input_masked = input_masked.cuda()
        target = target.cuda()

    out = netG(input_masked)
    count = count + 1

    mse = criterionMSE(out, target)
    psnr = 10 * math.log10(1 / mse.data[0])
    ssim_none = ssim(out, target)

    sum_psnr = sum_psnr + psnr
    sum_ssim = sum_ssim + ssim_none

    outstr = "[%4d]-> " % count
    outstr = outstr + "psnr:\t%f\t" % psnr
    outstr = outstr + "ssim_img :\t%f\t \n" % ssim_none
    print(outstr)
    f.write(outstr)


    out = out.cpu()
    out_img = out.data[0]

    def ToPilImage(image_tensor):
        # trans_norm = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        image_norm = (image_tensor.data + 1) / 2
        return image_norm

    input = input.cpu()
    target = target.cpu()
    in_img = input.data[0]
    target = target.data[0]

    merged_result = torch.cat((in_img,out_img,target), 1)

    #save_img(in_img, "result/{}/Input_{}.jpg".format(opt.dataset, opt.model, i))
    #save_img(out_img, "result/{}/Pred_{}.jpg".format(opt.dataset, opt.model, i))
    #save_img(target, "result/{}/Target_{}.jpg".format(opt.dataset, opt.model, i))
    save_img(merged_result, "result/{}/{}_{}_{}.jpg".format(opt.dataset, opt.model, i, opt.dataset))
    i=i+1

mean_psnr = sum_psnr/count
mean_ssim = sum_ssim / count
outresultstr = "Total-> psnr:\t%f\t" % mean_psnr
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % mean_ssim

print(outresultstr)
f.write(outresultstr)
f.close()

'''
netG = torch.load(opt.model)

image_dir = "dataset/{}/test/b/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)

    mask = torch.zeros(64 * 4, 64 * 4)
    mask_1 = torch.ones(64, 64)
    startx = 256 // 2 - (64 // 2)
    starty = 256 // 2 - (64 // 2)
    mask[startx:startx + 64, starty:starty + 64] = mask_1
    mask = torch.unsqueeze(mask, 0)

    img_masked = torch.cat((img, mask), 0)


    input = Variable(img_masked, volatile=True).view(1, -1, 256, 256)


    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name)

from __future__ import print_function
import argparse
import os

import torch
from ssim_torch import ssim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math

from Preprocess_data import is_image_file, load_img, save_img, get_test_set

# Testing settings
parser = argparse.ArgumentParser(description='RestoNet-PyTorch-implementation')
parser.add_argument('--dataset', default='facades_tvloss', help='facades')
parser.add_argument('--model', type=str, default='1700', help='model file to use')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--datasetPath', default='../dataset/Facade', help='facades')
opt = parser.parse_args()
print(opt)

if not os.path.exists("result"):
    os.mkdir("result")
if not os.path.exists(os.path.join("result/{}_{}".format(opt.model, opt.dataset))):
    os.mkdir(os.path.join("result/{}_{}".format(opt.model, opt.dataset)))

f= open("result/result_{}.txt".format(opt.model),'w')
avg_psnr = 0
sum_psnr = 0
avg_ssim = 0
sum_ssim = 0
prediction = 0
count = 0

test_set = get_test_set(opt.datasetPath)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

netG = torch.load("checkpoint/facades/netG_model_epoch_{}.pth".format(opt.model))
i =0

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

for batch in testing_data_loader:
    input, target, input_masked = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(
        batch[2], volatile=True)

    if opt.cuda:
        netG = netG.cuda()
        input_masked = input_masked.cuda()
        target = target.cuda()

    out = netG(input_masked)
    count = count + 1

    mse = criterionMSE(out, target)
    psnr = 10 * math.log10(1 / mse.data[0])
    ssim_none = ssim(out, target)

    sum_psnr = sum_psnr + psnr
    sum_ssim = sum_ssim + ssim_none

    outstr = "[%4d]-> " % count
    outstr = outstr + "psnr:\t%f\t" % psnr
    outstr = outstr + "ssim_img :\t%f\t \n" % ssim_none
    print(outstr)
    f.write(outstr)


    out = out.cpu()
    out_img = out.data[0]

    def ToPilImage(image_tensor):
        # trans_norm = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        image_norm = (image_tensor.data + 1) / 2
        return image_norm

    input = input.cpu()
    target = target.cpu()
    in_img = input.data[0]
    target = target.data[0]

    save_img(in_img, "result/{}_{}/Input_{}.jpg".format(opt.model, opt.dataset, i))
    save_img(out_img, "result/{}_{}/Pred_{}.jpg".format(opt.model, opt.dataset, i))
    save_img(target, "result/{}_{}/Target_{}.jpg".format(opt.model, opt.dataset, i))
    i=i+1

mean_psnr = sum_psnr/count
mean_ssim =I sum_ssim / count
outresultstr = "Total-> psnr:\t%f\t" % mean_psnr
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % mean_ssim

print(outresultstr)
f.write(outresultstr)
f.close()'''
