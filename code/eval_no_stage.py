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
parser.add_argument('--dataset', default='nostep_l1+vgg_unet', help='facades')
parser.add_argument('--model', type=str, default='facades_tvnogan_reselunet', help='model file to use')

parser.add_argument('--model_2nd', type=str, default='nostep_l1+vgg_unet', help='model file to use')
parser.add_argument('--ckpt_2nd', type=str, default='2200', help='model file to use')

parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--datasetPath', default='../dataset/Facade', help='facades')

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--masked_size', type=int, default=64)
parser.add_argument('--resize_ratio', type=int, default=8)

opt = parser.parse_args()
print(opt)

if not os.path.exists("result"):
    os.mkdir("result")
if not os.path.exists(os.path.join("result/{}".format(opt.dataset))):
    os.mkdir(os.path.join("result/{}".format(opt.dataset)))

f= open("result/{}_{}.txt".format(opt.dataset, opt.model),'w')
avg_psnr_1 = 0
sum_psnr_1 = 0
avg_ssim_1 = 0
sum_ssim_1 = 0
avg_psnr_2 = 0
sum_psnr_2 = 0
avg_ssim_2 = 0
sum_ssim_2 = 0
avg_l1 = 0
sum_l1 = 0
avg_l2 = 0
sum_l2 = 0
prediction = 0
count = 0

image_size = 128
masked_size= 64
resize_ratio = 8

test_set = get_test_set(opt.datasetPath, opt.image_size, opt.masked_size, opt.resize_ratio)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

netG_2nd = torch.load("checkpoint/{}/netG_model_epoch_{}.pth".format(opt.model_2nd ,opt.ckpt_2nd),  map_location={'cuda:1':'cuda:0'})
i =0

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

for batch in testing_data_loader:
    input = Variable(batch[0], volatile=True)
    target = Variable(batch[1], volatile=True)
    input_masked = Variable(batch[2], volatile=True)
    target_mosaic  = Variable(batch[3], volatile=True)
    input_2ndmasked = Variable(batch[7], volatile=True)

    count = count + 1

    ############## Stage_1 ##############
    if opt.cuda:
        input_masked = input_masked.cuda()
        target = target.cuda()


    ############## Stage_2 ##############
    if opt.cuda:
        netG_2nd = netG_2nd.cuda()

    out_2nd = netG_2nd(input_masked)
    #out_2nd = netG_2nd(input_2ndmasked)

    startx = (opt.image_size - opt.masked_size) // 2
    starty = (opt.image_size - opt.masked_size) // 2
    target_crop = target[:, :,startx:startx+opt.masked_size, starty:starty+opt.masked_size]
    out_2nd_crop = out_2nd[:, :, startx:startx + opt.masked_size, starty:starty + opt.masked_size]

    out_2nd = out_2nd.cpu().data[0]
    target = target.cpu().data[0]
    input = input.cpu().data[0]
    merged_result_2nd = torch.cat((input, out_2nd, target), 2)

    save_img(merged_result_2nd, "result/{}/{}_{}_{}.jpg".format(opt.dataset, opt.model_2nd, count, opt.dataset))

    p = 0
    l1 = 0
    l2 = 0
    fake = out_2nd_crop.cpu().data.numpy()
    real_center = target_crop.cpu().data.numpy()

    t = real_center - fake
    l2 = np.mean(np.square(t))
    l1 = np.mean(np.abs(t))
    real_center = (real_center + 1) * 127.5
    fake = (fake + 1) * 127.5

    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    for i in range(1):
        p = p + psnr(real_center[i].transpose(1, 2, 0), fake[i].transpose(1, 2, 0))

    #print(l2)
    #print(l1)
    #print(p / 1)

    ssim_none = ssim(out_2nd_crop, target_crop)

    sum_l1 = sum_l1 + l1
    sum_l2 = sum_l2 + l2
    sum_psnr_2 = sum_psnr_2 + p
    sum_ssim_2 = sum_ssim_2 + ssim_none

    outstr = "[stage 2] [%4d]-> " % count
    outstr = outstr + "l1:\t%f\t" % l1
    outstr = outstr + "l2:\t%f\t" % l2
    outstr = outstr + "psnr:\t%f\t" % p
    outstr = outstr + "ssim_img :\t%f\t \n" % ssim_none
    print(outstr)
    f.write(outstr)


outresultstr = "[stage 1] Total-> psnr:\t%f\t" % (sum_psnr_1 / count)
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % (sum_ssim_1 / count)
print(outresultstr)
f.write(outresultstr)

outresultstr = "[stage 2] Total-> psnr:\t%f\t" % (sum_psnr_2/ count)
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % (sum_ssim_2 / count)
outresultstr = outresultstr +  "l1 :\t%f\t \n" % (sum_l1 *100 / count)
outresultstr = outresultstr +  "l2 :\t%f\t \n" % (sum_l2 *100 / count)
print(outresultstr)
f.write(outresultstr)
f.close()