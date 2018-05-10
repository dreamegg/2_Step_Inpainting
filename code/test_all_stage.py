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
parser.add_argument('--dataset', default='facades_tvnogan_reselunet', help='facades')
parser.add_argument('--model', type=str, default='facades_tvnogan_reselunet', help='model file to use')

parser.add_argument('--model_1st', type=str, default='facades_tvnogan_reselunet', help='model file to use')
parser.add_argument('--ckpt_1st', type=str, default='2600', help='model file to use')
parser.add_argument('--model_2nd', type=str, default='l1+vgg_unet', help='model file to use')
parser.add_argument('--ckpt_2nd', type=str, default='1800', help='model file to use')

parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--datasetPath', default='../dataset/Facade', help='facades')

parser.add_argument('--image_size', type=int, default=256)
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

image_size = 256
masked_size= 64
resize_ratio = 8

test_set = get_test_set(opt.datasetPath, opt.image_size, opt.masked_size, opt.resize_ratio)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

netG_1st = torch.load("checkpoint/{}/netG_model_epoch_{}.pth".format(opt.model_1st ,opt.ckpt_1st))
netG_2nd = torch.load("checkpoint/{}/netG_model_epoch_{}.pth".format(opt.model_2nd ,opt.ckpt_2nd),  map_location={'cuda:1':'cuda:0'})
i =0

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()
up_sample = nn.Upsample(size=(image_size, image_size), mode='nearest')
down_sample = nn.AvgPool2d(resize_ratio, stride = resize_ratio)

for batch in testing_data_loader:
    input = Variable(batch[0], volatile=True)
    target = Variable(batch[1], volatile=True)
    input_masked = Variable(batch[2], volatile=True)
    target_mosaic  = Variable(batch[3], volatile=True)
    input_2ndmasked = Variable(batch[7], volatile=True)

    count = count + 1

    ############## Stage_1 ##############
    if opt.cuda:
        netG_1st = netG_1st.cuda()
        input_masked = input_masked.cuda()
        target = target.cuda()
        input_2ndmasked =input_2ndmasked.cuda()

    input_small = down_sample(input)
    target_small = down_sample(target)
    out_1st = netG_1st(down_sample(input_masked))
    out_1st_up = up_sample(out_1st)

    mse = criterionMSE(out_1st, target_small)
    psnr = 10 * math.log10(1 / mse.data[0])
    ssim_none = ssim(out_1st, target_small)

    sum_psnr_1 = sum_psnr_1 + psnr
    sum_ssim_1 = sum_ssim_1 + ssim_none

    outstr = "[stage 1] [%4d]-> " % count
    outstr = outstr + "psnr:\t%f\t" % psnr
    outstr = outstr + "ssim_img :\t%f\t \n" % ssim_none
    print(outstr)
    f.write(outstr)

    input_small = input_small.cpu().data[0]
    out_1st = out_1st.cpu().data[0]
    target_small = target_small.cpu().data[0]
    merged_result_1st = torch.cat((input_small, out_1st, target_small), 1)
    #save_img(merged_result_1st, "result/{}/{}_{}_{}.jpg".format(opt.dataset, opt.model_1st, count, opt.dataset))

    input = input.cpu().data[0]
    out_1st_up = out_1st_up.cpu().data[0]
    input_masked = input_masked.cpu().data[0]
    mask = input_masked[3,:, :]
    mask = torch.unsqueeze(mask, 0)
    input_2stage_img = (1-mask) * out_1st_up + (mask) * input

    input_2stage = torch.cat((input_2stage_img,mask), 0)
    input_2stage = torch.unsqueeze(input_2stage, 0)
    input_2stage = Variable(input_2stage).cuda()


    ############## Stage_2 ##############
    if opt.cuda:
        netG_2nd = netG_2nd.cuda()

    out_2nd = netG_2nd(input_2stage)
    #out_2nd = netG_2nd(input_2ndmasked)

    startx = (opt.image_size - opt.masked_size) // 2
    starty = (opt.image_size - opt.masked_size) // 2
    target_crop = target[:, :,startx:startx+opt.masked_size, starty:starty+opt.masked_size]
    out_2nd_crop = out_2nd[:, :, startx:startx + opt.masked_size, starty:starty + opt.masked_size]

    mse = criterionMSE(out_2nd_crop, target_crop)
    psnr = 10 * math.log10(1 / mse.data[0])
    ssim_none = ssim(out_2nd_crop, target_crop)

    sum_psnr_2 = sum_psnr_2 + psnr
    sum_ssim_2 = sum_ssim_2 + ssim_none

    outstr = "[stage 2] [%4d]-> " % count
    outstr = outstr + "psnr:\t%f\t" % psnr
    outstr = outstr + "ssim_img :\t%f\t \n" % ssim_none
    print(outstr)
    f.write(outstr)

    out_2nd = out_2nd.cpu().data[0]
    target = target.cpu().data[0]
    merged_result_2nd = torch.cat((input, input_2stage_img, out_2nd, target), 2)
    save_img(merged_result_2nd, "result/{}/{}_{}_{}.jpg".format(opt.dataset, opt.model_2nd, count, opt.dataset))

outresultstr = "[stage 1] Total-> psnr:\t%f\t" % (sum_psnr_1 / count)
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % (sum_ssim_1 / count)
print(outresultstr)
f.write(outresultstr)

outresultstr = "[stage 2] Total-> psnr:\t%f\t" % (sum_psnr_2/ count)
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % (sum_ssim_2 / count)
print(outresultstr)
f.write(outresultstr)
f.close()