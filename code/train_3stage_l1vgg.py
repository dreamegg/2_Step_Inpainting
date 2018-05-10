from __future__ import print_function
import argparse
import os
from math import log10

import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms


from nets_with_vgg import define_G, define_D, GANLoss, print_network
from Preprocess_data import get_training_set, get_test_set
from loss import GeneratorLoss

from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='2 Stage RestoNet-PyTorch-implementation')
parser.add_argument('--dataset', default='l1+vgg' , help='facades')
#parser.add_argument('--datasetPath', default='../../../DataSets/facade' , help='facades')
parser.add_argument('--datasetPath', default='../dataset/Facade' , help='facades')
parser.add_argument('--resume_epoch', default=3000 , help='facades')
parser.add_argument('--nEpochs', type=int, default=30*1000, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=16*2, help='training batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')

parser.add_argument('--model_1st', type=str, default='facades_tvnogan.notan_dilated', help='model file to use')
parser.add_argument('--ckpt_1st', type=str, default='3400', help='model file to use')
parser.add_argument('--model_2nd', type=str, default='l1+vgg_unet', help='model file to use')
parser.add_argument('--ckpt_2nd', type=str, default='3000', help='model file to use')
parser.add_argument('--G_model', type=str, default="unet")
parser.add_argument('--D_model', type=str, default="dual")

parser.add_argument('--input_nc', type=int, default=4, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--l1lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--vgglamb', type=int, default=1, help='weight on vgg term in objective')
parser.add_argument('--wlamb', type=int, default=0.1, help='weight on L1 term in objective')

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--masked_size', type=int, default=64)
parser.add_argument('--resize_ratio', type=int, default=8)

parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)



if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.datasetPath, opt.image_size, opt.masked_size, opt.resize_ratio)
test_set = get_test_set(opt.datasetPath, opt.image_size, opt.masked_size, opt.resize_ratio)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')

resume_epoch = opt.resume_epoch

netG_1st = torch.load("checkpoint/{}/netG_model_epoch_{}.pth".format(opt.model_1st, opt.ckpt_1st))
netG_2nd = torch.load("checkpoint/{}/netG_model_epoch_{}.pth".format(opt.model_2nd, opt.ckpt_2nd))
netD_2nd = torch.load("checkpoint/{}/netD_model_epoch_{}.pth".format(opt.model_2nd, opt.ckpt_2nd))

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
BCE_loss = nn.BCELoss()
generator_criterion = GeneratorLoss()
up_sample = nn.Upsample(size=(opt.image_size, opt.image_size), mode='nearest')
down_sample = nn.AvgPool2d(opt.resize_ratio, stride = opt.resize_ratio)

# setup optimizer
optimizerG = optim.Adam(netG_2nd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD_2nd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
print_network(netG_2nd)
print_network(netD_2nd)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
real_masked_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
origin_a = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
    netD_2nd = netD_2nd.cuda()
    netG_2nd = netG_2nd.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    generator_criterion = generator_criterion.cuda()
    BCE_loss = BCE_loss.cuda()
    real_a = real_a.cuda()
    real_masked_a = real_masked_a.cuda()
    real_b = real_b.cuda()
    origin_a = origin_a.cuda()

real_a = Variable(real_a)
real_masked_a = Variable(real_masked_a)
real_b = Variable(real_b)
origin_a = Variable(origin_a)


def train(epoch):
    for iteration, batch in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader)):
        # forward
        input_masked_cpu , target_cpu , input_cpu= batch[2], batch[1] , batch[0]

        #input = Variable(batch[0], volatile=True)
        #target = Variable(batch[1], volatile=True)
        #input_masked = Variable(batch[2], volatile=True)
        #target_mosaic = Variable(batch[3], volatile=True)
        #input_2ndmasked = Variable(batch[7], volatile=True)

        real_b.data.resize_(target_cpu.size()).copy_(target_cpu)
        real_a.data.resize_(input_cpu.size()).copy_(input_cpu)
        real_masked_a.data.resize_(input_masked_cpu.size()).copy_(input_masked_cpu)

        out_1st = netG_1st(down_sample(real_masked_a))
        out_1st_up = up_sample(out_1st)
        mask = real_masked_a[:, 3, :, :]
        mask = torch.unsqueeze(mask, 1)
        input_2stage_img = (1 - mask) * out_1st_up + (mask) * real_a

        input_2stage = torch.cat((input_2stage_img, mask), 1)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()

        pred_real = netD_2nd(real_b)
        D_real_loss = -torch.mean(pred_real)

        (batch_size,_) = pred_real.data.size()
        y_real_ = Variable(torch.ones(batch_size, 1).cuda())
        loss_d_real = BCE_loss(pred_real, y_real_)

        fake_b = netG_2nd(input_2stage)
        pred_fake = netD_2nd(fake_b)
        D_fake_loss = torch.mean(pred_fake)

        y_fake_ = Variable(torch.zeros(batch_size, 1).cuda())
        loss_d_fake = BCE_loss(pred_fake, y_fake_)

        alpha = torch.rand(real_b.size()).cuda()
        x_hat = Variable(alpha * real_b.data + (1 - alpha) * fake_b.data, requires_grad=True)
        pred_hat = netD_2nd(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = opt.wlamb * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        # Combined loss
        loss_d = (D_real_loss + D_fake_loss + gradient_penalty)

        loss_d.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()

        out_1st = netG_1st(down_sample(real_masked_a))
        out_1st_up = up_sample(out_1st)
        mask = real_masked_a[:, 3, :, :]
        mask = torch.unsqueeze(mask, 1)
        input_2stage_img = (1 - mask) * out_1st_up + (mask) * real_a

        input_2stage = torch.cat((input_2stage_img, mask), 1)

        fake_b = netG_2nd(input_2stage)
        pred_fake = netD_2nd(fake_b)

        loss_g_gan = -torch.mean(pred_fake)
        loss_g_vgg = generator_criterion(fake_b, real_b) * opt.vgglamb
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.l1lamb
        
        loss_g = loss_g_gan + loss_g_vgg + loss_g_l1
        
        loss_g.backward()

        optimizerG.step()

        #print("===> Epoch[{}]({}/{}): Loss_G: {:.4f}".format(
        #    epoch, iteration, len(training_data_loader), loss_g.data[0]))

        g_step = len(training_data_loader)*epoch + iteration

        writer.add_scalar('dataD/d_loss', loss_d.data[0], g_step)
        writer.add_scalar('dataD/D_real_loss', D_real_loss.data[0], g_step)
        writer.add_scalar('dataD/D_fake_loss', D_fake_loss.data[0], g_step)
        writer.add_scalar('dataD/gradient_penalty', gradient_penalty.data[0], g_step)

        if iteration == 1:
            writer.add_scalar('dataG/loss_g', loss_g.data[0], g_step)
            writer.add_scalar('dataG/loss_g_gan', loss_g_gan.data[0], g_step)
            writer.add_scalar('dataG/loss_g_vgg', loss_g_vgg.data[0], g_step)
            writer.add_scalar('dataG/loss_g_l1', loss_g_l1.data[0], g_step)

            writer.add_image('Train/input', ToPilImage(real_masked_a[0]), epoch)
            writer.add_image('Train/prediction_1st', ToPilImage(input_2stage[0]), epoch)
            writer.add_image('Train/prediction_2nd', ToPilImage(fake_b[0]), epoch)
            writer.add_image('Train/target', ToPilImage(real_b[0]), epoch)

def ToPilImage(image_tensor):
    #trans_norm = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    image_norm = (image_tensor.data + 1 )/2
    return  image_norm

def test(epoch):
    sum_psnr = 0
    prediction =0

    for batch in testing_data_loader:
        input = Variable(batch[0], volatile=True)
        target = Variable(batch[1], volatile=True)
        input_masked = Variable(batch[2], volatile=True)
        target_mosaic = Variable(batch[3], volatile=True)
        input_2ndmasked = Variable(batch[7], volatile=True)

        if opt.cuda:
            input = input.cuda()
            input_masked = input_masked.cuda()
            target = target.cuda()
            input_2ndmasked = input_2ndmasked.cuda()

        out_1st = netG_1st(down_sample(input_masked))

        out_1st_up = up_sample(out_1st)
        mask = input_masked[:, 3, :, :]
        mask = torch.unsqueeze(mask, 1)
        input_2stage_img = (1 - mask) * out_1st_up + (mask) * input

        input_2stage = torch.cat((input_2stage_img, mask), 1)

        out_2nd = netG_2nd(input_2stage)

        mse = criterionMSE(out_2nd, target)
        psnr = 10 * log10(1 / mse.data[0])
        sum_psnr += psnr

    avg_psnr = sum_psnr / len(testing_data_loader)
    print("[{}]===> Avg. PSNR: {:.4f} dB".format(epoch, avg_psnr))
    writer.add_image('Test/input', ToPilImage(input[0]), epoch)
    writer.add_image('Test/prediction_1st', ToPilImage(input_2stage[0]), epoch)
    writer.add_image('Test/prediction_2nd', ToPilImage(out_2nd[0]), epoch)
    writer.add_image('Test/target', ToPilImage(target[0]), epoch)
    writer.add_scalar('PSNR/PSNR', avg_psnr, epoch)


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint/{}_{}".format(opt.dataset,opt.G_model))) :
        os.mkdir(os.path.join("checkpoint/{}_{}".format(opt.dataset,opt.G_model)))
    net_g_model_out_path = "checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.dataset,opt.G_model, epoch)
    net_d_model_out_path = "checkpoint/{}_{}/netD_model_epoch_{}.pth".format(opt.dataset,opt.G_model, epoch)
    torch.save(netG_2nd, net_g_model_out_path)
    torch.save(netD_2nd, net_d_model_out_path)

    net_g_model_out_path = "checkpoint/{}_{}/netG_model_latest.pth".format(opt.dataset,opt.G_model)
    net_d_model_out_path = "checkpoint/{}_{}/netD_model_latest.pth".format(opt.dataset,opt.G_model)
    torch.save(netG_2nd, net_g_model_out_path)
    torch.save(netD_2nd, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs_stage3', opt.dataset+"_"+str(opt.resume_epoch)+"_"+current_time)
writer = SummaryWriter(log_dir)

for epoch in range(resume_epoch, opt.nEpochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 100 == 0:
        checkpoint(epoch)

writer.close()
