from __future__ import print_function
import argparse
import os
from math import log10

import numpy as np
from PIL import Image
import time
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
parser.add_argument('--dataset', default='facades_tvnogan_reselunet' , help='facades')
#parser.add_argument('--datasetPath', default='../../../DataSets/facade' , help='facades')
parser.add_argument('--datasetPath', default='../dataset/Facade' , help='facades')
parser.add_argument('--resume_epoch', default=0 , help='facades')
parser.add_argument('--nEpochs', type=int, default=5*1000, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=16*8, help='training batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')

#parser.add_argument('--G_model', type=str, default="dilated")
#parser.add_argument('--G_model', type=str, default="unet5")
#parser.add_argument('--G_model', type=str, default="reselunet")
parser.add_argument('--G_model', type=str, default="resnet")
#parser.add_argument('--G_model', type=str, default="densenet")
parser.add_argument('--D_model', type=str, default=None)

parser.add_argument('--input_nc', type=int, default=4, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--wlamb', type=int, default=0.25, help='weight on L1 term in objective')

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--masked_size', type=int, default=64)
parser.add_argument('--resize_ratio', type=int, default=8)

parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.datasetPath, opt.image_size, opt.masked_size, opt.resize_ratio)
test_set = get_test_set(opt.datasetPath, opt.image_size, opt.masked_size, opt.resize_ratio)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

print('===> Building model')

resume_epoch = 0
if opt.resume_epoch < 0  :
    net_g_model_out_path = "checkpoint/{}_{}/netG_model_latest.pth".format(opt.dataset,opt.G_model)
    if (opt.D_model != None): net_d_model_out_path = "checkpoint/{}_{}/netD_model_latest.pth".format(opt.dataset,opt.G_model)
    netG = torch.load(net_g_model_out_path)
    if (opt.D_model != None): netD = torch.load(net_d_model_out_path)

elif opt.resume_epoch > 0 :
    resume_epoch = opt.resume_epoch
    net_g_model_out_path = "checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.dataset,opt.G_model, resume_epoch)
    if (opt.D_model != None): net_d_model_out_path = "checkpoint/{}_{}/netD_model_epoch_{}.pth".format(opt.dataset,opt.G_model, resume_epoch)
    netG = torch.load(net_g_model_out_path)
    if (opt.D_model != None): netD = torch.load(net_d_model_out_path)
else :
    netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, [0], opt.G_model)
    if (opt.D_model != None) :
        netD = define_D(opt.output_nc, opt.ndf, 'batch', False, [0], opt.D_model)

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
BCE_loss = nn.BCELoss()
#generator_criterion = GeneratorLoss()
tv_loss = TVLoss()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
if (opt.D_model != None):  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
print_network(netG)
if (opt.D_model != None): print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
real_masked_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
    if (opt.D_model != None): netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    #generator_criterion = generator_criterion.cuda()
    BCE_loss = BCE_loss.cuda()
    tv_loss = TVLoss().cuda()

    real_a = real_a.cuda()
    real_masked_a = real_masked_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_masked_a = Variable(real_masked_a)
real_b = Variable(real_b)

#y_real_= Variable(torch.ones(opt.batchSize, 1).cuda())
#y_fake_ = Variable(torch.zeros(opt.batchSize, 1).cuda())


def train(epoch):
    for iteration, batch in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader)):
        # forward
        input_cpu, target_cpu, input_masked_cpu = batch[4], batch[5] , batch[6]
        real_b.data.resize_(target_cpu.size()).copy_(target_cpu)
        real_a.data.resize_(input_cpu.size()).copy_(input_cpu)
        real_masked_a.data.resize_(input_masked_cpu.size()).copy_(input_masked_cpu)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        if (opt.D_model != None):
            optimizerD.zero_grad()

            pred_real = netD(real_b)
            D_real_loss = -torch.mean(pred_real)

            (batch_size,_) = pred_real.data.size()
            y_real_ = Variable(torch.ones(batch_size, 1).cuda())
            loss_d_real = BCE_loss(pred_real, y_real_)

            fake_b = netG(real_masked_a)
            pred_fake = netD(fake_b)
            D_fake_loss = torch.mean(pred_fake)

            y_fake_ = Variable(torch.zeros(batch_size, 1).cuda())
            loss_d_fake = BCE_loss(pred_fake, y_fake_)

            alpha = torch.rand(real_b.size()).cuda()
            x_hat = Variable(alpha * real_b.data + (1 - alpha) * fake_b.data, requires_grad=True)
            pred_hat = netD(x_hat)
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

        fake_b = netG(real_masked_a)
        if (opt.D_model != None): pred_fake = netD(fake_b)
        if (opt.D_model != None): loss_g_gan = -torch.mean(pred_fake)

        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        tv_loss_g  = tv_loss(fake_b)

        loss_g = loss_g_l1 + tv_loss_g
        if (opt.D_model != None): loss_g +=  loss_g_gan

        loss_g.backward()

        optimizerG.step()

        #print("===> Epoch[{}]({}/{}): Loss_G: {:.4f}".format(
        #    epoch, iteration, len(training_data_loader), loss_g.data[0]))

        g_step = len(training_data_loader)*epoch + iteration

        if (opt.D_model != None):
            writer.add_scalar('dataD/d_loss', loss_d.data[0], g_step)
            writer.add_scalar('dataD/D_real_loss', D_real_loss.data[0], g_step)
            writer.add_scalar('dataD/D_fake_loss', D_fake_loss.data[0], g_step)
            writer.add_scalar('dataD/gradient_penalty', gradient_penalty.data[0], g_step)

        writer.add_scalar('dataG/loss_g', loss_g.data[0], g_step)
        if (opt.D_model != None): writer.add_scalar('dataG/loss_g_gan', loss_g_gan.data[0], g_step)
        writer.add_scalar('dataG/loss_g_l1', loss_g_l1.data[0], g_step)
        writer.add_scalar('dataG/tv_loss', tv_loss_g.data[0], g_step)

def ToPilImage(image_tensor):
    #trans_norm = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    image_norm = (image_tensor.data + 1 )/2
    return  image_norm

def test(epoch):
    sum_psnr = 0
    prediction =0
    for batch in testing_data_loader:
        input, target, input_masked = Variable(batch[4], volatile=True), Variable(batch[5], volatile=True), Variable(batch[6], volatile=True)
        if opt.cuda:
            input_masked = input_masked.cuda()
            target = target.cuda()

        prediction = netG(input_masked)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        sum_psnr += psnr

    avg_psnr = sum_psnr / len(testing_data_loader)
    print("[{}]===> Avg. PSNR: {:.4f} dB".format(epoch, avg_psnr))
    writer.add_image('Test/input', ToPilImage(input), epoch)
    writer.add_image('Test/prediction', ToPilImage(prediction), epoch)
    writer.add_image('Test/target', ToPilImage(target), epoch)
    writer.add_scalar('PSNR/PSNR', avg_psnr, epoch)


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint/{}_{}".format(opt.dataset,opt.G_model))) :
        os.mkdir(os.path.join("checkpoint/{}_{}".format(opt.dataset,opt.G_model)))

    net_g_model_out_path = "checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.dataset,opt.G_model, epoch)
    torch.save(netG, net_g_model_out_path)
    net_g_model_out_path = "checkpoint/{}_{}/netG_model_latest.pth".format(opt.dataset, opt.G_model)
    torch.save(netG, net_g_model_out_path)

    if (opt.D_model != None):
        net_d_model_out_path = "checkpoint/{}_{}/netD_model_epoch_{}.pth".format(opt.dataset, opt.G_model, epoch)
        torch.save(netD, net_d_model_out_path)
        net_d_model_out_path = "checkpoint/{}_{}/netD_model_latest.pth".format(opt.dataset, opt.G_model)
        torch.save(netD, net_d_model_out_path)


    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs_stage1_tv', opt.dataset+"_"+opt.G_model+"_"+str(opt.resume_epoch)+"_"+current_time)
writer = SummaryWriter(log_dir)

for epoch in range(resume_epoch, opt.nEpochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 200 == 0:
        checkpoint(epoch)

writer.close()
