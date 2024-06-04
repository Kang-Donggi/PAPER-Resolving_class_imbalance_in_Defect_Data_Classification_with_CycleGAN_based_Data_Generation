import wandb
wandb.login()
wandb.init(project='0526_cycle-gan-2000')

# In[1]:


import torch.autograd as autograd

import glob
import random
import os
import glob

import argparse
import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from datasets import UnpairedDataset
# from models import Generator, Discriminator
# from utils import init_weight, ImagePool, LossDisplayer

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch

import os
import numpy as np
import math
import itertools
import datetime
import time

from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torch.autograd import Variable

from tqdm import tqdm 

import easydict

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# os.environ['CUDA_VISIBLE_DEVICES']='1,2,3' # 1, 2번 gpu 사용
device = torch.device('cuda')


# In[2]:


def compute_r1_penalty(discriminator, real_images):
    real_images.requires_grad = True
    scores = discriminator(real_images)
    gradients = autograd.grad(outputs=scores, inputs=real_images,
                              grad_outputs=torch.ones_like(scores),
                              create_graph=True, retain_graph=True)[0]
    gradient_penalty = torch.sum(gradients.pow(2))
    return gradient_penalty


# In[3]:


# unpaired한 데이터셋 구성을 위해 두 스타일의 데이터를 랜덤으로 구성
class UnpairedDataset(DataLoader):
    def __init__(self, dataset_dir, styles, transforms):
        self.dataset_dir = dataset_dir
        self.styles = styles
        self.image_path_A = glob.glob(os.path.join(dataset_dir, styles[0]) + "/*")
        self.image_path_B = glob.glob(os.path.join(dataset_dir, styles[1]) + "/*")
        self.transform = transforms

    def __getitem__(self, index_A):
        index_B = random.randint(0, len(self.image_path_B) - 1)

        item_A = self.transform(Image.open(self.image_path_A[index_A]))
        item_B = self.transform(Image.open(self.image_path_B[index_B]))

        return [item_A, item_B]

    def __len__(self):
        return len(self.image_path_A)


# In[4]:


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# In[5]:



class Generator(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        model = []

        # 1, c7s1-64 # convolution-instancenorm-ReLU layer를 리스트로 추가
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
        ]

        # 2, dk # 두개의 dk layer로 크기를 줄임
        model += [self.__conv_block(64, 128), self.__conv_block(128, 256)]

        # 3, Rk # 여러개의 residualblock추가 블록수는 생성자의 매개변수로 지정
        model += [ResidualBlock()] * num_blocks

        # 4, uk # fractional-strided-convolution을 사용
                # 크기를 늘리는 점을 명시하기 위해 upsample인자로 전달
        model += [
            self.__conv_block(256, 128, upsample=True),
            self.__conv_block(128, 64, upsample=True),
        ]

        # 5, c7s1-3 제일 마지막 layer에서 than함수 사용 이미지 값 -1에서
                    # 1으로 유지하기 위함
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]

        # 6, 리스트에서 nn.Sequential으로 순서대로 통과하게 만들어 줌
             # 리스트에 *을 사용해서 리스트가 아닌 각 값이 입력되게 함
        self.model = nn.Sequential(*model)

    # 7, nn.sequential로 만든 모델 통과
    def forward(self, x):
        return self.model(x)
    
    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            # 8 feature map의 크기를 늘릴 때 nn.Transpose2d를 사용
                # 크기를 줄일때와 다르게 padding뿐 아니라
                # output_padding도 적용해야 정확하게 2배로 늘려짐
            conv = nn.ConvTranspose2d(
                in_features, out_features, 3, 2, 1, output_padding=1
            )
        else:
            conv = nn.Conv2d(in_features, out_features, 3, 2, 1)

        # 9, nn.sequential로 정의한 conv와 함께 반환해 줌
        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            # nn.ReLU(),
        )


# In[6]:


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 10, ck를 __conv_layer 메서드로 구현 장규화 사용하지 않기에 norm=False
              # stride=1
        self.model = nn.Sequential(
            self.__conv_layer(3, 64, norm=False),
            self.__conv_layer(64, 128),
            self.__conv_layer(128, 256),
            self.__conv_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    # 11, 매개변수로 받은 입축력 feature map수, stride, 정규화 유무에 따라
        # 모델 구조를 다르게 리스트로 모듈을 넣은 후 nn.sequential 하는 방식
    def __conv_layer(self, in_features, out_features, stride=2, norm=True):
        layer = [nn.Conv2d(in_features, out_features, 4, stride, 1)]

        if norm:
            layer.append(nn.InstanceNorm2d(out_features))

        layer.append(nn.LeakyReLU(0.2))

        layer = nn.Sequential(*layer)

        return layer

    def forward(self, x):
        return self.model(x)


# In[7]:


def init_weight(module):
    class_name = module.__class__.__name__

    if class_name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant(module.bias.data, 0.0)

if __name__ == "__main__":
    netG_A2B = Generator(6).to(device)
    netG_A2B.apply(init_weight)


# In[8]:


class ImagePool():
    

    def __init__(self, pool_size):
   
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
 
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


# In[9]:


def compute_r1_penalty(discriminator, real_images):
    real_images.requires_grad_(True)
    scores_real = discriminator(real_images)
    grads_real = torch.autograd.grad(
        outputs=scores_real.sum(), inputs=real_images, create_graph=True
    )[0]
    r1_penalty = torch.sum(grads_real ** 2)
    return r1_penalty


# In[10]:


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=2000)
# parser.add_argument("--batch_size", type=int, default=9)
parser.add_argument("--dataset_path", type=str, default="/home/kangdg22/meta_Assignment/paper/data")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--lambda_ide", type=float, default=10)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--pool_size", type=int, default=50)
parser.add_argument("--identity", action="store_true")
parser.add_argument("--lambda_r1", type=float, default=0.5)

args = parser.parse_args('')




# parser=easydict.EasyDict({
#     'epoch': 500,
#     'batch_size': 1,
#     'dataset_path':'meta_Assignment/paper/data',
#     'checkpoint_path':'meta_Assignment/paper/checkpoint', 
#     'size': int(256),
#     'lambda_ide': 10,
#     'lr': 2e-4,
#     'pool_size': 50,
#     'identity': 'store_true'
# })

# args=parser


def train():
    # os.environ['CUDA_VISIBLE_DEVICES']='2'
    device = torch.device('cuda')
    print(device)
    criterion_R1 = nn.MSELoss()
    # Model
    num_blocks = 6 if args.size <= 256 else 8
    _netG_A2B = Generator(num_blocks).to(device)
    _netG_B2A = Generator(num_blocks).to(device)
    _netD_A = Discriminator().to(device)
    _netD_B = Discriminator().to(device)
    # multi gpu사용
    netG_A2B = nn.DataParallel(_netG_A2B)
    netG_B2A = nn.DataParallel(_netG_B2A)
    netD_A = nn.DataParallel(_netD_A)
    netD_B = nn.DataParallel(_netD_B)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"])
        netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"])
        netD_A.load_state_dict(checkpoint["netD_A_state_dict"])
        netD_B.load_state_dict(checkpoint["netD_B_state_dict"])
        epoch = checkpoint["epoch"]
    else: # 가중치 초기화
        netG_A2B.apply(init_weight)
        netG_B2A.apply(init_weight)
        netD_A.apply(init_weight)
        netD_B.apply(init_weight)
        epoch = 0

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataloader = DataLoader(
        UnpairedDataset(args.dataset_path, ["trainA", "trainB"], transform), batch_size=44
    )
    dataset_name = os.path.basename(args.dataset_path)

    pool_fake_A = ImagePool(args.pool_size)
    pool_fake_B = ImagePool(args.pool_size)

    # Loss
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_GAN = nn.MSELoss()

    # disp = LossDisplayer(["G_GAN", "G_recon", "D"])
    # summary = SummaryWriter()

    # Optimizer, Schedular
    optim_G = optim.Adam(
        list(netG_A2B.parameters()) + list(netG_B2A.parameters()),
        lr=0.0003,
        betas=(0.5, 0.999),
    )
    optim_D_A = optim.Adam(netD_A.parameters(), lr=0.0005)
    optim_D_B = optim.Adam(netD_B.parameters(), lr=0.0005)
    
    # 스케쥴러
    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lr_lambda)
    scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer=optim_D_A, lr_lambda=lr_lambda
    )
    scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer=optim_D_B, lr_lambda=lr_lambda
    )

    os.makedirs(f"checkpoint/{dataset_name}", exist_ok=True)

    # Training
    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")

        for idx, (real_A, real_B) in enumerate(dataloader):
            print(f"{idx}/{len(dataloader)}", end="\r")
            # 1, 진짜 두 스타일의 이미지에서 가짜 이미지, cycle 이미지를 생성
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Forward model
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)
            cycle_A = netG_B2A(fake_B)
            cycle_B = netG_A2B(fake_A)
            # 2, 생성된 가짜이미지를 판별자를 통과해 점수를 계산
            pred_fake_A = netD_A(fake_A)
            pred_fake_B = netD_B(fake_B)


             # R1 패널티 계산
            r1_penalty_A = compute_r1_penalty(netD_A, real_A)
            r1_penalty_B = compute_r1_penalty(netD_B, real_B)

            # 3, cycle과 GAN 손실을 계산. 여기서 생성자가 학습할 때에는
                # 판별자의 점수가 1에 가깝게 학습되어야 하는 점에서 
                # ones_like 함수로 타겟 벡터를 생성
            loss_cycle_A = criterion_cycle(cycle_A, real_A)
            loss_cycle_B = criterion_cycle(cycle_B, real_B)
            loss_GAN_A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
            loss_GAN_B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            # lambda로 가중치를 조정해 생성자 손실을 계산
            loss_G = (
                args.lambda_ide * (loss_cycle_A + loss_cycle_B)
                + loss_GAN_A
                + loss_GAN_B
                + args.lambda_r1 * (r1_penalty_A + r1_penalty_B)  # R1 패널티 추가
            )
            
            # identity loss를 사용할 때 추가적으로 동작을 수행해 손실을 추가함
            if args.identity:
                identity_A = netG_B2A(real_A)
                identity_B = netG_A2B(real_B)
                loss_identity_A = criterion_identity(identity_A, real_A)
                loss_identity_B = criterion_identity(identity_B, real_B)
                loss_G += 0.5 * args.lambda_ide * (loss_identity_A + loss_identity_B)
            # optimizer로 역전파 알고리즘을 수행함
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # discriminator 학습
            # D_A학습
            # 진짜, 가짜 이미지 discriminator에 통과 
        for _ in range(50):
            pred_real_A = netD_A(real_A)
            pred_fake_A = netD_A(pool_fake_A.query(fake_A))
            # gan loss 계산 진짜 이미지에 대해서는 1,
            # 가짜 이미지에 대해서는 0으로 추정하게 학습 
            loss_D_A = 0.5 * (
                criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
                + criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            )

            optim_D_A.zero_grad()
            loss_D_A.backward()
            optim_D_A.step()

            # D_B학습 D_A와 동일
            pred_real_B = netD_B(real_B)
            pred_fake_B = netD_B(pool_fake_B.query(fake_B))

            loss_D_B = 0.5 * (
                criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
                + criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
            )

            optim_D_B.zero_grad()
            loss_D_B.backward()
            optim_D_B.step()

            # Record loss
            loss_G_GAN = loss_GAN_A + loss_GAN_B
            loss_G_recon = loss_G - loss_G_GAN
            loss_D = loss_D_A + loss_D_B
            # disp.record([loss_G_GAN, loss_G_recon, loss_D])
            wandb.log({
                'loss_G_GAN': loss_G_GAN.item(),
                'loss_G_recon': loss_G_recon.item(),
                'loss_D': loss_D.item()
            })

        # Step scheduler
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Record and display loss
        # avg_losses = disp.get_avg_losses()
        # summary.add_scalar("loss_G_GAN", avg_losses[0], epoch)
        # summary.add_scalar("loss_G_recon", avg_losses[1], epoch)
        # summary.add_scalar("loss_D", avg_losses[2], epoch)

        # disp.display()
        # disp.reset()

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "netG_A2B_state_dict": netG_A2B.state_dict(),
                    "netG_B2A_state_dict": netG_B2A.state_dict(),
                    "netD_A_state_dict": netD_A.state_dict(),
                    "netD_B_state_dict": netD_B.state_dict(),
                    "epoch": epoch,
                },
                os.path.join("checkpoint", dataset_name, f"{epoch}.pth"),
            )
