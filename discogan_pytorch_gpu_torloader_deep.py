import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import edges_dataset as ed
import os
from PIL import Image
import torchvision.utils as vutils
from itertools import chain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=300,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='the batch size')
    parser.add_argument('--load-epoch', type = int ,default =0,
                        help = 'load trained model')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    parser.add_argument('--unet', action='store_true',
                        help='unet network for generator')
    return parser.parse_args()


'''
variables
'''
x_size =64
y_size =64
in_ch = 3
out_ch =3
batch_size = parse_args().batch_size
save = parse_args().save
ini_ch_size = 64
outputsize = 30



'''
m  =model

'''
def drawlossplot( epoch,loss_g,loss_d,e):
    g_x = np.linspace(0, len(loss_g), len(loss_g))
    f, ax = plt.subplots(1)
   
    plt.plot(g_x, loss_g, label='loss_g')
    plt.plot(g_x, loss_d, label='loss_d')
    ax.set_xlim(0, epoch)

    plt.title('Generative Adversarial Network Loss Graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig("cifar_cdcgan_loss_epoch%d" %e)
    plt.close()


def weights_init(self):
    classname = self.__class__.__name__
    if classname.find('Conv') != -1:
        self.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        self.weight.data.normal_(1.0, 0.02)
        self.bias.data.fill_(0)

def convblocklayer(in_ch,out_ch,stride = 2):
     return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size =4, stride = stride,padding = 1, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.LeakyReLU(0.2)
                         )

def deconvblocklayer(in_ch,out_ch,pad,dropout = True):
    if dropout:
     return nn.Sequential(nn.ConvTranspose2d(in_ch,out_ch,kernel_size = 4, stride = 2,padding = pad, bias=False),
                         nn.BatchNorm2d(out_ch),
                          nn.Dropout(0.5),
                         nn.ReLU()
                         )
    else:
     return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=pad, bias=False),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU()
                             )


class generator(nn.Module):
    def __init__(self, img_channel, o_channel, ini_ch_size):
        super(generator, self).__init__()
        self.layer1 = convblocklayer(img_channel, ini_ch_size)
        self.layer2 = convblocklayer(ini_ch_size, 2 * ini_ch_size)
        self.layer3 = convblocklayer(2 * ini_ch_size, 4 * ini_ch_size)
        self.layer4 = convblocklayer(4 * ini_ch_size, 8 * ini_ch_size)
        self.layer5 = convblocklayer(8 * ini_ch_size, 8 * ini_ch_size)
        self.layer6 = nn.Conv2d(8 * ini_ch_size, 8 * ini_ch_size, kernel_size=4, stride=2, padding=1)
        self.layer7 = deconvblocklayer(8 * ini_ch_size, 8 * ini_ch_size, 1,False)
        self.layer8 = deconvblocklayer(8 * ini_ch_size, 4 * ini_ch_size, 1, False)
        self.layer9 = deconvblocklayer( 4 * ini_ch_size, 2 * ini_ch_size, 1, False)
        self.layer10 = deconvblocklayer( 2 * ini_ch_size, ini_ch_size, 1, False)
        self.deconv = nn.ConvTranspose2d(ini_ch_size, o_channel, kernel_size=4, stride=2, padding=1)
        self.activ = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer5(out5)
        out7 = self.layer5(out6)
        out10 = self.layer7(out7)
        out11 = self.layer7(out10)
        out12 = self.layer7(out11)
        out13 = self.layer8(out12)
        out14 = self.layer9(out13)
        out15 = self.layer10(out14)
        out16 = self.deconv(out15)
        out = self.activ(out16)
        return out


class u_generator(nn.Module):
    def __init__(self, img_channel, o_channel, ini_ch_size):
        super(u_generator, self).__init__()
        self.layer1 = convblocklayer(img_channel, ini_ch_size)
        self.layer2 = convblocklayer(ini_ch_size, 2 * ini_ch_size)
        self.layer3 = convblocklayer(2 * ini_ch_size, 4 * ini_ch_size)
        self.layer4 = convblocklayer(4 * ini_ch_size, 8 * ini_ch_size)
        self.layer5 = convblocklayer(8 * ini_ch_size, 8 * ini_ch_size)
        self.layer6 = nn.Conv2d(8 * ini_ch_size, 8 * ini_ch_size, kernel_size=4, stride=2, padding=1)
        self.layer7 = deconvblocklayer(2 * 8 * ini_ch_size, 8 * ini_ch_size, 1)
        self.layer7_no_drop = deconvblocklayer(2 * 8 * ini_ch_size, 8 * ini_ch_size, 1, False)
        self.layer8 = deconvblocklayer(2 * 8 * ini_ch_size, 4 * ini_ch_size, 1, False)
        self.layer9 = deconvblocklayer(2 * 4 * ini_ch_size, 2 * ini_ch_size, 1, False)
        self.layer10 = deconvblocklayer(2 * 2 * ini_ch_size, ini_ch_size, 1, False)
        self.layer11 = deconvblocklayer(8 * ini_ch_size, 8 * ini_ch_size, 1)
        self.deconv = nn.ConvTranspose2d(2 * ini_ch_size, o_channel, kernel_size=4, stride=2, padding=1)
        self.activ = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer5(out5)
        out7 = self.layer5(out6)
        out8 = self.relu(self.layer6(out7))
        out9 = self.layer11(out8)
        out10 = self.layer7(torch.cat((out9, out7), 1))
        out11 = self.layer7(torch.cat((out10, out6), 1))
        out12 = self.layer7_no_drop(torch.cat((out11, out5), 1))
        out13 = self.layer8(torch.cat((out12, out4), 1))
        out14 = self.layer9(torch.cat((out13, out3), 1))
        out15 = self.layer10(torch.cat((out14, out2), 1))
        out16 = self.deconv(torch.cat((out15, out1), 1))
        out = self.activ(out16)
        return out

class discriminator(nn.Module):
    def __init__(self, img_channel, ini_ch_size):
        super(discriminator, self).__init__()
        self.layer1 = convblocklayer(img_channel, ini_ch_size )
        self.layer2 = convblocklayer(ini_ch_size, 2 * ini_ch_size)
        self.layer3 = convblocklayer(2 * ini_ch_size, 4 * ini_ch_size)
        self.layer4 = convblocklayer(4 * ini_ch_size, 8 * ini_ch_size, 1)
        self.conv5 = nn.Conv2d(8 * ini_ch_size, 1, kernel_size=4, stride=1, padding=1)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        out1 = self.layer1(x)
        out = self.layer2(out1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv5(out)
        out = self.activ(out)
        return out


class GAN(object):
      def __init__(self,params,in_ch,o_ch,ini_ch_size):
          if params.unet :
              self.gab = u_generator(in_ch, o_ch, ini_ch_size)
              self.gba = u_generator(in_ch, o_ch, ini_ch_size)
          else :
              self.gab = generator(in_ch, o_ch, ini_ch_size)
              self.gba = generator(in_ch, o_ch, ini_ch_size)
          self.da = discriminator(in_ch,ini_ch_size)
          self.db = discriminator(in_ch, ini_ch_size)
          self.gab.apply(weights_init)
          self.gba.apply(weights_init)
          self.da.apply(weights_init)
          self.db.apply(weights_init)
          self.gab.cuda(0)
          self.gba.cuda(0)
          self.da.cuda(0)
          self.db.cuda(0)
          self.batch_size = params.batch_size
          self.lr = params.learning_rate
          self.l1loss = nn.L1Loss()
          self.mseloss =nn.MSELoss()
          params_g = chain(self.gab.parameters(),self.gba.parameters())
          params_d = chain(self.da.parameters(),self.db.parameters())
          self.g_opt = torch.optim.Adam(params_g, lr=self.lr, betas=(0.5, 0.999))
          self.d_opt = torch.optim.Adam(params_d, lr=self.lr, betas=(0.5, 0.999),weight_decay=0.0001)
          self.epoch = params.num_steps
      def save(self,i):
          torch.save(self.gab,"gab_epoch_"+str(i)+".pt")
          torch.save(self.gba,"gba_epoch_"+str(i)+".pt")
          torch.save(self.da, "da_epoch_" + str(i) + ".pt")
          torch.save(self.db, "db_epoch_" + str(i) + ".pt")
      def load(self,i):
          self.gab = torch.load("gab_epoch_"+str(i)+".pt")
          self.gba = torch.load("gba_epoch_" + str(i) + ".pt")
          self.da = torch.load("da_epoch_"+str(i)+".pt")
          self.db = torch.load("db_epoch_" + str(i) + ".pt")

def train(model,trl_a,trl_b,i,eval_a,eval_b):
    
    ones = Variable(torch.ones(model.batch_size,1,outputsize,outputsize).cuda())
    zeros = Variable(torch.zeros(model.batch_size,1,outputsize,outputsize).cuda())
    iter_a = trl_a.__iter__()
    iter_b = trl_b.__iter__()
    size = min(len(trl_a),len(trl_b))

    print("epoch :%s size:%s" %(i,size))

    e_loss_g = 0
    e_loss_d = 0
    for m in range(size):
        print("iteration:%s"%m)
        A = iter_a.next()
        B = iter_b.next()
        if A.shape[0] != model.batch_size:
            print("loop break %s" %m)
            break
        V_A = Variable(A.cuda())
        V_B = Variable(B.cuda())

        model.d_opt.zero_grad()

        xab = model.gab(V_A).detach()
        xba = model.gba(V_B).detach()
        xaba = model.gba(xab).detach()
        xbab = model.gab(xba).detach()

        da_r = model.da(V_A)
        da_f = model.da(xba)

      
        loss_da_r = model.mseloss(da_r,ones)
        loss_da_f = model.mseloss(da_f,zeros)

        db_r = model.db(V_B)
        db_f = model.db(xab)
        loss_db_r = model.mseloss(db_r,ones)
        loss_db_f = model.mseloss(db_f,zeros)

        loss_d_a = loss_da_r + loss_da_f
        loss_d_b = loss_db_r + loss_db_f

        loss = loss_d_a + loss_d_b
        loss.backward()
        model.d_opt.step()



        model.g_opt.zero_grad()


        xab = model.gab(V_A)
        xba = model.gba(V_B)
        xaba = model.gba(xab)
        xbab = model.gab(xba)

        da_f = model.da(xba)
        db_f = model.db(xab)

        l_const_a = model.l1loss(xaba,V_A)
        l_const_b = model.l1loss(xbab,V_B)

        l_g_a = model.mseloss(db_f,ones)
        l_g_b = model.mseloss(da_f,ones)



        loss_g = l_g_a + l_g_b + l_const_a + l_const_b
        loss_g.backward()
        model.g_opt.step()

        e_loss_g += loss_g.data[0]
        e_loss_d += loss.data[0]
        if m % 200 == 0 :
            eval(model, eval_a, i*size+m, 0)
            eval(model, eval_b, i*size+m, 1)



    return e_loss_g / size , e_loss_d / size

def eval(model, trl, i,mode = 0):
    print("eval epoch :%s" % i)
    iter = trl.__iter__()
    size = len(trl)
    g_data = []
    gb_data = []
    for m in range(size):
        A = iter.next()
        if A.shape[0] != model.batch_size:
            print("loop break %s" % m)
            break
        V_A = Variable(A.cuda())
        if mode :
         g = model.gba(V_A)
         gb= model.gab(g)
        else:
         g = model.gab(V_A)
         gb = model.gba(g)
        g_data.append(g.data.cpu())
        gb_data.append(gb.data.cpu())

    g_data   = torch.stack(g_data)
    gb_data  = torch.stack(gb_data)
    g_shape  = g_data.shape
    gb_shape = gb_data.shape
    g_data   = g_data.view(-1,g_shape[-3],g_shape[-2],g_shape[-1])
    gb_data  = gb_data.view(-1,gb_shape[-3],gb_shape[-2],gb_shape[-1])
    if mode :
         vutils.save_image(g_data, str(i) + '_eval_valid_x_BA.png')
         vutils.save_image(gb_data, str(i) + '_eval_valid_x_BAB.png')
    else :
         vutils.save_image(g_data, str(i) + '_eval_valid_x_AB.png')
         vutils.save_image(gb_data, str(i) + '_eval_valid_x_ABA.png')

def evaloriginal(model, trl,mode = 0):
    iter = trl.__iter__()
    size = len(trl)
    g_data = []
    for m in range(size):
        A = iter.next()
        if A.shape[0] != model.batch_size:
            print("loop break %s" % m)
            break
        g_data.append(A)

    g_data = torch.stack(g_data)
    g_shape = g_data.shape
    g_data = g_data.view(-1,g_shape[-3],g_shape[-2],g_shape[-1])
    if mode :
         vutils.save_image(g_data,'eval_valid_x_B.png')
    else :
         vutils.save_image(g_data, 'eval_valid_x_A.png')









def main(args):
    model = GAN(args,in_ch,out_ch,ini_ch_size)
    batchsize = args.batch_size
    train_dataset_a = ed.imgldr(batchsize,'./train_A/')
    train_dataset_b = ed.imgldr(batch_size, './train_B/')
    eval_dataset_a = ed.imgldr(batch_size,'./eval_A/')
    eval_dataset_b = ed.imgldr(batch_size, './eval_B/')


    train_a_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_a,
                                                batch_size=batchsize,
                                                shuffle=True,
                                                num_workers=2)
    train_b_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_b,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)

    eval_a_data_loader = torch.utils.data.DataLoader(dataset=eval_dataset_a,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=1)
    eval_b_data_loader = torch.utils.data.DataLoader(dataset=eval_dataset_b,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)
    evaloriginal(model,eval_a_data_loader,)

    a_loss_g = []
    a_loss_d = []
    evaloriginal(model, eval_a_data_loader, 0)
    evaloriginal(model, eval_b_data_loader, 1)


    if args.load_epoch != 0:
        model.load(args.load_epoch)
    for i in range(args.load_epoch,args.num_steps+args.load_epoch):
     e_loss_g, e_loss_d = train(model, train_a_data_loader,train_b_data_loader,i,eval_a_data_loader,eval_b_data_loader)
     a_loss_g.append(e_loss_g)
     a_loss_d.append(e_loss_d)
     print("discriminator loss :%s" %e_loss_d)

     if save:
             model.save(i)


if __name__ == '__main__':
    main(parse_args())










