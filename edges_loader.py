import os
import torch
from PIL import Image
from PIL import ImageOps
import numpy as np


mypath_A = '/data/edges2shoes/train/'
mypath_B = '/data/edges2handbags/train/'

resize_img = 64

class imgldr(object):
    def __init__(self,batch_size,path_a=mypath_A,path_b=mypath_B):
         self.path0 = path_a
         self.path1 = path_b
         self.bn   = batch_size
         self.cnt  = 0
         self.getlist(path_a,0)
         self.getlist(path_b,1)


    def getlist(self,path,i):
         setattr(self,"list"+str(i),[f for f in os.listdir(path)])
         #self.list = [f for f in os.listdir(self.path)]

         setattr(self, "lens" + str(i), len( getattr( self,"list"+str(i) ) ) )
         #self.lens = len(self.list)
         print("len : %s" %getattr( self,"lens"+str(i)) )
    def getbn(self,resize = False,flip = True):
          A_batch_img = []
          B_batch_img = []
          info_im = Image.open(self.path0 + self.list0[0])
          info_im = np.array(info_im)
          ch = info_im.shape[-1]
          a_rand_idx = np.random.randint(0,self.lens0-self.bn)
          b_rand_idx = np.random.randint(0,self.lens1-self.bn)

          for i in range(self.bn):
           file_a = self.path0 + self.list0[i + a_rand_idx]
           file_b = self.path1 + self.list1[i + b_rand_idx]
           im_a = Image.open(file_a)
           im_b = Image.open(file_b)

           if resize :
            im_a=im_a.resize((resize_img+12,resize_img+12),Image.BICUBIC)
            im_b = im_b.resize((resize_img + 12, resize_img + 12),Image.BICUBIC)
           offset_x = np.random.randint(0,5)
           offset_y = np.random.randint(0,5)
           A = im_a.crop((offset_x,offset_y,offset_x+resize_img,offset_y+resize_img))
           B = im_b.crop((offset_x,offset_y,offset_x+resize_img,offset_y+resize_img))
           A.show()

           if offset_x >2 and flip:
               A = ImageOps.mirror(A)
               B = ImageOps.mirror(B)
           A = np.moveaxis((np.array(A)/255.0).astype(np.float32),-1,0)
           B = np.moveaxis((np.array(B)/255.0).astype(np.float32), -1,0)
           A_batch_img.append(A)
           B_batch_img.append(B)

          A_batch_img = np.array(A_batch_img).reshape(-1,ch,resize_img,resize_img)
          B_batch_img = np.array(B_batch_img).reshape(-1, ch, resize_img, resize_img)

          '''
          print(real_batch_img[0].shape)
          temp = np.moveaxis(real_batch_img[0],0,-1)
          print(temp.shape)
          de_im = Image.fromarray(temp)
          de_im.show()
          '''

          return A_batch_img,B_batch_img
    '''
    def getsinglebn(self,m):
         if self.cnt+self.bn > getattr( self,"lens"+str(m) ):
          self.cnt = 0
          return []
         else :
          #print(self.cnt)
          A_batch_img = []
          info_im = Image.open(getattr( self,"path"+str(m) ) + getattr( self,"list"+str(m) )[0])
          info_im = np.array(info_im)


          ch = info_im.shape[-1]
          for i in range(self.bn):
           file = getattr( self,"path"+str(m) ) + getattr( self,"list"+str(m) )[i+self.cnt]
           im = Image.open(file)
           A = np.moveaxis((np.array(im)/255.0).astype(np.float32),-1,0)
           A_batch_img.append(A)

          A_batch_img = np.array(A_batch_img).reshape(-1,ch,resize_img,resize_img)
          
          self.cnt += self.bn
          return A_batch_img
     '''
    def getsinglebn(self,m,resize = True):
         if self.cnt >= getattr( self,"lens"+str(m) ):
          self.cnt = 0
          return []
         else :
          #print(self.cnt)
          A_batch_img = []
          info_im = Image.open(getattr( self,"path"+str(m) ) + getattr( self,"list"+str(m) )[0])
          info_im = np.array(info_im)


          ch = info_im.shape[-1]
          file = getattr( self,"path"+str(m) ) + getattr( self,"list"+str(m) )[self.cnt]
          im = Image.open(file)
          if resize:
              im = im.resize((resize_img, resize_img),Image.BICUBIC)
          im_np = np.moveaxis((np.array(im)/255.0).astype(np.float32),-1,0)
          A = im_np.reshape(-1,ch,resize_img,resize_img)

          self.cnt += 1
          return A
    def preprocess(self,i,folder='train',right = True):

          info_im = Image.open(getattr( self,"path"+str(i) ) + getattr( self,"list"+str(i) )[0])
          info_im = np.array(info_im)


          ch = info_im.shape[-1]
          for j in range(getattr( self,"lens"+str(i))):
           file = getattr( self,"path"+str(i) ) + getattr( self,"list"+str(i) )[j]
           im = Image.open(file)
           if right :
            real = im.crop((0,0,resize_img,resize_img))
           else:
            real = im.crop((resize_img,0,2*resize_img,resize_img))
           if not os.path.exists(folder):
               os.mkdir(folder)
           real.save(folder + '/'+ str(j) + '.jpg', 'JPEG')
          '''
          print(real_batch_img[0].shape)
          temp = np.moveaxis(real_batch_img[0],0,-1)
          print(temp.shape)
          de_im = Image.fromarray(temp)
          de_im.show()
          '''








if __name__ == '__main__':
 cel = imgldr(1,mypath_A.replace('train','val'),mypath_B.replace('train','val'))
 cel.preprocess(0,'eval_E_A');
 cel.preprocess(1,'eval_E_B');

     
