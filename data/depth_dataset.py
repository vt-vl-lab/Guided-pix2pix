import os.path
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
import random
import h5py
import scipy.io
import numpy as np

class DepthDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot                                 

        if (self.opt.isTrain):
            file = 'train'
        else:
            file = 'test'
        dir_A = os.path.join(opt.dataroot, 'input_%dx_%s.mat'%(opt.depthTask_scale,file))
        dir_B = os.path.join(opt.dataroot, 'depth_%s.mat'%(file))
        dir_guide = os.path.join(opt.dataroot, 'nyu_depth_v2_labeled.mat')
        self.As, self.guides, self.Bs = self.read_mat_files(dir_A, dir_B, dir_guide)

    def read_mat_files(self, dir_A, dir_B, dir_guide):
        print ('Reading Matlab files...')
        ##### A ##### 
        if (self.opt.isTrain):
            data = 'input_%dx_train'%self.opt.depthTask_scale
            with h5py.File(dir_A, 'r') as hf:
                LowRes = np.array(hf.get(data)).astype(np.float32)
            As = np.transpose(LowRes, (0, 3,2,1))
        else:
            data = 'input_%dx_test'%self.opt.depthTask_scale
            mat = scipy.io.loadmat(dir_A)
            LowRes =  mat[data].astype(np.float32)
            As = np.transpose(LowRes, (3, 0,1,2))
        As = np.clip(As, 0, 1)
        ##### Guide ######
        with h5py.File(dir_guide, 'r') as hf:
            images = np.array(hf.get('images'))
        images = np.transpose(images, (0, 3,2,1))
        guides = images.astype(np.float32)/255
        ##### B ######
        if (self.opt.isTrain):
            data = 'depth_train'
            with h5py.File(dir_B, 'r') as hf:
                depths = np.array(hf.get(data)).astype(np.float32)
            Bs = np.transpose(depths, (0, 3,2,1))
        else:
            data = 'depth_test'
            mat = scipy.io.loadmat(dir_B)
            depths =  mat[data].astype(np.float32)
            Bs = np.transpose(depths, (3, 0,1,2))
        return As, guides, Bs
       
    def randomFlip(self, input, guide, target):
        # random horizontal flip
        rand = random.randint(0, 1)
        if (rand==1):
            input = np.flip(input, axis=1).copy()
            guide = np.flip(guide, axis=1).copy()
            target = np.flip(target, axis=1).copy()
        return input, guide, target

    def __getitem__(self, index):
        if (self.opt.isTrain):
            shift = 0
        else:
            shift = 1000

        A = self.As[index]
        guide = self.guides[shift+index]
        B = self.Bs[index]

        # normalize [-1,1]
        A = A*2-1
        guide = guide*2-1
        B = B*2-1

        if self.opt.isTrain:
            A, guide, B = self.randomFlip(A, guide, B)

        # return as tensors
        t = transforms.Compose([transforms.ToTensor()])
        A = t(A)
        guide = t(guide)
        B = t(B)

        assert (self.opt.input_nc==1)
        assert (self.opt.output_nc==1)
        assert (self.opt.guide_nc==3)

        return {'A': A, 'guide': guide, 'B': B}

    def __len__(self):
        return len(self.As)

    def name(self):
        return 'DepthDataset'
