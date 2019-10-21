# heavily borrowed from: https://github.com/janesjanes/Pytorch-TextureGAN

from PIL import Image
import glob
import os
import os.path as osp
import random
from util import texture_transforms as custom_transforms
from data.base_dataset import BaseDataset
import torch
import math 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(directory):
    classes = [d for d in os.listdir(directory) if osp.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(directory, opt, erode_seg=True):
    # opt: 'train' or 'val'
    img = glob.glob(osp.join(directory, opt + '_img/*/*.jpg'))
    img = sorted(img)
    skg = glob.glob(osp.join(directory, opt + '_skg/*/*.jpg'))
    skg = sorted(skg)
    seg = glob.glob(osp.join(directory, opt + '_seg/*/*.jpg'))
    seg = sorted(seg)
    txt = glob.glob(osp.join(directory, opt + '_txt/*/*.jpg'))
    extended_txt = []
    for i in range(len(skg)):
        extended_txt.append(txt[i%len(txt)])
    random.shuffle(extended_txt)
    
    if erode_seg:
        eroded_seg = glob.glob(osp.join(directory, 'eroded_' + opt + '_seg/*/*.jpg'))
        eroded_seg = sorted(eroded_seg)
        return list(zip(img, skg, seg , eroded_seg, extended_txt))
    else:
        return list(zip(img, skg, seg, extended_txt))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    return pil_loader(path)

def train_get_transforms():
    transforms_list = [
        custom_transforms.RandomSizedCrop(128, 0.6, 1),
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.toLAB(),
        custom_transforms.toTensor(),
    ]
    transforms = custom_transforms.Compose(transforms_list)
    return transforms

def test_get_transforms():
    transforms_list = [
        custom_transforms.Scale(128),
        custom_transforms.CenterCrop(128),
        custom_transforms.toLAB(),
        custom_transforms.toTensor(),
    ]
    transforms = custom_transforms.Compose(transforms_list)
    return transforms

def get_coor(index, size):
    index = int(index)
    w,h = size
    return ((index%(w*h))/h, ((index%(w*h))%h))

def gen_input(img, skg, ini_texture, ini_mask, xcenter=64, ycenter=64, size=40):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh
    w, h = img.size()[1:3]
    xstart = max(int(xcenter - size / 2), 0)
    ystart = max(int(ycenter - size / 2), 0)
    xend = min(int(xcenter + size / 2), w)
    yend = min(int(ycenter + size / 2), h)
    input_texture = ini_texture  # torch.ones(img.size())*(1)
    input_sketch = skg[0:1, :, :]  # L channel from skg
    input_mask = ini_mask  # torch.ones(input_sketch.size())*(-1)
    input_mask[:, xstart:xend, ystart:yend] = 1
    input_texture[:, xstart:xend, ystart:yend] = img[:, xstart:xend, ystart:yend].clone()
    return torch.cat((input_sketch.cpu().float(), input_texture.float(), input_mask), 0)

def rand_between(a, b):
    return a + torch.round(torch.rand(1) * (b - a))[0]

def gen_input_rand(img, skg, seg, size_min=40, size_max=60, num_patch=1):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh
    bs, c, w, h = img.size()
    results = torch.Tensor(bs, 5, w, h)
    texture_info = []

    seg = seg / torch.max(seg) #make sure it's 0/1
    
    seg[:,0:int(math.ceil(size_min/2)),:] = 0
    seg[:,:,0:int(math.ceil(size_min/2))] = 0
    seg[:,:,int(math.floor(h-size_min/2)):h] = 0
    seg[:,int(math.floor(w-size_min/2)):w,:] = 0
    
    counter = 0
    for i in range(bs):
        counter = 0
        ini_texture = torch.ones(img[0].size()) * (1)
        ini_mask = torch.ones((1, w, h)) * (-1)
        temp_info = []
        
        for j in range(num_patch):
            crop_size = int(rand_between(size_min, size_max))
            
            seg_index_size = seg[i,:,:].view(-1).size()[0]
            seg_index = torch.arange(0,seg_index_size)
            seg_one = seg_index[seg[i,:,:].view(-1)==1]
            if len(seg_one) != 0:
                seg_select_index = int(rand_between(0,seg_one.view(-1).size()[0]-1))
                x,y = get_coor(seg_one[seg_select_index],seg[i,:,:].size())
            else:
                x,y = (w/2, h/2)
            
            temp_info.append([x, y, crop_size])
            res = gen_input(img[i], skg[i], ini_texture, ini_mask, x, y, crop_size)

            ini_texture = res[1:4, :, :]

        texture_info.append(temp_info)
        results[i, :, :, :] = res
    return results, texture_info

class TextureDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt

        # train or val
        if (opt.isTrain):
            choice = 'train'
            self.transform = train_get_transforms()
        else:
            choice = 'val'
            self.transform = test_get_transforms()
        
        self.erode_seg = True
        self.loader = default_loader

        self.imgs = make_dataset(opt.dataroot, choice, self.erode_seg)

    def __getitem__(self, index):
        if self.erode_seg:
            img_path, skg_path, seg_path, eroded_seg_path, txt_path = self.imgs[index]
        else:
            img_path, skg_path, seg_path, txt_path = self.imgs[index]
        
        img = self.loader(img_path)
        skg = self.loader(skg_path)
        seg = self.loader(seg_path)
        txt = self.loader(txt_path)

        if self.erode_seg:
            eroded_seg = self.loader(eroded_seg_path)
        else:
            eroded_seg = None

        if self.transform is not None:
            if self.erode_seg:
                img, skg, seg, eroded_seg, txt = self.transform([img, skg, seg, eroded_seg, txt])
            else:
                img, skg, seg, txt = self.transform([img, skg, seg, txt])
                eroded_seg = seg

        ## WE NOW HAVE: img, skg, seg, eroded_seg, txt
        # convert them to 1 batch for processing
        img = img.unsqueeze(0)
        skg = skg.unsqueeze(0)
        seg = seg.unsqueeze(0)
        eroded_seg = eroded_seg.unsqueeze(0)
        txt = txt.unsqueeze(0)

        # output img/skg/seg lab between 0-100, -128-128
        # normalize
        img = custom_transforms.normalize_lab(img)
        skg = custom_transforms.normalize_lab(skg)
        txt = custom_transforms.normalize_lab(txt)
        seg = custom_transforms.normalize_seg(seg)
        eroded_seg = custom_transforms.normalize_seg(eroded_seg)

        if not self.opt.use_segmentation_patch:
            seg.fill_(1)
        bs, w, h = seg.size()
        
        seg = seg.view(bs, 1, w, h)
        seg = torch.cat((seg, seg, seg), 1)
        eroded_seg = eroded_seg.view(bs, 1, w, h)

        temp = torch.ones(seg.size()) * (1 - seg).float()
        temp[:, 1, :, :] = 0  # torch.ones(seg[:,1,:,:].size())*(1-seg[:,1,:,:]).float()
        temp[:, 2, :, :] = 0  # torch.ones(seg[:,2,:,:].size())*(1-seg[:,2,:,:]).float()
        
        txt = txt.float() * seg.float() + temp
        
        if self.opt.input_texture_patch == 'original_image':
            inp, _ = gen_input_rand(img, skg, eroded_seg[:, 0, :, :], self.opt.patch_size_min, self.opt.patch_size_max, self.opt.num_input_texture_patch)
        elif self.opt.input_texture_patch == 'dtd_texture':
            inp, _ = gen_input_rand(txt, skg, eroded_seg[:, 0, :, :], self.opt.patch_size_min, self.opt.patch_size_max, self.opt.num_input_texture_patch)

        batch_size, _, _, _ = img.size()

        input_stack = inp
        target_img = img
        segment = seg
        target_texture = txt
        
        inv_idx = torch.arange(target_texture.size(0)-1, -1, -1).long()
        target_texture_inv = target_texture.index_select(0, inv_idx)

        assert torch.max(seg) <= 1
        assert torch.max(eroded_seg) <= 1
        
        A = input_stack[:,0,:,:].float().unsqueeze(1).squeeze(0)
        guide = input_stack[:,1:,:,:].float().squeeze(0)
        B = target_img.float().squeeze(0)

        return {'A':A, 'guide':guide, 'B':B}

    def __len__(self):
        return len(self.imgs)

    def name(self):
        return 'TextureDataset'
