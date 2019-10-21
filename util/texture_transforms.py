# source: https://raw.githubusercontent.com/janesjanes/Pytorch-TextureGAN/master/utils/transforms.py

from __future__ import division
import torchvision.transforms
import torch
import math
import random
from PIL import Image, ImageOps
from skimage import color
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections

class toLAB(object):
    """
    Transform to convert loaded into LAB space. 
    """
    
    def __init__(self):
        self.space = 'LAB'
        
    def __call__(self, images):
        lab_images = [color.rgb2lab(np.array(image)/255.0) for image in images]
        return lab_images


class toRGB_(object):
    """
    Transform to convert loaded into LAB space. 
    """
    
    def __init__(self):
        self.space = 'LAB'
        
    def __call__(self, images):
        images = np.transpose(images.numpy(), (1, 2, 0))
        rgb_images = [(np.array(image)/255.0) for image in images]
        return rgb_images


class toRGB(object):
    """
    Transform to convert loaded into RGB color space. 
    """
    
    def __init__(self, space ='LAB'):
        self.space = space
        
    def __call__(self, images):
        if self.space =='LAB':
            # npimg = np.transpose(np.array(images), (1, 2, 0))
            # print(image)
            rgb_img = [np.transpose(color.lab2rgb(np.transpose(image, (1,2,0))), (2,0,1)) for image in images]
        elif self.space =='RGB':
            # print np.shape(images)
            # images = np.transpose(images.numpy(), (1, 2, 0))
            rgb_img = [(np.array(image)/255.0) for image in images]

        return rgb_img


class toTensor(object):
    """Transforms a Numpy image to torch tensor"""
    
    def __init__(self):
        self.space = 'RGB'
        
    def __call__(self, pics):
        imgs = [torch.from_numpy(pic.transpose((2, 0, 1))) for pic in pics]
        return imgs


def normalize_lab(lab_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(lab_img.size())
    stds = torch.zeros(lab_img.size())
    
    mean[:,0,:,:] = 50
    mean[:,1,:,:] = 0
    mean[:,2,:,:] = 0
    
    stds[:,0,:,:] = 50
    stds[:,1,:,:] = 128
    stds[:,2,:,:] = 128
    
    return (lab_img.double() - mean.double())/stds.double()

def normalize_seg(seg):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    result = seg[:,0,:,:]
    if torch.max(result) >1:
        result = result/100.0
    result = torch.round(result)
    
    
    return result

def normalize_rgb(rgb_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(rgb_img.size())
    stds = torch.zeros(rgb_img.size())
    
    mean[:,0,:,:] = 0.485
    mean[:,1,:,:] = 0.456
    mean[:,2,:,:] = 0.406
    
    stds[:,0,:,:] = 0.229
    stds[:,1,:,:] = 0.224
    stds[:,2,:,:] = 0.225
    
    return (rgb_img.double() - mean.double())/stds.double()
   
    
def denormalize_lab(lab_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(lab_img.size())
    stds = torch.zeros(lab_img.size())
    
    mean[:,0,:,:] = 50
    mean[:,1,:,:] = 0
    mean[:,2,:,:] = 0
    
    stds[:,0,:,:] = 50
    stds[:,1,:,:] = 128
    stds[:,2,:,:] = 128

    return lab_img.double() *stds.double() + mean.double()


def denormalize_rgb(rgb_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(rgb_img.size())
    stds = torch.zeros(rgb_img.size())
    
    mean[:,0,:,:] = 0.485
    mean[:,1,:,:] = 0.456
    mean[:,2,:,:] = 0.406
    
    stds[:,0,:,:] = 0.229
    stds[:,1,:,:] = 0.224
    stds[:,2,:,:] = 0.225

    return rgb_img.double() *stds.double() + mean.double()


###########################################################################
# multiple images transformation -- based on transform from torchvision


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs

class Scale(object):
    """Rescale multiple input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.transform = torchvision.transforms.Scale(size)

    def __call__(self, imgs):
        """
        Args:
            imgs (list of PIL.Image): Images to be scaled.
        Returns:
            list of PIL.Image: Rescaled images.
        """       
        return [self.transform(img) for img in imgs]


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.transform = torchvision.transforms.CenterCrop(size)

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        return [self.transform(img) for img in imgs]


class Pad(object):
    """Pad the given PIL.Image on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        
        self.transform = torchvision.transforms.Pad(padding,fill)

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be padded.
        Returns:
            PIL.Image: Padded image.
        """
        
        return [self.transform(img) for img in imgs]


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]

        w, h = imgs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return imgs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, min_resize=0.08,max_resize=1.0,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.resize_size = (min_resize,max_resize)

    def __call__(self, imgs):
        for attempt in range(10):
            area = imgs[0].size[0] * imgs[0].size[1]
            target_area = random.uniform(self.resize_size[0], self.resize_size[1]) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
                x1 = random.randint(0, imgs[0].size[0] - w)
                y1 = random.randint(0, imgs[0].size[1] - h)

                imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
                assert([img.size == (w, h) for img in imgs])

                return [img.resize((self.size, self.size), self.interpolation) for img in imgs]

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(imgs))



   
    
