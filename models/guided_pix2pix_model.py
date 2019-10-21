import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util

class GuidedPix2PixModel(BaseModel):
    def name(self):
        return 'GuidedPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.use_GAN = opt.use_GAN
        if self.use_GAN:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        else:
            self.loss_names = ['G_L1']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            if self.use_GAN:
                self.model_names = ['G', 'D']
            else:
                self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        
        # load/define networks
        self.netG = networks.define_G(input_nc=opt.input_nc, guide_nc=opt.guide_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, n_layers=opt.n_layers, norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain & self.use_GAN:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(input_nc=opt.input_nc + opt.guide_nc + opt.output_nc, ndf=opt.ndf, netD=opt.netD, n_layers_D=opt.n_layers_D, 
                                                norm=opt.norm, use_sigmoid=use_sigmoid, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(0)
            
            # define loss functions
            if self.use_GAN:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.use_GAN:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.guide = input['guide'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A, self.guide)
    
    def get_output(self):
        return self.fake_B

    def get_class(self):
        return self.class_B

    def get_current_data(self):
        return {'A':self.real_A, 'guide':self.guide, 'B':self.real_B, 'output':self.fake_B}

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.guide, self.fake_B), 1))
        
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.guide, self.real_B), 1)

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        if self.use_GAN:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.guide, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.loss_G = self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        if self.use_GAN:
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            # update G
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            # update G
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
