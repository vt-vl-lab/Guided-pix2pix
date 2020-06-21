import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def affine_transformation(X, alpha, beta):
    x = X.clone()
    mean, std = calc_mean_std(x)
    mean = mean.expand_as(x)
    std = std.expand_as(x)
    return alpha * ((x-mean)/std) + beta


###############################################################################
# Defining G/D
###############################################################################

def define_G(input_nc, guide_nc, output_nc, ngf, netG, n_layers=8, n_downsampling=3, n_blocks=9, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'bFT_resnet':
        net = bFT_Resnet(input_nc, guide_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=n_blocks)
    elif netG == 'bFT_unet':
        net = bFT_Unet(input_nc, guide_nc, output_nc, n_layers, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = init_net(net, init_type, init_gain, gpu_ids)
   
    return net

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True)):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation)
    def build_conv_block(self, dim, padding_type, norm_layer, activation):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


##############################################################################
# Discriminators
##############################################################################

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)
    def forward(self, input):
        return self.net(input)


##############################################################################
# bFT Generators
##############################################################################

class bFT_Unet(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, bottleneck_depth=100):
        super(bFT_Unet, self).__init__()

        self.num_downs = num_downs

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.downconv1 = nn.Sequential(*[nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv2 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv3 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv4 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])

        downconv = [] ## this has #(num_downs - 5) layers each with [relu-downconv-norm]
        for i in range(num_downs - 5):
            downconv += [nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.downconv = nn.Sequential(*downconv)
        self.downconv5 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])

        ### bottleneck ------

        self.upconv1 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8)])
        upconv = [] ## this has #(num_downs - 5) layers each with [relu-upconv-norm]
        for i in range(num_downs - 5):
            upconv += [nn.ReLU(True), nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8)]
        self.upconv = nn.Sequential(*upconv)
        self.upconv2 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4)])
        self.upconv3 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2)])
        self.upconv4 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf)])
        self.upconv5 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), nn.Tanh()])

        ### guide downsampling
        self.G_downconv1 = nn.Sequential(*[nn.Conv2d(guide_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv2 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv3 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv4 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        G_downconv = [] ## this has #(num_downs - 5) layers each with [relu-downconv-norm]
        for i in range(num_downs - 5):
            G_downconv += [nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.G_downconv = nn.Sequential(*G_downconv)

        ### bottlenecks for param generation
        self.bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        bottleneck_alpha = []
        bottleneck_beta = []
        for i in range(num_downs - 5):
            bottleneck_alpha += self.bottleneck_layer(ngf * 8, bottleneck_depth)
            bottleneck_beta += self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.bottleneck_alpha = nn.Sequential(*bottleneck_alpha)
        self.bottleneck_beta = nn.Sequential(*bottleneck_beta)
        ### for guide
        self.G_bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.G_bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        G_bottleneck_alpha = []
        G_bottleneck_beta = []
        for i in range(num_downs - 5):
            G_bottleneck_alpha += self.bottleneck_layer(ngf * 8, bottleneck_depth)
            G_bottleneck_beta += self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_alpha = nn.Sequential(*G_bottleneck_alpha)
        self.G_bottleneck_beta = nn.Sequential(*G_bottleneck_beta)

    def bottleneck_layer(self, nc, bottleneck_depth):   
        return [nn.Conv2d(nc, bottleneck_depth, kernel_size=1), nn.ReLU(True), nn.Conv2d(bottleneck_depth, nc, kernel_size=1)]

    # per pixel
    def get_FiLM_param_(self, X, i, guide=False):
        x = X.clone()
        # bottleneck
        if guide:
            if (i=='2'):
                alpha_layer = self.G_bottleneck_alpha_2
                beta_layer = self.G_bottleneck_beta_2
            elif (i=='3'):
                alpha_layer = self.G_bottleneck_alpha_3
                beta_layer = self.G_bottleneck_beta_3
            elif (i=='4'): 
                alpha_layer = self.G_bottleneck_alpha_4
                beta_layer = self.G_bottleneck_beta_4
            else: # a number i will be given to specify which bottleneck to use
                alpha_layer = self.G_bottleneck_alpha[i:i+3]
                beta_layer = self.G_bottleneck_beta[i:i+3]
        else:
            if (i=='2'):
                alpha_layer = self.bottleneck_alpha_2
                beta_layer = self.bottleneck_beta_2
            elif (i=='3'):
                alpha_layer = self.bottleneck_alpha_3
                beta_layer = self.bottleneck_beta_3
            elif (i=='4'):
                alpha_layer = self.bottleneck_alpha_4
                beta_layer = self.bottleneck_beta_4
            else: # a number i will be given to specify which bottleneck to use
                alpha_layer = self.bottleneck_alpha[i:i+3]    
                beta_layer = self.bottleneck_beta[i:i+3]
            
        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta

    def forward (self, input, guide):
        ## downconv
        down1 = self.downconv1(input)
        G_down1 = self.G_downconv1(guide)
        
        down2 = self.downconv2(down1)
        G_down2 = self.G_downconv2(G_down1)

        g_alpha2, g_beta2 = self.get_FiLM_param_(G_down2, '2', guide=True)
        i_alpha2, i_beta2 = self.get_FiLM_param_(down2, '2')
        down2 = affine_transformation(down2, g_alpha2, g_beta2)
        G_down2 = affine_transformation(G_down2, i_alpha2, i_beta2)


        down3 = self.downconv3(down2)
        G_down3 = self.G_downconv3(G_down2)

        g_alpha3, g_beta3 = self.get_FiLM_param_(G_down3, '3', guide=True)
        i_alpha3, i_beta3 = self.get_FiLM_param_(down3, '3')
        down3 = affine_transformation(down3, g_alpha3, g_beta3)
        G_down3 = affine_transformation(G_down3, i_alpha3, i_beta3)

        down4 = self.downconv4(down3)
        G_down4 = self.G_downconv4(G_down3)

        g_alpha4, g_beta4 = self.get_FiLM_param_(G_down4, '4', guide=True)
        i_alpha4, i_beta4 = self.get_FiLM_param_(down4, '4')
        down4 = affine_transformation(down4, g_alpha4, g_beta4) 
        G_down4 = affine_transformation(G_down4, i_alpha4, i_beta4)
        
        ## (num_downs - 5) layers
        down = []
        G_down = []
        for i in range(self.num_downs - 5):
            layer = 2 * i
            bottleneck_layer = 3 * i
            downconv = self.downconv[layer:layer+2]
            G_downconv = self.G_downconv[layer:layer+2]
            if (layer == 0):
                down += [downconv(down4)]
                G_down += [G_downconv(G_down4)]
            else:
                down += [downconv(down[i-1])]
                G_down += [G_downconv(G_down[i-1])]

            g_alpha, g_beta = self.get_FiLM_param_(G_down[i], bottleneck_layer, guide=True)
            i_alpha, i_beta = self.get_FiLM_param_(down[i], bottleneck_layer)
            down[i] = affine_transformation(down[i], g_alpha, g_beta) 
            G_down[i] = affine_transformation(G_down[i], i_alpha, i_beta)

        down5 = self.downconv5(down[-1])

        ## concat and upconv
        up = self.upconv1(down5)
        num_down = self.num_downs - 5
        for i in range(self.num_downs - 5):
            layer = 3 * i
            upconv = self.upconv[layer:layer+3]
            num_down -= 1
            up = upconv(torch.cat([down[num_down], up], 1))
        up = self.upconv2(torch.cat([down4,up],1))
        up = self.upconv3(torch.cat([down3,up],1))
        up = self.upconv4(torch.cat([down2,up],1))
        up = self.upconv5(torch.cat([down1,up],1))

        return up

class bFT_Resnet(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', bottleneck_depth=100):
        super(bFT_Resnet, self).__init__()

        self.activation = nn.ReLU(True)

        n_downsampling=3
		
        ## input
        padding_in = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)]
        self.padding_in = nn.Sequential(*padding_in)
        self.conv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)

        ## guide
        padding_g = [nn.ReflectionPad2d(3), nn.Conv2d(guide_nc, ngf, kernel_size=7, padding=0)]
        self.padding_g = nn.Sequential(*padding_g)
        self.conv1_g = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2_g = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3_g = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)

        # bottleneck1
        self.bottleneck_alpha_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.G_bottleneck_alpha_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.bottleneck_beta_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.G_bottleneck_beta_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        # bottleneck2
        self.bottleneck_alpha_2 = self.bottleneck_layer(ngf*2, bottleneck_depth)
        self.G_bottleneck_alpha_2 = self.bottleneck_layer(ngf*2, bottleneck_depth)
        self.bottleneck_beta_2 = self.bottleneck_layer(ngf*2, bottleneck_depth)
        self.G_bottleneck_beta_2 = self.bottleneck_layer(ngf*2, bottleneck_depth)
        # bottleneck3
        self.bottleneck_alpha_3 = self.bottleneck_layer(ngf*4, bottleneck_depth)
        self.G_bottleneck_alpha_3 = self.bottleneck_layer(ngf*4, bottleneck_depth)
        self.bottleneck_beta_3 = self.bottleneck_layer(ngf*4, bottleneck_depth)
        self.G_bottleneck_beta_3 = self.bottleneck_layer(ngf*4, bottleneck_depth)
        # bottleneck4
        self.bottleneck_alpha_4 = self.bottleneck_layer(ngf*8, bottleneck_depth)
        self.G_bottleneck_alpha_4 = self.bottleneck_layer(ngf*8, bottleneck_depth)
        self.bottleneck_beta_4 = self.bottleneck_layer(ngf*8, bottleneck_depth)
        self.G_bottleneck_beta_4 = self.bottleneck_layer(ngf*8, bottleneck_depth)
        
        resnet = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            resnet += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=self.activation, norm_layer=norm_layer)]
        self.resnet = nn.Sequential(*resnet)
        decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), self.activation]
        self.pre_decoder = nn.Sequential(*decoder)
        self.decoder = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()])

    def bottleneck_layer(self, nc, bottleneck_depth):
        return nn.Sequential(*[nn.Conv2d(nc, bottleneck_depth, kernel_size=1), self.activation, nn.Conv2d(bottleneck_depth, nc, kernel_size=1)])

    def get_FiLM_param_(self, X, i, guide=False):
        x = X.clone()
        # bottleneck
        if guide:
            if (i==1):
                alpha_layer = self.G_bottleneck_alpha_1
                beta_layer = self.G_bottleneck_beta_1
            elif (i==2):
                alpha_layer = self.G_bottleneck_alpha_2
                beta_layer = self.G_bottleneck_beta_2
            elif (i==3):
                alpha_layer = self.G_bottleneck_alpha_3
                beta_layer = self.G_bottleneck_beta_3
            elif (i==4):
                alpha_layer = self.G_bottleneck_alpha_4
                beta_layer = self.G_bottleneck_beta_4
        else:
            if (i==1):
                alpha_layer = self.bottleneck_alpha_1
                beta_layer = self.bottleneck_beta_1
            elif (i==2):
                alpha_layer = self.bottleneck_alpha_2
                beta_layer = self.bottleneck_beta_2
            elif (i==3):
                alpha_layer = self.bottleneck_alpha_3
                beta_layer = self.bottleneck_beta_3
            elif (i==4):
                alpha_layer = self.bottleneck_alpha_4
                beta_layer = self.bottleneck_beta_4
        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta
	

    def forward(self, input, guidance):
        input = self.padding_in(input)  
        guidance = self.padding_g(guidance)

        g_alpha1, g_beta1 = self.get_FiLM_param_(guidance, 1, guide=True)
        i_alpha1, i_beta1 = self.get_FiLM_param_(input, 1)
        guidance = affine_transformation(guidance, i_alpha1, i_beta1)
        input = affine_transformation(input, g_alpha1, g_beta1)
        
        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv1(input)
        guidance = self.conv1_g(guidance)

        g_alpha2, g_beta2 = self.get_FiLM_param_(guidance, 2, guide=True)
        i_alpha2, i_beta2 = self.get_FiLM_param_(input, 2)
        input = affine_transformation(input, g_alpha2, g_beta2)
        guidance = affine_transformation(guidance, i_alpha2, i_beta2)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv2(input)
        guidance = self.conv2_g(guidance)

        g_alpha3, g_beta3 = self.get_FiLM_param_(guidance, 3, guide=True)
        i_alpha3, i_beta3 = self.get_FiLM_param_(input, 3)
        input = affine_transformation(input, g_alpha3, g_beta3)
        guidance = affine_transformation(guidance, i_alpha3, i_beta3)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv3(input)
        guidance = self.conv3_g(guidance)

        g_alpha4, g_beta4 = self.get_FiLM_param_(guidance, 4, guide=True)
        input = affine_transformation(input, g_alpha4, g_beta4)

        input = self.activation(input)

        input = self.resnet(input)
        input = self.pre_decoder(input)
        output = self.decoder(input)
        return output

##############################################################################
# uFT Generators
##############################################################################
class uFT_resnet(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', bottleneck_depth=100):
        super(uFT_resnet, self).__init__()
        
        self.activation = nn.ReLU(True)
        
        n_downsampling=3
        
        ## input
        padding_in = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)]
        self.padding_in = nn.Sequential(*padding_in)
        self.conv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)

        ## guide
        padding_g = [nn.ReflectionPad2d(3), nn.Conv2d(guide_nc, ngf, kernel_size=7, padding=0)]
        self.padding_g = nn.Sequential(*padding_g)
        self.conv1_g = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.norm1_g = norm_layer(ngf * 1)
        self.conv2_g = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.norm2_g = norm_layer(ngf * 2)
        self.conv3_g = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.norm3_g = norm_layer(ngf * 4)
        
        # bottleneck1
        self.bottleneck_alpha_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.bottleneck_beta_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        # bottleneck2
        self.bottleneck_alpha_2 = self.bottleneck_layer(ngf*2, bottleneck_depth)
        self.bottleneck_beta_2 = self.bottleneck_layer(ngf*2, bottleneck_depth)
        # bottleneck3
        self.bottleneck_alpha_3 = self.bottleneck_layer(ngf*4, bottleneck_depth)
        self.bottleneck_beta_3 = self.bottleneck_layer(ngf*4, bottleneck_depth)
        # bottleneck4
        self.bottleneck_alpha_4 = self.bottleneck_layer(ngf*8, bottleneck_depth)
        self.bottleneck_beta_4 = self.bottleneck_layer(ngf*8, bottleneck_depth)
        
        resnet = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            resnet += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=self.activation, norm_layer=norm_layer)]
        self.resnet = nn.Sequential(*resnet)
        decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), self.activation]
        self.pre_decoder = nn.Sequential(*decoder)
        self.decoder = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()])

    def bottleneck_layer(self, nc, bottleneck_depth):
        return nn.Sequential(*[nn.Conv2d(nc, bottleneck_depth, kernel_size=1), self.activation, nn.Conv2d(bottleneck_depth, nc, kernel_size=1)])

    def get_FiLM_param_(self, X, i):
        x = X.clone()
        # bottleneck
        if (i==1):
            alpha_layer = self.bottleneck_alpha_1
            beta_layer = self.bottleneck_beta_1
        elif (i==2):
            alpha_layer = self.bottleneck_alpha_2
            beta_layer = self.bottleneck_beta_2
        elif (i==3):
            alpha_layer = self.bottleneck_alpha_3
            beta_layer = self.bottleneck_beta_3
        elif (i==4):
            alpha_layer = self.bottleneck_alpha_4
            beta_layer = self.bottleneck_beta_4
        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta


    def forward(self, input, guidance):
        input = self.padding_in(input)
        guidance = self.padding_g(guidance)
        
        alpha1, beta1 = self.get_FiLM_param_(guidance, 1)
        input = affine_transformation(input, alpha1, beta1)
        guidance = self.norm1_g(guidance)

        input = self.activation(input)
        guidance = self.activation(guidance)
        
        input = self.conv1(input)
        guidance = self.conv1_g(guidance)

        alpha2, beta2 = self.get_FiLM_param_(guidance, 2)
        input = affine_transformation(input, alpha2, beta2)
        guidance = self.norm2_g(guidance)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv2(input)
        guidance = self.conv2_g(guidance)

        alpha3, beta3 = self.get_FiLM_param_(guidance, 3)
        input = affine_transformation(input, alpha3, beta3)
        guidance = self.norm3_g(guidance)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv3(input)
        guidance = self.conv3_g(guidance)

        alpha4, beta4 = self.get_FiLM_param_(guidance, 4)
        input = affine_transformation(input, alpha4, beta4)
        
        input = self.activation(input)
        
        input = self.resnet(input)
        input =  self.pre_decoder(input)
        output = self.decoder(input)
        return output

class uFT_Unet(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, bottleneck_depth=100):
        super(uFT_Unet, self).__init__()

        self.num_downs = num_downs

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.downconv1 = nn.Sequential(*[nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv2 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv3 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv4 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])

        downconv = [] ## this has #(num_downs - 5) layers each with [relu-downconv-norm]
        for i in range(num_downs - 5):
            downconv += [nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.downconv = nn.Sequential(*downconv)
        self.downconv5 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])

        ### bottleneck ------

        self.upconv1 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8)])
        upconv = [] ## this has #(num_downs - 5) layers each with [relu-upconv-norm]
        for i in range(num_downs - 5):
            upconv += [nn.ReLU(True), nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8)]
        self.upconv = nn.Sequential(*upconv)
        self.upconv2 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 4)])
        self.upconv3 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2)])
        self.upconv4 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ngf)])
        self.upconv5 = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), nn.Tanh()])

        ### guide downsampling
        self.G_downconv1 = nn.Sequential(*[nn.Conv2d(guide_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv2 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv3 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv4 = nn.Sequential(*[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        G_downconv = [] ## this has #(num_downs - 5) layers each with [relu-downconv-norm]
        for i in range(num_downs - 5):
            G_downconv += [nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.G_downconv = nn.Sequential(*G_downconv)

        ### bottlenecks for param generation
        ### for guide
        ### normalization
        self.G_bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.norm2_g = norm_layer(ngf * 2)
        self.G_bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.norm3_g = norm_layer(ngf * 4)
        self.G_bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.G_bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.norm4_g = norm_layer(ngf * 8)
        G_bottleneck_alpha = []
        G_bottleneck_beta = []
        self.norm_g = norm_layer(ngf * 8)
        for i in range(num_downs - 5):
            G_bottleneck_alpha += self.bottleneck_layer(ngf * 8, bottleneck_depth)
            G_bottleneck_beta += self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_alpha = nn.Sequential(*G_bottleneck_alpha)
        self.G_bottleneck_beta = nn.Sequential(*G_bottleneck_beta)
        
    def bottleneck_layer(self, nc, bottleneck_depth):   
        return [nn.Conv2d(nc, bottleneck_depth, kernel_size=1), nn.ReLU(True), nn.Conv2d(bottleneck_depth, nc, kernel_size=1)]

    # per pixel
    def get_FiLM_param_(self, X, i):
        x = X.clone()
        # bottleneck
        if (i=='2'):
            alpha_layer = self.G_bottleneck_alpha_2
            beta_layer = self.G_bottleneck_beta_2
        elif (i=='3'):
            alpha_layer = self.G_bottleneck_alpha_3
            beta_layer = self.G_bottleneck_beta_3
        elif (i=='4'): 
            alpha_layer = self.G_bottleneck_alpha_4
            beta_layer = self.G_bottleneck_beta_4
        else: # a number i will ve given to specify which bottleneck to use
            alpha_layer = self.G_bottleneck_alpha[i:i+3]
            beta_layer = self.G_bottleneck_beta[i:i+3]
            
        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta

    def forward (self, input, guide):
        ## downconv
        down1 = self.downconv1(input)
        G_down1 = self.G_downconv1(guide)
        
        down2 = self.downconv2(down1)
        G_down2 = self.G_downconv2(G_down1)

        g_alpha2, g_beta2 = self.get_FiLM_param_(G_down2, '2')
        down2 = affine_transformation(down2, g_alpha2, g_beta2)
        G_down2 = self.norm2_g(G_down2)
        
        down3 = self.downconv3(down2)
        G_down3 = self.G_downconv3(G_down2)

        g_alpha3, g_beta3 = self.get_FiLM_param_(G_down3, '3')
        down3 = affine_transformation(down3, g_alpha3, g_beta3)
        G_down3 = self.norm3_g(G_down3)

        down4 = self.downconv4(down3)
        G_down4 = self.G_downconv4(G_down3)

        g_alpha4, g_beta4 = self.get_FiLM_param_(G_down4, '4')
        down4 = affine_transformation(down4, g_alpha4, g_beta4) 
        G_down4 = self.norm4_g(G_down4)
        
        ## (num_downs - 5) layers
        down = []
        G_down = []
        for i in range(self.num_downs - 5):
            layer = 2 * i
            bottleneck_layer = 3 * i
            downconv = self.downconv[layer:layer+2]
            G_downconv = self.G_downconv[layer:layer+2]
            if (layer == 0):
                down += [downconv(down4)]
                G_down += [G_downconv(G_down4)]
            else:
                down += [downconv(down[i-1])]
                G_down += [G_downconv(G_down[i-1])]

            g_alpha, g_beta = self.get_FiLM_param_(G_down[i], bottleneck_layer)
            down[i] = affine_transformation(down[i], g_alpha, g_beta) 
            G_down[i] = self.norm_g(G_down[i])

        down5 = self.downconv5(down[-1])

        ## concat and upconv
        up = self.upconv1(down5)
        num_down = self.num_downs - 5
        for i in range(self.num_downs - 5):
            layer = 3 * i
            upconv = self.upconv[layer:layer+3]
            num_down -= 1
            up = upconv(torch.cat([down[num_down], up], 1))
        up = self.upconv2(torch.cat([down4,up],1))
        up = self.upconv3(torch.cat([down3,up],1))
        up = self.upconv4(torch.cat([down2,up],1))
        up = self.upconv5(torch.cat([down1,up],1))

        return up
