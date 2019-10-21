from .base_model import BaseModel
from . import networks
from .guided_pix2pix_model import GuidedPix2PixModel


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = GuidedPix2PixModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        
        self.opt = opt

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks.define_G(input_nc=opt.input_nc, guide_nc=opt.guide_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, n_layers=opt.n_layers,
                                      norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.guide = input['guide'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A, self.guide)

    def get_output(self):
        return self.fake_B

