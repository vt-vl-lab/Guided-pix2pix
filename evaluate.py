import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util 
from models import texturegan
import torch

def load_network(model, save_path):
    print('loading the TextureGAN model from %s' % save_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module 
    model_state = torch.load(save_path)
    if "state_dict" in model_state:
        model.load_state_dict(model_state["state_dict"])
    else:
        model.load_state_dict(model_state)
        model_state = {
            'state_dict': model.cpu().state_dict(),
            'epoch': epoch,
            'iteration': iteration,
            'model': args.model,
            'color_space': args.color_space,
            'batch_size': args.batch_size,
            'dataset': dataset,
            'image_size': args.image_size
        }
    model.cuda()

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    data_loader = CreateDataLoader(opt)
    dataset_size = len(data_loader)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)

    # which task
    task = opt.task
    
    # results are saved in task_results
    results_path = opt.results_dir
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    repeat = 1
    if (task == 'texture'):
        textureGAN_model = texturegan.TextureGAN(5, 3, 32)
        model_location = './resources/textureD_final_allloss_handbag_3300.pth'
        load_network(textureGAN_model, model_location)
        textureGAN_model.eval()
        repeat = 10
        
    output_list = []
    target_list = []

    for r in range(repeat):
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
        
            # process
            print('processing (%d/%d)-th image...' % (i+1, dataset_size))
            input = data['A']
            target = data['B']
            guide = data['guide']
            model.set_input(data)
            model.test()
            output = model.get_output()

            if (task == 'depth' or task == 'pose'):
                # save target/output in a list for evaluation
                output_list.append(output)
                target_list.append(target)
            elif (task == 'texture'):
                # save output of textureGAN for evaluation
                inpv = torch.cat([input,guide],1)
                textureGAN_output = textureGAN_model(inpv.cuda())
                util.save_texture_out(textureGAN_output, os.path.join(results_path, 'textureGAN'), i+1+(r*dataset_size))
            
            # save results
            util.save(input, guide, target, output, results_path, i+1+(r*dataset_size), opt)
        
        
    print ('Results saved in %s'%results_path)
    
    
    ## evaluation
    if (task == 'depth'):
        print ('Evaluating results...')
        util.evaluate_depth(output_list, target_list, opt)
    elif (task == 'pose'):
        print ('Evaluating results...')
        util.evaluate_pose(output_list, target_list)

