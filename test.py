import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util 


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    model = create_model(opt)
    model.setup(opt)

    # results are saved in task_results
    results_path = opt.results_dir
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    output_list = []
    target_list = []

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

		# save results
        util.save(input, guide, target, output, results_path, i+1, opt)
    print ('Results saved in %s'%results_path)
    
