from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import texture_transforms as custom_transforms
from skimage.measure import compare_ssim
from eval.InceptionScore import get_inception_score

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

 
def print_current_losses(epoch, i, losses, t, t_data):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_pose(input_batch, guide_batch, target_batch, output_batch):
	# convert to numpy image
    input = tensor2im(input_batch)
    target = tensor2im(target_batch)
    output = tensor2im(output_batch)
    pose = guide_batch.data
    pose_np = pose[0].cpu().float().numpy()
    pose_np = np.transpose(pose_np, (1, 2, 0))
    pose = np.amax(np.expand_dims(pose_np,0), axis=-1, keepdims=True)
    pose = (pose.squeeze(0)+1.0)*127.5
    guide = np.tile(pose, (1, 1, 3)).astype(np.uint8)
    return input, guide, target, output
    

def depth2im(depth):
    depth = depth.data
    depth_np = depth[0].squeeze().cpu().float().numpy()
    depth_np = np.stack((depth_np, )*3, -1)
    depth = ((depth_np+1.0)*127.5).round().astype(np.uint8)
    return depth
    
def visualize_depth(input_batch, guide_batch, target_batch, output_batch):
	# process the depth (input/target/output) 
    input = depth2im(input_batch)
    target = depth2im(target_batch)
    output = depth2im(output_batch)
    guide = tensor2im(guide_batch)
    return input, guide, target, output


def visualize_texture(input_batch, guide_batch, target_batch, output_batch):
    input_batch = input_batch.cpu()
    guide_batch = guide_batch.cpu()
    target_batch = target_batch.cpu()
    output_batch = output_batch.detach().cpu()
    # process the guide
    # Concatenation of [channel for texture intensity + two channels for color + channel for binary location mask]
    mask_batch = guide_batch[:,3,:,:].unsqueeze(3)
    # mask = -1 of not included and 1 if included
    # change this to 0 and 1 respectively
    mask_batch[mask_batch==-1] = 0
    guide_batch = guide_batch[:,0:3,:,:]
    ## denormalize to LAB
    guide_batch = custom_transforms.denormalize_lab(guide_batch)
    target_batch = custom_transforms.denormalize_lab(target_batch)
    output_batch = custom_transforms.denormalize_lab(output_batch)
    # convert to RGB for visualization
    guide_batch = vis_image(guide_batch)
    target_batch = vis_image(target_batch)
    output_batch = vis_image(output_batch)
    # get [h,w,c]
    output = np.transpose(output_batch[0], (1,2,0))
    input = np.stack((input_batch[0].numpy().squeeze(), )*3, -1) # [h,w,3]
    target = np.transpose(target_batch[0], (1,2,0))
    guide = np.transpose(guide_batch[0], (1,2,0))
    # get [0,255]
    output = (output*255)
    input = (input*255)
    target = (target*255)
    guide = (guide*255)
    # put texture on top of sketch
    current_mask = mask_batch[0].numpy()
    guide = guide * current_mask + (1-current_mask) * input
    # round and to uint8
    output = output.round().astype(np.uint8)
    guide = guide.round().astype(np.uint8)
    target = target.round().astype(np.uint8)
    input = input.round().astype(np.uint8)
    # return processed values        
    return input, guide, target, output

def save_texture_out(output_batch, path, index):
    output_batch = output_batch.detach().cpu()
    ## denormalize to LAB
    output_batch = custom_transforms.denormalize_lab(output_batch)
    # convert to RGB for visualization
    output_batch = vis_image(output_batch)
    # get [h,w,c]
    output = np.transpose(output_batch[0], (1,2,0))
    # get [0,255]
    output = (output*255)
    # round and to uint8
    output = output.round().astype(np.uint8)
    # save
    mkdir(path)
    save_image(output, os.path.join(path, '%d.png'%index))

def vis_image(img):
    # returb RGB numpy image [0,1]
    if torch.cuda.is_available():
        img = img.cpu()
    img = img.numpy()
    ToRGB = custom_transforms.toRGB()
    img_np = ToRGB(img)
    return img_np

        
def save(input, guide, target, output, path, index, opt):
    task = opt.task
    if (task == 'depth'):
    	input, guide, target, output = visualize_depth(input, guide, target, output)
    elif (task == 'pose'):
    	input, guide, target, output = visualize_pose(input, guide, target, output)
    elif (task == 'texture'):
    	input, guide, target, output = visualize_texture(input, guide, target, output)
    else:
    	print ('Invalid task. Valid tasks are [depth, pose, or texture].')
    
    input_path = os.path.join(path, 'input')
    guide_path = os.path.join(path, 'guide')
    target_path = os.path.join(path, 'target')
    output_path = os.path.join(path, 'output')

    mkdir(input_path)
    mkdir(guide_path)
    mkdir(target_path)
    mkdir(output_path)
	
    save_image(input, os.path.join(input_path, '%d.png'%index))
    save_image(guide, os.path.join(guide_path, '%d.png'%index))
    save_image(target, os.path.join(target_path, '%d.png'%index))
    save_image(output, os.path.join(output_path, '%d.png'%index))


def evaluate_depth(output_list, target_list, opt):
    import scipy.io
    # read min/max for normalization
    file = os.path.join(opt.dataroot, 'max_16x.mat')
    mat = scipy.io.loadmat(file)
    max_ =  mat['max_16x'].squeeze()
    file = os.path.join(opt.dataroot, 'min_16x.mat')
    mat = scipy.io.loadmat(file)
    min_ =  mat['min_16x'].squeeze()
    rmse = []
    for i in range(len(output_list)):
        target = target_list[i][0][0]
        output = output_list[i][0][0]
        # reverse normalization
        cur_max = max_[i]
        cur_min = min_[i]
        real = ((target.data.cpu().numpy()+1)/2)    	# back to [0,1]
        real = (real*(cur_max-cur_min))+cur_min     	# convert to meters
        real = real * 100                   			# convert to cm
        fake = ((output.data.cpu().numpy()+1)/2)
        fake = (fake*(cur_max-cur_min))+cur_min
        fake = fake * 100
        # remove the edge for fair comparison with state-of-the-art
        real = real.squeeze()
        fake = fake.squeeze()
        h = 480
        w = 640
        real = real[7:h-6, 7:w-6]
        fake = fake[7:h-6, 7:w-6]
        current_rmse = np.sqrt(np.mean((real-fake)**2))
        rmse.append(current_rmse)
    print ('RMSE = %f'%np.mean(rmse))

	
def evaluate_pose(output_list, target_list):
    ssim = []
    for i in range(len(output_list)):
        ssim.append(compare_ssim(output_list[i], target_list[i], multichannel=True))
    print ('SSIM Score: mean = %f, std= %f' %(np.mean(ssim),np.std(ssim)))
    mean, std = get_inception_score(output_list)
    print ('Inception Score: mean = %f, std= %f' %(mean,std))
	
	
		
	

    
