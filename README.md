# Guided Image-to-Image Translation with Bi-Directional Feature Transformation
**[[Project](https://filebox.ece.vt.edu/~Badour/guided_pix2pix.html) |  [Paper](https://filebox.ece.vt.edu/~Badour/guided_pix2pix.pdf)]**

Official Pytorch implementation for **Guided Image-to-Image Translation with Bi-Directional Feature Transformation**. 
Please contact Badour AlBahar (badour@vt.edu) if you have any questions.

<img src='https://filebox.ece.vt.edu/~Badour/figures/teaser.png'>

## Prerequisites
This codebase was developed and tested with:
- Python2.7
- Pytorch 0.4.1.post2
- CUDA 8.0

## Installation

## Datasets
- **Pose transfer:**  
We use DeepFashion dataset. We follow the train/test splits provided by [Pose guided person image generation](https://github.com/charliememory/Pose-Guided-Person-Image-Generation). We provide the data in pickle format [here](https://filebox.ece.vt.edu/~Badour/datasets/DeepFashion.zip).

- **Texture transfer:** 
We use the dataset provided by [textureGAN](https://github.com/janesjanes/Pytorch-TextureGAN). \[[shoes dataset](https://s3-us-west-2.amazonaws.com/texturegan/training_shoe.tar.gz), [handbags dataset](https://s3-us-west-2.amazonaws.com/texturegan/training_handbag.tar.gz), [clothes dataset](https://s3-us-west-2.amazonaws.com/texturegan/training_cloth.tar.gz)\].

- **Depth Upsampling:**  
We use the NYU v2 dataset. We provide the mat files [here](https://filebox.ece.vt.edu/~Badour/datasets/NYU_RGBD_matfiles.zip).

## Train
**1. Pose transfer:**  

```bash
python train.py --dataroot /root/DeepFashion/ --name exp_name --netG bFT_resnet --dataset_mode pose --input_nc 3 --guide_nc 18 --output_nc 3 --lr 0.0002 --niter 100 --niter_decay 0 --batch_size 8 --use_GAN --netD basic --beta1 0.9 --checkpoints_dir ./pose_checkpoints
```

**2. Texture transfer:** 

```bash
python train.py --dataroot /root/training_handbags_pretrain/ --name exp_name --netG bFT_unet --dataset_mode texture --input_nc 1 --guide_nc 4 --output_nc 3 --niter 100 --niter_decay 0 --batch_size 256 --lr 0.0002 --use_GAN --netD basic --n_layers 7 --beta1 .9 --checkpoints_dir ./texture_checkpoints
```

**3. Depth Upsampling:** 

```bash
python train.py --dataroot /root/NYU_RGBD_matfiles/ --name exp_name --netG bFT_resnet --dataset_mode depth --input_nc 1     --guide_nc 3 --output_nc 1 --lr 0.0002 --niter 500 --niter_decay 0 --batch_size 2 --checkpoints_dir ./depth_checkpoints --depthTask_scale [4, 8, or 16]
```

## Test
You can specify which epoch to test by specifying ```--epoch``` or use the default which is the latest epoch. Results will be saved in ```--results_dir```.

**1. Pose transfer:**  

```bash
python test.py --dataroot /root/DeepFashion/ --name exp_name --netG bFT_resnet --dataset_mode pose --input_nc 3 --guide_nc 18 --output_nc 3 --checkpoints_dir ./pose_checkpoints --task pose --results_dir ./pose_results
```

**2. Texture transfer:** 

```bash
python test.py --dataroot /root/training_handbags_pretrain/ --name exp_name --netG bFT_unet --n_layers 7 --dataset_mode texture --input_nc 1 --guide_nc 4 --output_nc 3 --checkpoints_dir ./texture_checkpoints --task texture --results_dir ./texture_results
```

**3. Depth Upsampling:** 

```bash
python test.py --dataroot /root/NYU_RGBD_matfiles/ --name exp_name --netG bFT_resnet --dataset_mode depth --input_nc 1 --guide_nc 3 --output_nc 1 --checkpoints_dir ./depth_checkpoints --task depth  --depthTask_scale [4, 8, or 16] --results_dir ./depth_results
```


## Pretrained checkpoints
- Download the pretrained checkpoints [here](https://filebox.ece.vt.edu/~Badour/guided_pix2pix_checkpoints/pretrained.zip).

- Test: For example, to test the depth upsampling task with scale 16:

```bash
python test.py --dataroot /root/NYU_RGBD_matfiles/ --name depth_16 --netG bFT_resnet --dataset_mode depth --input_nc 1 --guide_nc 3 --output_nc 1 --checkpoints_dir ./checkpoints/pretrained/ --task depth  --depthTask_scale 16 --results_dir ./depth_results
```

## Evaluate
You can specify which epoch to evaluate by specifying ```--epoch``` or use the default which is the latest epoch. Results will be saved in ```--results_dir```.

**1. Pose transfer:**  

```bash
python evaluate.py --dataroot /root/DeepFashion/ --name pose --netG bFT_resnet --dataset_mode pose --input_nc 3 --guide_nc 18 --output_nc 3 --checkpoints_dir ./checkpoints/pretrained/ --task pose --results_dir ./pose_results
```

This will save the results in ```--results_dir``` and compute both SSIM and IS metrics.

**2. Texture transfer:** 
 Please download the pretrained model of textureGAN in  ```./resources``` from [bags](https://s3-us-west-2.amazonaws.com/texturegan/textureD_final_allloss_handbag_3300.pth), [shoes](https://s3-us-west-2.amazonaws.com/texturegan/textureD_final_allloss_shoes_200.pth), and [clothes](https://s3-us-west-2.amazonaws.com/texturegan/final_cloth_finetune.pth). For example, to test the pretrained texture transfer model for the bags dataset:

```bash
python evaluate.py --dataroot /root/training_handbags_pretrain/ --name texture_bags --netG bFT_unet --n_layers 7 --dataset_mode texture --input_nc 1 --guide_nc 4 --output_nc 3 --checkpoints_dir ./checkpoints/pretrained/ --task texture --results_dir ./texture_results
```

This will save the output of bFT and textureGAN in ```--results_dir``` for 10 random input texture patches per test image. The results can then be used to compute FID and LPIPS. 

**3. Depth Upsampling:** 

```bash
python evaluate.py --dataroot /root/NYU_RGBD_matfiles/ --name depth_16 --netG bFT_resnet --dataset_mode depth --input_nc 1 --guide_nc 3 --output_nc 1 --checkpoints_dir ./checkpoints/pretrained/ --task depth  --depthTask_scale 16 --results_dir ./depth_results
```
This will save the results in ```--results_dir``` and compute their RMSE metric.


## Acknowledgments
This code is heavily borrowed from [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
