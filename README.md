# Generative Adversarial Perturbations (GAP)
Code for the paper "[Generative Adversarial Perturbations](http://openaccess.thecvf.com/content_cvpr_2018/papers/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.pdf)",  CVPR 2018. 

# Usage
Separate files are used for generating perturbations for classification and segmentation models.

## Classification
First specify the paths to both training and validation folders for ImageNet `--imagenetTrain` and `--imagenetVal`. We use the class IDs listed [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 

Now we can run the following to start training a generative network for universal perturbations with target 805 (soccer ball):

```text
CUDA_VISIBLE_DEVICES=0,1 python GAP_clf.py \
--expname test_incv3_universal_targeted_linf10_twogpu \
--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel incv3 --mode train \
--perturbation_type universal --target 805 --gpu_ids 0,1 --nEpochs 10
```

Note the `CUDA_VISIBLE_DEVICES` flag is set to make both GPUs 0 and 1 visible. After the above model is done running, you can test a checkpoint like so (set `--checkpoint` to your model's checkpoint, change `--mode train` to `--mode test`, and set `--MaxIterTest <number of test iters>` sufficiently high):

```text
CUDA_VISIBLE_DEVICES=0,1 python GAP_clf.py \
--expname test_incv3_universal_targeted_linf10_twogpu \
--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel incv3 --mode test \
--perturbation_type universal --target 805 --gpu_ids 0,1 --nEpochs 10 --MaxIterTest 1700
```

#### Testing Universal Perturbation Tensor
If you have trained a universal perturbation model and want to just test one of the output perturbation tensors in the `U_out` folder of your experiment folder, you can specify it explicitly with `explicit_U` and just test without U noise or a checkpiont:

```text
CUDA_VISIBLE_DEVICES=0,1 python GAP_clf.py \
--expname test_incv3_universal_targeted_linf10_twogpu_saving_U \
--testBatchSize 16 --mag_in 10 --foolmodel incv3 --mode test \
--perturbation_type universal --target 805 --gpu_ids 0,1 \
--explicit_U images_test_incv3_universal_targeted_linf10_twogpu_saving_U/U_out/U_epoch_10_top1target_54.54192546583851.pth
```

Full usage details are given below. You can similarly generate image-dependent perturbations, and make the perturbation non-targeted by specifying `--target` to be -1.

```text
usage: GAP_clf.py [-h] [--imagenetTrain IMAGENETTRAIN]
                  [--imagenetVal IMAGENETVAL] [--batchSize BATCHSIZE]
                  [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS]
                  [--ngf NGF] [--optimizer OPTIMIZER] [--lr LR]
                  [--beta1 BETA1] [--threads THREADS] [--seed SEED]
                  [--MaxIter MAXITER] [--MaxIterTest MAXITERTEST]
                  [--mag_in MAG_IN] [--expname EXPNAME]
                  [--checkpoint CHECKPOINT] [--foolmodel FOOLMODEL]
                  [--mode MODE] [--perturbation_type PERTURBATION_TYPE]
                  [--target TARGET] [--gpu_ids GPU_IDS]
                  [--path_to_U_noise PATH_TO_U_NOISE]
                  [--explicit_U EXPLICIT_U]

generative adversarial perturbations

optional arguments:
  -h, --help            show this help message and exit
  --imagenetTrain IMAGENETTRAIN
                        ImageNet train root
  --imagenetVal IMAGENETVAL
                        ImageNet val root
  --batchSize BATCHSIZE
                        training batch size
  --testBatchSize TESTBATCHSIZE
                        testing batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --ngf NGF             generator filters in first conv layer
  --optimizer OPTIMIZER
                        optimizer: "adam" or "sgd"
  --lr LR               Learning Rate. Default=0.002
  --beta1 BETA1         beta1 for adam. default=0.5
  --threads THREADS     number of threads for data loader to use
  --seed SEED           random seed to use. Default=123
  --MaxIter MAXITER     Iterations in each Epoch
  --MaxIterTest MAXITERTEST
                        Iterations in each Epoch
  --mag_in MAG_IN       l_inf magnitude of perturbation
  --expname EXPNAME     experiment name, output folder
  --checkpoint CHECKPOINT
                        path to starting checkpoint
  --foolmodel FOOLMODEL
                        model to fool: "incv3", "vgg16", or "vgg19"
  --mode MODE           mode: "train" or "test"
  --perturbation_type PERTURBATION_TYPE
                        "universal" or "imdep" (image dependent)
  --target TARGET       target class: -1 if untargeted, 0..999 if targeted
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 or 0,1 or 1,2.
  --path_to_U_noise PATH_TO_U_NOISE
                        path to U_input_noise.txt (only needed for universal)
  --explicit_U EXPLICIT_U
                        Path to a universeal perturbation to use
```

## Segmentation

### Training
First download the [Cityscapes dataset](https://www.cityscapes-dataset.com/) and then downsample the images and label maps to 1024x512 using bilinear and nearest-neighbor interpolation, respectively ([augmentation](https://github.com/Lextal/pspnet-pytorch/blob/master/augmentation.py)). Next either download the [pretrained model](https://drive.google.com/file/d/1fh2qSLAAMzX0J27Megisehz9tV62zKtv/view?usp=sharing) (FCN-8s trained to segment Cityscapes) or provide your own pretrained model on Cityscapes. Then run `GAP_seg.py` with suitable arguments. Specify the root folder in which the cityscapes dataset is stored using `--dataroot` and the pretrained model's path with `--pretrained_cityscapes`. If you are running a targeted experiment, you need to specify a target label map using `--target_path`.

You can check `python GAP_seg.py -h` for descriptions of all arguments.

#### Examples

* Training a model with resnet generator:
  * `CUDA_VISIBLE_DEVICES=0,1 python GAP_seg.py -g 'resnet' --eps 10 --name 'test' --gpu_ids '0,1' --ngf 200 --lr 0.0001 --metric 0 --task 3 --display_freq 10 --block 5 --dataset cityscapes` 
* Training a model with unet generator:
  * ` CUDA_VISIBLE_DEVICES=0,1 python GAP_seg.py -g 'unet' --eps 10 --name 'test' --gpu_ids '0,1' --ngf 200 --lr 0.0001 --metric 2 --task 2 --dataset cityscapes`
  
#### Tasks:

We specify tasks by different numbers: 

* Task Mapping
  * 0 --- Non-targeted Image Dependent Task
  * 1 --- Targeted Image Dependent Task
  * 2 --- Non-targeted Universal Task
  * 3 --- Targeted Universal Task
  
We use 'Mean IoU' for non-targeted tasks and 'Success Rate' for targeted tasks:

* Metric Mapping
  * 0 --- Success Rate (for task 1 and 3)
  * 2 --- Mean IoU (for task 0 and 2)

## Architecture

### Universal Perturbations:

<p align="center">
<img src="https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations/blob/master/material/architecture/Universal.png" width="750">
</p>

### Image-dependent Perturbations:
<p align="center">
<img src="https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations/blob/master/material/architecture/Image_Dependent.png" width="750">
</p>


# Abstract
In this paper, we propose novel generative models for
creating adversarial examples, slightly perturbed images
resembling natural images but maliciously crafted to fool
pre-trained models. We present trainable deep neural networks
for transforming images to adversarial perturbations.
Our proposed models can produce image-agnostic
and image-dependent perturbations for targeted and nontargeted
attacks. We also demonstrate that similar architectures
can achieve impressive results in fooling both classification
and semantic segmentation models, obviating the
need for hand-crafting attack methods for each task. Using
extensive experiments on challenging high-resolution
datasets such as ImageNet and Cityscapes, we show that
our perturbations achieve high fooling rates with small perturbation
norms. Moreover, our attacks are considerably
faster than current iterative methods at inference time.

# Citation

If you use the code in this repository in your paper, please consider citing:
```
@InProceedings{Poursaeed_2018_CVPR,
  author = {Poursaeed, Omid and Katsman, Isay and Gao, Bicheng and Belongie, Serge},
  title = {Generative Adversarial Perturbations},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```
