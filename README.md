# KTN: Knowledge Transfer Network for Learning Multi-person 2D-3D Correspondences

# Introduction
This is the implementation of KTN: Knowledge Transfer Network for Multi-person Densepose Estimation.In this work, we address the multi-person densepose estimation problem, which aims at learning dense correspondences between 2D pixels of human body and 3D human body template. It still poses several challenges due to practical scenarios where real-world scenes are complex and only partial annotations are available, leading to incompelete or false estimations. In this work, we present a novel framework to detect the densepose of multiple people in an image. The proposed method, which we refer to Knowledge Transfer Network (KTN), tackles two main problems: 1) how to refine image representation for alleviating incomplete estimations, and 2) how to reduce false estimation caused by the low-quality training labels (i.e., limited annotations and class-imbalance labels). Unlike existing works directly propagating the pyramidal features of regions for densepose estimation, the KTN uses a refinement of pyramidal representation, where it simultaneously maintains feature resolution and suppresses background pixels, and this strategy results in a substantial increase in accuracy. Moreover, the KTN enhances the ability of 3D based body parsing with external knowledges, where it casts 2D based body parsers trained from sufficient annotations as a 3D based body parser through a structural body knowledge graph. In this way, it significantly reduces the adverse effects caused by the low-quality annotations. Effectiveness of KTN is demonstrated by its superior performance to the state-of-the-art methods on DensePose-COCO dataset. 
![](/figures/KTNv2.png)
# Main Results on Densepose-COCO validation set
## KTNv2 with ResNet50 backbone
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP(50)</th>
<th valign="bottom">AP(75)</th>
<th valign="bottom">AP(M)</th>
<th valign="bottom">AP(L)</th>

<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_s1x_legacy -->
<tr><td align="center">DensePose RCNN*</td>
<td align="center">58.8</td>
<td align="center">89.3</td>
<td align="center">67.2</td>
<td align="center">55.0</td>
<td align="center">60.2</td>

<tr><td align="center">DensePose RCNN* with MID</td>
<td align="center">64.4</td>
<td align="center">90.8</td>
<td align="center">73.6</td>
<td align="center">60.2</td>
<td align="center">65.7</td>

<tr><td align="center">DensePose RCNN* with KTM</td>
<td align="center">63.4</td>
<td align="center">91.6</td>
<td align="center">72.2</td>
<td align="center">61.0</td>
<td align="center">64.8</td>

<tr><td align="center">KTNv2</td>
<td align="center">68.3</td>
<td align="center">92.1</td>
<td align="center">77.1</td>
<td align="center">63.8</td>
<td align="center">70.0</td>

</tr>

</tbody></table>

# Environment
The code is developed based on the [Detectron2](https://github.com/facebookresearch/detectron2) platform. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards or Tesla V100 cards. Other platforms or GPU cards are not fully tested.
# Installation
## Requirements
- Linux with Python=3.7
- Pytorch = 1.4 and torchvision that matches the Pytorch installation. Please install them together at [pytorch.org](https://pytorch.org/)
- OpenCV is needed by demo and visualization
- We recommend using **anaconda3** for environment management

## Build detectron2 from KTN
```
git clone https://github.com/stoa-xh91/KTN
cd KTN/
python -m pip install -e .
```

# Prepare

## Data prepare


1. Request dataset here: [DensePose](https://github.com/facebookresearch/detectron2) and [UltraPose](https://github.com/MomoAILab/ultrapose)


2. Please download dataset under datasets

Make sure to put the files as the following structure:

```
  ├─configs
  ├─datasets
  │  ├─coco
  │  │  ├─annotations
  │  │  ├─train2014
  │  │  ├─val2014
  │  ├─Ultrapose
  │  │  ├─annotations
  │  │  ├─train2014
  │  │  ├─val2014
  ├─demo
  ├─detectron2
```

## Pretrain and weight prepare
Please download the models from [Baidu](https://pan.baidu.com/s/1OyuimZ4Xd6rtC3iD4SbyZQ) (fccy)

Make sure to put the files as the follwing structure

```
  ├─detectron2
  ├─detectron2.egg-info
  ├─dev
  ├─docs
  ├─projects
  │  ├─KTNv2 
  │     ├─comm_knowledge
  │          ├─script_crkg_generate.py
  │          ├─glove.6B.300d.txt
  │          ├─kpt_surf_crkg.pkl
  │          ├─part_surf_crkg.pkl
  │          ├─person_surf_crkg.pkl
  ├─pretrained_weights
  │  ├─KTN_models
  │     ├─R50-MIDv2-kpt.pth
  │     ├─KTNv2_R50.pth
  │     ├─KTNv2_HRNet_w32.pth
```

# Training 
- Change the config file depending on what you want. Here, we provide a way to train KTN models
```
# Example: training KTNv2 with ResNet-50 backbone on DensePose-COCO with GPU 0 1 2 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/KTNv2/train_net.py \
--num-gpus 4 \
--config-file projects/KTNv2/configs/densepose_rcnn_R_50_KTNv2.yaml \
OUTPUT_DIR work_dirs/densepose_rcnn_R_50_KTNv2
```
After training, the final model is saved in OUTPUT_DIR.

# Testing
- To test the trained models saved in <work_dir>, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/KTN/train_net.py \
--num-gpus 4 \
--config-file projects/KTNv2/configs/densepose_rcnn_R_50_KTNv2.yaml \
--eval-only \
MODEL.WEIGHTS work_dirs/densepose_rcnn_R_50_KTNv2/model_final.pth
```

- Alternatively, you can test our pre-trained model saved in <pretrained_weights/KTN_models>. Run the following command:

```
# Example: testing KTNv2 with ResNet-50 backbone
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/KTN/train_net.py \
--num-gpus 4 \
--config-file projects/KTNv2/configs/densepose_rcnn_R_50_KTNv2.yaml \
--eval-only \
MODEL.WEIGHTS pretrained_weights/KTN_models/KTNv2_R50.pth
```