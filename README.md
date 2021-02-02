# HumanDensePose

# Introduction
This is the implementation of KTN: Knowledge Transfer Network for Multi-person Densepose Estimation.In this work, we address the multi-person densepose estimation problem, which aims at learning dense correspondences between 2D pixels of human body and 3D human body template. It still poses several challenges due to real-world scenes with scale variations, occlusion, and insufficient annotations. 
In particular, we address two main problems: 1) how to design a simple yet effective pipeline to alleviate incomplete densepose estimation, and 2) how to equip this pipeline with the ability to handle the issue of limited annotations and class-imbalance labels. 
To tackle these problems, we develop a novel densepose estimation framework based on a two-stage pipeline, which is called ***Knowledge Transfer Network (KTN)***. 
Unlike existing works directly propagating the pyramidal base features of regions, we enhance their representation power by awell-designed ***multi-instance decoder (MID)***. 
Then, we introduce a plug-and-play ***knowledge transfer machine (KTM)***, which facilitates densepose estimationby utilizing the external commonsense knowledge.  
Notably, with the help of our knowledge transfer machine (KTM), current densepose estimation systems (either based on RCNN orfully-convolutional frameworks) can be promoted in terms of the accuracy of human densepose estimation.
Solid experiments on densepose estimation benchmarks demonstrate the superiority and generalizability of our approach. 
![](https://github.com/cfm-wxh/TSN/blob/master/visualization/KTN.png)
# Main Results on Densepose-COCO validation set
![](https://github.com/cfm-wxh/TSN/blob/master/visualization/main_result.jpg)
# Environment
The code is developed based on the [Detectron2](https://github.com/facebookresearch/detectron2) platform. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards or Tesla V100 cards. Other platforms or GPU cards are not fully tested.
# Installation
Please follow the [installation instruction](https://github.com/facebookresearch/detectron2) to build environment.
# Training and Testing
- Training on COCO dataset using pretrained keypont models([Baidu](https://pan.baidu.com/s/1OyuimZ4Xd6rtC3iD4SbyZQ 
). Extraction Code:fccy)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/KTN/train_net.py \
--num-gpus 4 \
--config-file projects/KTN/configs/densepose_rcnn_R_50_KTNv2.yaml \
OUTPUT_DIR work_dirs/densepose_rcnn_R_50_KTNv2
```
After training, the final model is saved in OUTPUT_DIR.
- Testing on COCO dataset using provided models([Baidu](https://pan.baidu.com/s/1OyuimZ4Xd6rtC3iD4SbyZQ). Extraction Code:fccy)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/KTN/train_net.py \
--num-gpus 4 \
--config-file projects/KTN/configs/densepose_rcnn_R_50_KTNv2.yaml \
--eval-only \
MODEL.WEIGHTS models/DensePose_KTN_Weights.pth
```
