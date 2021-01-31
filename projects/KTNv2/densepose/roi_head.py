# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from typing import Dict, List, Optional
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference
from detectron2.structures import ImageList, Instances

from .densepose_head import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
    dp_keypoint_rcnn_loss, 
    ASPPConv, 
    initialize_module_params, 
    build_ktn_losses
)

class FeatureAdaptation(torch.nn.Module):
    def __init__(self, cfg, feature_dims):
        super(FeatureAdaptation, self).__init__()
        norm = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        self.adt_conv = Conv2d(
                feature_dims*4,
                feature_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, feature_dims),
                activation=F.relu,
            )
        weight_init.c2_msra_fill(self.adt_conv)
    def forward(self, x):
        n, c, h, w = x.size()
        x_group1 = x[:, :, 0::2, :]
        x_group2 = x[:, :, 1::2, :]
        x1 = x_group1[:, :, :, 0::2]
        x2 = x_group1[:, :, :, 1::2]
        x3 = x_group2[:, :, :, 0::2]
        x4 = x_group2[:, :, :, 1::2]
        new_x = torch.cat([x1, x2, x3, x4], dim=1)
        new_x = self.adt_conv(new_x)
        return new_x

class MultiInstanceDecoder(torch.nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(MultiInstanceDecoder, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_out_dims          = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
        conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        self.multi_scale_on   = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_MULTI_SCALE_ON
        # fmt: on
        self.bottom_up = []
        for i, in_feat in enumerate(self.in_features):
            if i+1 == len(self.in_features):
                break
            fa = FeatureAdaptation(cfg, feature_channels[in_feat])
            self.bottom_up.append(fa)
            self.add_module('bottom_up_'+in_feat, self.bottom_up[-1])
        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    conv = Conv2d(
                        conv_dims,
                        conv_dims*4,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=not norm,
                        norm=get_norm(norm, conv_dims),
                        activation=F.relu,
                    )
                    weight_init.c2_msra_fill(conv)
                    head_ops.append(conv)
                    head_ops.append(
                        nn.PixelShuffle(2)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        if self.multi_scale_on:
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(conv_dims, conv_dims, 1, bias=False),
                    nn.GroupNorm(32, conv_dims),
                    nn.ReLU(),
                )
            )
            modules.append(ASPPConv(conv_dims, conv_dims, 1))
            modules.append(ASPPConv(conv_dims, conv_dims, 2))
            modules.append(ASPPConv(conv_dims, conv_dims, 3))

            self.trident_convs = nn.ModuleList(modules)
            self.predictor = Conv2d(4*conv_dims, num_out_dims, kernel_size=1, stride=1, padding=0)
        else:
            self.predictor = Conv2d(conv_dims, num_out_dims, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)
        # self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)

    def forward(self, features):
        adt_features = []
        num_stages = len(features)

        for i in range(num_stages):
            if i == 0:
                latent_feature = features[i]
            else:
                # print(i,' latent:',latent_feature.size())
                latent_feature = features[i] + self.bottom_up[i-1](latent_feature)
            adt_features.append(latent_feature)
        features = adt_features
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        if self.multi_scale_on:
            res = []
            for conv in self.trident_convs:
                res.append(conv(x))
            x = torch.cat(res, dim=1)
        x = self.predictor(x)
        return x

class Decoder(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(Decoder, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_classes           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
        conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features: List[torch.Tensor]):
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        x = self.predictor(x)
        return x

def parsing_inference(parsing_outputs, instances):
    s = parsing_outputs
    s = F.softmax(s, dim=1)
    k = 0
    for instance in instances:
        n_i = len(instance)
        s_i = s[k : k + n_i]
        instance.pred_parsings = s_i
        k += n_i

@ROI_HEADS_REGISTRY.register()
class DensePoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)
        self.mask_thresh = cfg.MODEL.ROI_DENSEPOSE_HEAD.FG_MASK_THRESHOLD
        if self.mask_thresh >= 0:
            print("-+"*10, "Mask thresh = ", self.mask_thresh, "+-"*10)

    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features, proposals = self.densepose_data_filter(features, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]

                if self.use_decoder:
                    features = [self.decoder(features)]

                features_dp = self.densepose_pooler(features, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _, confidences, _ = self.densepose_predictor(densepose_head_outputs)
                densepose_loss_dict = self.densepose_losses(proposals, densepose_outputs, confidences)
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                features = [self.decoder(features)]

            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _, confidences, _ = self.densepose_predictor(
                    densepose_head_outputs
                )
            else:
                # If no detection occurred instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 4)
                confidences = tuple([empty_tensor] * 6)

            parsing_inference(densepose_outputs[0], instances)
            # densepose_inference(densepose_outputs, confidences, instances, self.mask_thresh)
            return instances

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_densepose(features, instances))
        return instances, losses

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """

        instances = super().forward_with_given_boxes(features, instances)
        instances = self._forward_densepose(features, instances)
        return instances

@ROI_HEADS_REGISTRY.register()
class DensePoseKTNHeads(DensePoseROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_dp_keypoint_head(cfg,input_shape)


    def _init_dp_keypoint_head(self, cfg,input_shape):
        # fmt: off

        self.dp_keypoint_on                      = cfg.MODEL.ROI_DENSEPOSE_HEAD.KPT_ON
        if not self.dp_keypoint_on:
            return
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        self.positive_sample_fraction = 0.25

    def _forward_dp_keypoint(self, keypoint_logits, instances):

        if not self.dp_keypoint_on:
            return {} if self.training else instances
        num_images = len(instances)
        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = dp_keypoint_rcnn_loss(
                keypoint_logits,
                instances,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def _forward_densepose(self, features, instances):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features, proposals = self.densepose_data_filter(features, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]
                
                if self.use_decoder:
                    features = [self.decoder(features)]

                features_dp = self.densepose_pooler(features, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, densepose_outputs_lowres, confidences, _ = self.densepose_predictor(densepose_head_outputs)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_outputs+densepose_outputs_lowres[1:2], confidences
                )
                if self.dp_keypoint_on:
                    kpt_loss_dict = self._forward_dp_keypoint(keypoints_output, proposals)
                    for _, k in enumerate(kpt_loss_dict.keys()):
                        densepose_loss_dict[k] = kpt_loss_dict[k]
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            if self.use_decoder:
                features = [self.decoder(features)]

            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _, confidences, _ = self.densepose_predictor(densepose_head_outputs)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                    instances = self._forward_dp_keypoint(keypoints_output, instances)
            else:
                # If no detection occured instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 4)
                confidences = tuple([empty_tensor] * 6)

            densepose_inference(densepose_outputs[:4], confidences, instances, self.mask_thresh)
            return instances

@ROI_HEADS_REGISTRY.register()
class DensePoseKTNv2Heads(DensePoseROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_dp_keypoint_head(cfg,input_shape)
        self.decoder = None
        self.mid_decoder = MultiInstanceDecoder(cfg, input_shape, self.in_features)
        self.densepose_losses = build_ktn_losses(cfg)

    def _init_dp_keypoint_head(self, cfg,input_shape):
        # fmt: off

        self.dp_keypoint_on                      = cfg.MODEL.ROI_DENSEPOSE_HEAD.KPT_ON
        if not self.dp_keypoint_on:
            return
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        self.positive_sample_fraction = 0.25

    def _forward_dp_keypoint(self, keypoint_logits, instances):

        if not self.dp_keypoint_on:
            return {} if self.training else instances
        num_images = len(instances)
        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = dp_keypoint_rcnn_loss(
                keypoint_logits,
                instances,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def _forward_densepose(self, features, instances):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances
        bbox_locs_params = self.box_predictor.bbox_pred.weight
        bbox_cls_params = self.box_predictor.cls_score.weight
        with torch.no_grad():
            bbox_params = torch.cat([bbox_cls_params, bbox_locs_params], dim=0)
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features, proposals = self.densepose_data_filter(features, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]
                
                if self.use_decoder:
                    features = [self.mid_decoder(features)]

                features_dp = self.densepose_pooler(features, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, dp_outputs_from_kpt, dp_outputs_from_bbox = self.densepose_predictor(densepose_head_outputs, bbox_params)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_outputs
                )
                if self.dp_keypoint_on:
                    kpt_loss_dict = self._forward_dp_keypoint(keypoints_output, proposals)
                    for _, k in enumerate(kpt_loss_dict.keys()):
                        densepose_loss_dict[k] = kpt_loss_dict[k]
                if dp_outputs_from_kpt is not None:
                    dp_kpt_loss_dict = self.densepose_losses(proposals, dp_outputs_from_kpt)
                    densepose_loss_dict['loss_densepose_I_from_kpt'] = dp_kpt_loss_dict['loss_densepose_I']
                if dp_outputs_from_bbox is not None:
                    dp_box_loss_dict = self.densepose_losses(proposals, dp_outputs_from_bbox)
                    densepose_loss_dict['loss_densepose_I_from_box'] = dp_box_loss_dict['loss_densepose_I']
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            if self.use_decoder:
                features = [self.mid_decoder(features)]

            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, confidences, _ = self.densepose_predictor(densepose_head_outputs, bbox_params)
                confidences = (None, None, None, None, None, None)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                    instances = self._forward_dp_keypoint(keypoints_output, instances)
            else:
                # If no detection occured instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 4)
                confidences = tuple([empty_tensor] * 6)

            # parsing_inference(densepose_outputs[4], instances)
            densepose_inference(densepose_outputs[:4], confidences, instances, self.mask_thresh)
            return instances