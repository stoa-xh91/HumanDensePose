# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_dataset_category_config(cfg: CN):
    """
    Add config for additional category-related dataset options
     - category whitelisting
     - category mapping
    """
    _C = cfg
    _C.DATASETS.CATEGORY_MAPS = CN(new_allowed=True)
    _C.DATASETS.WHITELISTED_CATEGORIES = CN(new_allowed=True)


def add_bootstrap_config(cfg: CN):
    """
    """
    _C = cfg
    _C.BOOTSTRAP_DATASETS = []
    _C.BOOTSTRAP_MODEL = CN()
    _C.BOOTSTRAP_MODEL.WEIGHTS = ""
    _C.BOOTSTRAP_MODEL.DEVICE = "cuda"


def get_bootstrap_dataset_config() -> CN:
    _C = CN()
    _C.DATASET = ""
    # ratio used to mix data loaders
    _C.RATIO = 0.1
    # image loader
    _C.IMAGE_LOADER = CN(new_allowed=True)
    _C.IMAGE_LOADER.TYPE = ""
    _C.IMAGE_LOADER.BATCH_SIZE = 4
    _C.IMAGE_LOADER.NUM_WORKERS = 4
    # inference
    _C.INFERENCE = CN()
    # batch size for model inputs
    _C.INFERENCE.INPUT_BATCH_SIZE = 4
    # batch size to group model outputs
    _C.INFERENCE.OUTPUT_BATCH_SIZE = 2
    # sampled data
    _C.DATA_SAMPLER = CN(new_allowed=True)
    _C.DATA_SAMPLER.TYPE = ""
    # filter
    _C.FILTER = CN(new_allowed=True)
    _C.FILTER.TYPE = ""
    return _C


def load_bootstrap_config(cfg: CN):
    """
    Bootstrap datasets are given as a list of `dict` that are not automatically
    converted into CfgNode. This method processes all bootstrap dataset entries
    and ensures that they are in CfgNode format and comply with the specification
    """
    if not cfg.BOOTSTRAP_DATASETS:
        return

    bootstrap_datasets_cfgnodes = []
    for dataset_cfg in cfg.BOOTSTRAP_DATASETS:
        _C = get_bootstrap_dataset_config().clone()
        _C.merge_from_other_cfg(CN(dataset_cfg))
        bootstrap_datasets_cfgnodes.append(_C)
    cfg.BOOTSTRAP_DATASETS = bootstrap_datasets_cfgnodes


def add_densepose_config(cfg: CN):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.DENSEPOSE_ON = True

    _C.MODEL.ROI_DENSEPOSE_HEAD = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.NAME = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR = "DensePosePredictor"
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS = 8
    # Number of parts used for point labels
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES = 24
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL = 4
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM = 512
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL = 3
    _C.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE = 112
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION = 28
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS = 2  # 15 or 2
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    _C.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD = 0.7
    # Loss weights for annotation masks.(14 Parts)
    _C.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS = 5.0
    # Loss weights for surface parts. (24 Parts)
    _C.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS = 1.0
    # Loss weights for UV regression.
    _C.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS = 0.01
    # Coarse segmentation is trained using instance segmentation task data
    _C.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS = False
    # For Decoder
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON = True
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_MULTI_SCALE_ON = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE = 4
    # For DeepLab head
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM = "GN"
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NONLOCAL_ON = 0
    # Confidences
    # Enable learning UV confidences (variances) along with the actual values
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE = CN({"ENABLED": False})
    # UV confidence lower bound
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.EPSILON = 0.01
    # Enable learning segmentation confidences (variances) along with the actual values
    _C.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE = CN({"ENABLED": False})
    # Segmentation confidence lower bound
    _C.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE.EPSILON = 0.01
    # Statistical model type for confidence learning, possible values:
    # - "iid_iso": statistically independent identically distributed residuals
    #    with isotropic covariance
    # - "indep_aniso": statistically independent residuals with anisotropic
    #    covariances
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.TYPE = "iid_iso"
    # List of angles for rotation in data augmentation during training
    _C.INPUT.ROTATION_ANGLES = [0]
    _C.TEST.AUG.ROTATION_ANGLES = ()  # Rotation TTA
    _C.MODEL.ROI_DENSEPOSE_HEAD.FG_MASK_THRESHOLD = 0.5 # 0: use s_bbox, make nonsense; 0.3\0.5: use s_score_bbox

    # KTN module
    _C.MODEL.ROI_DENSEPOSE_HEAD.KPT_ON = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.KPT_UP_SCALE = 1
    _C.MODEL.ROI_DENSEPOSE_HEAD.KPT_CLASSIFIER_WEIGHT_DIR = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.KPT_SURF_RELATION_DIR = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.BBOX_SURF_RELATION_DIR = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.PART_SURF_RELATION_DIR = ""

    # For Res2Conv head
    _C.MODEL.ROI_DENSEPOSE_HEAD.RES2 = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.RES2.GN = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.RES2.SCALE = 4

    # For mixup
    _C.MODEL.ROI_DENSEPOSE_HEAD.MIXUP = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.MIXUP.ENABLED = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.MIXUP.WEIGHTS_DECAY = 0.01
    _C.MODEL.ROI_DENSEPOSE_HEAD.MIXUP.ALPHA = 0.5 # 0 means random sample alpha in (0, 1)

    # For Triplet Loss
    _C.MODEL.ROI_DENSEPOSE_HEAD.TRIPLET_LOSS = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.TRIPLET_LOSS.ENABLED = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.TRIPLET_LOSS.WEIGHTS = 0.1
    _C.MODEL.ROI_DENSEPOSE_HEAD.TRIPLET_LOSS.MARGIN = 1.0
    _C.MODEL.ROI_DENSEPOSE_HEAD.TRIPLET_LOSS.USE_XY_DIST = False

    # For Lowres Loss
    _C.MODEL.ROI_DENSEPOSE_HEAD.LOWRES_LOSS = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.LOWRES_LOSS.ENABLED = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.LOWRES_LOSS.LOWRES_WEIGHTS = 0.1

    # add smooth_l1_loss with weights for balance uv distribution
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.U = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.U.ENABLED = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.U.W0 = 3
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.U.W1 = 3
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.U.X1 = 0.1
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.U.X2 = 0.7
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.V = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.V.ENABLED = False
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.V.W0 = 6
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.V.W1 = 4
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.V.X1 = 0.2
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_LOSS_WITH_WEIGHTS.V.X2 = 0.8

    _C.MODEL.ROI_DENSEPOSE_HEAD.BODY_MASK_WEIGHTS = 5.0

def add_hrnet_config(cfg: CN):
    """
    Add config for HRNet backbone.
    """
    _C = cfg

    # For HigherHRNet w32
    _C.MODEL.HRNET = CN()
    _C.MODEL.HRNET.STEM_INPLANES = 64
    _C.MODEL.HRNET.STAGE2 = CN()
    _C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.HRNET.STAGE2.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
    _C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
    _C.MODEL.HRNET.STAGE2.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE3 = CN()
    _C.MODEL.HRNET.STAGE3.NUM_MODULES = 4
    _C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.HRNET.STAGE3.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
    _C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
    _C.MODEL.HRNET.STAGE3.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE4 = CN()
    _C.MODEL.HRNET.STAGE4.NUM_MODULES = 3
    _C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.HRNET.STAGE4.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    _C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    _C.MODEL.HRNET.STAGE4.FUSE_METHOD = "SUM"

    _C.MODEL.HRNET.HRFPN = CN()
    _C.MODEL.HRNET.HRFPN.OUT_CHANNELS = 256
