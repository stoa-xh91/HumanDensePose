# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_FLIP_MAP, COCO_PERSON_KEYPOINT_NAMES, KEYPOINT_CONNECTION_RULES


def get_densepose_metadata():
    meta = {
        "thing_classes": ["person"],
        "densepose_transform_src": "detectron2://densepose/UV_symmetry_transforms.mat",
        "densepose_smpl_subdiv": "detectron2://densepose/SMPL_subdiv.mat",
        "densepose_smpl_subdiv_transform": "detectron2://densepose/SMPL_SUBDIV_TRANSFORM.mat",
        "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
        "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
        "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
    }

    return meta


SPLITS = {
    "densepose_coco_2014_train": ("coco/train2014", "coco/annotations/densepose_train2014.json"),
    "densepose_coco_2014_minival": ("coco/val2014", "coco/annotations/densepose_minival2014.json"),
    "densepose_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/densepose_minival2014_100.json",
    ),
    "densepose_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/densepose_valminusminival2014.json",
    ),
    "densepose_ultrapose_train": ("ultrapose/train2014", "ultrapose/annotations/densepose_train2014.json"),
    "densepose_ultrapose_minival": ("ultrapose/val2014", "ultrapose/annotations/densepose_valminusminival2014.json"),
}

for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    register_coco_instances(
        key,
        get_densepose_metadata(),
        os.path.join("./datasets", json_file),
        os.path.join("./datasets", image_root),
    )
