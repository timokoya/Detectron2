import torch, detectron2
from detectron2.utils.logger import setup_logger


# import some common libraries
import numpy as np
import os, json, cv2, random

import requests

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__':

    # Print torch, detectron and cuda versions
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    # Setup detectron2 logger
    setup_logger()


    # Download an example image from the COCO Dataset 
    #url = 'http://images.cocodataset.org/val2017/000000439715.jpg'
    #r = requests.get(url, allow_redirects=True)
    #open('input.jpg', 'wb').write(r.content)

    # Load Image, show image and destroy image with escape key
    #im = cv2.imread("./input.jpg")
    im = cv2.imread("/home/timi/Desktop/sheep.png")
    resize = ResizeWithAspectRatio(im, width=1280)
    
    cv2.imshow('Image', resize)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

    # Create a detectron2 config and a detectron2 DefaultPredictor to run inference on image
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    resize2 = ResizeWithAspectRatio(out.get_image()[:, :, ::-1], width=1280)
    cv2.imshow('Image Window', resize2)
    #cv2.imshow(out)

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

    # from detectron2.data.datasets import register_coco_instances
    # register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
    # register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")