import torch, detectron2
from detectron2.utils.logger import setup_logger

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

def get_fiftyone_dicts(samples):
    samples.compute_metadata()

    dataset_dicts = []
    for sample in samples.select_fields(["id", "filepath", "metadata", "segmentations"]):
        height = sample.metadata["height"]
        width = sample.metadata["width"]
        record = {}
        record["file_name"] = sample.filepath
        record["image_id"] = sample.id
        record["height"] = height
        record["width"] = width

        objs = []
        for det in sample.segmentations.detections:
            tlx, tly, w, h = det.bounding_box
            bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
            fo_poly = det.to_polyline()
            poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
# Resize with Aspect Ratio
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

    # Load dataset from Google Open Images v7
    dataset2 = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["points", "segmentations", "detections"],
        classes=["Fish"],
        dataset_name="Fishv9",
        max_samples=100,
    )

    # Remove other classes and existing tags
    dataset2.filter_labels("points", F("label") == "Fish").save()
    dataset2.untag_samples("validation")

    # Split dataset into training and validation
    four.random_split(dataset2, {"train": 0.8, "val": 0.2})
    
    for d in ["train", "val"]:
        view = dataset2.match_tags(d)
        DatasetCatalog.register("fiftyone_" + d, lambda view=view: get_fiftyone_dicts(view))
        MetadataCatalog.get("fiftyone_" + d).set(thing_classes=["Fish"])

    metadata = MetadataCatalog.get("fiftyone_train")
    # Load Image and show image
    im = cv2.imread("/home/timi/Desktop/fishes2.jpg")
    resize = ResizeWithAspectRatio(im, width=1280)

    # Destroy Image with escape key
    cv2.imshow('Image Window', resize)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

    # Create a detectron2 config and a detectron2 DefaultPredictor to run inference on image
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
    cfg.OUTPUT_DIR = "./output"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # Print predictions
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # Draw instance predictions
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE  )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Resize and show image
    resize2 = ResizeWithAspectRatio(out.get_image()[:, :, ::-1], width=1280)
    cv2.imshow('Image Window', resize2)

    # Destroy Image with escape key
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
