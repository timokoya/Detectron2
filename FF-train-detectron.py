import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os

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

if __name__ == '__main__':

    dataset2 = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["points", "segmentations", "detections"],
        classes=["Fish"],
        dataset_name="Fishv8",
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

    # dataset_dicts = get_fiftyone_dicts(dataset2.match_tags("train"))
    # ids = [dd["image_id"] for dd in dataset_dicts]
    # dataset2.save()
    dataset2.persistent = True
    # view = dataset2.select(ids)
    # session = fo.launch_app(view)
    # session = fo.launch_app(dataset2)
    
    # session = fo.launch_app(dataset2)
    # session.wait()

# Create a detectron2 config and a detectron2 DefaultPredictor to run inference on image
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("fiftyone_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this Fish dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this Fish dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (Fish). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()