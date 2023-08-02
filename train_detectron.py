from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


register_coco_instances("fish_dataset_val", {}, "_annotations.coco.json", "/home/timi/Fish/valid")
register_coco_instances("fish_dataset_train", {}, "_annotations.coco.json", "/home/timi/Fish/train")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("fish_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2                    # "batch size" as commonly known in the deep learning community
cfg.SOLVER.BASE_LR = 0.00025                    # A good Learning Rate
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []                           # 300 iterations seems good enough for this fish dataset; you will need to train longer for a more robust dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # The fish dataset has 13 classes