from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os, cv2

import fiftyone as fo
import fiftyone.zoo as foz

import random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

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

    # Register the coco-format dataset 
    register_coco_instances("fish_dataset_train", {}, "/home/timi/Fish_coco/train_annotations.coco.json", "/home/timi/Fish_coco/train")
    register_coco_instances("fish_dataset_val", {}, "/home/timi/Fish_coco/val_annotations.coco.json", "/home/timi/Fish_coco/valid")
    
    # Set config settings
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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13            # The fish dataset has 13 classes

    # Get Metadata
    # nuts_metadata = MetadataCatalog.get('fish_dataset_train')
    # dataset_dicts = DatasetCatalog.get("fish_dataset_train")


    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     img_resize = ResizeWithAspectRatio(img, width=1280)
    #     visualizer = Visualizer(img_resize[:, :, ::-1], metadata=nuts_metadata , scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     plt.imshow(vis.get_image()[:, :, ::-1])
    #     print("Here!")

    #     # Destroy Image with escape key
    #     k = cv2.waitKey(0) & 0xFF
    #     if k == 27:
    #         cv2.destroyAllWindows()

    # Initiate training
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg.DATASETS.TEST = ("fish_dataset_val")
    # predictor = DefaultPredictor(cfg)

    # img = cv2.imread("demo/input.jpg")
    # outputs = predictor(img)

    # for d in random.sample(dataset_dicts, 1):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1],
    #                     metadata=metadata,
    #                     scale=0.8)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite('demo/output_retrained.jpg', out.get_image()[:, :, ::-1])