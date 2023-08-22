import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import os, cv2

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

# Convert outputs from detectron2 to FiftyOne format, then add them to our FiftyOne dataset.
def detectron_to_fo(outputs, img_w, img_h):
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    for pred_box, score, c, mask in zip(
        instances.pred_boxes, instances.scores, instances.pred_classes, instances.pred_masks,
    ):
        x1, y1, x2, y2 = pred_box
        fo_mask = mask.numpy()[int(y1):int(y2), int(x1):int(x2)]
        bbox = [float(x1)/img_w, float(y1)/img_h, float(x2-x1)/img_w, float(y2-y1)/img_h]
        detection = fo.Detection(label="Fish", confidence=float(score), bounding_box=bbox, mask=fo_mask)
        detections.append(detection)

    return fo.Detections(detections=detections)

if __name__ == '__main__':

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

    dataset2.persistent = True
    session = fo.launch_app(dataset2)

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

# Load saved model for prediction
cfg.OUTPUT_DIR = "./output"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Generate predictions on each sample in the validation set
val_view = dataset2.match_tags("val")
dataset_dicts = get_fiftyone_dicts(val_view)
predictions = {}
for d in dataset_dicts:
    img_w = d["width"]
    img_h = d["height"]
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    detections = detectron_to_fo(outputs, img_w, img_h)
    predictions[d["image_id"]] = detections
    
# Visualize predictions
    v = Visualizer(img[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE   
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the predictions to CPU from the GPU
    resize = ResizeWithAspectRatio(v.get_image()[:, :, ::-1], width=1280)
    #cv2.imshow('Image Window', resize)         # display image with predictions
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

########## Prediction on a single specific image ##########
# Load Image
    im = cv2.imread("/home/timi/Desktop/11.jpeg")
    resize = ResizeWithAspectRatio(im, width=1280)

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Print predictions
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# Draw instance predictions
v = Visualizer(im[:, :, ::-1], metadata, scale=0.8, instance_mode=ColorMode.IMAGE)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Resize and show image
resize2 = ResizeWithAspectRatio(out.get_image()[:, :, ::-1], width=1280)
cv2.imshow('Image Window', resize2)

# Destroy Image with escape key
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
###########

dataset2.set_values("predictions", predictions, key_field="id")

# Evaluate and view predictions
results = dataset2.evaluate_detections(
    "predictions",
    gt_field="segmentations",
    eval_key="eval",
    use_masks=True,
    compute_mAP=True,
)

# Print Mean Average precision
print('Mean Average Precision: ' + str(results.mAP()))

# Plot a PR curve
plot = results.plot_pr_curves()
plot.show()

# Generate a confusion matrix for the specified class
plot = results.plot_confusion_matrix(classes=["Fish"])
plot.show()

# View Fiftyone portal with filter labels
session.view = dataset2.filter_labels("predictions", (F("eval") == "fp") & (F("confidence") > 0.8))