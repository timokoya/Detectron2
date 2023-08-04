import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

# # The directory containing the source images
# data_path = "/home/timi/Fish_coco/train"

# # The path to the COCO labels JSON file
# labels_path = "/home/timi/Fish_coco/train_annotations.coco.json"

# # Import the dataset
# dataset = fo.Dataset.from_dir(
#     dataset_type=fo.types.COCODetectionDataset,
#     data_path=data_path,
#     labels_path=labels_path,
# )

# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split="validation",
#     label_types=["points","detections"],
#     classes=["Fish"],
#     max_samples=100,
#     dataset_name="fish-dataset",
# )
# dataset.persistent = True

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
        label_types=["points", "detections"],
        classes=["Fish"],
        dataset_name="Fish2",
    )

    for d in ["train", "val"]:
        view = dataset2.match_tags(d)
        DatasetCatalog.register("fiftyone_" + d, lambda view=view: get_fiftyone_dicts(view))
        MetadataCatalog.get("fiftyone_" + d).set(thing_classes=["Fish"])

    metadata = MetadataCatalog.get("fiftyone_train")
    # Remove other classes and existing tags
    dataset2.filter_labels("detections", F("label") == "Fish").save()
    dataset2.untag_samples("validation")

    four.random_split(dataset2, {"train": 0.8, "val": 0.2})

    dataset2.persistent = True
    session = fo.launch_app(dataset2)
    session.wait()