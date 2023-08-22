# Detectron2
Required Installation
---------------------

- Install Conda

    https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

- Install Torch:

    [GPU] pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

    [CPU] pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

- Install Detectron2: 

    $ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  
- Install Fiftyone: 

    $ pip install fiftyone

- Install ipywidgets: 

    $ pip install ipywidgets

- Install shapely: 

    $ pip install shapely

Useful Tutorials
----------------

- Run Inference on Images or Videos, with an existing detectron2 model:

    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=QHnVupBBn9eR
  
- Training and Evaluating FiftyOne Datasets with Detectron2:

    https://docs.voxel51.com/tutorials/detectron2.html


Scripts
-------

- FF-train-detectron.py: Train Detectron2 with a dataset loaded via Fiftyone

- predict-detectron.py: Inference on an image using a model from the Model-Zoo

- test-FF.py: Test script to load a sample fiftyone dataset and view via the fiftyone portal

- train_detectron.py: Train Detectron2 with a obscure fish dataset in COCO format and make inference

Solved Problems
---------------

- (training) RuntimeError: CUDA error: out of memory (after multiple consecutive training attempts)

    Simply restart the kernel

Image
-----

- Before.png: Inference on Image with "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"

- After.png: Inference on Image with "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml" trained with a fish dataset extracted from Google Open Images Dataset V7

- original.jpg: Original Image

Dataset
-------

Fiftyone was used to extract a fish detection dataset from Google Open Images v7

https://storage.googleapis.com/openimages/web/index.html


Training
--------

Approx Time: 2.30 min