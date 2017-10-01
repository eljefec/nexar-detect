# nexar-detect
Jeffrey Liu's work on the Nexar Car Detection Challenge

I trained 2 models: Faster R-CNN and RFCN 

Here are the papers:
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
* [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)

# Prerequisites
* Install conda environment with environment-gpu.yml

# Generating Detections in Nexet Format
* python generate_detections.py

# How to Train
* Faster R-CNN
  * In keras_frcnn_lib, run "sh train_nexet.sh"
* R-FCN
  * In RFCN_tensorflow, run "python main.py -dataset <path_to_folder> -annotation <path_to_csv>"

# Model Weights
https://drive.google.com/open?id=0BwUkl7_YvqGrWk5HSk4xSmFwa28
