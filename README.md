# Fingerprint pore detection
This repository contains the original implementation of the fingerprint pore detection from ["Improving Fingerprint Pore Detection with a Small FCN"](https://arxiv.org/abs/1811.06846). It also contains code for repeating our evaluation protocol for pore detection in PolyU-HRF _GroundTruth_ and our unofficial reimplementation of ["A deep learning approach towards pore extraction for high-resolution fingerprint recognition"](https://ieeexplore.ieee.org/document/7952518/).

## PolyU-HRF dataset
The Hong Kong Polytechnic University (PolyU) High-Resolution-Fingerprint (HRF) Database is a high-resolution fingerprint dataset for fingerprint recognition. We ran all of our experiments in the PolyU-HRF dataset, so it is required to reproduce them. PolyU-HRF can be obtained by following the instructions from its authors [here](http://www4.comp.polyu.edu.hk/~biometrics/HRF/HRF_old.htm).

Assuming PolyU-HRF is inside a local directory named `polyu_hrf`, its internal organization must be as following in order to reproduce our experiments with the code in this repository as it is:
```
polyu_hrf/
  GroundTruth/
    PoreGroundTruth/
      PoreGroundTruthMarked/
      PoreGroundTruthSampleimage/
```

## Requirements
The code in this repository was tested for Ubuntu 16.04 and Python 3.5.2, but we believe any newer version of both will do.

We recomend installing Python's venv (tested for version 15.0.1) to run the experiments. To do it in Ubuntu 16.04:
```
sudo apt install python3-venv
```

Then, create and activate a venv:
```
python3 -m venv env
source env/bin/activate
```

To install the requirements either run, for CPU usage:
```
pip install -r cpu-requirements.txt
```
or run, for GPU usage, which requires the [Tensorflow GPU dependencies](https://www.tensorflow.org/install/gpu):
```
pip install -r gpu-requirements.txt
```

## Proposed FCN model
### Training
Throught our experiments, we will assume that PolyU-HRF is inside a local folder name `polyu_hrf`. To train a pore detection FCN with our best found parameters, run:
```
python3 -m train.fcn --polyu_dir_path polyu_hrf --log_dir_path log --dropout 0.2
```
This will create a folder inside `log` for the trained model's resources. We will call it `[det_model_dir]` for the rest of the instructions.

The options for training the detection net are:
```
usage: train/fcn.py [-h] --polyu_dir_path POLYU_DIR_PATH
                    [--learning_rate LEARNING_RATE] [--log_dir_path LOG_DIR_PATH]
                    [--dropout DROPOUT] [--tolerance TOLERANCE]
                    [--batch_size BATCH_SIZE] [--steps STEPS]
                    [--label_size LABEL_SIZE] [--label_mode LABEL_MODE]
                    [--patch_size PATCH_SIZE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --learning_rate LEARNING_RATE
                        learning rate
  --log_dir_path LOG_DIR_PATH
                        logging directory
  --dropout DROPOUT     dropout rate in last convolutional layer
  --tolerance TOLERANCE
                        early stopping tolerance
  --batch_size BATCH_SIZE
                        batch size
  --steps STEPS         maximum training steps
  --label_size LABEL_SIZE
                        pore label size
  --label_mode LABEL_MODE
                        how to convert pore coordinates into labels
  --patch_size PATCH_SIZE
                        pore patch size
  --seed SEED           random seed
```
for more details, refer to the code documentation.

### Validation
To evaluate a trained FCN model using the proposed post-processing, based on thresholding and non-maximum suppression, run:
```
python3 validate.fcn --polyu_dir_path polyu_hrf --model_dir_path log/[det_model_dir]
```
To evaluate a trained model using traditional post-processing, based on merging connected components, like we did for our ablation study, run:
```
python3 validate.fcn --polyu_dir_path polyu_hrf --model_dir_path log/[det_model_dir] --post traditional
```
The results will most likely differ from the ones reported in the paper. To reproduce those, read below about the trained models.

The options for validating the detection model are:
```
usage: validate/fcn.py [-h] --polyu_dir_path POLYU_DIR_PATH --model_dir_path
                       MODEL_DIR_PATH [--post POST] [--patch_size PATCH_SIZE]
                       [--results_path RESULTS_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --model_dir_path MODEL_DIR_PATH
                        logging directory
  --post POST           how to post-process detections. Can be either
                        'traditional' or 'proposed'
  --patch_size PATCH_SIZE
                        pore patch size
  --results_path RESULTS_PATH
                        path in which to save results
  --seed SEED           random seed

```

### Detecting pores in arbitrary input images
To detect pores in a arbitrary fingerprint image named `image.png` with the trained FCN model stored in a folder `log/model_dir`, simply run:
```
python3 detect.py --image_path image.png --model_dir_path log/model_dir
```

This script has the parameters that achieve best results for the trained model provided below as default, but they can be passed as arguments for each run. Other options for it are:
```
usage: detect.py [-h] --image_path IMAGE_PATH --model_dir_path MODEL_DIR_PATH
                 [--patch_size PATCH_SIZE] [--save_path SAVE_PATH]
                 [--prob_thr PROB_THR] [--inter_thr INTER_THR]

optional arguments:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
                        path to image in which to detect pores
  --model_dir_path MODEL_DIR_PATH
                        path from which to restore trained model
  --patch_size PATCH_SIZE
                        pore patch size
  --save_path SAVE_PATH
                        path to file in which detections should be saved
  --prob_thr PROB_THR   probability threshold to filter detections
  --inter_thr INTER_THR
                        nms intersection threshold
```

## Su _et al._'s (2017) CNN reimplementation model
### Training
To train a pore detection CNN as specified in ["A deep learning approach towards pore extraction for high-resolution fingerprint recognition"](https://ieeexplore.ieee.org/document/7952518/), run:
```
python3 -m train.cnn --polyu_dir_path polyu_hrf --log_dir_path log
```
Like it did for the FCN model, this will create a directory to store the trained model.
Options for this script are:
```
usage: train/cnn.py [-h] --polyu_dir_path POLYU_DIR_PATH
                    [--learning_rate LEARNING_RATE] [--log_dir_path LOG_DIR_PATH]
                    [--tolerance TOLERANCE] [--batch_size BATCH_SIZE]
                    [--steps STEPS] [--label_size LABEL_SIZE]
                    [--label_mode LABEL_MODE] [--patch_size PATCH_SIZE]
                    [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --learning_rate LEARNING_RATE
                        learning rate
  --log_dir_path LOG_DIR_PATH
                        logging directory
  --tolerance TOLERANCE
                        early stopping tolerance
  --batch_size BATCH_SIZE
                        batch size
  --steps STEPS         maximum training steps
  --label_size LABEL_SIZE
                        pore label size
  --label_mode LABEL_MODE
                        how to convert pore coordinates into labels
  --patch_size PATCH_SIZE
                        pore patch size
  --seed SEED           random seed
```

### Validation
To evaluate the trained model, run:
```
python3 -m validate.cnn --polyu_dir_path polyu_hrf --model_dir_path log/[cnn_model_dir]
```
The same considerations about reproducibility from FCN models are valid here. Read above and below for more info.
Options for validation here are:
```
usage: validate/cnn.py [-h] --polyu_dir_path POLYU_DIR_PATH --model_dir_path
                       MODEL_DIR_PATH [--patch_size PATCH_SIZE]
                       [--results_path RESULTS_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --model_dir_path MODEL_DIR_PATH
                        logging directory
  --patch_size PATCH_SIZE
                        pore patch size
  --results_path RESULTS_PATH
                        path in which to save results
  --seed SEED           random seed

```

## Pre-trained models and reproducing paper results
The trained models for the [FCN](https://drive.google.com/file/d/15GRs23KQVc_yhJzL1fVzZ8uI4-U-xqcg/view?usp=sharing) and [Su _et al._'s CNN reimplementation](https://drive.google.com/file/d/1Qb1s7g1kcMYPdkOoUjXOlc6iNn1cCrHq/view?usp=sharing) are required to ensure that you get the exact same results as those of the paper. After downloading them, follow the validation steps for each model, replacing model directory paths where appropriate.

## Evaluating arbitrary detections with the proposed protocol
To evaluate arbitrary detections with the proposed protocol, generate `txt` detection files in PolyU-HRF _GroundTruth_ format (1-indexed, "pixel_row pixel_column" integer coordinates, one detection per line) for the images in the protocol test set (the last ten images of _GroundTruth_ in lexicographic order) and put them in a folder, say `detections`. To compute the metrics using the evaluation protocol, run:
```
python3 -m validate.detections polyu_hrf/GroundTruth/PoreGroundTruth/PoreGroundTruthMarked detections
```
This will output the true detection rate (TDR), false detection rate (FDR), and corresponding F-score. For more details, read the paper.

## Reference
If you find the code in this repository useful for your research or use the proposed evaluation protocol, please consider citing:
```
@article{dahia2018improving,
  title={Improving Fingerprint Pore Detection with a Small FCN},
  author={Dahia, Gabriel and Segundo, Maur{\'\i}cio Pamplona},
  journal={arXiv preprint arXiv:1811.06846},
  year={2018}
}
```

## License
See the [LICENSE](LICENSE) file for details.
