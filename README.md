# Fingerprint pore detection
This repository contains the original implementation of the fingerprint pore detection from [Improving Fingerprint Pore Detection with a Small FCN]().

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

## Training the model
Throught our experiments, we will assume that PolyU-HRF is inside a local folder name `polyu_hrf`. To train a pore detection network with our best found parameters, run:
```
python3 train.py --polyu_dir_path polyu_hrf --log_dir_path log --dropout 0.2
```
This will create a folder inside `log` for the trained model's resources. We will call it `[det_model_dir]` for the rest of the instructions.

The options for training the detection net are:
```
usage: train.py [-h] --polyu_dir_path POLYU_DIR_PATH
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

## Validating the trained model
To evaluate the model trained above, run:
```
python3 validate --polyu_dir_path polyu_hrf --model_dir_path log/[det_model_dir]
```
The results will most likely differ from the ones reported in the paper. To reproduce those, read below about the trained models.

The options for validating the detection model are:
```
usage: validate.py [-h] --polyu_dir_path POLYU_DIR_PATH --model_dir_path
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

## Pre-trained models and reproducing paper results
The pre-trained [model]() is required to ensure that you get the exact same results as those of the paper. After downloading it, follow the validation steps, replacing `[det_model_dir]` where appropriate.

## Detecting pores in single images

## Reference
If you find the code in this repository useful for your research, please consider citing:
```
```

## License
See the [LICENSE](LICENSE) file for details.
