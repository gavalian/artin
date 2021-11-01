# Track Classification Project
This project aims to classify tracks from a large list
of track candidates based on the hits in the tracking detector.
Originally developed for CLAS12 Drift Chambers, it uses segment
(cluster) positions in each of the 6 super-layers to form a track
candidate and return the probability of it being a good track.

Paper publication of this work is available [here](https://arxiv.org/abs/2008.12860).

# Requirements
* Anaconda (tested with Python 3.7)

# Installation
Create a new Anaconda environment using the YML file provided:

`conda env create -f conda_environment.yml`

# Usage
**Make sure to activate the conda environment.**

Execute `ml-cli.py` under `src/` with the necessary CLI arguments.

## Primary Arguments
```
ml-cli.py [-h] {train,test,predict} ...
```
positional arguments:
```
{train,test,predict}
train               Train a model, perform testing, and serialize it.
test                Load a model for testing.
predict             Load a model and use it for predictions.

optional arguments:
-h, --help            show this help message and exit 
```

The script makes use of subcommands. 

To get help for any subcommand, such as the `train` subcommand for example, execute `./ml-cli.py train -h`.

## Train Subcommand Arguments
```
ml-cli.py train [-h] --training-file TRAINING_FILE_PATH --testing-file
                       TESTING_FILE_PATH --num-features {6,36,4032}
                       --out-model OUTPUT_MODEL_PATH --model-type {cnn,mlp,et}
                       [--epochs TRAINING_EPOCHS]
                       [--batchSize TRAINING_BATCH_SIZE]
                       [--testing-batchSize EVALUATION_BATCH_SIZE]

Required arguments:
  --training-file TRAINING_FILE_PATH, -t TRAINING_FILE_PATH
                        Path to the file containing the training data.
  --testing-file TESTING_FILE_PATH, -e TESTING_FILE_PATH
                        Path to the file containing the testing data.
  --num-features {6,36,4032}, -f {6,36,4032}
                        Path to the directory containing the testing data.
  --out-model OUTPUT_MODEL_PATH, -m OUTPUT_MODEL_PATH
                        Name of the file in which to save the model.
  --model-type {cnn,mlp,et}
                        The type of the model to train.

Optional arguments:
  --epochs TRAINING_EPOCHS
                        How many training epochs to go through.
  --batchSize TRAINING_BATCH_SIZE
                        Size of the training batch.
  --testing-batchSize EVALUATION_BATCH_SIZE
                        Size of the evaluation batch.
```

## Test Subcommand Arguments
```
ml-cli.py test [-h] --testing-file TESTING_FILE_PATH --num-features
                      {6,36,4032} --model MODEL_PATH --model-type {cnn,mlp,et}
                      [--batchSize EVALUATION_BATCH_SIZE]

Required arguments:
  --testing-file TESTING_FILE_PATH, -e TESTING_FILE_PATH
                        Path to the file containing the testing data.
  --num-features {6,36,4032}, -f {6,36,4032}
                        Path to the directory containing the testing data.
  --model MODEL_PATH, -m MODEL_PATH
                        The name of the file from which to load the model.
  --model-type {cnn,mlp,et}
                        The type of the model to load.

Optional arguments:
  --batchSize EVALUATION_BATCH_SIZE
                        Size of the evaluation batch.
```

## Predict Subcommand Arguments
```
ml-cli.py predict [-h] --prediction-file PREDICTION_FILE_PATH --model
                         MODEL_PATH --model-type {cnn,mlp,et}
                         [--batchSize PREDICTION_BATCH_SIZE]
                         [--c-lib C_LIBRARY] [--softmax]

Required arguments:
  --prediction-file PREDICTION_FILE_PATH, -p PREDICTION_FILE_PATH
                        Path to the file containing the prediction data.
  --model MODEL_PATH, -m MODEL_PATH
                        The name of the file from which to load the model.
  --model-type {cnn,mlp,et}
                        The type of the model to load.

Optional arguments:
  --batchSize PREDICTION_BATCH_SIZE
                        Size of the prediction batch.
  --c-lib C_LIBRARY     Path to the C library reader interface
  --softmax, -s         Use this flag to pass the output probabilities from
                        softmax before returning them
```

# Background
## Problem

Traditional tracking algorithms are computationally intensive, especially for high luminosity experiments with multi-track final states where all combinations of segments in drift chambers have to be considered for firing best track candidates. At high luminosity the number of random segments (unrelated to the tracks) are increasing and as a result the number of possible combinations also increases, making the whole process longer.

![Tracking](https://crtc.cs.odu.edu/images/5/5f/Tracking.png)

### Goal

Using machine learning one can recognize the patterns that are valid in order to find the correct track faster. 
The model will be trained on real pre-labeled data and as an outcome, it will
be able to label track combinations as valid or not.

### Data

The drift chambers consist of 6 layers, of 6 wires, of 112 sensors each for a total of 4032 sensors (see picture below). The data provided let us know whether a sensor
has detected a hit or not. Those detections might be part of the trajectory that we want to track or can be irrelevant (noise). The labeled data consist of all the possible combinations that form a track as rows (events) and the state of each sensor (detected something or not) as columns (features). The label provides information on whether a combination produces the valid track or not.
An example of the input to be used by the model can be found in [https://userweb.jlab.org/~gavalian/ML/2021/Classifier].

![Drift Chambers](https://crtc.cs.odu.edu/images/b/bf/Drift_Chambers.png)

### Data Samples

The testing data are given in samples. A sample consists of multiple rows, the first row which is a valid track (labeled as 1) and all the invalid ones (labeled as 0) that come after it, the next valid track represents the beginning of a new sample. For example, consider the following dataset:

 | labels |     features     |
 |--------|------------------|
 |   1    |   x11 x12 x13 ...|
 |   0    |   x21 x22 x23 ...|
 |   0    |   x31 x32 x33 ...|
 |   0    |   x41 x42 x43 ...|
 |   1    |   x51 x52 x53 ...|
 |   0    |   x61 x62 x63 ...|
 
 
This dataset contains two samples:
          
           Sample 1
 | labels |     features     |   
 |--------|------------------|   
 |   1    |   x11 x12 x13 ...|   
 |   0    |   x21 x22 x23 ...|   
 |   0    |   x31 x32 x33 ...|
 |   0    |   x41 x42 x43 ...|
 
           Sample 2
 | labels |     features     |
 |--------|------------------|
 |   1    |   x51 x52 x53 ...|
 |   0    |   x61 x62 x63 ...|

### Predictions and Accuracy

A prediction is performed with respect to a sample as a whole and is considered correct or not based on whether the model was able to identify which of the rows in a sample is a valid track. The accuracy is then measured by counting the number of samples where the model correctly identified the valid track and dividing this number by the total number of samples. In the previous example, if we have the following predicitions:

           Sample 1
 | labels |     features     | Prediction |  
 |--------|------------------|------------|   
 |   **1**    |   x11 x12 x13 ...|      **1**     |
 |   0    |   x21 x22 x23 ...|      0     |
 |   0    |   x31 x32 x33 ...|      1     |
 |   0    |   x41 x42 x43 ...|      0     |
  
           Sample 2
 | labels |     features     | Prediction | 
 |--------|------------------|------------| 
 |   **1**    |   x51 x52 x53 ...|      **0**     |
 |   0    |   x61 x62 x63 ...|      1     |
 

The model has correctly predicted the valid track of the 1st sample, but not for the second. So the accuracy will be 50%. Notice that even though we have also mispredicted some invalid tracks as valid ones this does not hurt accuracy since we only care about not losing a valid track.

### Possible Issue with Dataset

In the dataset of 2019-10-1 we have at least one experiment that looks to be incorrectly labeled. The track can be seen on the picture below.

![Incorrect Label Track](https://crtc.cs.odu.edu/images/2/2e/Incorrect_label_track.png)

## Phase 2
In Phase 2 we want to use Recurrent Neural Networks (RNNs) to generate data based on previous observations. Specifically, the RNNs will be trained using the observations on the first 24 layers of the drift chambers to predict the remaining 12. This prediction will be used to narrow down the search space of the last two segments, the ones which draw most of the noise.

### Goal
The goal of this phase is to predict the track that the particle followed in the last 2 segments of the drift chamber. Having this kind of information, the search space will be narrow enough, so that the models built in the previous phase will be able to make correct inference about which track is the valid one.

![RNN ex 3210 labeled](https://crtc.cs.odu.edu/images/7/77/RNN_ex_3210_labeled.png)

### Data
For this phase, only the valid tracks are taken into account. The input to the RNNs will be the first 24 values of each valid row (with label 1) in the 36cols/row format as data and the rest columns will be used as the prediction targets.

For example, a dataset like this:

 1 1:29.00000 2:30.00000 3:29.00000 4:29.00000 5:29.00000 6:0.00000 7:30.00000 8:30.00000 9:29.50000 10:30.00000 11:29.00000 12:30.00000 ...
 0 1:23.00000 2:23.00000 3:23.00000 4:23.00000 5:22.00000 6:23.00000 7:26.00000 8:27.00000 9:26.00000 10:26.00000 11:26.00000 12:26.00000 ...
 1 1:12.00000 2:12.50000 3:12.00000 4:12.00000 5:11.00000 6:11.50000 7:13.00000 8:13.00000 9:13.00000 10:13.00000 11:12.00000 12:12.00000 ...
 0 1:5.00000 2:5.00000 3:4.00000 4:0.00000 5:4.00000 6:4.00000 7:13.00000 8:13.00000 9:13.00000 10:13.00000 11:12.00000 12:12.00000 ...
 1 1:64.00000 2:64.00000 3:64.00000 4:64.00000 5:64.00000 6:64.00000 7:71.00000 8:71.00000 9:71.00000 10:71.00000 11:71.00000 12:71.00000 ...
 0 1:54.00000 2:54.00000 3:54.00000 4:54.00000 5:54.00000 6:54.00000 7:60.00000 8:60.00000 9:60.00000 10:60.00000 11:60.00000 12:60.00000 ...
 1 1:23.00000 2:23.00000 3:23.00000 4:23.00000 5:22.00000 6:22.00000 7:26.00000 8:26.00000 9:26.00000 10:26.00000 11:25.00000 12:26.00000 ...
 0 1:14.00000 2:15.00000 3:14.00000 4:14.00000 5:13.00000 6:14.00000 7:17.00000 8:17.00000 9:17.00000 10:17.00000 11:16.00000 12:16.00000 ...
 1 1:55.00000 2:55.00000 3:55.00000 4:56.00000 5:55.00000 6:55.50000 7:59.00000 8:60.00000 9:59.00000 10:60.00000 11:59.00000 12:60.00000 ...
 0 1:38.00000 2:39.00000 3:38.00000 4:39.00000 5:39.00000 6:40.00000 7:43.00000 8:44.00000 9:43.00000 10:44.00000 11:44.00000 12:45.00000 ...
 1 1:12.00000 2:12.00000 3:12.00000 4:12.00000 5:11.00000 6:11.00000 7:14.00000 8:14.00000 9:13.00000 10:13.00000 11:12.50000 12:13.00000 ...
 0 1:12.00000 2:12.00000 3:12.00000 4:12.00000 5:11.00000 6:11.00000 7:0.00000 8:18.00000 9:0.00000 10:17.00000 11:17.00000 12:17.00000 ...
 1 1:43.00000 2:44.00000 3:43.00000 4:43.00000 5:43.00000 6:43.00000 7:43.00000 8:43.00000 9:43.00000 10:43.00000 11:43.00000 12:43.00000 ...
 0 1:32.00000 2:33.00000 3:32.00000 4:33.00000 5:32.00000 6:33.00000 7:34.00000 8:34.00000 9:34.00000 10:34.00000 11:34.00000 12:34.00000 ...
 1 1:25.00000 2:26.00000 3:25.00000 4:25.00000 5:25.00000 6:25.00000 7:25.00000 8:26.00000 9:25.00000 10:25.00000 11:24.00000 12:25.00000 ...

Will be converted to:

 1:29.00000 2:30.00000 3:29.00000 4:29.00000 5:29.00000 6:0.00000 7:30.00000 8:30.00000 9:29.50000 10:30.00000 11:29.00000 12:30.00000 ...
 1:12.00000 2:12.50000 3:12.00000 4:12.00000 5:11.00000 6:11.50000 7:13.00000 8:13.00000 9:13.00000 10:13.00000 11:12.00000 12:12.00000 ...
 1:64.00000 2:64.00000 3:64.00000 4:64.00000 5:64.00000 6:64.00000 7:71.00000 8:71.00000 9:71.00000 10:71.00000 11:71.00000 12:71.00000 ...
 1:23.00000 2:23.00000 3:23.00000 4:23.00000 5:22.00000 6:22.00000 7:26.00000 8:26.00000 9:26.00000 10:26.00000 11:25.00000 12:26.00000 ...
 1:55.00000 2:55.00000 3:55.00000 4:56.00000 5:55.00000 6:55.50000 7:59.00000 8:60.00000 9:59.00000 10:60.00000 11:59.00000 12:60.00000 ...
 1:12.00000 2:12.00000 3:12.00000 4:12.00000 5:11.00000 6:11.00000 7:14.00000 8:14.00000 9:13.00000 10:13.00000 11:12.50000 12:13.00000 ...
 1:43.00000 2:44.00000 3:43.00000 4:43.00000 5:43.00000 6:43.00000 7:43.00000 8:43.00000 9:43.00000 10:43.00000 11:43.00000 12:43.00000 ...
 1:25.00000 2:26.00000 3:25.00000 4:25.00000 5:25.00000 6:25.00000 7:25.00000 8:26.00000 9:25.00000 10:25.00000 11:24.00000 12:25.00000 ...

The features, in this case, are the first 24 columns, while the prediction targets are the rest 12, e.g.:

Training features:

 1:29.00000 2:30.00000 3:29.00000 4:29.00000 5:29.00000 6:0.00000 7:30.00000 8:30.00000 9:29.50000 10:30.00000 11:29.00000 12:30.00000 13:25.00000 14:26.00000 15:25.00000 16:25.00000 17:24.00000 18:0.00000 19:24.00000 20:24.00000 21:24.00000 22:24.00000 23:23.00000 24:23.00000 

Prediction targets:
 25:14.50000 26:14.50000 27:13.50000 28:14.00000 29:13.00000 30:13.00000 31:13.00000 32:13.00000 33:12.00000 34:12.00000 35:11.00000 36:11.00000
