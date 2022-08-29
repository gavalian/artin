# Track Parameter Estimation

This project aims to estimate track parameters, including momentum, polar and azimuthal angles, from the reconstructed cluster positions. Originally developed for CLAS12 Drift Chambers, it uses cluster positions in each of the six super-layers to estimate the aforementioned parameters.

# Requirements
* python 3
* anacoda

# Installation

Create a new anaconda environment using the provide YML file:

```bash
conda env create -f conda_environment.yml
```

# Usage

**Make sure to activate the conda environment.**

Execute ``` ml-cli.py ``` with the necessary command-line arguments (presented below).

## Primary Arguments

``` bash
ml-cli.py [-h] {train, test} ...
```

positional arguments:
```
{train, test}
train       # Train a model on a dataset and store it.
test        # Load a model and perform testing/predictions
```

The script makes use of subcommands.

To get hel for any subcommand, such as the ```train``` subcommand for example, execute ```ml-cli.py train -h```

## Train Subcommand Arguments

```bash
usage: ml-cli.py train [-h] --training-file TRAINING_FILE_PATH --charge-value
                       {-1,1} --sector-values SECTORS_TRAIN --out-model
                       OUTPUT_MODEL_PATH --model-type {et,mlp,xgb}
                       [--activation-functions ACTIVATION_FUNCTIONS]

Required arguments:
  --training-file TRAINING_FILE_PATH, -t TRAINING_FILE_PATH
                        Path to the file containing the training data.
  --charge-value {-1,1}, -c {-1,1}
                        Whether to use the negative (-1) or positive (1)
                        charged data.
  --sector-values SECTORS_TRAIN, -s SECTORS_TRAIN
                        Comma separated values of sectors to use for training.
  --out-model OUTPUT_MODEL_PATH, -m OUTPUT_MODEL_PATH
                        Name of the file in which to save the model.
  --model-type {et,mlp,xgb}
                        The type of the model to train.

Optional arguments:
  --activation-functions ACTIVATION_FUNCTIONS, -af ACTIVATION_FUNCTIONS
                        Main activation function used for all hidden layers
                        and output activation function, comma-separated;
                        supports all activation functions available in
                        Tensorflow Keras API
```

## Test Subcommand Arguments

```bash
usage: ml-cli.py test [-h] --testing-file TESTING_FILE_PATH --charge-value
                      {-1,1} --sector-values SECTORS_TEST --model MODEL_PATH
                      --model-type {et,mlp,xgb}
                      [--prediction-file PREDICTION_PATH]

Required arguments:
  --testing-file TESTING_FILE_PATH, -e TESTING_FILE_PATH
                        Path to the file containing the testing data.
  --charge-value {-1,1}, -c {-1,1}
                        Whether to use the negative (-1) or positive (1)
                        charged data.
  --sector-values SECTORS_TEST, -s SECTORS_TEST
                        Comma separated values of sectors to use for training.
  --model MODEL_PATH, -m MODEL_PATH
                        The name of the file from which to load the model.
  --model-type {et,mlp,xgb}
                        The type of the model to load.

Optional arguments:
  --prediction-file PREDICTION_PATH, -p PREDICTION_PATH
                        File to store predictions.

```

# Dataset
The dataset used to test and validate this project can be found [here](https://userweb.jlab.org/~gavalian/ML/2022/RegressionFinal/).