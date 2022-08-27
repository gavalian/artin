#!/usr/bin/env python3

from json import load
import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse
import os.path
from sklearn.metrics import confusion_matrix
from termcolor import colored
import os.path
from data import *

from models.ExtraTreesModel import ExtraTreesModel
from models.XGBoostModel import XGBoostModel
from models.MlpModel import MlpModel

## Script entry point
def main():

    args = parse_arguments()
    print(args)

    subroutine = get_subroutine(args)
    subroutine(args)

def parse_arguments():
    """
    Parse CLI arguments and return an object containing the values.
    """

    # Create main program parser
    parser = argparse.ArgumentParser(description="CRTC-JLab Machine Learning CLI")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "subprogram"

    # Create training subprogram parser
    parser_train = subparsers.add_parser("train", help="Train a model, perform testing, and serialize it.")
    parser_train._action_groups.pop()
    parser_train_required_args = parser_train.add_argument_group("Required arguments")
    parser_train_optional_args = parser_train.add_argument_group("Optional arguments")
    parser_train_required_args.add_argument("--training-file", "-t", required=True, help="Path to the file containing the training data.", dest="training_file_path")
    # parser_train_required_args.add_argument("--testing-file", "-e", required=True, help="Path to the file containing the testing data.", dest="testing_file_path")
    parser_train_required_args.add_argument("--charge-value", "-c", type=int, required=True, choices=[-1, 1], help="Whether to use the negative (-1) or positive (1) charged data.", dest="charge_train")
    parser_train_required_args.add_argument("--sector-values", "-s", required=True,  help="Comma separated values of sectors to use for training.", dest="sectors_train")
    parser_train_required_args.add_argument("--out-model", "-m", required=True, help="Name of the file in which to save the model.", dest="output_model_path")
    parser_train_required_args.add_argument("--model-type", choices=["et", "mlp", "xgb"], required=True, help="The type of the model to train.", dest="model_type")
    parser_train_optional_args.add_argument("--activation-functions", "-af", required=False, type=str, default="tanh,linear", help="Main activation function used for all hidden layers and output activation function, comma-separated; supports all activation functions available in Tensorflow Keras API", dest="activation_functions")
    
    # TODO replace with --model-config
    # parser_train_optional_args.add_argument("--epochs", required=False, type=int, default="10", help="How many training epochs to go through.", dest="training_epochs")
    # parser_train_optional_args.add_argument("--batchSize", required=False, type=int, default="16", help="Size of the training batch.", dest="training_batch_size")
    # parser_train_optional_args.add_argument("--testing-batchSize", required=False, type=int, default="16", help="Size of the evaluation batch.", dest="evaluation_batch_size")

    # Create evaluation subprogram parser
    parser_test = subparsers.add_parser("test", help="Load a model for testing.")
    parser_test._action_groups.pop()
    parser_test_required_args = parser_test.add_argument_group("Required arguments")
    parser_test_optional_args = parser_test.add_argument_group("Optional arguments")
    parser_test_required_args.add_argument("--testing-file", "-e", required=True, help="Path to the file containing the testing data.", dest="testing_file_path")
    parser_test_required_args.add_argument("--charge-value", "-c", type=int, required=True, choices=[-1, 1], help="Whether to use the negative (-1) or positive (1) charged data.", dest="charge_test")
    parser_test_required_args.add_argument("--sector-values", "-s", required=True,  help="Comma separated values of sectors to use for training.", dest="sectors_test")

    parser_test_required_args.add_argument("--model", "-m", required=True, help="The name of the file from which to load the model.", dest="model_path")
    parser_test_required_args.add_argument("--model-type", choices=["et", "mlp", "xgb"], required=True, help="The type of the model to load.", dest="model_type")
    # parser_test_optional_args.add_argument("--batchSize", required=False, type=int, default="16", help="Size of the evaluation batch.", dest="evaluation_batch_size")
    parser_test_optional_args.add_argument("--prediction-file", "-p", required=False, type=str,  help="File to store predictions.", dest="prediction_path")

    # TODO disable predict until we learn about the C interface
    # Create prediction subprogram parser
    parser_predict = subparsers.add_parser("predict", help="Load a model and use it for predictions.")
    parser_predict._action_groups.pop()
    parser_predict_required_args = parser_predict.add_argument_group("Required arguments")
    parser_predict_optional_args = parser_predict.add_argument_group("Optional arguments")
    parser_predict_required_args.add_argument("--prediction-file", "-p", required=True, help="Path to the file containing the prediction data.", dest="prediction_file_path")
    # parser_predict_required_args.add_argument("--prediction-output-file", "-o", required=True, help="File in which to save the predictions.", dest="prediction_output_file")
    parser_predict_required_args.add_argument("--model", "-m", required=True, help="The name of the file from which to load the model.", dest="model_path")
    parser_predict_required_args.add_argument("--model-type", choices=["cnn", "mlp", "et"], required=True, help="The type of the model to load.", dest="model_type")
    parser_predict_optional_args.add_argument("--batchSize", required=False, type=int, default="32", help="Size of the prediction batch.", dest="prediction_batch_size")
    parser_predict_optional_args.add_argument("--c-lib", required=False, default="libc_reader.so", help="Path to the C library reader interface", dest="c_library")
    parser_predict_optional_args.add_argument("--softmax", "-s", action='store_true', help="Use this flag to pass the output probabilities from softmax before returning them", dest="use_softmax_output")

    return parser.parse_args()

def enumerate_data_files_in_dir(dir_path):
    return glob.glob(f'{dir_path}/*.txt')

def read_input_data(input_type, args) -> dict:
    """
    Reads all of the SVM files in the training and testing directories,
    concatenates them, and creates a dictionary that references them.

    Args:
        input_type: Type of reading and processing of the input for training, evaluation, or prediction.
        args: The object that contains all the parsed CLI arguments. 

    Returns:
        A dictionary containing the read and processed input data. This includes the features and targets when appropriate.
    """

    if input_type == "train":
        # Read training and testing data
        chargefilter = [args.charge_train]
        secfilter = [int(n) for n in args.sectors_train.split(',')]
        data, _ = load_data(args.training_file_path, chargefilter, secfilter)
        X_train, y_train = data[:, 2:8], data[:, 8:]
        # X_test, y_test = read_concat_svm_files([args.testing_file_path], int(args.num_features))

        # X_test_segmented, y_test_segmented = segment_svm_data(X_test, y_test)
        # total_test_samples = len(y_test_segmented)
        
        return {
            "training": {"data": X_train, "labels": y_train}
            # "training": {"data": X_train, "labels": y_train, "epochs": args.training_epochs, "batch_size": args.training_batch_size}, 
            # "testing": {"data": X_test, "labels": y_test,"batch_size": args.evaluation_batch_size},
            # "testing_segmented": {"data": X_test_segmented, "labels": y_test_segmented},
            # "total_test_samples": total_test_samples,
            # "num_classes": int(np.max(y_train)) + 1
        }

    elif input_type == "test":
        # Read testing data
        chargefilter = [args.charge_test]
        secfilter = [int(n) for n in args.sectors_test.split(',')]
        data, filtered_raw_input = load_data(args.testing_file_path, chargefilter, secfilter)
        X_test, y_test = data[:, 2:8], data[:, 8:]
        # X_test, y_test = read_concat_svm_files([args.testing_file_path], int(args.num_features))

        # X_test_segmented, y_test_segmented = segment_svm_data(X_test, y_test)

        # total_test_samples = len(y_test_segmented)

        return {
            "testing": {"data": X_test, "labels": y_test},
            "prediction_path" : args.prediction_path,
            "filtered_input" : filtered_raw_input 
            # "testing": {"data": X_test, "labels": y_test, "batch_size": args.evaluation_batch_size},
            # "testing_segmented": {"data": X_test_segmented, "labels": y_test_segmented},
            # "total_test_samples": total_test_samples,
            # "num_classes": int(np.max(y_test)) + 1
        }
    
    #elif input_type == "predict":
    #    return {"prediction": {"data": prediction_data}}

    else:
        print(colored("Error: Wrong input type.", "red"))
        quit()

def print_training_report(training_dict):
    training_time = training_dict["training_time"]
    training_accuracy = training_dict["accuracy_training"]

    print("\nTraining Report")
    print("================================")
    print(f'{colored("Training accuracy:", "blue")} {training_accuracy}')
    print(f'{colored("Training time:", "blue")} {training_time}s')

def print_testing_report(testing_dict):
    testing_accuracy = testing_dict["accuracy_testing"]
    testing_prediction_time = testing_dict["testing_prediction_time"]


    print("\nTesting Report")
    print("================================")
    print(f'{colored("Testing accuracy:", "blue")} {testing_accuracy}')
    print(f'{colored("Testing prediction time:", "blue")} {testing_prediction_time}s')

def train_model(args):
    """
    Trains the model with the input data specified in the CLI arguments followed by an evaluation of the model.

    Args:
        args: The object that contains all the parsed CLI arguments. 
    """

    print(colored("\nReading input data...", "green"))
    training_files = []
    model  = None

    input_dict = read_input_data("train", args)

    if args.model_type == "et":
        model = ExtraTreesModel()
    elif args.model_type == "xgb":
        model = XGBoostModel()
    elif (args.model_type == "mlp"):
        model = MlpModel()
    #     elif (args.model_type == "cnn"):
    #         model = CnnModel(input_dict)

    if (args.model_type == "mlp"):
        afs = [str(n) for n in args.activation_functions.split(',')]
        af1 = afs[0]
        af2 = afs[1]
        model.build_new_model(af1, af2)
    else:
        model.build_new_model()

    training_dict = model.train(input_dict)
    print_training_report(training_dict)

    model.save_model(args.output_model_path)

def write_predictions(input_dict, predictions):
    pred_path = input_dict["prediction_path"]
    filter_input = input_dict["filtered_input"]
    
    if pred_path:
        print(colored("\nWriting predictions to {} ...".format(pred_path), "green"))
        with open(pred_path, "w+") as f:
            for i, line in enumerate(filter_input):
                f.write(line + ":\t{} {} {}\n".format(predictions[i][0], predictions[i][1], predictions[i][2]) )
        

def test_model(args):
    """
    Evaluates the model with the input data specified in the CLI arguments.

    Args:
        args: The object that contains all the parsed CLI arguments. 
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("test", args)

    model = None

    if (args.model_type == "et"):
        model = ExtraTreesModel()
    elif (args.model_type == "xgb"):
        model = XGBoostModel()
    elif (args.model_type == "mlp"):
        model = MlpModel()
    
    model.load_model(args.model_path)

    testing_dict = model.test(input_dict)

    print_testing_report(testing_dict)
    
    write_predictions(input_dict, testing_dict["testing_predictions"])

# # TODO implement
# def predict(args):
#     from c_py_interface.c_interface import CLibInterface

#     if (args.model_type == "et"):
#         model = ExtraTreesModel()
#     elif (args.model_type == "mlp"):
#         model = MlpModel()
#     elif (args.model_type == "cnn"):
#         model = CnnModel()

#     model.load_model(args.model_path)

#     clib = CLibInterface(args.c_library)
#     clib.open_file(args.prediction_file_path)
    
#     dict_input = {}
#     dict_input['softmax'] = args.use_softmax_output
#     dict_input['prediction'] = {}
    
#     while(True):
#         N = clib.read_next()

#         if (N < 0):
#             break

#         for i in range(0,N):
#             k = clib.count_roads(i)
#             if (k>0):
#                 roads = clib.read_roads(k,i)
#                 dict_input['prediction']['data'] = roads
#                 pred = model.predict(dict_input)
#                 preds = pred['predictions']
#                 clib.write_roads(preds,i)

#     # TODO continue implementation
#     # raise NotImplementedError

def get_subroutine(args):
    """
    Processes the CLI subprogram utilized and return the matching function for training, evaluation, or prediction.

    Args:
        args: The object that contains all the parsed CLI arguments. 

    Returns:
        The functions corresponding the the CLI subprogram utilized.
    """

    if args.subprogram == "train":
        print(colored("Executing training subprogram.", "green"))
        return train_model
    elif args.subprogram == "test":
        print(colored("Executing testing subprogram.", "green"))
        return test_model
    # elif args.subprogram == "predict":
    #     print(colored("Executing prediction subprogram.", "green"))
    #     return predict
    else:
        print(colored("Fatal Error: Wrong subprogram specified.", "red"))
        quit()

if __name__ == "__main__": main()
