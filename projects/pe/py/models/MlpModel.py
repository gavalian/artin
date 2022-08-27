from timeit import default_timer as timer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from termcolor import colored
import numpy as np
import math

def NormalizeData(data, min=None, max=None):
    if min is None and max is None:
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        return (data - min) / (max - min)

def DeNormalizeData(data, min=None, max=None):
    if min is None and max is None:
        return data * (np.max(data) - np.min(data)) + np.min(data)
    else:
        return data * (max - min) + min

def MapData(data):
    for x in data:
        x = (2.0 - math.log(7.5 - x)) / 2.0
    return data

def UnmapData(data):
    for x in data:
        x = -1 * (math.pow(math.e, (-2 * x + 2)) - 7.5)
    return data

class MlpModel():

    def __init__(self, **kwargs):
        #super().__init__(**kwargs)
        self.model = None

    def save_model(self, path):
        print(colored(f'\nSaving tensorflow keras model in {path}.p', "green"))
        self.model.save(path)
	
    def load_model(self, path):
        print(colored(f'\nLoading tensorflow keras model from {path}\n', "green"))
        self.model = keras.models.load_model(path)

    def build_new_model(self, af1, af2):
        inputs = tf.keras.Input(shape=(6,))
        dense = tf.keras.layers.Dense(12, activation=af1)
        x = dense(inputs)
        x = tf.keras.layers.Dense(24,activation=af1)(x)
        x = tf.keras.layers.Dense(24,activation=af1)(x)
        x = tf.keras.layers.Dense(12,activation=af1)(x)
        outputs = tf.keras.layers.Dense(3,activation=af2)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MLP_model")
        opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,metrics=['accuracy', 'mse'])
        self.model.summary()

    def train(self, input_dict) -> dict:
        print(colored("Training MLP model...", "green"))
        X_train = input_dict["training"]["data"]
        y_train = input_dict["training"]["labels"]

        # Normalize 6 Inputs, training data
        X_train = NormalizeData(X_train, 0, 112)
        #print(input_dict)
        #print(y_train)

        # Normalize 3 Outputs, training data
        y3_data_DeNormalized_1 = y_train[:,0:1]
        y3_data_DeNormalized_2 = y_train[:,1:2]
        y3_data_DeNormalized_3 = y_train[:,2:3]
        #print(y3_data_DeNormalized_1)
        #print(y3_data_DeNormalized_2)
        #print(y3_data_DeNormalized_3)

        # Normalize all y data, each feature individually, then concatenate data for all three features
        y3_data_Normalized_1 = MapData(y3_data_DeNormalized_1)
        y3_data_Normalized_1 = NormalizeData(y3_data_Normalized_1, 0.5, 6.5)
        y3_data_Normalized_2 = NormalizeData(y3_data_DeNormalized_2, 5, 45)
        y3_data_Normalized_3 = NormalizeData(y3_data_DeNormalized_3, 40, 120)
        y3_data_Normalized_1_2 = np.concatenate((y3_data_Normalized_1, y3_data_Normalized_2),axis=1)
        y3_data_Normalized = np.concatenate((y3_data_Normalized_1_2, y3_data_Normalized_3),axis=1)

        y_train = y3_data_Normalized

        start = timer()
        self.model.fit(X_train, y_train, epochs=1500, batch_size=64, validation_split = 0.1)
        end = timer()
        training_time = end - start

        accuracy_training = self.model.evaluate(X_train, y_train, batch_size=64)

        return {
            "training_time": training_time,
            "accuracy_training": accuracy_training
        }
        #return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing MLP model...", "green"))
        X_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]

        # Normalize 6 Inputs, training data
        X_test = NormalizeData(X_test, 0, 112)
        #print(input_dict)
        #print(y_test)

        # Normalize 3 Outputs, testing data
        y3_data_DeNormalized_1 = y_test[:,0:1]
        y3_data_DeNormalized_2 = y_test[:,1:2]
        y3_data_DeNormalized_3 = y_test[:,2:3]
        #print(y3_data_DeNormalized_1)
        #print(y3_data_DeNormalized_2)
        #print(y3_data_DeNormalized_3)

        # Normalize all y data, each feature individually, then concatenate data for all three features
        y3_data_Normalized_1 = MapData(y3_data_DeNormalized_1)
        y3_data_Normalized_1 = NormalizeData(y3_data_Normalized_1, 0.5, 6.5)
        y3_data_Normalized_2 = NormalizeData(y3_data_DeNormalized_2, 5, 45)
        y3_data_Normalized_3 = NormalizeData(y3_data_DeNormalized_3, 40, 120)
        y3_data_Normalized_1_2 = np.concatenate((y3_data_Normalized_1, y3_data_Normalized_2),axis=1)
        y3_data_Normalized = np.concatenate((y3_data_Normalized_1_2, y3_data_Normalized_3),axis=1)

        y_test = y3_data_Normalized

        accuracy_testing = self.model.evaluate(X_test, y_test, batch_size=64)

        start = timer()
        y_pred = self.model.predict(X_test)
        end = timer()
        testing_prediction_time = end - start

        y_pred_1 = y_pred[:,0:1]
        y_pred_2 = y_pred[:,1:2]
        y_pred_3 = y_pred[:,2:3]
        y_pred_1 = DeNormalizeData(y_pred_1, 0.5, 6.5)
        y_pred_1 = UnmapData(y_pred_1)
        y_pred_2 = DeNormalizeData(y_pred_2, 5, 45)
        y_pred_3 = DeNormalizeData(y_pred_3, 40, 120)
        y_pred_1_2 = np.concatenate((y_pred_1, y_pred_2),axis=1)
        y_pred = np.concatenate((y_pred_1_2, y_pred_3),axis=1)
        #print(y_pred.shape)
        #print(y_pred)

        output_dict = {
            "accuracy_testing": accuracy_testing, 
            "testing_prediction_time": testing_prediction_time, 
            "testing_predictions": y_pred
        }

        return output_dict

    def predict(self, input_dict) -> dict:
        return super().predict(input_dict)
