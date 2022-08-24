from timeit import default_timer as timer
# from sklearn.metrics import confusion_matrix
from termcolor import colored
# from sklearn.discriminant_analysis import softmax
import pickle

from models.AbstractModel import AbstractModel
# from utils.accuracy_utils import *

class AbstractScikitLearnRegressor(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def load_model(self, path):
        print(colored(f'\nLoading scikit-learn model from {path}\n', "green"))
        self.model = pickle.load(open(f'{path}', "rb"))

    def save_model(self, path):
        print(colored(f'\nSaving scikit-learn model in {path}.p', "green"))
        pickle.dump(self.model, open(f'{path}.p', "wb"))

    def preprocess_input(self, input_dict):
        None

    def train(self, input_dict) -> dict:
        X_train = input_dict["training"]["data"]
        y_train = input_dict["training"]["labels"]

        start = timer()
        self.model.fit(X_train, y_train)
        end = timer()
        training_time = end - start

        accuracy_training = self.model.score(X_train, y_train)

        return {
            "training_time": training_time,
            "accuracy_training": accuracy_training
        }

    def test(self, input_dict) -> dict:

        X_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]

        accuracy_testing = self.model.score(X_test, y_test)

        start = timer()
        y_pred = self.model.predict(X_test)
        end = timer()
        testing_prediction_time = end - start

        output_dict = {
            "accuracy_testing": accuracy_testing, 
            "testing_prediction_time": testing_prediction_time, 
            "testing_predictions": y_pred
        }

        return output_dict