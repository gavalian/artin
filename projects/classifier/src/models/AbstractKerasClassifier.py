from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import softmax
from termcolor import colored


from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

from models.AbstractModel import AbstractModel
from utils.accuracy_utils import *

class AbstractKerasClassifier(AbstractModel):

    def __init__(self, in_dict=None):
        super().__init__(in_dict)
        self.model = None
        self.n_classes = 0

    def load_model(self, path):
        print(colored(f'\nLoading keras model from {path}\n', "green"))
        self.model = self.model = load_model(path)

    def save_model(self, path):
        print(colored(f'\nSaving keras model in {path}.p', "green"))
        self.model.save(f'{path}.p')

    def preprocess_input(self, input_dict):
        self.n_classes = input_dict["num_classes"]

    def compute_accuracy_metrics(self, input_dict) -> dict:
        conf_matrix = input_dict["confusion_matrix"]
        total_test_samples = input_dict["total_test_samples"]
        y_pred = input_dict["y_pred"]
        y_pred_proba = input_dict["y_pred_proba"]
        y_test_segmented = input_dict["y_test_segmented"]

        accuracy_A1 = get_accuracy_A1(conf_matrix, total_test_samples)
        accuracy_Ac = get_accuracy_Ac(conf_matrix, y_pred, y_test_segmented)
        accuracy_Ah = get_accuracy_Ah(y_pred_proba, y_test_segmented)
        accuracy_Af = get_accuracy_Af(conf_matrix, total_test_samples)

        return {
            "accuracy_A1": accuracy_A1,
            "accuracy_Ac": accuracy_Ac,
            "accuracy_Ah": accuracy_Ah,
            "accuracy_Af": accuracy_Af
        }

    def train(self, input_dict) -> dict:
        X_train = input_dict["training"]["data"]
        y_train = input_dict["training"]["labels"]
        epochs = input_dict["training"]["epochs"]
        batch_size = input_dict["training"]["batch_size"]

        start = timer()
        self.model.fit(X_train, y_train,epochs=epochs,batch_size=batch_size)
        end = timer()
        training_time = end - start

        accuracy_training = self.model.evaluate(X_train, y_train,batch_size=batch_size)[1]
        return {
            "training_time": training_time,
            "accuracy_training": accuracy_training
        }

    def test(self, input_dict) -> dict:

        X_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]
        batch_size = input_dict["testing"]["batch_size"]


        accuracy_testing = self.model.evaluate(X_test, y_test, batch_size=batch_size)[1]
        start = timer()
        y_pred_proba = self.model.predict(X_test)
        end = timer()
        testing_prediction_time = end - start

        y_pred =  np.argmax(y_pred_proba,1)
        y_test =  np.argmax(y_test,1)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Compute accuracy metrics
        input_dict_accuracy_metrics = {
            "confusion_matrix": conf_matrix, 
            "y_pred": y_pred, 
            "y_pred_proba": y_pred_proba,
            "total_test_samples": input_dict["total_test_samples"], 
            "y_test_segmented": input_dict["testing_segmented"]["labels"]
        }

        accuracy_metrics_dict = self.compute_accuracy_metrics(input_dict_accuracy_metrics)

        output_dict = {
            "accuracy_testing": accuracy_testing, 
            "testing_prediction_time": testing_prediction_time, 
            "confusion_matrix": conf_matrix,
            "prediction_matrix": y_pred_proba
        }
        output_dict.update(accuracy_metrics_dict)

        return output_dict

    # TODO implement
    def predict(self, input_dict)  -> dict:
        X = input_dict["prediction"]["data"]
        y_pred_proba = self.model.predict(X)
        y_valid = y_pred_proba[:,1]

        if (input_dict['softmax']):
            output_dict = {
                "predictions": softmax(y_valid.reshape(1,-1),copy=False)[0]
            }
        else:
            output_dict = {
                "predictions": y_valid
            }

        return output_dict
