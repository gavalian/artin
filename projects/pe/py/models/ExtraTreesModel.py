from sklearn.ensemble import ExtraTreesRegressor
from models.AbstractScikitLearnRegressor import AbstractScikitLearnRegressor
from termcolor import colored

class ExtraTreesModel(AbstractScikitLearnRegressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self):
        self.model = ExtraTreesRegressor(n_estimators=100, max_features=None,
                                   n_jobs=-1, verbose=1, random_state=3333)

    def train(self, input_dict) -> dict:
        print(colored("Training ExtraTrees model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing ExtraTrees model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        return super().predict(input_dict)
