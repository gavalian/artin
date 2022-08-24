import xgboost as xgb
from models.AbstractScikitLearnRegressor import AbstractScikitLearnRegressor
from termcolor import colored

class XGBoostModel(AbstractScikitLearnRegressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self):
        self.model = xgb.XGBRegressor(n_estimators=1300, objective='reg:squarederror', max_depth=6, learning_rate=0.1, max_delta_step=0, gamma=0,
                                   n_jobs=-1, verbosity =1, min_child_weight=7, num_parallel_tree=4)

    def train(self, input_dict) -> dict:
        print(colored("Training XGBoost model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing XGBoost model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        return super().predict(input_dict)
