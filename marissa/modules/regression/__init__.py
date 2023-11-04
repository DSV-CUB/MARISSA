import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor

class Model:
    def __init__(self, **kwargs):
        regressor = kwargs.get("load", None)
        self.ytype = kwargs.get("ytype", "relative")

        if regressor is None:
            self.regression = ExtraTreesRegressor(n_estimators=1000)  # exchange the regressor with your own
        else:
            self.regression = None
            self.load(regressor)
        return

    def get(self):
        return pickle.dumps(self.regression)

    def load(self, regressor):
        self.regression = pickle.loads(regressor)
        return

    def train(self, x, y):
        self.regression.fit(x, y)

        fit_y = self.predict(x)

        result = {}
        result["rmse"] = np.sqrt(np.sum((y - fit_y) ** 2) / len(y))
        result["p"] = -1
        result["rsquared"] = r2_score(y, fit_y)
        return result

    def predict(self, x):
        return self.regression.predict(x)

    def apply(self, x, y, ytype=None):
        dy = self.predict(x)

        if not ytype is None:
            self.ytype = ytype

        if self.ytype == "absolute":
            result = y - dy
        else: #percentage
            result = y / (1 + dy/100)
        return result
