import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class Model:
    def __init__(self, **kwargs):
        regressor = kwargs.get("load", None)
        self.ytype = kwargs.get("ytype", "relative")

        if regressor is None:
            self.regression = LinearRegression()
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

    def feature_weights(self):
        return self.regression.coef_ / np.sum(self.regression.coef_)


if __name__ == "__main__":
    mod = Model()

    print("Linear")
    error = []
    for i in range(100):
        x = (np.zeros(100).reshape((10,10)) + np.arange(0,10,1).transpose()).transpose().flatten()
        y = ((np.random.randn(100)/100).reshape((10,10)) + np.arange(0,10,1).transpose()).transpose().flatten()
        res = mod.run(x,y,x[0])
        y_pred = mod.apply(x, y, res["coeff"])
        error.append(np.mean(-y_pred))
    print("ME = " + str(np.mean(error)))
    print("MAE = " + str(np.mean(np.abs(error))))
    print("RMSE = " + str(np.sqrt(np.mean(np.array(error)**2))))

    print("Exponential")
    error = []
    for i in range(100):
        x = (np.zeros(100).reshape((10,10)) + np.arange(0,10,1).transpose()).transpose().flatten()
        y = ((np.random.randn(100)/100).reshape((10,10)) + (np.exp(2 * np.arange(0,10,1)) - 1).transpose()).transpose().flatten()
        res = mod.run(x,y,x[0])
        y_pred = mod.apply(x, y, res["coeff"])
        error.append(np.mean(-y_pred))
    print("ME = " + str(np.mean(error)))
    print("MAE = " + str(np.mean(np.abs(error))))
    print("RMSE = " + str(np.sqrt(np.mean(np.array(error)**2))))

'''
import numpy as np
from scipy.stats import linregress

class Model:
    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", "absolute")
        return

    def run(self, x, y, reference_x=None):
        result = {}

        if not reference_x is None:
            reference_x = np.array([reference_x]).astype(x.dtype)[0]

        if reference_x is None or np.sum(x==reference_x):
            reference_x = np.min(x)

        reference_y = np.mean(y[x==reference_x]).astype(float)

        if reference_y < 1e-6:
            reference_y = 1e-6

        if self.mode.lower() == "absolute":
            train_y = y.astype(float) - reference_y
        else: #percentage
            train_y = 100 * (y.astype(float) - reference_y) / reference_y

        regression = linregress(x, train_y)

        result["coeff"] = [regression.intercept, regression.slope]
        result["se"] = regression.stderr
        result["p"] = regression.pvalue
        result["rsquared"] = regression.rvalue**2
        result["reference"] = [reference_x, reference_y]

        return result

    def apply(self, x, y, coeff):
        dy = np.array(x * coeff[1] + coeff[0]).flatten() # just in case if x is provided as a lis

        if self.mode.lower() == "absolute":
            result = y - dy
        else: #percentage
            result = y / (1 + dy/100)
        return result

if __name__ == "__main__":
    mod = Model()

    print("Linear")
    error = []
    for i in range(100):
        x = (np.zeros(100).reshape((10,10)) + np.arange(0,10,1).transpose()).transpose().flatten()
        y = ((np.random.randn(100)/100).reshape((10,10)) + np.arange(0,10,1).transpose()).transpose().flatten()
        res = mod.run(x,y,x[0])
        y_pred = mod.apply(x, y, res["coeff"])
        error.append(np.mean(-y_pred))
    print("ME = " + str(np.mean(error)))
    print("MAE = " + str(np.mean(np.abs(error))))
    print("RMSE = " + str(np.sqrt(np.mean(np.array(error)**2))))

    print("Exponential")
    error = []
    for i in range(100):
        x = (np.zeros(100).reshape((10,10)) + np.arange(0,10,1).transpose()).transpose().flatten()
        y = ((np.random.randn(100)/100).reshape((10,10)) + (np.exp(2 * np.arange(0,10,1)) - 1).transpose()).transpose().flatten()
        res = mod.run(x,y,x[0])
        y_pred = mod.apply(x, y, res["coeff"])
        error.append(np.mean(-y_pred))
    print("ME = " + str(np.mean(error)))
    print("MAE = " + str(np.mean(np.abs(error))))
    print("RMSE = " + str(np.sqrt(np.mean(np.array(error)**2))))
    
'''