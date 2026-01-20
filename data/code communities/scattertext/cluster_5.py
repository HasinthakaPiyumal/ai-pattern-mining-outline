# Cluster 5

class Sigmoidal:

    def __init__(self):
        self.popt = None

    def fit(self, x, y):
        assert len(x) == len(y)
        p0 = [max(y), np.median(x), 1, min(y)]
        self.popt, pcov = curve_fit(sigmoid, x, y, p0, method='dogbox', maxfev=10000)
        return self

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)

    def predict(self, x):
        return sigmoid(x, *self.popt)

def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y

