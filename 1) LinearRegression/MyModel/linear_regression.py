import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = 1
        self.interceptor = 0

    def fit(self, x, y):
        # Mean values
        x_mean = x.mean()
        y_mean = y.mean()
        x_diff = (x - x_mean)
        y_diff = (y - y_mean)

        # Compute slope and interceptor
        slope = (x_diff*y_diff).sum()/(x_diff**2).sum()
        interceptor = y_mean - slope * x_mean 

        # Set slope and interceptor to model
        self.slope = slope
        self.interceptor = interceptor

    def predict(self, x):
        return x*self.slope + self.interceptor

    def correlation(self, x, y):
        prediction = self.predict(x)

        sst = ((y - y.mean())**2).sum()
        sse = ((y - prediction)**2).sum()
        ssr = (sst - sse)

        r_2 = ssr/sst 
        sign = self.slope/np.abs(self.slope)

        return np.round(sign * np.sqrt(r_2), 2)