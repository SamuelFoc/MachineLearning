import numpy as np


class DataGenerator:
    def linear_data(slope:float, bias:float, start:int, end:int, points:int, noise_rate:float):
        x = np.linspace(start, end, points)
        noise = noise_rate * np.random.random(size=len(x))
        y = slope * x + bias + noise
        return([x, y])