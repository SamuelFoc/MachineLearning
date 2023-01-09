import numpy as np

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def nearest_value(array, value):
    index = (np.abs(array - value)).argmin()
    return array[index]

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, x, y):
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def _predict(self, x):
        # Find the distances of given point x from all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # Sort the indices of distances by asscend order and pick first k elements
        k_indices = np.argsort(distances)[:self.k]
        
        # Find the assigned value of the k elements
        k_assigned_values = [self.y_train[i] for i in k_indices]
        print(k_assigned_values)
        # Find the mean value and return prediction
        common_value = np.bincount(k_assigned_values).argmax()
        
        # Find the nearest value from assigned values possibilities
        unique_values = unique(self.y_train)
        value = nearest_value(unique_values, common_value)

        return value