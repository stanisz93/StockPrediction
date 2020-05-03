import pandas as pd
import numpy as np






class DataSet:
    MEAN = "mean"
    STD = "std"

    def __init__(self, data: pd.array):
        self.data = data
        self.learned_vals = {}
        self.train, self.test = None, None
    
    def get(self, _key):
        return self.learned_vals[_key]

    def transform(self, dataset: np.array):
        assert DataSet.MEAN in self.learned_vals
        assert DataSet.STD in self.learned_vals
        return (dataset - self.get(DataSet.MEAN)) / self.get(DataSet.STD)

    def fit(self, dataset: np.array):
        if DataSet.MEAN in self.learned_vals or DataSet.STD in self.learned_vals:
            print("Warning: Mean and std already learned!!!! Your value might be changed.")
        self.learned_vals[DataSet.MEAN] = dataset.mean()
        self.learned_vals[DataSet.STD] = dataset.std()
        return (dataset - self.get(DataSet.MEAN)) / self.get(DataSet.STD)


    def split_data(self):
        train_size = int(len(self.data) * 0.9)
        self.train, self.test = self.data[0:train_size], self.data[train_size:len(self.data)]
        return self.train, self.test

    def revert_transform(self, dataset):
        return (dataset * self.get(DataSet.STD))+self.get(DataSet.MEAN)

    @staticmethod
    def univariate_data(dataset: np.array, history_size: int, target_size: int):
        data = []
        labels = []

        start_index = history_size
        #warning, for end iteration should be reconsidered referring to 
        
        for i in range(start_index, len(dataset)):
            indices = range(i-history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)