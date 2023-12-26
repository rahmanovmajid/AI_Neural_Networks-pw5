#class is used to take the path and convert it to the matrix

import numpy as np

class DataLib:
    def __init__(self,path):
        self.path = path
        self.matrix = self.from_data_to_matrix()
    def from_data_to_matrix(self):
        file = open(self.path,"r") 
        raw_data = file.readlines()
        real_data = [None]*len(raw_data)
        for i in range(len(raw_data)):
            real_data[i] = raw_data[i][:-1].split()
            real_data[i] = raw_data[i].split(',')
            real_data[i] = [float(j) for j in real_data[i]]
            real_data[i][-1] = int(real_data[i][-1])
        return real_data
data = DataLib('iris_num.data')







