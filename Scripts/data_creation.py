import tensorflow as tf
from tensorflow import keras
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

index = 1


# Y_train = np.array([])
    
# with open("./Data/Y.csv",'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for value in csv_reader:
#         index = index + 1
#         if(index <= 3341):
#             Y_train = np.append(Y_train, value)
            
        
# Y_val = np.array([])
# index = 3342
# with open("./Data/Y.csv",'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for value in csv_reader:
#         index = index + 1
#         if(index<=3760):
#             Y_val = np.append(Y_val, value)

# index = 3761
# Y_test = np.array([])
# with open("./Data/Y.csv",'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for value in csv_reader:
#         index = index + 1
#         if(index<=4176):
#             Y_test = np.append(Y_val, value)
            
# Y_train = Y_train.reshape(3340,1)
# Y_val = Y_val.reshape(418,1)
# Y_test = Y_test.reshape(419,1)


X_train = np.array([])
index = 2

df  = pd.read_csv("./Data/X.csv")

all_features = df[["LongestShell","Diameter","Height","WholeWeight","VisceraWeight","ShellWeight","Rings"]].to_numpy()


X_train = all_features.reshape(-1,7)


print(X_train)


