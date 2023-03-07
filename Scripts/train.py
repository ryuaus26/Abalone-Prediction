import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd

# # 1 / 1+(np.exp(-z))
# # relu -> softmax

X_train = np.array([])
index = 2

df  = pd.read_csv("./Data/X.csv")

all_features = df[["LongestShell","Diameter","Height","WholeWeight","VisceraWeight","ShellWeight","Rings"]].to_numpy()


X_train = all_features.reshape(-1,7)

m = X_train.shape[1]
n = X_train.shape[0]




e = 0.01
#parameters
#I can also vectorize these parameters





hidden1_size = 8
hidden2_size = 6
hidden3_size = 4

w1 = np.random.randn(1,hidden1_size) * e
w2 = np.random.randn(1,hidden2_size) * e
w3 = np.random.randn(1,hidden3_size) * e

b1 = np.random.randn(1,hidden1_size) * e
b2 = np.random.randn(1,hidden2_size) * e
b3 = np.random.randn(1,hidden3_size) * e






def activation_z(X,w,b):
    z = np.dot(X,w.T)
    z = z + b
    return z

def leaky_relu(z):
    return max((0.01* z),z)


def softmax(Z):
    s = np.exp(Z)
    s = s/np.sum(np.exp(Z))
    

def forward_propagation(X):
    Z1 = np.dot(X,w1) + b1
    A1 = leaky_relu(Z1)
    
    Z2 = np.dot(A1,w2) + b2
    A2 = leaky_relu(Z2)
    
    Z3 = np.dot(A2,w3) + b3
    A3 = leaky_relu(Z3)
   

    

    
    
    
# def compute_cost(X,Y,yhat):
#     m = X.shape[0]
#     cost (Y - yhat) ** 2
#     cost = cost / m
#     return cost

#def backpropagation():

