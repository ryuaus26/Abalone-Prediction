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



w = np.array([])


w = (np.random.randn(1,7) *  e)

b1 = np.random.rand(1,1) * e
b2 = np.random.rand(1,1) * e
b3 = np.random.rand(1,1) * e
b4 = np.random.rand(1,1) * e
b5 = np.random.rand(1,1) * e
b6 = np.random.rand(1,1) * e
b7 = np.random.rand(1,1) * e

hidden1_size = 8
hidden2_size = 6


def activation_z(X,w,b):
    z = np.dot(X,w.T)
    z = z + b
    return z

def leaky_relu(z):
    return max((0.01* z),z)




def forward_propagation(X,z,w,b):
    



def softmax(a,k):
    #a is a vector
    #k = 3
    softmax = 

# def compute_cost(X,Y,yhat):
#     m = X.shape[0]
#     cost (Y - yhat) ** 2
#     cost = cost / m
#     return cost

# def compute_gradient_descent(X,Y,yhat,w,b,a = 0.0001,iterations = 1000,e = 0.1):
  
    
#     m = X.shape[1]
    
#     cost = compute_cost(X,Y,yhat)
    
#     while(cost < e):
#         dj_dw = (Y - yhat) ** 2
#         dj_dw = dj_dw / (m * 2)
    
#         dj_db = np.sum(X)
#         dj_db = dj_db / m
    
#         w = w - a * dj_dw
#         b = b - a * dj_db
    
#     return w,b

