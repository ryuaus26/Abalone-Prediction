import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd
import csv

#Hyperparameters
EPOCHS = 20
BATCHES = 1000
LEARNING_RATE = 0.01

X_train = np.array([])
index = 2

df  = pd.read_csv("./Data/X.csv")


Y_train = np.array([])
    
with open("./Data/Y.csv",'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for value in csv_reader:
        index = index + 1
        if(index <= 3341):
            Y_train = np.append(Y_train, value)
print(Y_train.shape)

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

#w should have shape of (n[l],n[l-1]) l = layer #

w1 = np.random.randn(X_train.shape[1],hidden1_size) * e
w2 = np.random.randn(hidden1_size,hidden2_size) * e
w3 = np.random.randn(hidden2_size,3) * e
#b should have shape of (n[l],1)
b1 = np.random.randn(1,hidden1_size) * e
b2 = np.random.randn(1,hidden2_size) * e
b3 = np.random.randn(1,3) * e





def leaky_relu(z):
    return np.maximum((0.1*z),z)


def softmax(Z):
    s = np.exp(Z)
    s = s/np.sum(s,axis=0)
    return s


def forward_propagation(X):
    Z1 = np.dot(X,w1) + b1
    A1 = leaky_relu(Z1)
  
    Z2 = np.dot(A1,w2) + b2
    A2 = leaky_relu(Z2)
    
    Z3 = np.dot(A2,w3) + b3
    yhat = softmax(Z3)
    
    #Dictionary
    cache ={"Z1":Z1,
            "A1":A1,
            "Z2":Z2,
            "A2":A2,
            "Z3":Z3,
            }
    return yhat
    
    


    
def compute_cost(Y,yhat):
    m = Y.shape[0]
    cost = -np.sum(Y*np.log(yhat))
    cost = cost / m
    cost = np.squeeze(cost)
    return cost

def gradient_descent(w,b,dw,db,learning_rate = 0.01):
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
    return w,b

def backpropagation(X,Y,yhat,A1,A2,Z1,Z2,w1,w2,w3,b1,b2,b3,learning_rate = 0.01):
    #Third layer
    dZ3 = yhat - Y
    dW3 = np.dot(dZ3,A2.T) * (1/m)
    dB3 = np.sum(dZ3,axis=1,keepdims=True) * (1/m)
    
    #Second layer
    dZ2 = np.multiply((np.dot(w2.T,dZ3)),np.where(Z2 < 0,0.1,1))
    dW2 = np.dot(dZ2,A1.T) * (1/m)
    dB2 = np.sum(dZ2,axis=1,keepdims=True) * (1/m)
    
    #First layer

    dZ1 = np.multiply(np.dot(Z1.T,dZ2),np.where(Z1 < 0, 0.1,1))
    dW1 = np.dot(dZ1,X.T)  * (1/m)
    dB1 = np.sum(dZ1,axis=1,keepdims=True) * (1/m)
    
    w1,b1 = gradient_descent(w1,b1,dW1,dB1,learning_rate)
    w2,b2 = gradient_descent(w2,b2,dW2,dB2,learning_rate)
    w3,b3 = gradient_descent(w3,b3,dW3,dB3,learning_rate)
    
    return w1,b1,w2,b2,w3,b3


def model_fit(X,Y,cache,learning_rate = 0.01):
    
    for i in range(EPOCHS):
        for j in range(0,n,BATCHES):
        
       
        #Select the next batch
            X_batch = X_train[j:j + BATCHES]
        
            #Get cache
        
            Z1 = cache["Z1"]
            A1 = cache["A1"]
            Z2 = cache["Z2"]
            A2 = cache["A2"]
            Z3 = cache["Z3"]
            
        
        
            #One hot encoding
            Y_batch = np.zeros((X_batch.shape[0],3))
            Y_batch[np.arange(X_batch.shape[0]), Y_train[j:j+BATCHES]] = 1
        
            # Forward propagation
            yhat_batch = forward_propagation(X_batch)
        
            # Compute the cost for the batch
            cost = compute_cost(Y_batch, yhat_batch)
            # Backpropagation
            w1, b1, w2, b2, w3, b3 = backpropagation(X_batch, Y_batch, yhat_batch, A1, A2, Z1, Z2, Z3, w1, w2, w3, b1, b2, b3, learning_rate)
        
            # Print the cost every 100 batches
            if j % 100 == 0:
                print(f"Epoch {i+1}, Batch {j+1}/{n}, Cost: {cost:.4f}")
        
    



model_fit(X_train,Y_train,forward_propagation(X_train),LEARNING_RATE)
