import numpy as np
import math

def relu(array):
    return np.maximum(0,array)
def reluDeriv_vectorial(  array):
    grad = np.zeros((array.shape))
    for i in range(len(array)):
        if array[i]>0:
            grad[i]=1
    return grad
def reluDeriv_matricial(  matrix):
    grad = np.zeros((matrix.shape))
    for i in range(matrix.shape[1]):
        grad[:,i] =   reluDeriv_vectorial(matrix[:,i])
    return grad
def tanh(  array):
    return 2.0/(1+np.exp(-2*array)) - 1
    
def tanhDeriv(  array):
    return 1-np.power(  tanh(array),2)

def softmax_vectorial(  array):
    sft = np.zeros((array.shape))
    for i in range(len(array)):
        sft[i]=math.exp(array[i])
    sft/=np.sum(sft)
    return sft
def softmax_matricial(  matrix):
    grad = np.zeros((matrix.shape))
    for i in range(matrix.shape[1]):
        grad[:,i] =   softmax_vectorial(matrix[:,i])
    return grad
    
    # Loss and Cost Functions
def loss(  yhat,y):
    return -np.sum(y*np.log(yhat))
def crossEntropy(  yhat,y):
    m = yhat.shape[1]
    cost = 0
    for i in range(y.shape[1]):
        cost +=   loss(yhat[:,i],y[:,i])
    return cost/m
def checkAccuracy(y,y_pred):
    accuracy = 0
    for i in range(y_pred.shape[1]):
        cpy = np.zeros(y_pred[:,i].shape)
        cpy[np.argmax(y_pred[:,i])] = 1
        y_pred[:,i] = cpy
    for i in range(y_pred.shape[1]):
        if(list(y_pred[:,i])==list(y[:,i])):
            accuracy+=1
    return accuracy/y_pred.shape[1]
