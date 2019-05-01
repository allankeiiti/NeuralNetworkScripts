import numpy as np

inputs = np.array([1,7,5])
weight = np.array([0.8,0.1,0])

def soma(input, weight):
    return input.dot(weight)
# dot product / produto escalar


s = soma(inputs, weight)

def stepFunction(soma):
    if(soma >=1):
        return 1
    return 0

r = stepFunction(s)