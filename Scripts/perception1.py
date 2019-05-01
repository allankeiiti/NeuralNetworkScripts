inputs = [1,7,5]
weight = [0.8,0.1,0]

def soma(input, weight):
    sum = 0;
    for i in range(len(input)):
        print(input[i])
        print(weight[i])
        sum += input[i] * weight[i]
    return sum

s = soma(inputs, weight)

def stepFunction(soma):
    if(soma >=1):
        return 1
    return 0

r = stepFunction(s)