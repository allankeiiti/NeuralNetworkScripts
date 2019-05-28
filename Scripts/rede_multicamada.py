import numpy as np

#   Sigmoid Formula
#
#   Y =  ____1____
#               -x
#         1 - e

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

#As entradas, saídas, pesos0, pesos1 e epocas foram retiradas da video-aula 25. Implementação rede multicamada I e II
entradas = ([[0,0],
             [0,1],
             [1,0],
             [1,1]])

saidas = ([[0],[1],[1],[0]])

pesos0 = ([[-0.424, -0.740, -0.961],
           [0.358, -0.577, -0.469]])

pesos1 = ([[-0.017], [-0.893], [0.148]])

epocas = 100

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    RespostaFinal = sigmoid(somaSinapse1)


# Lembrando, a função NumPy.DOT equivale às linhas abaixo, porém mais ágil:
#DotProductcomFor(entradas, pesos):
#    sum = 0
#    for i in range(len(entradas)):
#        print(entradas[i])
#        print(pesos[i])
#        sum += entradas[i] * pesos[i]
#    return sum

#Erro = respostaCorreta - respostaCalculada