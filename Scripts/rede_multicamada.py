import numpy as np

#   Sigmoid Formula
#
#   y =  ____1____
#               -x
#         1 - e


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

#   Derivada
#   d = y * (1 - y)

def sigmoidDerivada(slg):
    return slg * (1 - slg)



#As entradas, saídas, pesos0, pesos1 e epocas foram retiradas da video-aula 25. Implementação rede multicamada I e II
entradas = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]])

saidas = np.array([[0],[1],[1],[0]])

# pesos0 = np.array([[-0.424, -0.740, -0.961],
#           [0.358, -0.577, -0.469]])

# pesos1 = np.array([[-0.017], [-0.893], [0.148]])

#2 neurônios na camada de entrada e 3 neurônios na camada de saída
pesos0 = 2 * np.random.random((2,3)) - 1
pesos1 = 2 * np.random.random((3,1)) - 1

epocas = 100000

#Valores definidos pelo usuário
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
    camadaEntrada = entradas

    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    CamadaFinal = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - CamadaFinal
    #Media absoluta envolve os valores de Erro da camada de saída positivo. Verificar Imagem Rede_Neural_Multicamda_foto.PNG
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print('Erro: ' + str(mediaAbsoluta))

    #Implementando o cálculo de DeltaSaida
    derivadaSaida = sigmoidDerivada(CamadaFinal)
    deltaSaida = erroCamadaSaida * derivadaSaida

    # Matriz Transposta
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta =deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    # Backpropagation 1
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    # Backpropagation 0
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)

# Lembrando, a função NumPy.DOT equivale às linhas abaixo, porém mais ágil:

#DotProductcomFor(entradas, pesos):
#    sum = 0
#    for i in range(len(entradas66)):
#        print(entradas[i])
#        print(pesos[i])
#        sum += entradas[i] * pesos[i]
#    return sum

#Erro = respostaCorreta - respostaCalculada.
