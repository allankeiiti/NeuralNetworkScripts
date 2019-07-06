import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(slg):
    return slg * (1 - slg)

base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
saidas = np.empty([569, 1], dtype=int)
for i in range(569):
    saidas[i] = valoresSaida[i]

pesos0 = 2 * np.random.random((30,3)) - 1
pesos1 = 2 * np.random.random((5,1)) - 1

epocas = 10
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
