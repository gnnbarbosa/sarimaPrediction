# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

#Importação das bibliotecas para execução do exemplo
df = pd.read_csv('/home/guilherme/international-airline-passengers.csv', engine='python')

#reducao do dataset apenas com as features necessarias para treinar o modelo
df = df["Passengers"]

#Separacao do dataset para treinamento e validacao
train = df[:int(0.67*(len(df)))]
validation = df[int(0.67*(len(df))):]

#Paramêtros do modelo
model = auto_arima(train,  start_p=1, d=1, start_q=1,
                   max_p=2, max_d=1, max_q=1,
                   start_P=1, D=1, start_Q=1,
                   max_P=2, max_D=1, max_Q=1,
                   max_order=5, m=12,
                   seasonal=True,
                   information_criterion='aic',
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)
# Treinamento
model.fit(validation)

# Predicao
prediction = model.predict(n_periods=len(validation))
prediction = pd.DataFrame(prediction,index = validation.index,columns=['Prediction'])

#Plotagem do grafico comparativo
plt.plot(df, label='Original Serie', color="deepskyblue")
plt.plot(prediction, label='Forecast', linestyle="--", color="darkred")
plt.ylabel('NonLinear components of proto entropy',fontsize=14)
plt.xlabel('Samples',fontsize=14)
plt.legend(loc="upper left")
plt.show()
