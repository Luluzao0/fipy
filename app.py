#importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Símbolo da ação ou moeda 
symbol = 'NVDC34'

# Obter os dados históricos
data = yf.download(symbol, start='2015-01-01', end='2024-09-15')

# Verificar se os dados foram obtidos com sucesso
if not data.empty:
    print(data.head())
else:
    print("Não foi possível obter os dados.")
    exit()

close_prices = data['Adj Close']

values = close_prices.values.reshape(-1, 1)

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Definir o tamanho de treinamento (80% dos dados)
training_data_len = int(np.ceil(len(scaled_data) * 0.9))

# Criar o conjunto de treinamento
train_data = scaled_data[0:training_data_len, :]

# Dividir os dados em x_train e y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Converter para arrays numpy e redimensionar
x_train, y_train = np.array(x_train), np.array(y_train)

# Redimensionar os dados para o formato [samples, time_steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Construir o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(x_train, y_train, batch_size=1, epochs=2)

# Criar os dados de teste
test_data = scaled_data[training_data_len - 60:, :]

# Criar x_test e y_test
x_test = []
y_test = values[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Converter para array numpy
x_test = np.array(x_test)

# Redimensionar para [samples, time_steps, features]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Obter as previsões do modelo
predictions = model.predict(x_test)

# Desnormalizar os dados
predictions = scaler.inverse_transform(predictions)

# Calcular o erro médio absoluto (MAE)
mae = np.mean(np.abs(predictions - y_test))
print(f'Erro Médio Absoluto (MAE): {mae}')

# Preparar os dados para visualização
train = data[:training_data_len]
valid = data[training_data_len:].copy()  # Usar .copy() para evitar SettingWithCopyWarning
valid['Predictions'] = predictions

# Visualizar os dados
plt.figure(figsize=(16,8))
plt.title('Modelo LSTM - Previsão de Preço de Fechamento')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento em USD')
plt.plot(train['Adj Close'], label='Treino')
plt.plot(valid['Adj Close'], label='Validação')
plt.plot(valid['Predictions'], label='Previsões')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Obter os últimos 60 dias
last_60_days = scaled_data[-60:]

# Criar uma lista
X_future = [last_60_days[:, 0]]

# Converter para array numpy
X_future = np.array(X_future)

# Redimensionar
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

# Obter a previsão escalonada
future_price = model.predict(X_future)

# Desnormalizar o preço
future_price = scaler.inverse_transform(future_price)

print(f'Preço previsto para o próximo dia: {future_price[0][0]}')
