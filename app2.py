# Importando as bibliotecas
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objs as go
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(layout="wide")

# Símbolo da ação ou moeda 
symbol = 'NVDA'

# Obter os dados históricos
data = yf.download(symbol, start='2015-01-01', end='2024-09-15')

# Verificar se os dados foram obtidos com sucesso
if not data.empty:
    st.write("### Dados Históricos:")
    st.dataframe(data.head().style.format({"Open": "{:.4f}", "High": "{:.4f}", "Low": "{:.4f}", 
                                           "Close": "{:.4f}", "Adj Close": "{:.4f}", 
                                           "Volume": "{:,.0f}"}))
else:
    st.error("Não foi possível obter os dados.")
    exit()

close_prices = data['Adj Close']
values = close_prices.values.reshape(-1, 1)

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Definir o tamanho de treinamento (90% dos dados)
training_data_len = int(np.ceil(len(scaled_data) * 1.0))

# Criar o conjunto de treinamento
train_data = scaled_data[0:training_data_len, :]
#
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

# Construir o modelo LSTM com camadas Dropout
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dropout(0.1))
model.add(Dense(units=25))
model.add(Dense(units=2))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=0)

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

# Calcular o erro médio absoluto (MAE) e o erro quadrático médio (RMSE)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

st.write(f'**Erro Médio Absoluto (MAE):** {mae:.4f}')
st.write(f'**Root Mean Square Error (RMSE):** {rmse:.4f}')

# Preparar os dados para visualização
train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

# Gráfico interativo usando Plotly
fig = go.Figure()

# Dados de Treino
fig.add_trace(go.Scatter(x=train.index, y=train['Adj Close'], mode='lines', name='Treino'))

# Dados de Validação
fig.add_trace(go.Scatter(x=valid.index, y=valid['Adj Close'], mode='lines', name='Validação'))

# Previsões
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Previsões'))

# Atualizar layout
fig.update_layout(title='Modelo LSTM - Previsão de Preço de Fechamento',
                  xaxis_title='Data',
                  yaxis_title='Preço de Fechamento em USD',
                  legend_title='Legenda',
                  hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)

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

st.write(f'**Preço previsto para o próximo dia:** {future_price[0][0]:.4f}')

# Salvar as previsões em um arquivo Excel
output_df = valid[['Adj Close']].copy()
output_df['Predictions'] = predictions
output_df.to_excel('previsoes.xlsx', index=True)
st.success('Previsões salvas no arquivo "previsoes.xlsx"')
