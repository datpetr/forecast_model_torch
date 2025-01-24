# N-BEATS Time Series Forecasting

This repository contains an implementation of the N-BEATS model for time series forecasting using PyTorch. The model predicts future values of a target variable based on historical data.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Dataset and DataLoader](#dataset-and-dataloader)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Forecasting](#forecasting)
7. [Visualization](#visualization)

---

## Introduction
The N-BEATS model is a neural network-based approach designed for time series forecasting. It processes input sequences to predict future values and handles multiple variables. This implementation trains the model on a dataset and generates forecasts for unseen data.

---

## Data Preparation
We load the dataset from an Excel file, rename the columns for better readability, and prepare the data by separating it into training and forecasting sets. Missing target values in the forecast dataset are handled.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('BST_DataSet.xlsx')
columns_russian = [
    "Прирост вкладов физических лиц в рублях (млн руб)",
    "Доходность ОФЗ по сроку до 1 года",
    "Ключевая ставка",
    "Ставка по вкладам в долларах до 1 года",
    "Ставка по вкладам в рублях до 1 года",
    "Нефть марки Юралс, долл./барр",
    "Индекс потребительских цен, ед.",
    "М0, млрд руб",
    "М2, млрд руб",
    "Курс доллара к рублю, руб",
    "Номинальная средняя з/п, руб"
]

columns_english = [
    "deposit_growth_rubles_mln",
    "ofz_yield_upto_1_year",
    "key_rate",
    "deposit_rate_dollars_upto_1_year",
    "deposit_rate_rubles_upto_1_year",
    "urals_oil_usd_per_barrel",
    "consumer_price_index",
    "m0_bln_rubles",
    "m2_bln_rubles",
    "usd_to_rub_exchange_rate",
    "average_nominal_salary_rubles"
]

df.rename(columns=dict(zip(columns_russian, columns_english)), inplace=True)
df.rename(columns={df.columns[0]: "date"}, inplace=True)
df["time_idx"] = df.index

# Separate training and forecast datasets
target_column = "deposit_growth_rubles_mln"
df_train = df[df[target_column].notna()]
df_forecast = df[df[target_column].isna()]

# Generate time series and normalize
sequence_length = 2

def generate_time_series_data_from_df(df, target_column, sequence_length):
    series = df.drop(columns=["date", "time_idx", target_column]).values
    y = df[target_column].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series)
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X, Y = [], []
    for i in range(len(series) - sequence_length):
        X.append(series[i:i + sequence_length])
        Y.append(y[i + sequence_length])
    return np.array(X), np.array(Y), scaler, y_scaler

X, Y, scaler, y_scaler = generate_time_series_data_from_df(df_train, target_column, sequence_length)
```

---

## Dataset and DataLoader
The dataset is prepared for PyTorch by creating a custom dataset class and using DataLoader for batch processing.

```python
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
```

---

## Model Architecture
The N-BEATS model consists of blocks that process the input and generate forecasts. Each block uses fully connected layers and ReLU activation.

```python
import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, num_hidden, forecast_length):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, forecast_length)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class NBeats(nn.Module):
    def __init__(self, input_size, num_blocks, num_hidden, forecast_length):
        super(NBeats, self).__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, num_hidden, forecast_length) for _ in range(num_blocks)])

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = torch.zeros(x.size(0), 1).to(x.device)
        for block in self.blocks:
            out += block(x)
        return out

input_size = X.shape[2] * sequence_length
num_blocks = 2
num_hidden = 8
forecast_length = 1

model = NBeats(input_size, num_blocks, num_hidden, forecast_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## Training the Model
The model is trained using the Mean Squared Error (MSE) loss and the Adam optimizer. Training progress is printed for each epoch.

```python
epochs = 400
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.reshape(batch_X.size(0), -1)
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze(-1)
        loss = criterion(predictions, batch_Y)
        if torch.isnan(loss):
            raise ValueError("Loss value is NaN. Check your model input or target data for invalid values.")
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
```

---

## Forecasting
The trained model is used to predict future values. The forecast process includes appending the last known value to the forecast dataset and normalizing the features.

```python
model.eval()

last_known_value = df_train.drop(columns=["date", "time_idx", target_column]).iloc[-1].values
X_forecast = df_forecast.drop(columns=["date", "time_idx", target_column]).values
X_forecast = np.vstack([last_known_value, X_forecast])
X_forecast = scaler.transform(X_forecast)

forecast_sequence = []
for i in range(len(X_forecast) - sequence_length + 1):
    forecast_sequence.append(X_forecast[i:i + sequence_length])

forecast_sequence = np.array(forecast_sequence)
predictions = []
with torch.no_grad():
    for i in range(forecast_sequence.shape[0]):
        input_seq = torch.from_numpy(forecast_sequence[i].reshape(1, -1)).float()
        pred = model(input_seq).squeeze(-1)
        predictions.append(pred.item())

predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

start_idx =

