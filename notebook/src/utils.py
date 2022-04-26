import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_data(stock, START, END, col='Close'):
    data = yf.download(stock, START, END)[[col]]
    return data


def gen_seq(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return Variable(torch.Tensor(np.array(x))), Variable(torch.Tensor(np.array(y)))



def create_forecast_index(start, horizon=30, freq="M"):
    return pd.date_range(start + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq, name="Time")




def recursive_forecast(input_data, model, n=20, responses=1):
    forecast = np.empty((n, responses))  # where we'll store our forecasts
    for i, n in enumerate(range(n)):     # loop for making forecasts one at a time
        forecast[i] = model.predict(input_data.reshape(1, -1))  # model forecast
        input_data = np.append(forecast[i], input_data[:-responses])  # append forecast to input data for next forecast
    return forecast.reshape((-1, responses))



def lag_df(df, lag=1, cols=None):
    if cols is None:
        cols = df.columns
    return df.assign(
        **{f"{col}-{n}": df[col].shift(n) for n in range(1, lag + 1) for col in cols}
    )