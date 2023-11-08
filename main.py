import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings

today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download(
    'GOOG',
    start=start_date,
    end=end_date,
    progress=False
)

data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())

data = data[["Date", "Close"]]
print(data.head())

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
""" plt.plot(data["Date"], data["Close"])
plt.show() """

""" result = seasonal_decompose(
    data["Close"],
    model='multiplicative',
    period=30
)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(15, 10) """
""" plt.show() """

""" autocorr = pd.plotting.autocorrelation_plot(data["Close"])
autocorr.plot() """
""" plt.show() """

""" plot_pacf(data["Close"], lags=100) """
""" plt.show() """

p, d, q = 5, 1, 2
""" model = ARIMA(data["Close"], order=(p,d,q))
fitted = model.fit()
print(fitted.summary())
predictions = fitted.predict()
print(predictions) """

model = sm.tsa.statespace.SARIMAX(
    data["Close"],
    order=(p, d, q),
    seasonal_order=(p, d, q, 12)
)
model = model.fit()
""" print(model.summary()) """
predictions = model.predict(len(data), len(data)+10)
""" print(predictions) """

data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")
plt.show()