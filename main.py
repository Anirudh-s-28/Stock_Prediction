from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import yfinance as yf
import numpy as np
import pandas as pd

app = FastAPI()

# Load models
lstm_model = joblib.load("lstm_model.pkl")
arima_model = joblib.load("arima_model.pkl")

class StockRequest(BaseModel):
    symbol: str
    timeframe: str

@app.post("/predict")
def predict_stock(req: StockRequest):
    # Fetch stock data
    data = yf.download(req.symbol, period=req.timeframe)

    selected_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    stock_data = data[selected_columns]

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    stock_data.to_csv('final_stock_data.csv')

    df = pd.read_csv('final_stock_data.csv', index_col='Date', parse_dates=True)

    full_range = pd.date_range(start=df.index.min(), end=df.index.max())

    df_full = df.reindex(full_range)

    df_full = df_full.interpolate(method='linear')

    df_full.index.name = 'Date'

    df_full['Date']=pd.to_datetime(df_full['Date'])
    df_full.set_index('Date',inplace=True)

    df_full.to_csv('final_stock_data_filled.csv')

    
    # Preprocess for ARIMA/LSTM as you did in your notebook
    # Example placeholder:
    arima_pred = arima_model.forecast(steps=7)
    lstm_input = np.array(data['Close'][-60:]).reshape(1, 60, 1)
    lstm_pred = lstm_model.predict(lstm_input)
    
    # Combine ARIMA + LSTM predictions (Hybrid)
    final_pred = (arima_pred[-len(lstm_pred):] + lstm_pred.flatten()) / 2
    
    return {
        "symbol": req.symbol,
        "predictions": final_pred.tolist()
    }
