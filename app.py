from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
import matplotlib.pyplot as plt
import traceback

app = Flask(__name__)
CORS(app)

def create_lstm_model(X_train):
    # Create and compile LSTM model with improved architecture
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.3),
        LSTM(75, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def prepare_lstm_data(df, time_step=60):
    # We'll use the 'Close' prices for prediction
    data = df['Close'].values.reshape(-1, 1)
    
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Determine training size (80% of data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, data

def predict_future(model, last_sequence, scaler, days_to_predict=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        # Reshape for model input
        current_sequence_reshaped = current_sequence.reshape(1, current_sequence.shape[0], 1)
        
        # Get prediction (scaled)
        next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        
        # Store prediction
        future_predictions.append(next_pred_scaled)
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], next_pred_scaled)
    
    # Inverse transform to get actual prices
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_predictions)
    
    return future_prices.flatten()

def plot_price_ma(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label="Closing Price", color='blue')
    plt.plot(df.index, df['MA100'], label="100-Day MA", color='green', linestyle='dashed')
    plt.plot(df.index, df['MA200'], label="200-Day MA", color='red', linestyle='dashed')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Stock Price & Moving Averages")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('static/price_ma_chart.png')
    plt.close()

def plot_actual_vs_predicted(df, test_predictions, ticker):
    plt.figure(figsize=(12, 6))
    pred_dates = df.index[len(df) - len(test_predictions):]
    plt.plot(pred_dates, df['Close'].values[-len(test_predictions):], label="Actual Prices", color='blue')
    plt.plot(pred_dates, test_predictions.flatten(), label="Predicted Prices", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Actual vs Predicted Prices")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('static/actual_vs_predicted_chart.png')
    plt.close()

def plot_future_predictions(future_dates, future_prices, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_prices, marker='o')
    plt.title(f"{ticker} 30-Day Future Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/future_prediction_chart.png')
    plt.close()

@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        # Get ticker from query parameter
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400

        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # Last 2 years
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return jsonify({'error': 'No data found for this ticker'}), 404

        # Calculate moving averages
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        # Prepare data for LSTM
        time_step = 60
        X_train, X_test, y_train, y_test, scaler, data = prepare_lstm_data(df, time_step)

        # Train LSTM model
        model = create_lstm_model(X_train)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        # Predictions on test data
        test_predictions = model.predict(X_test, verbose=0)
        test_predictions = scaler.inverse_transform(test_predictions)

        # Future predictions
        last_sequence = scaler.transform(data[-time_step:])
        future_prices = predict_future(model, last_sequence.flatten(), scaler, days_to_predict=30)

        # Generate future dates
        last_date = df.index[-1]
        future_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(len(future_prices))]

        # Plot charts
        plot_price_ma(df, ticker)
        plot_actual_vs_predicted(df, test_predictions, ticker)
        plot_future_predictions(future_dates, future_prices, ticker)

        # Response JSON with image paths
        response_data = {
            'current_price': float(df['Close'].iloc[-1]),
            'volume': int(df['Volume'].iloc[-1]),
            'high_52week': float(df['High'].max()),
            'low_52week': float(df['Low'].min()),
            'future_dates': future_dates,
            'future_prices': future_prices.tolist(),
            'predicted_next_day': float(future_prices[0]),
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Make sure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

if __name__ == '__main__':
    app.run(debug=True)