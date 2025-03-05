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

app = Flask(__name__)
CORS(app)

def create_lstm_model(X_train):
    # Create and compile LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
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

@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get ticker from form
        ticker = request.form.get('ticker')
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
        
        # Get dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years of data for better training
        
        # Load data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return jsonify({'error': 'No data found for this ticker'}),404
        
        # Calculate moving averages
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Prepare data for LSTM
        time_step = 60  # Use 60 days of data to predict the next day
        X_train, X_test, y_train, y_test, scaler, data = prepare_lstm_data(df, time_step)
        
        # Create and train the model
        model = create_lstm_model(X_train)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # Make predictions on test data
        test_predictions = model.predict(X_test, verbose=0)
        test_predictions = scaler.inverse_transform(test_predictions)
        
        # Prepare data for plotting test predictions (actual vs predicted)
        pred_dates = df.index[len(df) - len(test_predictions):].strftime('%Y-%m-%d').tolist()
        actual_prices = df['Close'].values[-len(test_predictions):].tolist()
        predicted_prices = test_predictions.flatten().tolist()
        
        # Predict future prices
        last_sequence = scaler.transform(data[-time_step:])
        future_prices = predict_future(model, last_sequence.flatten(), scaler, days_to_predict=30)
        
        # Generate dates for future predictions
        last_date = df.index[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(future_prices))]
        
        # Convert pandas Series to lists properly
        response_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'closing_prices': df['Close'].values.tolist(),
            'ma100': df['MA100'].fillna(0).tolist(),
            'ma200': df['MA200'].fillna(0).tolist(),
            'current_price': float(df['Close'].iloc[-1]),
            'volume': int(df['Volume'].iloc[-1]),
            'high_52week': float(df['High'].max()),
            'low_52week': float(df['Low'].min()),
            'pred_dates': pred_dates,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices,
            'future_dates': future_dates,
            'future_prices': future_prices.tolist(),
            'predicted_next_day': float(future_prices[0]),
            'ohlc_data': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close'])
                }
                for date, row in df.iterrows()
            ]
        }
        
        return jsonify(response_data),200
    
    except Exception as e:
        print(f"Error: {str(e)}")  # This will print to your terminal
        return jsonify({'error': f'Server error: {str(e)}'}),500

# Make sure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

if __name__ == '__main__':
    app.run(debug=True)