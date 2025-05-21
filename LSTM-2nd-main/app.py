import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS  
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
import os
import matplotlib.pyplot as plt
import traceback
from flask_sqlalchemy import SQLAlchemy
import json
import firebase_admin
from firebase_admin import credentials, auth
import functools
from sqlalchemy import text  

app = Flask(__name__)
CORS(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


cred = credentials.Certificate({
    "type": "service_account",
    "project_id": "stocksage-dee21",
    "private_key_id": "b90ad8ed6d9518eb6ba0972411262546b94b7292",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDSEvpYv2X9Gpw6\nmLzqG3itfj2WT4/4VpxWV/uw7j3W6jNwcp34jkUIrYrTiZHAWL4WmidGjvTtixnd\nDPI/4K7AT5Xp2BtULABsVzUhSHgN6dIeZ2MzKzsN2LjDpLmkumA50RwfxrEsnF/D\n+sTjIBCthrvhaRjzWbNmCSpCg4KeRIFWWOqxV5RDOkBfdXbSpY9OYkaxP0QmtQjR\nOGrgN3oF57X21Tl6NrAye6HfK9+onpAm0Z0lxbdFG0W6zNDhgK4VLf8DSKnZNt35\np7NA+B47CQnY1h0EsR+zjGQoU7VMrB9CM2ME/rI8OZGNAoQP5YhVVgEp/5z+owdY\nYX5yAs1tAgMBAAECggEAZI/yKL7C51UwOXq5YIYzEDZQ7B39KN/pKgonMczybGmj\noy4zAfO1B1DwFpCR5ZZ1tKMprwSrKfV/NYrHgp2fee7/MDij6HjB4RskE2E6G0r1\nq1lJqwqEU/4NgJZfygPwIW1BPdLjKJxY2Zu3WZSVd0P6t46Ug5XKo8s6YCnNQs5h\nVgIOI3LptWrXhuykhtqLqalACpWBPzNPpoHFVQhC0Dk6yRtzdRzEsXER0jL0aVgU\nWv7z887K738RrMzspqeGUeWjQBNed3loidSHi2qVtVfHhCNqZbGokz6pBk/sYyDy\nHVe1l9IhwucbOA7A7wmWxS+DK0Rnx7mPfrSXGNz6fwKBgQDtn5UCo5TiXIEMV5Dh\n2uQszaRwWO5z0sPsTS7fV6/Gh05rJ8+gMMHaQBUVEfzonMblUe8glZ/9bnrZrBKX\nrwdtU/0FMmW21It+fFfCfYOvRSMuM2dcMhBJoRxbdEONltozIWFJt3IoDcVzTwk/\n3ujSiYwBTPoQ6c7J7CwqlQKh8wKBgQDiUftZNWfQyc3P0t4k4LgXI7CG0qutSYE0\nyP5+XmYwceGNmpJLDm7kix3+PFhV2+1LNkFw5GKgTrOByN1bVHE9cke+wp0act4J\n1gLUUnyOQ0/gy72IpO6Gh20CLf6Y46G2BmRQhot6cdDJTJFvMkKKgDiQkygYpyCM\n8rBPCoVLHwKBgFt15fvbDK5H3p/qgGh+QP8Bs/5OLJHGGQU1/AmZDZ4DbGqQKzlE\nKwkfSz4N62N5HJEdhUXscWUxhfWMZqedheVNw4ChkkbZ7ksj/v9sIihvGwhoXqs8\n1coRSgHlcS7pVlkByxl9k2HZUxt721qLKOAIcyv4/fNOmvpbffBx5E6DAoGAR+3g\nYajsltoclGmjersFJy2LpD3+nDOZYgpjgAxGlC0Nj7DJwBsVOYPRg4TwWlFsqJPU\n1qlvgx2gRaZuW+GJoArbLJCz170cPqoK+ipBNgHEuGBom643tQADsEan/TWTpsN/\nTGyGleLohaHVMy8ZIOXBlImAm906JjHXwTdJx+cCgYEA4wrq4fG4AiLXOr8dvmB6\nWYf9q90/eLmZezUuLnaWp24VjTUtrsNj4b235JSzEGH14o9Zc4fXkpbawM79gTqR\nkFfNe4XGjJCPQjk88H5OuxqQkbBOxW9gn5nUcPLaqF3REyCVHQbWV+yT/7mC7SV9\nhDQSBBSMM3SX3r5bfk4uwCE=\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-fbsvc@stocksage-dee21.iam.gserviceaccount.com",
    "client_id": "111284639930692208020",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40stocksage-dee21.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
})

try:
    firebase_admin.initialize_app(cred)
except ValueError:
    # App already initialized
    pass

# Authentication middleware
def auth_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized: No token provided'}), 401
        
        token = auth_header.split('Bearer ')[1]
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(token)
            # Add user_id to kwargs
            kwargs['user_id'] = decoded_token['uid']
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': f'Unauthorized: {str(e)}'}), 401
            
    return decorated_function

# Database Models
class StockPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)  # Firebase UID
    ticker = db.Column(db.String(10), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.now)
    current_price = db.Column(db.Float, nullable=False)
    predicted_prices = db.Column(db.Text, nullable=False)  # JSON string of predictions
    prediction_dates = db.Column(db.Text, nullable=False)  # JSON string of dates
    
    def __repr__(self):
        return f'<StockPrediction {self.ticker}>'

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)  # Firebase UID
    ticker = db.Column(db.String(10), nullable=False)
    added_date = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<UserPreference {self.user_id}:{self.ticker}>'

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)  # Firebase UID
    ticker = db.Column(db.String(10), nullable=False)
    analysis_date = db.Column(db.DateTime, default=datetime.now)
    successful = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text, nullable=True)
    
    def __repr__(self):
        return f'<AnalysisHistory {self.ticker}>'

# Create and initialize the database tables
def init_db():

    inspector = db.inspect(db.engine)
    
    db.create_all()
    
    existing_tables = inspector.get_table_names()
    
    if 'analysis_history' in existing_tables:
        columns = [col['name'] for col in inspector.get_columns('analysis_history')]
        if 'user_id' not in columns:
            print("Migrating analysis_history table to include user_id column...")
            
            # Backup existing data
            existing_data = []
            try:
                existing_records = AnalysisHistory.query.all()
                for record in existing_records:
                    existing_data.append({
                        'ticker': record.ticker,
                        'analysis_date': record.analysis_date,
                        'successful': record.successful,
                        'error_message': record.error_message
                    })
            except Exception as e:
                print(f"Error backing up analysis_history data: {str(e)}")
            
            # Drop and recreate the table using text() function
            db.session.execute(text('DROP TABLE analysis_history'))
            db.session.commit()
            db.create_all()
            
            # Restore data with a default user_id
            for record_data in existing_data:
                new_record = AnalysisHistory(
                    user_id='default',
                    ticker=record_data['ticker'],
                    analysis_date=record_data['analysis_date'],
                    successful=record_data['successful'],
                    error_message=record_data['error_message']
                )
                db.session.add(new_record)
            
            db.session.commit()
            print("Migration complete.")
    
    if 'stock_prediction' in existing_tables:
        columns = [col['name'] for col in inspector.get_columns('stock_prediction')]
        if 'user_id' not in columns:
            # Similar migration for stock_prediction table
            print("Migrating stock_prediction table to include user_id column...")
            # Implement similar migration logic as above
            
            # Backup existing data
            existing_data = []
            try:
                existing_records = StockPrediction.query.all()
                for record in existing_records:
                    existing_data.append({
                        'ticker': record.ticker,
                        'prediction_date': record.prediction_date,
                        'current_price': record.current_price,
                        'predicted_prices': record.predicted_prices,
                        'prediction_dates': record.prediction_dates
                    })
            except Exception as e:
                print(f"Error backing up stock_prediction data: {str(e)}")
            
            # Drop and recreate the table using text() function
            db.session.execute(text('DROP TABLE stock_prediction'))
            db.session.commit()
            db.create_all()
            
            # Restore data with a default user_id
            for record_data in existing_data:
                new_record = StockPrediction(
                    user_id='default',
                    ticker=record_data['ticker'],
                    prediction_date=record_data['prediction_date'],
                    current_price=record_data['current_price'],
                    predicted_prices=record_data['predicted_prices'],
                    prediction_dates=record_data['prediction_dates']
                )
                db.session.add(new_record)
            
            db.session.commit()
            print("Migration complete.")
    
    if 'user_preference' in existing_tables:
        columns = [col['name'] for col in inspector.get_columns('user_preference')]
        if 'user_id' not in columns:
            # Similar migration for user_preference table
            print("Migrating user_preference table to include user_id column...")
            
            # Backup existing data
            existing_data = []
            try:
                existing_records = UserPreference.query.all()
                for record in existing_records:
                    existing_data.append({
                        'ticker': record.ticker,
                        'added_date': record.added_date
                    })
            except Exception as e:
                print(f"Error backing up user_preference data: {str(e)}")
            
            # Drop and recreate the table using text() function
            db.session.execute(text('DROP TABLE user_preference'))
            db.session.commit()
            db.create_all()
            
            # Restore data with a default user_id
            for record_data in existing_data:
                new_record = UserPreference(
                    user_id='default',
                    ticker=record_data['ticker'],
                    added_date=record_data['added_date']
                )
                db.session.add(new_record)
            
            db.session.commit()
            print("Migration complete.")

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

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/analyze', methods=['GET'])
@auth_required
def analyze(user_id):
    analysis_record = None
    try:
        # Get ticker from query parameter
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
            
        # Create analysis history record with user_id
        analysis_record = AnalysisHistory(user_id=user_id, ticker=ticker)
        db.session.add(analysis_record)
        db.session.commit()

        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # Last 2 years
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            if analysis_record:
                analysis_record.successful = False
                analysis_record.error_message = 'No data found for this ticker'
                db.session.commit()
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

        # Store prediction in database with user_id
        prediction = StockPrediction(
            user_id=user_id,
            ticker=ticker,
            current_price=float(df['Close'].iloc[-1]),
            predicted_prices=json.dumps(future_prices.tolist()),
            prediction_dates=json.dumps(future_dates)
        )
        db.session.add(prediction)
        db.session.commit()

        # Response JSON with image paths
        response_data = {
            'current_price': float(df['Close'].iloc[-1]),
            'volume': int(df['Volume'].iloc[-1]),
            'high_52week': float(df['High'].max()),
            'low_52week': float(df['Low'].min()),
            'future_dates': future_dates,
            'future_prices': future_prices.tolist(),
            'predicted_next_day': float(future_prices[0]),
            'prediction_id': prediction.id  # Include prediction ID for reference
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        if analysis_record:
            analysis_record.successful = False
            analysis_record.error_message = str(e)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
        db.session.rollback()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predictions/history/<ticker>', methods=['GET'])
@auth_required
def prediction_history(ticker, user_id):
    try:
        # Get history limit from query params, default to 10
        limit = request.args.get('limit', 10, type=int)
        
        # Query for the user's predictions for the given ticker
        predictions = StockPrediction.query.filter_by(
            user_id=user_id, 
            ticker=ticker
        ).order_by(StockPrediction.prediction_date.desc()).limit(limit).all()
        
        if not predictions:
            return jsonify({'message': f'No prediction history found for {ticker}'}), 404
            
        # Format the predictions for the response
        result = []
        for pred in predictions:
            result.append({
                'id': pred.id,
                'ticker': pred.ticker,
                'prediction_date': pred.prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': pred.current_price,
                'prediction_dates': json.loads(pred.prediction_dates),
                'predicted_prices': json.loads(pred.predicted_prices)
            })
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/user/preferences', methods=['GET', 'POST', 'DELETE'])
@auth_required
def manage_preferences(user_id):
    if request.method == 'GET':
        # Get user preferences
        preferences = UserPreference.query.filter_by(user_id=user_id).all()
        result = [{'id': pref.id, 'ticker': pref.ticker, 'added_date': pref.added_date.strftime('%Y-%m-%d')} 
                 for pref in preferences]
        return jsonify(result), 200
        
    elif request.method == 'POST':
        # Add new preference
        data = request.get_json()
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
            
        # Check if already exists
        existing = UserPreference.query.filter_by(user_id=user_id, ticker=ticker).first()
        if existing:
            return jsonify({'message': 'Preference already exists'}), 200
            
        new_pref = UserPreference(user_id=user_id, ticker=ticker)
        db.session.add(new_pref)
        db.session.commit()
        
        return jsonify({'message': 'Preference added', 'id': new_pref.id}), 201
        
    elif request.method == 'DELETE':
        # Remove preference
        data = request.get_json()
        pref_id = data.get('id')
        
        if not pref_id:
            return jsonify({'error': 'Preference ID is required'}), 400
            
        pref = UserPreference.query.get(pref_id)
        if not pref or pref.user_id != user_id:
            return jsonify({'error': 'Preference not found'}), 404
            
        db.session.delete(pref)
        db.session.commit()
        
        return jsonify({'message': 'Preference removed'}), 200

@app.route('/analysis/history', methods=['GET'])
@auth_required
def get_analysis_history(user_id):
    try:
        # Get history limit from query params, default to 20
        limit = request.args.get('limit', 20, type=int)
        
        # Query for the user's analysis history
        history = AnalysisHistory.query.filter_by(
            user_id=user_id
        ).order_by(AnalysisHistory.analysis_date.desc()).limit(limit).all()
        
        result = []
        for entry in history:
            result.append({
                'id': entry.id,
                'ticker': entry.ticker,
                'analysis_date': entry.analysis_date.strftime('%Y-%m-%d %H:%M:%S'),
                'successful': entry.successful,
                'error_message': entry.error_message
            })
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

db_initialized = False

@app.before_request
def before_request_func():
    global db_initialized
    if not db_initialized:
        init_db()
        db_initialized = True

# Make sure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

if __name__ == '__main__':
    with app.app_context():
        init_db()  # Initialize and migrate database tables as needed
    app.run(debug=True)