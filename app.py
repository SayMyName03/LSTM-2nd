from flask import Flask, render_template, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get ticker from form
        ticker = request.form.get('ticker')
        if not ticker:
            return jsonify({'error': 'No ticker provided'})
        
        # Get dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Load data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return jsonify({'error': 'No data found for this ticker'})
        
        # Convert pandas Series to lists properly
        response_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'closing_prices': df['Close'].values.tolist(),  # Fixed this line
            'current_price': float(df['Close'].iloc[-1]),
            'volume': int(df['Volume'].iloc[-1]),
            'high_52week': float(df['High'].max()),
            'low_52week': float(df['Low'].min()),
            'prediction': float(df['Close'].iloc[-1]) * 1.01,  # Dummy prediction
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
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error: {str(e)}")  # This will print to your terminal
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)