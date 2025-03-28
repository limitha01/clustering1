from flask import Flask, render_template, request
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the models
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values with correct column names
        duration_minutes = float(request.form['feature1'])
        duration_seasons = float(request.form['feature2'])
        show_id = float(request.form['feature3'])
        listed_in = float(request.form['feature4'])
        
        # Convert date_added to days since a reference date (1970-01-01)
        date_added_str = request.form['date_added']
        date_added = datetime.strptime(date_added_str, '%Y-%m-%d')
        days_since = (date_added - datetime(1970, 1, 1)).days
        
        # Combine all features for prediction
        features = [duration_minutes, duration_seasons, show_id, listed_in, days_since]
        scaled_features = scaler.transform([features])
        
        # Predict using K-Means
        cluster = kmeans_model.predict(scaled_features)[0]

        return render_template('result.html', prediction=f"Cluster {cluster}")
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
