from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib

# Use 'Agg' backend for non-GUI plotting
matplotlib.use('Agg')

app = Flask(__name__)

# Load the models
try:
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print(f"Model loaded successfully. Model expects {kmeans_model.n_features_in_} features.")
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'kmeans_model.pkl' and 'scaler.pkl' exist.")
    exit()

# Load dataset for visualization
try:
    df = pd.read_csv('Netflix_Dataset.csv')
    numeric_data = df[['duration_minutes', 'duration_seasons', 'show_id', 'listed_in']].copy()

    # Convert 'date_added' to days since 1970-01-01
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['days_since'] = (df['date_added'] - pd.Timestamp('1970-01-01')).dt.days

    numeric_data['days_since'] = df['days_since']

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(numeric_data.fillna(0))
    labels = kmeans_model.predict(numeric_data.fillna(0))
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
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
        cluster = kmeans_model.predict(scaled_features)[0]

        # Perform PCA on features for visualization
        input_data_2d = pca.transform([features])

        plt.figure(figsize=(8, 6))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', edgecolors='k')
        plt.scatter(input_data_2d[0, 0], input_data_2d[0, 1], color='red', s=200, edgecolors='k', label='Your Data')
        plt.title(f'Cluster Visualization using PCA - Predicted Cluster {cluster}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)

        # Save the image
        image_path = os.path.join('static', 'prediction_plot.png')
        plt.savefig(image_path)
        plt.close()

        return render_template('result.html', prediction=f'Cluster {cluster}', image_path=image_path)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
