#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import matplotlib

# Use 'Agg' backend for non-GUI plotting
matplotlib.use('Agg')

# Initialise Flask app
app = Flask(__name__)

# Load trained model
try:
    model = joblib.load('netflix_kmeans_model.pkl')
    model_name = 'K-Means (netflix_kmeans_model.pkl)'
    print(f"Model loaded successfully. Model expects {model.n_features_in_} features.")
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'netflix_kmeans_model.pkl' exists.")
    exit()

# Load dataset for visualization
try:
    df = pd.read_csv('Netflix_Dataset.csv')
    numeric_data = df.select_dtypes(include=[np.number])

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(numeric_data)
    labels = model.predict(numeric_data.values)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get 4 feature values from the form
        features = [float(request.form[f'feature{i}']) for i in range(1, 5)]

        # Predict using the model
        prediction = model.predict([features])[0]

        # Perform PCA on features for visualization
        input_data_2d = pca.transform([features])

        plt.figure(figsize=(8, 6))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', edgecolors='k')
        plt.scatter(input_data_2d[0, 0], input_data_2d[0, 1], color='red', s=200, edgecolors='k', label='Your Data')
        plt.title(f'Cluster Visualization using PCA - Predicted Cluster {prediction}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)

        # Save the image
        image_path = os.path.join('static', 'prediction_plot.png')
        plt.savefig(image_path)
        plt.close()

        return render_template('result.html', prediction=f'Cluster {prediction}', image_path=image_path)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

