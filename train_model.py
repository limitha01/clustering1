#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
try:
    df = pd.read_csv('Netflix_Dataset.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❗ Error: Netflix_Dataset.csv not found.")
    exit()

# Select specific columns for clustering
selected_columns = ['duration_minutes', 'duration_seasons', 'show_id', 'listed_in', 'date_added']

# Handle date column by converting to days since 1970-01-01
if 'date_added' in df.columns:
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['days_since'] = (df['date_added'] - pd.Timestamp('1970-01-01')).dt.days
    selected_columns.remove('date_added')
    selected_columns.append('days_since')

# Select only numeric columns
numeric_data = df[selected_columns].dropna()

if numeric_data.empty:
    print("❗ Error: No valid numeric data found for training.")
    exit()

print("Selected Columns for Training:", numeric_data.columns)

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Train the K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(scaled_data)

# Save the model and scaler
joblib.dump(kmeans, 'netflix_kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully.")
