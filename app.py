from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from google.cloud import storage
import pickle
import io

app = Flask(__name__)

class PokerModel:
    def __init__(self, bucket_name, data_file):
        self.bucket_name = bucket_name  # GCS bucket name
        self.data_file = data_file      # Path to .pkl file in GCS

        # Initialize the Google Cloud Storage client
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket_name)

        # Load existing data from GCS or create a new DataFrame
        if self.file_exists_in_gcs(self.data_file):
            self.df = self.load_data_from_gcs(self.data_file)
        else:
            columns = ['card1_rank', 'card2_rank', 'flop1', 'flop2', 'flop3', 'turn', 'river',
                       'position_encoded', 'is_suited', 'is_pair', 'high_card', 'low_card',
                       'Opponent A', 'Opponent B', 'Opponent C', 'Opponent D', 'Opponent E', 'Opponent F', 'action']
            self.df = pd.DataFrame(columns=columns)

        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.actions = {0: "Fold", 1: "Check", 2: "Call", 3: "Raise", 4: "All_in"}

    def file_exists_in_gcs(self, file_name):
        """Check if the file exists in the GCS bucket."""
        blob = self.bucket.blob(file_name)
        return blob.exists()

    def load_data_from_gcs(self, file_name):
        """Load pickle data from Google Cloud Storage."""
        blob = self.bucket.blob(file_name)
        data = blob.download_as_bytes()
        return pd.read_pickle(io.BytesIO(data))

    def save_data_to_gcs(self, file_name):
        """Save pickle data to Google Cloud Storage."""
        blob = self.bucket.blob(file_name)
        buffer = io.BytesIO()
        self.df.to_pickle(buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')

    def feature_engineering(self):
        """Create additional poker features for hand strength."""
        self.df['is_pair'] = (self.df['card1_rank'] == self.df['card2_rank']).astype(int)
        self.df['high_card'] = self.df[['card1_rank', 'card2_rank']].max(axis=1)
        self.df['low_card'] = self.df[['card1_rank', 'card2_rank']].min(axis=1)

    def prepare_data(self):
        self.feature_engineering()
        opponent_regions = ['A', 'B', 'C', 'D', 'E', 'F'] 
        opponent_move_columns = [f'Opponent {region}' for region in opponent_regions]

        X = self.df[['card1_rank', 'card2_rank', 'flop1', 'flop2', 'flop3', 'turn', 'river', 
                     'position_encoded', 'is_suited', 'is_pair', 'high_card', 'low_card'] + opponent_move_columns].to_numpy()
        y = self.df['action'].to_numpy()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.clf.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def predict_hand(self, new_hand):
        predicted_action = self.clf.predict(new_hand)
        return self.actions[predicted_action[0]]

    def append_real_game_data(self, new_game_data):
        """Append new game data and save to Google Cloud Storage."""
        new_data_df = pd.DataFrame([new_game_data], columns=self.df.columns)
        self.df = self.df._append(new_data_df, ignore_index=True)

        # Save the updated data back to GCS
        self.save_data_to_gcs(self.data_file)

        # Optionally, also save a backup
        appended_data_file = "new_train_data.pkl"
        self.save_data_to_gcs(appended_data_file)


# Initialize the PokerModel with the GCS bucket and data file
poker_model = PokerModel(bucket_name='your-gcs-bucket-name', data_file='train_data.pkl')

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    poker_model.prepare_data()
    poker_model.train_model()
    accuracy = poker_model.evaluate_model()
    return jsonify({'message': 'Model trained successfully!', 'accuracy': f'{accuracy*100:.2f}%'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Expecting a list of data for a new hand
    new_hand = np.array([data])
    prediction = poker_model.predict_hand(new_hand)
    return jsonify({'prediction': prediction})

@app.route('/append_game_data', methods=['POST'])
def append_data():
    data = request.json['data']  # New game data
    if len(data) == len(poker_model.df.columns):
        poker_model.append_real_game_data(data)
        return jsonify({'message': 'Game data appended successfully!'})
    else:
        return jsonify({'error': 'Column mismatch'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
