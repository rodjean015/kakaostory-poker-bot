import requests
import json

class PokerModelAPI:
    def __init__(self, base_url):
        """
        Initializes the API class with the base URL.
        
        Args:
            base_url (str): The base URL of the API.
        """
        self.base_url = base_url

    def train_model(self):
        """
        Sends a POST request to the /train endpoint to train the model.
        """
        response = requests.post(f"{self.base_url}/train")
        if response.status_code == 200:
            print(f"Model trained successfully: {response.json()}")
        else:
            print(f"Failed to train the model: {response.status_code}")

    def predict_hand(self, data):
        """
        Sends a POST request to the /predict endpoint to predict the action for a hand.
        
        Args:
            data (list): A list of features representing a poker hand.
        """
        headers = {'Content-Type': 'application/json'}
        payload = {'data': data}

        response = requests.post(f"{self.base_url}/predict", headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"Prediction: {response.json()['prediction']}")
        else:
            print(f"Failed to predict: {response.status_code}, {response.text}")

    def append_game_data(self, new_game_data):
        """
        Sends a POST request to the /append_game_data endpoint to append new game data.
        
        Args:
            new_game_data (list): A list of new game data to be appended to the data file.
        """
        headers = {'Content-Type': 'application/json'}
        payload = {'data': new_game_data}

        response = requests.post(f"{self.base_url}/append_game_data", headers=headers, json=payload)
        
        if response.status_code == 200:
            print(response.json()['message'])
        else:
            print(f"Failed to append game data: {response.status_code}, {response.json()}")

if __name__ == '__main__':
    # Initialize the API with the base URL
    api = PokerModelAPI(base_url='https://flask-api-902323753008.us-central1.run.app')

    # 1. Train the model
    api.train_model()

    # 2. Predict a hand: Provide sample input data for a poker hand
    sample_hand = [10, 14, 2, 3, 4, 5, 6, 2, 1, 0, 14, 10, 1, 2, 3, 4, 5, 4]  # 17 features
    api.predict_hand(sample_hand)

    # 3. Append new game data
    new_game_data = [10, 14, 2, 3, 4, 5, 6, 2, 1, 0, 14, 10, 1, 2, 3, 4, 5, 3, 2]  # Example includes the action
    api.append_game_data(new_game_data)
