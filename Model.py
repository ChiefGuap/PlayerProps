from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('nba_data_regular_season.csv')
# Drop rows with any NaN values
data= data.dropna(axis=0, how="any")


# Select relevant features and target variable
X = data[['MIN', 'FGM', 'FGA', 'FTM', 'FTA', 'EFF', 'TOV', 'AST', 'FG3A', 'FG3M', 'DREB']]  # input features
y = data['PTS']  # Target variable


# Initialize the Label Encoder
label_encoder = LabelEncoder()

# Fit the encoder with the player names and transform them into integers
data['PLAYER_encoded'] = label_encoder.fit_transform(data['PLAYER'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train) # training 

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# test our model on the averages CSV file
# Create a mapping from 'PLAYER_encoded' to 'PLAYER'
player_mapping = data[["PLAYER_encoded", "PLAYER"]].drop_duplicates().set_index("PLAYER_encoded")

# Example player name input by the user
player_name = "LeBron James"  # Change to USER INPUT/function arguments!!!

# Find the encoded value for the given player name
player_encoded = player_mapping[player_mapping["PLAYER"] == player_name].index[0]  # Get the encoded value

averages = pd.read_csv('average_values.csv')
row = averages.iloc[player_encoded]

# Features used in the model
features = ['MIN', 'FGM', 'FGA', 'FTM', 'FTA', 'EFF', 'TOV', 'AST', 'FG3A', 'FG3M', 'DREB']

# Extract the row for the specified player (after correction)
row = averages.loc[player_encoded, features]

# Reshape the data and ensure it's in DataFrame format with proper feature names
row_features_df = pd.DataFrame([row], columns=features)  # Single-row DataFrame with feature names

# Predict using the model
prediction = model.predict(row_features_df)

print(f"Prediction for {player_name}: {prediction[0]}")