# Food Recommendation System

## Overview

The Food Recommendation System is a machine learning-based web application designed to predict and recommend foods to users based on their eating patterns and the time of day. This system leverages TensorFlow for model building, Pandas for data manipulation, and Flask to create a RESTful API for serving predictions.

## Features

- **Data Fetching:** Fetches user food log data from a PostgreSQL database.
- **Data Preprocessing:** Preprocesses data including encoding categorical features and scaling recency of food consumption.
- **Model Building:** Creates a neural network model using user embedding and feature inputs to predict top food recommendations.
- **REST API:** Provides a REST API for making predictions based on user ID and time of day.


## API Endpoints

### Predict Food Recommendations

- **Endpoint:** `/predict`
- **Method:** `GET`
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "time_of_day": "string"  # "Morning", "Afternoon", "Evening", "Night"
  }
  ```
- **Response**
  ```json
  [
    {
      "rank": 1,
      "time_of_day": "Morning",
      "food_name": "apple",
      "probability": "0.9"
    },
    {
      "rank": 2,
      "time_of_day": "Morning",
      "food_name": "banana",
      "probability": "0.7"
    },
    {
      "rank": 3,
      "time_of_day": "Morning",
      "food_name": "oatmeal",
      "probability": "0.6"
    },
    {
      "rank": 4,
      "time_of_day": "Morning",
      "food_name": "toast",
      "probability": "0.5"
    },
    {
      "rank": 5,
      "time_of_day": "Morning",
      "food_name": "yogurt",
      "probability": "0.4"
    }
  ]
  ```
## Project Structure

- `app.py`: Flask application file.
- `predictor.py`: Contains the Predictor class for data fetching, preprocessing, model training, and predictions.
- `requirements.txt`: Lists the Python dependencies.
- `.env`: Environment variables for database credentials (not included in the repository).
- `data/`: Directory to store fetched data.
- `models/`: Directory to store the trained model.

## How It Works

1. **Data Fetching:** The predictor fetches user food log data from the PostgreSQL database.
2. **Data Preprocessing:** Data is preprocessed, including encoding categorical features (time of day) and scaling the recency of food consumption.
3. **Model Creation:** A neural network model is created using user embeddings and feature inputs.
4. **Model Training:** The model is trained to predict food recommendations.
5. **Predictions:** Given a user ID and time of day, the model predicts the top N food recommendations.
