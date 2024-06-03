

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
import psycopg2
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from dotenv import load_dotenv
load_dotenv()

# Function to apply MinMaxScaler by user
def scale_recency(group):
    scaler = MinMaxScaler()
    group['recency_days'] = scaler.fit_transform(group[['recency_days']])
    return group

class Predictor:
    def __init__(self):
        print('Initializing predictor')
        self.model_path = './models/predict_model.keras' #os.getenv("MODEL_PATH")
        self.top_n = 5  # Define the top_n value

    
    def model_exists(self):
        if os.path.exists(self.model_path):
            return True
        return False    
    
    def fetch_data(self):
        print('Fetching data')
        DBNAME = os.getenv("DBNAME")
        HOST = os.getenv("DBHOST")
        USER = os.getenv("DBUSER")
        PASSWORD = os.getenv("DBPASSWORD")
        PORT = os.getenv("DBPORT")

        conn = psycopg2.connect(dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
        cursor = conn.cursor()
        try: 
            cursor.execute("""--sql
            SELECT
                f.user_uuid,
                log_id,
                protein,
                carbohydrates,
                fats,
                calories,
                servings,
                log_date,
                LOWER(TRIM(food_name)) as food_name,
                serving_size,
                group_id,
                CASE
                    WHEN EXTRACT(HOUR FROM log_date AT TIME ZONE 'UTC' AT TIME ZONE u.timezone) >= 0 AND EXTRACT(HOUR FROM log_date AT TIME ZONE 'UTC' AT TIME ZONE u.timezone) < 6 THEN 'Night'
                    WHEN EXTRACT(HOUR FROM log_date AT TIME ZONE 'UTC' AT TIME ZONE u.timezone) >= 6 AND EXTRACT(HOUR FROM log_date AT TIME ZONE 'UTC' AT TIME ZONE u.timezone) < 12 THEN 'Morning'
                    WHEN EXTRACT(HOUR FROM log_date AT TIME ZONE 'UTC' AT TIME ZONE u.timezone) >= 12 AND EXTRACT(HOUR FROM log_date AT TIME ZONE 'UTC' AT TIME ZONE u.timezone) < 18 THEN 'Afternoon'
                    ELSE 'Evening'
                END AS time_of_day
            FROM
                food_log f
            JOIN 
                users u
            ON
                f.user_uuid = u.user_uuid
            GROUP BY 
                f.user_uuid,
                log_id,
                protein,
                carbohydrates,
                fats,
                calories,
                servings,
                log_date,
                food_name,
                serving_size,
                group_id,
                time_of_day
            ORDER BY
                log_date DESC

            """)

            records = cursor.fetchall()

            # Create a DataFrame from the fetched records
            df = pd.DataFrame(records, columns=[
                'user_uuid',
                'log_id',
                'protein',
                'carbohydrates',
                'fats',
                'calories',
                'servings',
                'log_date',
                'food_name',
                'serving_size',
                'group_id',
                'time_of_day'
            ])


            cursor.close()
            conn.close()
            # # Save to file
            df.to_csv('./data/ALL_users.csv', index=False)
        except Exception as e:
            print('ERROR',e)
            conn.rollback()
        return

    
    def preprocess_data(self):
        print('Preprocessing data')
        # Load the data
        users = pd.read_csv('./data/ALL_users.csv')

        # Preprocess the data
        usersData = users[['user_uuid', 'food_name', 'log_date', 'time_of_day']].copy()  # Make a copy of the DataFrame
        usersData['food_name'] = usersData['food_name'].str.lower()  # Convert food names to lowercase

        # Convert log_date to datetime
        usersData['log_date'] = pd.to_datetime(usersData['log_date'])

        # Get today's date
        today = pd.to_datetime('today')

        # Compute recency_days by user
        usersData['recency_days'] = usersData.groupby('user_uuid')['log_date'].transform(lambda x: (today - x).dt.days.clip(lower=0))

        usersData = usersData.groupby('user_uuid').apply(scale_recency)

        # Apply weight to recency_days
        WEIGHT_RECENCY = 1
        usersData['recency_days'] = (1 - usersData['recency_days']) * WEIGHT_RECENCY

        # Drop log_date column
        usersData.drop(columns=['log_date'], inplace=True)

        # Save the updated DataFrame to a CSV file
        usersData.to_csv('./data/ALLusersDataRecency.csv', index=False)

        return
        
    def load_data(self):
        print('Loading data')

        df = pd.read_csv('./data/ALLusersDataRecency.csv')

        df['time_of_day'] = pd.Categorical(df['time_of_day'], categories=['Morning', 'Afternoon', 'Evening', 'Night'])

        # rename user_uuid to user_id
        df.rename(columns={'user_uuid': 'user_id'}, inplace=True)

        data_encoded = pd.get_dummies(df[['time_of_day']]) # time of day is Night, Morning, Afternoon, Evening
        data_encoded = pd.concat([df[['food_name']], data_encoded], axis=1) # food_name is a string 

        data_encoded = pd.concat([data_encoded, df['recency_days']], axis=1)

        y = df['food_name'].values

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)

        self.user_encoder = LabelEncoder()
        self.user_encoder.fit(df['user_id'])
        encoded_user = self.user_encoder.transform(df['user_id'])


        data_encoded['food_encoded'] = encoded_y

        processed_data = data_encoded.drop(['food_name'], axis=1)

        user_id = pd.DataFrame(encoded_user, columns=['user_id'])

        processed_data = pd.concat([user_id, processed_data], axis=1)

        self.data = processed_data


    def create_model(self): 
        print('Creating model')
        features = ['time_of_day_Morning', 'time_of_day_Afternoon', 'time_of_day_Evening', 'time_of_day_Night', 'recency_days']
        target = 'food_encoded'
        user_feature = 'user_id'
        # self.top_n = 5

        # # Split the data
        # X = self.data[features + [user_feature]]
        # y = self.data[target]

        num_features = len(features)

        num_users = len(self.user_encoder.classes_)

        embedding_dim = 8

        num_classes = len(self.label_encoder.classes_)

        user_input = Input(shape=(1,), name='user_input')
        feature_input = Input(shape=(num_features,), name='feature_input')
        # Define the user embedding layer
        user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
        user_embedding = Flatten()(user_embedding)

        # Concatenate the user embedding with the feature input
        concat = Concatenate()([user_embedding, feature_input])

        # Define the rest of the model
        hidden1 = Dense(128, activation='relu')(concat)
        # hidden2 = Dense(64, activation='relu')(hidden1)
        # hidden3 = Dense(32, activation='relu')(hidden2)
        output = Dense(num_classes, activation='softmax', name='output')(hidden1)

        # Create the model
        model = Model(inputs=[user_input, feature_input], outputs=output)

        # Compile the model
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=self.top_n)])

        model.summary()

        print('Fitting model')
        model.fit([self.data[user_feature], self.data[features]], self.data[target], epochs=100, batch_size=32)


        model.save(self.model_path)

    def predict(self, user_id, time_of_day):
        print('Predicting')
        model = tf.keras.models.load_model(self.model_path)
        # Get the maximum recency_days value
        maxDay = 1

        # # Prepare permutations for different times of day with max recency_days
        
        # Define time of day categories
        time_of_day_categories = ['Morning', 'Afternoon', 'Evening', 'Night']

         # Create the permutation based on the input time_of_day
        time_of_day_index = time_of_day_categories.index(time_of_day)
        permutations_day = np.zeros((1, len(time_of_day_categories) + 1), dtype=np.float32)
        permutations_day[0, time_of_day_index] = 1
        permutations_day[0, -1] = maxDay


        # Encode the specific user ID
        user_encoded = self.user_encoder.transform([user_id])[0]
        user_encoded_array = np.array([user_encoded])
        # Make predictions
        predictions = model.predict([user_encoded_array, permutations_day])

        # Get top N predictions
        top_n_indices = np.argsort(predictions, axis=1)[:, -self.top_n:][:, ::-1]
        top_n_probabilities = np.sort(predictions, axis=1)[:, -self.top_n:][:, ::-1]

        # Decode the top N predictions
        flattened = self.label_encoder.inverse_transform(top_n_indices.flatten())
        top_n_labels = flattened.reshape(top_n_indices.shape)

        # Prepare results
        results = []


        for i, (prediction, probability) in enumerate(zip(top_n_labels[0], top_n_probabilities[0]), 1):
                    results.append({
                    'rank': i,
                    'time_of_day': time_of_day,
                    'food_name': prediction,
                    'probability': str(probability)
                })
             
        for result in results:
            print(result)
        return results

   




# if __name__ == '__main__':
    # predictor = Predictor()

    # predictor.preprocess_data()
    # predictor.load_data()
    # # predictor.create_model()

    # USER = 'cac93e86-cad0-4030-839b-0415300528ab'

    # predictor.predict(USER, 'Morning')
    # predictor.predict(USER, 'Afternoon')
    # predictor.predict(USER, 'Evening')
    # predictor.predict(USER, 'Night')

    
    
        