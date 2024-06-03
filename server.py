import os
from flask import Flask, request, jsonify
from predictor import Predictor

# Initialize the Flask application
app = Flask(__name__)

# Initialize the predictor
predictor = Predictor()

@app.route('/predict', methods=['GET'])

def predict():
    data = request.get_json()

    user_id = data.get('user_id')
    time_of_day = data.get('time_of_day')
    
    if not user_id or not time_of_day:
        return jsonify({'error': 'Please provide both user_id and time_of_day'}), 400

    try:
        predictions = predictor.predict(user_id, time_of_day)
        return jsonify(predictions)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if(not os.path.exists('data')):
        os.makedirs('data')
    
    # check All_users.csv exists
    if(not os.path.exists('data/All_users.csv')):
        predictor.fetch_data()

    if(not os.path.exists('models')):
        os.makedirs('models')

    predictor.preprocess_data()
    predictor.load_data()

    if not predictor.model_exists():
        print('Model Does not exist. Creating a new one.')
        predictor.create_model()
    app.run(debug=True, host='0.0.0.0', port=2000)
