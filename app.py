from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

#initialize the flask app
app = Flask(__name__)

#load the trained model and the encoder
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('labelEncoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

#route to render the form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)
        prediction_encoded = model.predict(features_array)
        prediction = encoder.inverse_transform(prediction_encoded)
        return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)