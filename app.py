from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import google.generativeai as genai
import re
import os
import pandas as pd

#initialize the flask app
app = Flask(__name__)

#load the trained model and the encoder
diseasepred = tf.keras.models.load_model('models/plant_disease_cnn.h5')
with open('models/yield.pkl', 'rb') as file:
    predcropYield = pickle.load(file)
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('models/labelEncoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Tomato___Bacterial_spot', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

#Chatbot
genai.configure(api_key="AIzaSyC06QOO0Zlf2BvTJAr-Baovt1bGlXvoBjc")

def chat_with_gemini(input_text):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    response = chat.send_message(input_text)
    return response.text

# Define agriculture-related keywords or patterns
AGRICULTURE_KEYWORDS = [
    "crop", "soil", "fertilizer", "irrigation", "pest", "harvest",
    "farming", "yield", "weather", "agriculture", "plant", "cultivation"
]

def is_agriculture_related(message):
    """Check if the user's message is agriculture-related."""
    message = message.lower()
    for keyword in AGRICULTURE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", message):
            return True
    return False

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/home')
def landing():
    return render_template('home.html')
@app.route('/cropClass')
def cropClass():
    return render_template('crop.html')
@app.route('/predictYield')
def predictYield():
    return render_template('yield.html')
@app.route('/disease')
def diseaseDetection():
    return render_template('disease.html')
@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')

#route to render the form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)
        prediction_encoded = model.predict(features_array)
        prediction = encoder.inverse_transform(prediction_encoded)
        return render_template('crop.html', prediction_text=f'Prediction: {prediction[0]}')
    except Exception as e:
        return render_template('crop.html', prediction_text=f'Error: {str(e)}')
@app.route('/predYield', methods=['POST'])
def predYield():
    try:
        # Get form data
        Region = request.form['Region']
        Soil_Type = request.form['Soil_Type']
        Crop = request.form['Crop']
        Rainfall_mm = float(request.form['Rainfall_mm'])
        Temperature_Celsius = float(request.form['Temperature_Celsius'])
        Fertilizer_Used = request.form['Fertilizer_Used']
        Irrigation_Used = request.form['Irrigation_Used']
        Weather_Condition = request.form['Weather_Condition']
        Days_to_Harvest = float(request.form['Days_to_Harvest'])

        # Convert to DataFrame
        input_data = pd.DataFrame([[Region, Soil_Type, Crop, Rainfall_mm, Temperature_Celsius, Fertilizer_Used, Irrigation_Used, Weather_Condition, Days_to_Harvest]], columns=['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest'])
        for column in input_data.select_dtypes(include=['object']).columns:
            input_data[column] = encoder.fit_transform(input_data[column])
        # Predict
        prediction = predcropYield.predict(input_data)

        return render_template('yield.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/diseasePred', methods=['GET', 'POST'])
def diseasePred():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            img_array = preprocess_image(file_path)
            prediction = diseasepred.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_class_label = class_labels[predicted_class[0]]

            return render_template('disease.html', file_path=file_path, prediction=predicted_class_label)

    return render_template('disease.html')
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Check if the message is agriculture-related
    if not is_agriculture_related(user_message):
        return jsonify({"response": "I specialize in agriculture-related topics. Please ask me about farming, crops, soil, or pests."})

    # Generate response from OpenAI API
    ai_response = chat_with_gemini(user_message)
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run()