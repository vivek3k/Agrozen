# Importing essential libraries and modules
from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9

# ==================== MODEL LOADING ==========================
# Disease classification model
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                   'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                   'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                   'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu')))
disease_model.eval()

# Crop recommendation model
crop_recommendation_model = pickle.load(open('models/RandomForest.pkl', 'rb'))

# ================ WEATHERSTACK API CONFIG ===================
API_KEY = "4b30b66ecb57e68b429a5bf5b555ff8b"
WEATHERSTACK_URL = "http://api.weatherstack.com/current"

def weather_fetch(city_name):
    params = {
        'access_key': API_KEY,
        'query': city_name
    }
    try:
        response = requests.get(WEATHERSTACK_URL, params=params)
        data = response.json()

        if "current" in data:
            temperature = data["current"]["temperature"]
            humidity = data["current"]["humidity"]
            return temperature, humidity
        else:
            return None
    except Exception as e:
        print("Error fetching weather:", e)
        return None

# ============== IMAGE PREDICTION FUNCTION ====================
def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# ===================== FLASK APP =============================
app = Flask(__name__)

@app.route('/')
def home():
    city = "Hyderabad"  # You can later make this dynamic
    weather_data = weather_fetch(city)

    weather = {}
    if weather_data:
        temperature, humidity = weather_data
        weather = {
            'location': city,
            'temperature': temperature,
            'humidity': humidity
        }

    return render_template('index.html', title='AgroZen - Home', weather=weather)

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='AgroZen - Crop Recommendation')

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='AgroZen - Fertilizer Suggestion')

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AgroZen - Crop Recommendation'
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        weather = weather_fetch(city)
        if weather:
            temperature, humidity = weather
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)[0]
            return render_template('crop-result.html', prediction=prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgroZen - Fertilizer Suggestion'
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_val = temp[max(temp.keys())]

    if max_val == "N":
        key = 'NHigh' if n < 0 else 'Nlow'
    elif max_val == "P":
        key = 'PHigh' if p < 0 else 'Plow'
    else:
        key = 'KHigh' if k < 0 else 'Klow'

    recommendation = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=recommendation, title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgroZen - Disease Detection'
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            explanation = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=explanation, title=title)
        except:
            return render_template('disease.html', title=title)
    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
