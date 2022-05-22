# Importing libs and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle

from io import BytesIO
from PIL import Image
import tensorflow as tf

from utils.fertilizers import fertilizer_dic
from utils.diseases import disease_dic

# -------------------------LOADING THE TRAINED MODELS -------------------------


# Loading crop recommendation model

model_path = 'models/randomf.pkl'
model = pickle.load(open(model_path,'rb'))


MODEL = tf.keras.models.load_model("models/1")


# Loading Weather data from OpenWeatherAPI

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.WEATHER_API_KEY
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    res = response.json()

    if res["cod"] != "404":
        y = res["main"]

        temperature = round((y["temp"] - 273.15), 2) #kelvin to Celcius
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page

@ app.route('/')
def home():
    title = 'Cropy - Home'
    return render_template('index.html', title=title)


# render crop recommendation form page

@ app.route('/crop_recommend')
def crop_recommend():
    title = 'Cropy - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page

@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Cropy - Fertilizer Recommendation'

    return render_template('fertilizer.html', title=title)

@ app.route('/joy')
def joy():
    title = 'joy'

    return render_template('disease-result.html', title=title)


# ===============================================================================================


# render crop recommendation result page

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Cropy - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            
            # data = np.array([[45, 65, 34, 16.95, 22, 5, 23]])
            my_prediction = model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:
            return render_template('try-again.html', title=title)


# render fertilizer recommendation result page

@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Cropy - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('data/my_fertilizers.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


# render disease result page

# disease names
CLASS_NAMES = [
'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    #np array <- pillow image <- binary stream <- bytes
    return image


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Cropy - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            file = file.read() #byte data
            image = read_file_as_image(file)

            image = tf.image.resize(image, [256,256]).numpy()

            image = np.expand_dims(image, 0) # 1d to 2d
            predictions = MODEL.predict(image)

            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = round(np.max(predictions[0])*100,2) # 0 because provided only one image

            data = {
                'confidence': float(confidence),
                'data': Markup(str(disease_dic[predicted_class]))
            }
            return render_template('disease-result.html', prediction=data, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True);
