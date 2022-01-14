# Importing libs and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle

from utils.fertilizers import fertilizer_dic

# -------------------------LOADING THE TRAINED MODELS -------------------------


# Loading crop recommendation model

model_path = 'models/randomf.pkl'
model = pickle.load(open(model_path,'rb'))


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

#render try again
@ app.route('/na')
def tryagain():
    title = 'page not found'
    return render_template('try-again.html', title=title)

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
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/my_fertilizers.csv')

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


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)


'''

<center>
    <p>
        The P value of your soil is <b>Low</b><br>
        <i>Please consider the following suggestions:</i>
    </p>
</center>
<ol>
    <li><b>Bone Meal</b>- a fast acting source that is made from ground animal bones which is rich in phosphorous.</li>

    <li>Rock phosphate – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.</li>

    <li>Phosphorus Fertilizers – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).</li>
</ol>  
'''