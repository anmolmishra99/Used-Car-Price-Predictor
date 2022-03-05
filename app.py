from flask import Flask, render_template, request,redirect
# from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
car = pd.read_csv('car_price_clean.csv')
# pipe = pickle.load(open('C:\\Users\\User\\Desktop\\python files\\ML Project\\9. Second Hand Car Price Predict\\env\\models.pkl', 'rb'))
pipe = pickle.load(open('models.pkl', 'rb'))
@app.route("/")
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=pipe.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))


if __name__ == '__main__': 
    app.run()