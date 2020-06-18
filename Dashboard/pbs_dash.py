from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

df = pd.read_csv('online_shoppers_intention.csv')

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/dataset', methods=("POST", "GET"))
def dataset():
    return render_template('dataset.html',  
        tables=[df[:500].to_html(classes='data')]
        )

@app.route('/dataviz', methods=("POST", "GET"))
def dataviz():
    return render_template('dataviz.html')

@app.route('/feature', methods=("POST", "GET"))
def feature():
    return render_template('feature.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input = request.form
        robust = scaler.transform([[float(input['Administrative_Duration']),
                           float(input['Informational_Duration']),
                           float(input['ProductRelated_Duration']),
                           float(input['BounceRates']),
                           float(input['PageValues']),
                           float(input['SpecialDay'])
                           ]])
        sample = robust.ravel().tolist()
        sample.extend([float(input['Month_Mar']),
                       float(input['Month_May']),
                       float(input['Month_Nov']),
                       float(input['OperatingSystems_3']),
                       float(input['TrafficType_1']),
                       float(input['TrafficType_2']),
                       float(input['TrafficType_3']),
                       float(input['TrafficType_13']),
                       float(input['VisitorType_New_Visitor']),
                       float(input['VisitorType_Returning_Visitor'])
                       ])
        prediction = model.predict(np.array([sample]))
        if prediction == 0:
            result = "This visitor is predicted not to purchase! Let's give them a dicount code to seal the deal!"
        else:
            result = "This visitor is predicted to purchase! Great!"
        return render_template('predict.html', data = input, pred = result)

if __name__ == '__main__':
    model = joblib.load('my_model')
    scaler = joblib.load('my_scaler')
    app.run(debug=True)