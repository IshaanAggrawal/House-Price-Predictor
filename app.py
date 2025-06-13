from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb
import os

app = Flask(__name__)

# Load XGBoost model (saved using .save_model())
model = xgb.XGBRegressor()
model.load_model('houseprediction.json')  # Make sure you convert your .pkl to .json first

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        output = round(float(prediction[0]), 2)
        return render_template('index.html', prediction_text=f'Predicted price: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # needed for Render
    app.run(debug=True, host='0.0.0.0', port=port)
