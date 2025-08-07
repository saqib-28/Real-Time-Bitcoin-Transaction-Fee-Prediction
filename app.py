from flask import Flask, render_template, request
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

scaler_path = 'scaler123.pkl'
scaler = joblib.load(scaler_path)
print("Scaler loaded successfully!")

model_path = 'new_transaction_model.keras'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Make sure 'index.html' exists in the templates folder

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract values from the form
        size = float(request.form['size'])
        weight = float(request.form['weight'])
        input_count = float(request.form['input_count'])
        output_count = float(request.form['output_count'])
        fee_per_kb = float(request.form['fee_per_kb'])
        
        # Preprocess data (log transformation, scaling)
        log_size = np.log(size + 1)
        log_weight = np.log(weight + 1)
        log_input_count = np.log(input_count + 1)
        log_output_count = np.log(output_count + 1)
        log_fee_per_kb = np.log(fee_per_kb + 1)
        
        # Scale the features
        scaled_features = scaler.transform([[log_size, log_weight, log_input_count, log_output_count, log_fee_per_kb]])

        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Reverse log transformation on predicted fee
        predicted_fee = np.expm1(prediction[0][0]) + 500
       

        return render_template('index.html', prediction=predicted_fee)

if __name__ == "__main__":
    app.run(debug=True)
