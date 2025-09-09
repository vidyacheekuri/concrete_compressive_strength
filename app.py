''' import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import warnings
warnings.filterwarnings('ignore')

# --- Import the class definition from your train.py file ---
from train import EnhancedCatBoostPredictor

app = Flask(__name__, template_folder='templates')

# --- Load the pre-trained model file ---
MODEL_PATH = os.path.join('models', 'enhanced_catboost_model.joblib')
predictor = EnhancedCatBoostPredictor.load_model(MODEL_PATH)
print("✅ Predictor loaded successfully for the app.")

# --- Define Flask Routes ---
@app.route('/')
def home():
    # Assumes you have an index.html file in a 'templates' folder
    return render_template('index1.html') 

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data_from_form = request.get_json(force=True)
        input_data = {
            'Cement (component 1)(kg in a m^3 mixture)': float(data_from_form.get('Cement') or 0),
            'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': float(data_from_form.get('Blast_Furnace_Slag') or 0),
            'Fly Ash (component 3)(kg in a m^3 mixture)': float(data_from_form.get('Fly_Ash') or 0),
            'Water  (component 4)(kg in a m^3 mixture)': float(data_from_form.get('Water') or 0),
            'Superplasticizer (component 5)(kg in a m^3 mixture)': float(data_from_form.get('Superplasticizer') or 0),
            'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': float(data_from_form.get('Coarse_Aggregate') or 0),
            'Fine Aggregate (component 7)(kg in a m^3 mixture)': float(data_from_form.get('Fine_Aggregate') or 0),
            'Age (day)': float(data_from_form.get('Age') or 0)
        }
        input_df = pd.DataFrame([input_data])
        final_prediction = predictor.predict(input_df) 
        output = round(float(final_prediction[0]), 2)
        print(f"✅ Final Prediction: {output} MPa")
        return jsonify({'prediction': output})
    except Exception as e:
        print(f"🚨 An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) '''
    
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import warnings
warnings.filterwarnings('ignore')

# --- Import the class definition from your train.py file ---
from train import EnhancedCatBoostPredictor

app = Flask(__name__, template_folder='templates')

# --- Load the pre-trained model file ---
MODEL_PATH = os.path.join('models', 'enhanced_catboost_model.joblib')
predictor = EnhancedCatBoostPredictor.load_model(MODEL_PATH)
print("✅ Predictor loaded successfully for the app.")

# --- Define Flask Routes ---
@app.route('/')
def home():
    # This route will serve the professor's version
    return render_template('index1.html') 

@app.route('/personal')
def personal_home():
    # This NEW route will serve your personal version
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data_from_form = request.get_json(force=True)
        input_data = {
            'Cement (component 1)(kg in a m^3 mixture)': float(data_from_form.get('Cement') or 0),
            'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': float(data_from_form.get('Blast_Furnace_Slag') or 0),
            'Fly Ash (component 3)(kg in a m^3 mixture)': float(data_from_form.get('Fly_Ash') or 0),
            'Water  (component 4)(kg in a m^3 mixture)': float(data_from_form.get('Water') or 0),
            'Superplasticizer (component 5)(kg in a m^3 mixture)': float(data_from_form.get('Superplasticizer') or 0),
            'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': float(data_from_form.get('Coarse_Aggregate') or 0),
            'Fine Aggregate (component 7)(kg in a m^3 mixture)': float(data_from_form.get('Fine_Aggregate') or 0),
            'Age (day)': float(data_from_form.get('Age') or 0)
        }
        input_df = pd.DataFrame([input_data])
        final_prediction = predictor.predict(input_df) 
        output = round(float(final_prediction[0]), 2)
        print(f"✅ Final Prediction: {output} MPa")
        return jsonify({'prediction': output})
    except Exception as e:
        print(f"🚨 An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)