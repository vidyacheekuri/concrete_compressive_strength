# --- Import Necessary Libraries ---
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# --- Initialize the Flask App ---
app = Flask(__name__, template_folder='templates')

# --- The Complete and Correct EnhancedCatBoostPredictor Class (Copied from Notebook) ---
class EnhancedCatBoostPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def engineer_features(self, X):
        X_engineered = X.copy()
        cement = X['Cement (component 1)(kg in a m^3 mixture)']
        blast_slag = X['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']
        fly_ash = X['Fly Ash (component 3)(kg in a m^3 mixture)']
        water = X['Water  (component 4)(kg in a m^3 mixture)']
        superplast = X['Superplasticizer (component 5)(kg in a m^3 mixture)']
        coarse_agg = X['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']
        fine_agg = X['Fine Aggregate (component 7)(kg in a m^3 mixture)']
        age = X['Age (day)']
        
        X_engineered['water_cement_ratio'] = water / (cement + 1e-5)
        X_engineered['total_cementitious'] = cement + blast_slag + fly_ash
        X_engineered['water_cementitious_ratio'] = water / (X_engineered['total_cementitious'] + 1e-5)
        
        std_dev = X_engineered['water_cement_ratio'].std()
        if pd.isna(std_dev):
            std_dev = 0
        X_engineered['abnormal_mix_factor'] = np.abs((X_engineered['water_cement_ratio'] - X_engineered['water_cement_ratio'].mean()) / (std_dev + 1e-5))

        X_engineered['agg_cement_ratio'] = (coarse_agg + fine_agg) / (cement + 1e-5)
        X_engineered['fine_coarse_ratio'] = fine_agg / (coarse_agg + 1e-5)
        X_engineered['cementitious_superplast_ratio'] = X_engineered['total_cementitious'] / (superplast + 1e-5)
        X_engineered['cement_binder_ratio'] = cement / (X_engineered['total_cementitious'] + 1e-5)
        X_engineered['log_age'] = np.log1p(age)
        X_engineered['sqrt_age'] = np.sqrt(age)
        X_engineered['age_28d_ratio'] = age / 28.0
        X_engineered['paste_volume'] = (cement / 3.15 + blast_slag / 2.9 + fly_ash / 2.3 + water) / ((cement / 3.15 + blast_slag / 2.9 + fly_ash / 2.3 + water + coarse_agg / 2.7 + fine_agg / 2.6) + 1e-5)
        X_engineered['slump_indicator'] = water + 10 * superplast
        X_engineered['flow_indicator'] = X_engineered['slump_indicator'] / X_engineered['total_cementitious']
        X_engineered['maturity_index'] = age * (1 - np.exp(-0.05 * age))
        X_engineered['supplementary_fraction'] = (blast_slag + fly_ash) / (X_engineered['total_cementitious'] + 1e-5)
        X_engineered['early_age_factor'] = np.where(age < 7, (7 - age) / 7, 0)
        X_engineered['very_early_strength'] = age**0.5 * cement
        X_engineered['early_hydration_rate'] = np.where(age < 7, cement / (age + 0.5), 0)
        X_engineered['late_age_factor'] = np.where(age > 28, np.log1p(age - 28) / 4, 0)
        X_engineered['very_low_correction'] = np.where(X_engineered['total_cementitious'] < X_engineered['total_cementitious'].mean(), -0.05 * X_engineered['water_cementitious_ratio'], 0)
        X_engineered['high_correction'] = np.where(X_engineered['total_cementitious'] > X_engineered['total_cementitious'].mean() * 1.2, 0.05 * X_engineered['cement_binder_ratio'], 0)
        X_engineered['medium_correction'] = np.where((X_engineered['total_cementitious'] >= 350) & (X_engineered['total_cementitious'] <= 450) & (X_engineered['water_cement_ratio'] <= 0.5), -0.1 * X_engineered['total_cementitious'], 0)
        X_engineered['water_excess_indicator'] = np.where(X_engineered['water_cement_ratio'] > 0.6, X_engineered['water_cement_ratio'] - 0.6, 0)
        return X_engineered

    @classmethod
    def load_model(cls, filepath):
        model_data = joblib.load(filepath)
        predictor = cls(random_state=model_data.get('random_state', 42))
        for key, value in model_data.items():
            setattr(predictor, key, value)
        if hasattr(predictor, 'meta_learner') and hasattr(predictor, '_create_meta_feature_generator'):
            predictor._create_meta_feature_generator()
        return predictor

    def _create_meta_feature_generator(self):
        def generate_meta_features(self, X):
            meta_features = []; deep_preds = self.deep_catboost.predict(X); meta_features.append(deep_preds)
            if hasattr(self, 'range_models'):
                for model in self.range_models.values(): meta_features.append(model.predict(X))
            if hasattr(self, 'boundary_models'):
                for model in self.boundary_models.values(): meta_features.append(model.predict(X))
            if hasattr(self, 'age_models'):
                for model in self.age_models.values(): meta_features.append(model.predict(X))
            if hasattr(self, 'very_low_specialized_models'):
                for model in self.very_low_specialized_models.values(): meta_features.append(model.predict(X))
            if hasattr(self, 'medium_bias_model'):
                bias_corrected_preds = deep_preds.copy(); medium_mask = (deep_preds >= 40) & (deep_preds < 60)
                if np.any(medium_mask):
                    medium_indices = np.where(medium_mask)[0]; X_medium = X.iloc[medium_indices]
                    if not X_medium.empty:
                        bias_predictions = self.medium_bias_model.predict(X_medium)
                        for idx, i in enumerate(medium_indices): bias_corrected_preds[i] -= bias_predictions[idx] * 0.7
                meta_features.append(bias_corrected_preds)
            meta_features_array = np.column_stack(meta_features); estimated_ranges = pd.cut(deep_preds, bins=self.strength_bins, labels=self.strength_labels); range_indicators = pd.get_dummies(pd.Series(estimated_ranges)).reindex(columns=self.strength_labels, fill_value=0).values; final_meta_features = np.column_stack([meta_features_array, range_indicators, X.values])
            expected_cols = len(self.meta_feature_names)
            if final_meta_features.shape[1] < expected_cols:
                padding = np.zeros((final_meta_features.shape[0], expected_cols - final_meta_features.shape[1])); final_meta_features = np.hstack([final_meta_features, padding])
            return pd.DataFrame(final_meta_features, columns=self.meta_feature_names)
        self.generate_meta_features = generate_meta_features.__get__(self, self.__class__)
    
    def detect_and_correct_outliers(self, X, predictions):
        corrected_predictions = predictions.copy()
        if 'water_cement_ratio' in X.columns and 'abnormal_mix_factor' in X.columns:
            wcr = X['water_cement_ratio']; abnormal_factor = X['abnormal_mix_factor']; wcr_array = np.array(wcr); abnormal_factor_array = np.array(abnormal_factor)
            wcr_high = wcr_array > np.quantile(wcr_array, 0.95); wcr_low = wcr_array < np.quantile(wcr_array, 0.05); abnormal_high = abnormal_factor_array > 2.0
            potential_outliers = wcr_high | wcr_low | abnormal_high; outlier_indices = np.where(potential_outliers)[0]
            if len(outlier_indices) > 0:
                for i in outlier_indices:
                    pred_value = predictions[i]
                    if pred_value < 20: strength_range = 'very_low'
                    elif pred_value < 40: strength_range = 'low'
                    elif pred_value < 60: strength_range = 'medium'
                    else: strength_range = 'high'
                    if hasattr(self, 'range_models') and strength_range in self.range_models:
                        range_pred = self.range_models[strength_range].predict(X.iloc[[i]])[0]
                        corrected_predictions[i] = 0.3 * predictions[i] + 0.7 * range_pred
        return corrected_predictions

    def predict(self, X_new):
        if not hasattr(self, 'meta_learner'): raise ValueError("Meta-learner is not available in the loaded model.")
        X_engineered = self.engineer_features(X_new)
        X_engineered = X_engineered.reindex(columns=self.all_features, fill_value=0)
        X_scaled = self.scaler.transform(X_engineered)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.all_features)
        
        meta_features = self.generate_meta_features(X_scaled_df)
        predictions = self.meta_learner.predict(meta_features)
        
        # This is the full post-processing logic that was missing before
        predictions = self.detect_and_correct_outliers(X_scaled_df, predictions)
        final_predictions = []
        for i, pred in enumerate(predictions):
            if pred < 20:
                if hasattr(self, 'very_low_specialized_models'):
                    if pred < 15 and 'ultra_low' in self.very_low_specialized_models:
                        specialized_pred = self.very_low_specialized_models['ultra_low'].predict(X_scaled_df.iloc[[i]])[0]
                        pred = 0.4 * pred + 0.6 * specialized_pred
                    elif pred >= 15 and 'mid_low' in self.very_low_specialized_models:
                        specialized_pred = self.very_low_specialized_models['mid_low'].predict(X_scaled_df.iloc[[i]])[0]
                        pred = 0.4 * pred + 0.6 * specialized_pred
            elif pred < 60:
                if hasattr(self, 'medium_bias_model'):
                    estimated_bias = self.medium_bias_model.predict(X_scaled_df.iloc[[i]])[0]
                    if estimated_bias > 5:
                        pred -= estimated_bias * 0.7
            elif pred >= 60:
                pred *= 1.05
            final_predictions.append(pred)
        return np.array(final_predictions)

# --- Load the Entire Predictor Object at Startup ---
predictor = None
try:
    model_path = os.path.join('models', 'enhanced_catboost_model.joblib')
    predictor = EnhancedCatBoostPredictor.load_model(model_path)
    print("✅ Enhanced CatBoost predictor loaded successfully!")
except Exception as e:
    print(f"🚨 Error loading predictor: {e}")

# --- Define Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if predictor is None: return jsonify({'error': 'Model not loaded.'}), 500
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
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
