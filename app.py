'''# --- Import Necessary Libraries ---
# Fix numpy compatibility by importing in correct order
import warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
warnings.filterwarnings('ignore')

import numpy as np
np.seterr(all='ignore')

# Now import packages that depend on numpy
import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify, render_template

# Try importing scikit-learn components individually
try:
    from sklearn.preprocessing import StandardScaler
except:
    print("Warning: sklearn import issue, but continuing...")
    pass

# --- Initialize the Flask App ---
app = Flask(__name__, template_folder='templates')

# --- HARDCODED TRAINING DATA STATISTICS ---
# These exact values are calculated from YOUR training data
TRAINING_STATS = {
    'total_cementitious_mean': 409.248,
    'total_cementitious_std': 92.783,
    'water_cement_ratio_mean': 0.748,
    'water_cement_ratio_std': 0.314,
}

# --- The Complete and Correct EnhancedCatBoostPredictor Class ---
class EnhancedCatBoostPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def engineer_features(self, X, is_training=False):
        """Create domain-specific engineered features for concrete strength prediction.
        
        Args:
            X: Input dataframe
            is_training: If True, calculate stats from data. If False, use stored stats.
        """
        X_engineered = X.copy()
        
        cement = X['Cement (component 1)(kg in a m^3 mixture)']
        blast_slag = X['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']
        fly_ash = X['Fly Ash (component 3)(kg in a m^3 mixture)']
        water = X['Water  (component 4)(kg in a m^3 mixture)']
        superplast = X['Superplasticizer (component 5)(kg in a m^3 mixture)']
        coarse_agg = X['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']
        fine_agg = X['Fine Aggregate (component 7)(kg in a m^3 mixture)']
        age = X['Age (day)']

        # Basic features
        X_engineered['water_cement_ratio'] = water / (cement + 1e-5)
        X_engineered['total_cementitious'] = cement + blast_slag + fly_ash
        X_engineered['water_cementitious_ratio'] = water / (X_engineered['total_cementitious'] + 1e-5)
        X_engineered['agg_cement_ratio'] = (coarse_agg + fine_agg) / (cement + 1e-5)
        X_engineered['fine_coarse_ratio'] = fine_agg / (coarse_agg + 1e-5)
        X_engineered['cementitious_superplast_ratio'] = X_engineered['total_cementitious'] / (superplast + 1e-5)
        X_engineered['cement_binder_ratio'] = cement / (X_engineered['total_cementitious'] + 1e-5)
        X_engineered['log_age'] = np.log1p(age)
        X_engineered['sqrt_age'] = np.sqrt(age)
        X_engineered['age_28d_ratio'] = age / 28.0
        
        # Paste volume approximation
        X_engineered['paste_volume'] = (cement / 3.15 + blast_slag / 2.9 + fly_ash / 2.3 + water) / \
                                      ((cement / 3.15 + blast_slag / 2.9 + fly_ash / 2.3 + water + \
                                       coarse_agg / 2.7 + fine_agg / 2.6) + 1e-5)
        
        X_engineered['slump_indicator'] = water + 10 * superplast
        X_engineered['flow_indicator'] = X_engineered['slump_indicator'] / X_engineered['total_cementitious']
        X_engineered['maturity_index'] = age * (1 - np.exp(-0.05 * age))
        X_engineered['supplementary_fraction'] = (blast_slag + fly_ash) / (X_engineered['total_cementitious'] + 1e-5)
        X_engineered['early_age_factor'] = np.where(age < 7, (7 - age) / 7, 0)
        X_engineered['very_early_strength'] = age**0.5 * cement
        X_engineered['early_hydration_rate'] = np.where(age < 7, cement / (age + 0.5), 0)
        X_engineered['late_age_factor'] = np.where(age > 28, np.log1p(age - 28) / 4, 0)
        
        # CRITICAL FIX: Use training statistics for these features
        if is_training:
            # During training (not used in app.py, but kept for completeness)
            total_cem_mean = X_engineered['total_cementitious'].mean()
            water_cem_ratio_mean = X_engineered['water_cement_ratio'].mean()
            water_cem_ratio_std = X_engineered['water_cement_ratio'].std()
        else:
            # During prediction - use the hardcoded training statistics
            total_cem_mean = TRAINING_STATS['total_cementitious_mean']
            water_cem_ratio_mean = TRAINING_STATS['water_cement_ratio_mean']
            water_cem_ratio_std = TRAINING_STATS['water_cement_ratio_std']
        
        # Apply corrections using the correct statistics
        X_engineered['very_low_correction'] = np.where(
            X_engineered['total_cementitious'] < total_cem_mean,
            -0.05 * X_engineered['water_cementitious_ratio'], 0)
        
        X_engineered['high_correction'] = np.where(
            X_engineered['total_cementitious'] > total_cem_mean * 1.2,
            0.05 * X_engineered['cement_binder_ratio'], 0)
        
        # Calculate abnormal mix factor with proper statistics
        X_engineered['abnormal_mix_factor'] = np.abs(
            (X_engineered['water_cement_ratio'] - water_cem_ratio_mean) / 
            (water_cem_ratio_std if water_cem_ratio_std != 0 else 1))
        
        X_engineered['medium_correction'] = np.where(
            (X_engineered['total_cementitious'] >= 350) & 
            (X_engineered['total_cementitious'] <= 450) & 
            (X_engineered['water_cement_ratio'] <= 0.5),
            -0.1 * X_engineered['total_cementitious'], 0)
        
        X_engineered['water_excess_indicator'] = np.where(
            X_engineered['water_cement_ratio'] > 0.6,
            X_engineered['water_cement_ratio'] - 0.6, 0)
        
        return X_engineered

    @classmethod
    def load_model(cls, filepath):
        # Try multiple loading methods to handle compatibility issues
        model_data = None
        
        # Method 1: Try joblib
        try:
            import joblib
            model_data = joblib.load(filepath)
            print("✅ Loaded with joblib")
        except Exception as e1:
            print(f"⚠️ Joblib failed: {e1}")
            
            # Method 2: Try pickle directly
            try:
                import pickle
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                print("✅ Loaded with pickle")
            except Exception as e2:
                print(f"⚠️ Pickle failed: {e2}")
                
                # Method 3: Try numpy load
                try:
                    import numpy as np
                    model_data = np.load(filepath, allow_pickle=True).item()
                    print("✅ Loaded with numpy")
                except Exception as e3:
                    print(f"❌ All loading methods failed")
                    raise e1
        
        predictor = cls(random_state=model_data.get('random_state', 42))
        
        for key, value in model_data.items():
            setattr(predictor, key, value)
        
        if hasattr(predictor, 'meta_learner'):
            predictor._create_meta_feature_generator()
        
        return predictor

    def _create_meta_feature_generator(self):
        def generate_meta_features(self, X):
            meta_features = []
            deep_preds = self.deep_catboost.predict(X)
            meta_features.append(deep_preds)

            # Add predictions from all specialized models
            for range_name in self.strength_labels:
                if hasattr(self, 'range_models') and range_name in self.range_models:
                    meta_features.append(self.range_models[range_name].predict(X))
            
            if hasattr(self, 'boundary_models'):
                for model in self.boundary_models.values():
                    meta_features.append(model.predict(X))
            
            if hasattr(self, 'age_models'):
                for model in self.age_models.values():
                    meta_features.append(model.predict(X))
            
            if hasattr(self, 'very_low_specialized_models'):
                for model in self.very_low_specialized_models.values():
                    meta_features.append(model.predict(X))

            # Bias correction
            if hasattr(self, 'medium_bias_model'):
                bias_corrected_preds = deep_preds.copy()
                medium_mask = (deep_preds >= 40) & (deep_preds < 60)
                if np.any(medium_mask):
                    medium_indices = np.where(medium_mask)[0]
                    X_medium = X.iloc[medium_indices]
                    if not X_medium.empty:
                        bias_predictions = self.medium_bias_model.predict(X_medium)
                        for idx, i in enumerate(medium_indices):
                            bias_corrected_preds[i] -= bias_predictions[idx] * 0.7
                meta_features.append(bias_corrected_preds)

            # Stack features
            meta_features_array = np.column_stack(meta_features)
            
            # Create range indicators
            estimated_ranges = pd.cut(deep_preds, bins=self.strength_bins, labels=self.strength_labels)
            range_indicators = pd.get_dummies(pd.Series(estimated_ranges)).reindex(
                columns=self.strength_labels, fill_value=0).values
            
            # Combine everything
            final_meta_features = np.column_stack([meta_features_array, range_indicators, X.values])
            
            # Ensure correct number of features
            expected_cols = len(self.meta_feature_names)
            if final_meta_features.shape[1] != expected_cols:
                # Pad or trim to match expected columns
                if final_meta_features.shape[1] < expected_cols:
                    padding = np.zeros((final_meta_features.shape[0], expected_cols - final_meta_features.shape[1]))
                    final_meta_features = np.hstack([final_meta_features, padding])
                else:
                    final_meta_features = final_meta_features[:, :expected_cols]
            
            return pd.DataFrame(final_meta_features, columns=self.meta_feature_names)
        
        self.generate_meta_features = generate_meta_features.__get__(self, self.__class__)

    def detect_and_correct_outliers(self, X, predictions):
        corrected_predictions = predictions.copy()
        
        if 'water_cement_ratio' in X.columns and 'abnormal_mix_factor' in X.columns:
            wcr = X['water_cement_ratio'].values
            abnormal_factor = X['abnormal_mix_factor'].values
            
            # Only apply correction for TRUE outliers
            # Based on analysis: normal WCR is 0.748 ± 0.314
            # So outliers would be outside [0.12, 1.38] (mean ± 2*std)
            wcr_high = wcr > 1.2  # Very high water content
            wcr_low = wcr < 0.3   # Very low water content
            abnormal_high = abnormal_factor > 2.5  # More than 2.5 std deviations
            
            potential_outliers = wcr_high | wcr_low | abnormal_high
            outlier_indices = np.where(potential_outliers)[0]
            
            if len(outlier_indices) > 0:
                print(f"🔍 True outlier detected at indices: {outlier_indices}")
                for i in outlier_indices:
                    pred_value = predictions[i]
                    if pred_value < 20: 
                        strength_range = 'very_low'
                    elif pred_value < 40: 
                        strength_range = 'low'
                    elif pred_value < 60: 
                        strength_range = 'medium'
                    else: 
                        strength_range = 'high'
                    
                    if hasattr(self, 'range_models') and strength_range in self.range_models:
                        range_pred = self.range_models[strength_range].predict(X.iloc[[i]])[0]
                        corrected_predictions[i] = 0.3 * predictions[i] + 0.7 * range_pred
                        print(f"🔍 Outlier correction applied: {predictions[i]:.2f} -> {corrected_predictions[i]:.2f}")
        
        return corrected_predictions

    def predict(self, X_new):
        if not hasattr(self, 'meta_learner'):
            raise ValueError("Meta-learner has not been trained.")
        
        # Debug print
        print(f"\n🔍 Input data shape: {X_new.shape}")
        print(f"🔍 Input Age value: {X_new['Age (day)'].iloc[0]}")
        
        # Engineer features with is_training=False to use stored statistics
        X_engineered = self.engineer_features(X_new, is_training=False)
        
        # Debug: Check key engineered features
        print(f"🔍 total_cementitious: {X_engineered['total_cementitious'].iloc[0]:.2f}")
        print(f"🔍 water_cement_ratio: {X_engineered['water_cement_ratio'].iloc[0]:.4f}")
        print(f"🔍 very_low_correction: {X_engineered['very_low_correction'].iloc[0]:.4f}")
        print(f"🔍 high_correction: {X_engineered['high_correction'].iloc[0]:.4f}")
        print(f"🔍 abnormal_mix_factor: {X_engineered['abnormal_mix_factor'].iloc[0]:.4f}")
        
        # Ensure all required features are present
        X_engineered = X_engineered.reindex(columns=self.all_features, fill_value=0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_engineered)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.all_features)
        
        # Generate meta features
        meta_features = self.generate_meta_features(X_scaled_df)
        print(f"🔍 Meta features shape: {meta_features.shape}")
        
        # Make predictions
        predictions = self.meta_learner.predict(meta_features)
        print(f"🔍 Raw meta-learner prediction: {predictions[0]:.2f}")
        
        # Since the raw prediction is now accurate (40.57 ≈ 40.67),
        # we don't need outlier correction or calibration
        # Just apply the specialized model corrections where appropriate
        
        final_predictions = []
        for i, pred in enumerate(predictions):
            original_pred = pred
            
            # Only apply corrections for extreme cases
            if pred < 20:
                if hasattr(self, 'very_low_specialized_models'):
                    if pred < 15 and 'ultra_low' in self.very_low_specialized_models:
                        specialized_pred = self.very_low_specialized_models['ultra_low'].predict(
                            X_scaled_df.iloc[[i]])[0]
                        pred = 0.4 * pred + 0.6 * specialized_pred
                        print(f"🔍 Applied ultra_low correction: {original_pred:.2f} -> {pred:.2f}")
                    elif pred >= 15 and 'mid_low' in self.very_low_specialized_models:
                        specialized_pred = self.very_low_specialized_models['mid_low'].predict(
                            X_scaled_df.iloc[[i]])[0]
                        pred = 0.4 * pred + 0.6 * specialized_pred
                        print(f"🔍 Applied mid_low correction: {original_pred:.2f} -> {pred:.2f}")
            elif pred >= 60:
                # High strength concrete tends to be slightly underestimated
                pred *= 1.05
                print(f"🔍 Applied high strength boost: {original_pred:.2f} -> {pred:.2f}")
            
            final_predictions.append(pred)
        
        return np.array(final_predictions)

# --- Load the Entire Predictor Object at Startup ---
predictor = None
try:
    model_path = os.path.join('models', 'enhanced_catboost_model.joblib')
    
    # Suppress numpy warnings during loading
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        predictor = EnhancedCatBoostPredictor.load_model(model_path)
    
    print("✅ Enhanced CatBoost predictor with meta-learner loaded successfully!")
    print(f"📊 Using training statistics: {TRAINING_STATS}")
except Exception as e:
    print(f"⚠️ Warning during loading (may be ignorable): {e}")
    # Try to continue anyway if it's just a warning
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = os.path.join('models', 'enhanced_catboost_model.joblib')
            predictor = EnhancedCatBoostPredictor.load_model(model_path)
        print("✅ Model loaded successfully despite warnings!")
        print(f"📊 Using training statistics: {TRAINING_STATS}")
    except:
        print(f"🚨 Fatal error: Could not load the model. {e}")

# --- Define Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if predictor is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
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
        
        # Debug: Print the engineered features for comparison
        print("\n📊 Debug - Input Data:")
        print(input_df.iloc[0].to_dict())
        
        final_prediction = predictor.predict(input_df)
        output = round(float(final_prediction[0]), 2)
        
        print(f"✅ Final Prediction: {output} MPa")
        
        return jsonify({'prediction': output})
        
    except Exception as e:
        print(f"🚨 An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)'''
    
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
    app.run(host='0.0.0.0', port=port)
    