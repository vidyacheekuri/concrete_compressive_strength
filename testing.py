from train import EnhancedCatBoostPredictor
import pandas as pd

# 1. Initialize the predictor
predictor = EnhancedCatBoostPredictor(random_state=42)

# 2. Load and preprocess the data
print("--- Loading and Preprocessing Data ---")
predictor.load_and_preprocess("Concrete_Data.xls")

# 3. Train the base model
print("\\n--- Training Base CatBoost Model ---")
predictor.train_deep_catboost()

# 4. Train all specialized models
print("\\n--- Training Specialized Models ---")
predictor.train_range_specific_models()
predictor.train_boundary_models()
predictor.train_age_specific_models()
predictor.train_very_low_specialized_models()
predictor.train_medium_bias_correction()

# 5. Train the final meta-learner
print("\\n--- Training Meta-Learner ---")
predictor.train_meta_learner()

# 6. Save the complete, trained model
print("\\n--- Saving Model ---")
predictor.save_model("models/enhanced_catboost_model.joblib")

print("\\n✅ Training pipeline complete. Model is saved and ready for deployment.")

# Save the model
predictor.save_model("models/enhanced_catboost_model.joblib")

# Test it
test_data = {
    'Cement (component 1)(kg in a m^3 mixture)': 332.0,
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 142.5,
    'Fly Ash (component 3)(kg in a m^3 mixture)': 0.0,
    'Water  (component 4)(kg in a m^3 mixture)': 228.0,
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 0,
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 932.0,
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 594.0,
    'Age (day)': 270.0
}
input_df = pd.DataFrame([test_data])
prediction = predictor.predict(input_df)
print(f"Prediction: {prediction[0]:.2f} MPa")