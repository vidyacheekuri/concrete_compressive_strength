#!/usr/bin/env python
"""
Minimal loader to extract and test the model
"""
import warnings
warnings.filterwarnings('ignore')

# Import in specific order
import numpy as np
print(f"NumPy version: {np.__version__}")

import pandas as pd
print(f"Pandas version: {pd.__version__}")

# Try different loading methods
def load_model_safely(filepath):
    """Try multiple methods to load the model"""
    
    # Method 1: Standard joblib
    try:
        import joblib
        print("Trying joblib...")
        model = joblib.load(filepath)
        print("✅ Joblib worked!")
        return model
    except Exception as e:
        print(f"❌ Joblib failed: {e}")
    
    # Method 2: Pickle with different protocols
    try:
        import pickle
        print("Trying pickle...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("✅ Pickle worked!")
        return model
    except Exception as e:
        print(f"❌ Pickle failed: {e}")
    
    # Method 3: NumPy with allow_pickle
    try:
        print("Trying numpy load...")
        model = np.load(filepath, allow_pickle=True)
        if hasattr(model, 'item'):
            model = model.item()
        print("✅ NumPy load worked!")
        return model
    except Exception as e:
        print(f"❌ NumPy load failed: {e}")
    
    # Method 4: Load with encoding
    try:
        import pickle
        print("Trying pickle with latin1 encoding...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print("✅ Pickle with encoding worked!")
        return model
    except Exception as e:
        print(f"❌ Pickle with encoding failed: {e}")
    
    return None

# Try to load the model
model_path = 'models/enhanced_catboost_model.joblib'
model_data = load_model_safely(model_path)

if model_data:
    print("\n✅ Model loaded successfully!")
    print(f"Model keys: {list(model_data.keys())[:5]}...")  # Show first 5 keys
    
    # Test prediction with your specific values
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
    print("\n✅ Ready to test predictions!")
else:
    print("\n❌ Could not load model with any method")
    print("\nTry reinstalling packages in this order:")
    print("pip uninstall numpy pandas scikit-learn catboost joblib -y")
    print("pip install numpy==2.0.2")
    print("pip install pandas scikit-learn catboost joblib")