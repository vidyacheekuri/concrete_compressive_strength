#!/usr/bin/env bash
set -e

echo "Installing packages with NumPy 2.x compatibility fix..."

# Install everything except catboost first
pip install flask==3.0.3 gunicorn==22.0.0 numpy==2.0.2 pandas==2.2.2 scikit-learn==1.5.2 joblib==1.4.2 xlrd==2.0.1 openpyxl==3.1.5

# Force install catboost, ignoring dependency conflicts
pip install catboost==1.2.7 --force-reinstall --no-deps

# Install catboost dependencies that were skipped
pip install graphviz matplotlib plotly scipy six

echo "Build completed successfully!"