import os
import joblib

MODELS_DIR = 'Models'
EXTENSION = '.pkl'

def save_model(model, filename):
    if not filename.endswith(EXTENSION):
        filename += EXTENSION

    filepath = os.path.join(MODELS_DIR, filename)

    if os.path.exists(filepath):
        raise FileExistsError(f"The file '{filename}' already exists. Choose a different name or delete it.")

    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filename):
    if not filename.endswith(EXTENSION):
        filename += EXTENSION

    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    model = joblib.load(filepath)
    return model

""" How to use: 
ml.save_model(model, 'linear_regression_model')
model_whatever = ml.load_model('linear_regression_model')
"""