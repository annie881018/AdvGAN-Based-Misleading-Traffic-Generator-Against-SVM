import joblib

def build_discriminator():
    # d_model_name = 'logistic_regression_1_6_1_20250512.pkl'
    d_model_name = 'svm_linear_1_6_1_20250429.pkl'
    model = joblib.load(d_model_name)
    return model