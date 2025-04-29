import joblib

def build_discriminator():
    d_model_name = 'svm_linear_1_6_1.pkl'
    model = joblib.load(d_model_name)
    return model