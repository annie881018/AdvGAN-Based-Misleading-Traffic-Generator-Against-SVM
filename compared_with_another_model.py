import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import sklearn.metrics as skm

def test_xgb_model():
    model_name = 'xgb_20250528.pkl'
    model = joblib.load(model_name)
    return model

def test_rf_model():
    model_name = 'rf_20250528.pkl'
    model = joblib.load(model_name)
    return model

def test_svm_model():
    model_name = 'svm_linear_1_6_1_20250429.pkl'
    model = joblib.load(model_name)
    return model

def Measurements(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print(f"Score: {score}")
    
    # 測量ML數據
    predictions = model.predict(X_test)
    print(model.predict_proba(X_test))
    print(predictions)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("Test Result:")
    print(cm)
    print(skm.classification_report(y_test, predictions))
    tn, fp, fn, tp = cm.ravel().tolist()
    return fp / (tn + fp)

def traversal_test(filtered_dir):
    xgb = test_xgb_model()
    rf = test_rf_model()
    svm = test_svm_model()
    results = []
    for f in os.listdir(filtered_dir):
        if "Compared_with_another_model" in f:
            continue
        print("-")
        print(f'File: {f}')
        data = pd.read_csv(os.path.join(filtered_dir, f), header=0)
        data = data.drop(index=0)
        if len(data) == 0:
            print(f'Empty Dataframe')
        else:
            print(f'Total # of label 0 data: {len(data)}')
            X_test = data.drop(columns=["predict", "gen_loss", "disc_loss"])
            y_test = data["predict"]
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)
            # X_test = X_test.round(0).astype(int)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            output_path = os.path.join(filtered_dir, "Compared_with_another_model.csv")
             
            print(f'Test on SVM...')
            success_rate = Measurements(svm, X_test, y_test)
            results.append(["SVM", len(data), f'{success_rate*100:.2f}%', f])
            
            print(f'Test on XGB...')
            success_rate = Measurements(xgb, X_test, y_test)
            columns = ["Model", "# label 0 data", "Success Rate", "File Name"]
            results.append(["XGB", len(data), f'{success_rate*100:.2f}%', f])
            
            print(f'Test on Random Forest...')
            success_rate = Measurements(rf, X_test, y_test)
            results.append(["RF", len(data), f'{success_rate*100:.2f}%', f])
            
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, mode='a', index=False, encoding="utf-8")
date = ["0608(g_loss_perturb改成zero-like)", "0609", "0610"]
root = "result"

for d in date:
    date_path = os.path.join(root, d)
    filtered_path = os.path.join(date_path, "filtered_data")
    # print(filtered_path)
    print("-" * 10)
    print(f'Date: {d}')
    traversal_test(filtered_path)