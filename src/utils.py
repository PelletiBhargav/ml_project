import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        best_model = None
        best_score = -1

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1
            )

            gs.fit(x_train, y_train)

            y_test_pred = gs.best_estimator_.predict(x_test)
            score = r2_score(y_test, y_test_pred)

            report[model_name] = score

            if score > best_score:
                best_score = score
                best_model = gs.best_estimator_

        return report, best_model

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)